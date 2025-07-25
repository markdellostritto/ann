// chem
#include "chem/units.hpp"
// pot
#include "torch/pot_lj_long.hpp"
// thread
#include "thread/dist.hpp"

namespace ptnl{

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotLJLong& pot){
	return out<<static_cast<const Pot&>(pot);
}
	
//==== member functions ====

void PotLJLong::read(Token& token){
	if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"PotLJLong::read(Token&):\n";
	static_cast<Pot&>(*this).read(token);
	if(!token.end()) prec_=std::atof(token.next().c_str());
	ksl_.prec()=prec_;
	ksl_.rc()=rc_;
}

void PotLJLong::coeff(Token& token){
	if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLJLong::coeff(Token&):\n";
	//coeff lj_cut type1 type2 eps sigma
	const int t1=std::atoi(token.next().c_str())-1;
	const int t2=std::atoi(token.next().c_str())-1;
	const double eps=std::atof(token.next().c_str());
	const double sigma=std::atof(token.next().c_str());
	
	if(t1>=ntypes_) throw std::invalid_argument("Invalid type.");
	if(t2>=ntypes_) throw std::invalid_argument("Invalid type.");
	
	int t1min=t1,t1max=t1;
	int t2min=t2,t2max=t2;
	if(t1<0){t1min=0;t1max=ntypes_-1;}
	if(t2<0){t2min=0;t2max=ntypes_-1;}
	for(int i=t1min; i<=t1max; ++i){
		for(int j=t2min; j<=t2max; ++j){
			const double s3=sigma*sigma*sigma;
			const double s6=s3*s3;
			s_(i,j)=sigma; s_(j,i)=sigma;
			s6_(i,j)=s6; s6_(j,i)=s6;
			e_(i,j)=eps; e_(j,i)=eps;
			f_(i,j)=1; f_(j,i)=1;
		}
	}
}

void PotLJLong::resize(int ntypes){
	if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLJLong::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotLJLong::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::MatrixXi::Zero(ntypes_,ntypes_);
		s_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		s6_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		e_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotLJLong::init(){
	if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLJLong::init():\n";
	for(int i=0; i<ntypes_; ++i){
		for(int j=i+1; j<ntypes_; ++j){
			if(f_(i,j)==0){
				//s_(i,j)=0.5*(s_(i,i)+s_(j,j));
				s_(i,j)=sqrt(s_(i,i)*s_(j,j));
				s_(j,i)=s_(i,j);
				e_(i,j)=sqrt(e_(i,i)*e_(j,j));
				e_(j,i)=e_(i,j);
				f_(i,j)=1; f_(j,i)=1;
			}			
		}
	}
	for(int i=0; i<ntypes_; ++i){
		for(int j=i; j<ntypes_; ++j){
			const double s3=s_(i,j)*s_(i,j)*s_(i,j);
			s6_(i,j)=s3*s3;
			s6_(j,i)=s6_(i,j);
		}
	}
}

double PotLJLong::energy(const Structure& struc, const NeighborList& nlist){
	if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLJLong::energy(const Structure&,const NeighborList&):\n";
	//compute k-space energy
	const Eigen::MatrixXd c6m=4.0*e_.cwiseProduct(s6_);
	ksl_.init(struc,c6m);
	const double energyK_=ksl_.energy(struc);
	const double a=ksl_.alpha();
	//compute real-space energy
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			if(nlist.neigh(i,j).dr()<rc_){
				/*
				const double dri=1.0/nlist.neigh(i,j).dr();
				const double dri3=dri*dri*dri;
				const double dri6=dri3*dri3;
				const double b=nlist.neigh(i,j).dr()*a;
				const double b2=b*b;
				const double c6=s6_(ti,tj)*dri6;
				energyR_+=4.0*e_(ti,tj)*c6*(c6-exp(-b2)*(1.0+b2*(1.0+0.5*b2)));
				*/
				const double du=s_(ti,tj)/nlist.neigh(i,j).dr();
				const double du3=du*du*du;
				const double du6=du3*du3;
				const double b=nlist.neigh(i,j).dr()*a;
				const double b2=b*b;
				energyR_+=4.0*e_(ti,tj)*du6*(du6-exp(-b2)*(1.0+b2*(1.0+0.5*b2)));
			}
		}
	}
	energyR_*=0.5;
	//return total energy
	return energyR_+energyK_;
}

double PotLJLong::compute(Structure& struc, const NeighborList& nlist){
	if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLJLong::compute(const Structure&,const NeighborList&):\n";
	//compute k-space energy
	const Eigen::MatrixXd c6m=4.0*e_.cwiseProduct(s6_);
	ksl_.init(struc,c6m);
	const double energyK_=ksl_.energy(struc);
	const double a=ksl_.alpha();
	//compute real-space energy
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			if(nlist.neigh(i,j).dr()<rc_){
				const Eigen::Vector3d& r=nlist.neigh(i,j).r();
				/*
				const double dri=1.0/nlist.neigh(i,j).dr();
				const double dri3=dri*dri*dri;
				const double dri6=dri3*dri3;
				const double b=nlist.neigh(i,j).dr()*a;
				const double b2=b*b;
				const double s6r=s6_(ti,tj)*dri6;
				const double expf=exp(-b2);
				energyR_+=4.0*e_(ti,tj)*s6r*(s6r-expf*(1.0+b2*(1.0+0.5*b2)));
				const Eigen::Vector3d fij=24.0*e_(ti,tj)*s6r*(2.0*s6r-1.0/6.0*expf*(6.0+b2*(6.0+b2*(3.0+b2))))*dri*dri*r;
				struc.force(i).noalias()+=fij;
				*/
				const double dri=1.0/nlist.neigh(i,j).dr();
				const double du=s_(ti,tj)*dri;
				const double du3=du*du*du;
				const double du6=du3*du3;
				const double b=nlist.neigh(i,j).dr()*a;
				const double b2=b*b;
				const double expf=exp(-b2);
				energyR_+=4.0*e_(ti,tj)*du6*(du6-expf*(1.0+b2*(1.0+0.5*b2)));
				const Eigen::Vector3d fij=24.0*e_(ti,tj)*du6*(2.0*du6-1.0/6.0*expf*(6.0+b2*(6.0+b2*(3.0+b2))))*dri*dri*r;
				struc.force(i).noalias()+=fij;
			}
		}
	}
	energyR_*=0.5;
	//return total energy
	return energyR_+energyK_;
}

double PotLJLong::energy(const Structure& struc, const verlet::List& vlist){
	if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLJLong::energy(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	//compute k-space energy
	const Eigen::MatrixXd c6m=4.0*e_*s6_;
	ksl_.init(struc,c6m);
	const double energyK_=ksl_.energy(struc);
	const double a=ksl_.alpha();
	//compute real-space energy
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dri=1.0/sqrt(dr2);
				const double dri3=dri*dri*dri;
				const double dri6=dri3*dri3;
				const double b2=dr2*a*a;
				const double c6=s6_(ti,tj)*dri6;
				energyR_+=4.0*e_(ti,tj)*c6*(c6-exp(-b2)*(1.0+b2*(1.0+0.5*b2)));
			}
		}
	}
	energyR_*=0.5;
	//return total energy
	return energyR_+energyK_;
}

double PotLJLong::compute(Structure& struc, const verlet::List& vlist){
	if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLJLong::compute(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	//compute k-space energy
	const Eigen::MatrixXd c6m=4.0*e_*s6_;
	ksl_.init(struc,c6m);
	const double energyK_=ksl_.energy(struc);
	const double a=ksl_.alpha();
	//compute real-space energy
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dri=1.0/sqrt(dr2);
				const double dri3=dri*dri*dri;
				const double dri6=dri3*dri3;
				const double b2=dr2*a*a;
				const double s6r=s6_(ti,tj)*dri6;
				const double expf=exp(-b2);
				energyR_+=4.0*e_(ti,tj)*s6r*(s6r-expf*(1.0+b2*(1.0+0.5*b2)));
				const Eigen::Vector3d fij=24.0*e_(ti,tj)*s6r*(2.0*s6r-1.0/6.0*expf*(6.0+b2*(6.0+b2*(3.0+b2))))*dri*dri*drv;
				struc.force(i).noalias()+=fij;
			}
		}
	}
	energyR_*=0.5;
	//return total energy
	return energyR_+energyK_;
}

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotLJLong& obj){
		if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotLJLong&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(int);//name_
		size+=sizeof(int);//ntypes_
		const int nt=obj.ntypes();
		size+=nt*nt*sizeof(int);//f_
		size+=nt*nt*sizeof(double);//s_
		size+=nt*nt*sizeof(double);//s6_
		size+=nt*nt*sizeof(double);//e_
		return size;
	}
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLJLong& obj, char* arr){
		if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotLJLong&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.name(),sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		const int nt=obj.ntypes();
		if(nt>0){
			std::memcpy(arr+pos,obj.f().data(),nt*nt*sizeof(int)); pos+=nt*nt*sizeof(int);//f_
			std::memcpy(arr+pos,obj.s().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//s_
			std::memcpy(arr+pos,obj.s6().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//s6_
			std::memcpy(arr+pos,obj.e().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//e_
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotLJLong& obj, const char* arr){
		if(POT_LJ_LONG_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotLJLong&,const char*):\n";
		int pos=0,nt=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		ptnl::Pot::Name name;
		std::memcpy(&name,arr+pos,sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		if(name!=ptnl::Pot::Name::LJ_LONG) throw std::invalid_argument("serialize::unpack(ptnl::PotLJLong&,const char*): Invalid name.");
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.f().data(),arr+pos,nt*nt*sizeof(int)); pos+=nt*nt*sizeof(int);//f_
			std::memcpy(obj.s().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//s_
			std::memcpy(obj.s6().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//s6_
			std::memcpy(obj.e().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//e_
		}
		obj.init();
		return pos;
	}
	
}