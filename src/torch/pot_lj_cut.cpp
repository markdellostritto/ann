// chem
#include "chem/units.hpp"
// pot
#include "torch/pot_lj_cut.hpp"
// thread
#include "thread/dist.hpp"

namespace ptnl{

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotLJCut& pot){
	return out<<static_cast<const Pot&>(pot);
}
	
//==== member functions ====

void PotLJCut::read(Token& token){
	static_cast<Pot&>(*this).read(token);
}

void PotLJCut::coeff(Token& token){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::coeff(Token&):\n";
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
			s_(i,j)=sigma; s_(j,i)=sigma;
			e_(i,j)=eps; e_(j,i)=eps;
			f_(i,j)=1; f_(j,i)=1;
		}
	}
}

void PotLJCut::resize(int ntypes){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotLJCut::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::MatrixXi::Zero(ntypes_,ntypes_);
		s_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		e_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotLJCut::init(){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::init():\n";
	for(int i=0; i<ntypes_; ++i){
		for(int j=i; j<ntypes_; ++j){
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
}

double PotLJCut::energy(const Structure& struc, const NeighborList& nlist){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::energy(const Structure&,const NeighborList&):\n";
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			if(nlist.neigh(i,j).dr()<rc_){
				const double du=s_(ti,tj)/nlist.neigh(i,j).dr();
				const double du3=du*du*du;
				const double du6=du3*du3;
				energy_+=4.0*e_(ti,tj)*du6*(du6-1.0);
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotLJCut::energy(const Structure& struc, const NeighborList& nlist, int i){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::energy(const Structure&,const NeighborList&,int):\n";
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<nlist.size(i); ++j){
		const int tj=struc.type(nlist.neigh(i,j).index());
		if(nlist.neigh(i,j).dr()<rc_){
			const double du=s_(ti,tj)/nlist.neigh(i,j).dr();
			const double du3=du*du*du;
			const double du6=du3*du3;
			energy_+=4.0*e_(ti,tj)*du6*(du6-1.0);
			
		}
	}
	return energy_;
}

double PotLJCut::compute(Structure& struc, const NeighborList& nlist){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::compute(const Structure&,const NeighborList&):\n";
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			if(nlist.neigh(i,j).dr()<rc_){
				const Eigen::Vector3d& r=nlist.neigh(i,j).r();
				const double dri=1.0/nlist.neigh(i,j).dr();
				const double du=s_(ti,tj)/nlist.neigh(i,j).dr();
				const double du3=du*du*du;
				const double du6=du3*du3;
				energy_+=4.0*e_(ti,tj)*du6*(du6-1.0);
				const Eigen::Vector3d fij=24.0*e_(ti,tj)*du6*(2.0*du6-1.0)*dri*dri*r;
				struc.force(i).noalias()+=fij;
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotLJCut::compute(Structure& struc, const NeighborList& nlist, int i){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::compute(const Structure&,const NeighborList&,int):\n";
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<nlist.size(i); ++j){
		const int tj=struc.type(nlist.neigh(i,j).index());
		if(nlist.neigh(i,j).dr()<rc_){
			const Eigen::Vector3d& r=nlist.neigh(i,j).r();
			const double dri=1.0/nlist.neigh(i,j).dr();
			const double du=s_(ti,tj)/nlist.neigh(i,j).dr();
			const double du3=du*du*du;
			const double du6=du3*du3;
			energy_+=4.0*e_(ti,tj)*du6*(du6-1.0);
			const Eigen::Vector3d fij=24.0*e_(ti,tj)*du6*(2.0*du6-1.0)*dri*dri*r;
			struc.force(i).noalias()+=fij;
		}
	}
	return energy_;
}

double PotLJCut::energy(const Structure& struc, const verlet::List& vlist){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::energy(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double du=s_(ti,tj)/sqrt(dr2);
				const double du3=du*du*du;
				const double du6=du3*du3;
				energy_+=4.0*e_(ti,tj)*du6*(du6-1.0);
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotLJCut::energy(const Structure& struc, const verlet::List& vlist, int i){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::energy(const Structure&,const verlet::List&,int):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<vlist.size(i); ++j){
		const int jj=vlist.neigh(i,j).index();
		const int tj=struc.type(jj);
		const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
		if(dr2<rc2_){
			const double du=s_(ti,tj)/sqrt(dr2);
			const double du3=du*du*du;
			const double du6=du3*du3;
			energy_+=4.0*e_(ti,tj)*du6*(du6-1.0);
		}
	}
	return energy_;
}

double PotLJCut::compute(Structure& struc, const verlet::List& vlist){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::compute(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dri=1.0/sqrt(dr2);
				const double du=s_(ti,tj)*dri;
				const double du3=du*du*du;
				const double du6=du3*du3;
				energy_+=4.0*e_(ti,tj)*du6*(du6-1.0);
				const Eigen::Vector3d fij=24.0*e_(ti,tj)*du6*(2.0*du6-1.0)*dri*dri*drv;
				struc.force(i).noalias()+=fij;
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotLJCut::compute(Structure& struc, const verlet::List& vlist, int i){
	if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLJCut::compute(const Structure&,const verlet::List&,int):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<vlist.size(i); ++j){
		const int jj=vlist.neigh(i,j).index();
		const int tj=struc.type(jj);
		struc.diff(struc.posn(i),struc.posn(jj),drv);
		drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
		const double dr2=drv.squaredNorm();
		if(dr2<rc2_){
			const double dri=1.0/sqrt(dr2);
			const double du=s_(ti,tj)*dri;
			const double du3=du*du*du;
			const double du6=du3*du3;
			energy_+=4.0*e_(ti,tj)*du6*(du6-1.0);
			const Eigen::Vector3d fij=24.0*e_(ti,tj)*du6*(2.0*du6-1.0)*dri*dri*drv;
			struc.force(i).noalias()+=fij;
		}
	}
	return energy_;
}

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotLJCut& obj){
		if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotLJCut&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(int);//name_
		size+=sizeof(int);//ntypes_
		const int nt=obj.ntypes();
		size+=nt*nt*sizeof(int);//f_
		size+=nt*nt*sizeof(double);//s_
		size+=nt*nt*sizeof(double);//e_
		return size;
	}
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLJCut& obj, char* arr){
		if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotLJCut&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.name(),sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		const int nt=obj.ntypes();
		if(nt>0){
			std::memcpy(arr+pos,obj.f().data(),nt*nt*sizeof(int)); pos+=nt*nt*sizeof(int);//f_
			std::memcpy(arr+pos,obj.s().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//s_
			std::memcpy(arr+pos,obj.e().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//e_
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotLJCut& obj, const char* arr){
		if(POT_LJ_CUT_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotLJCut&,const char*):\n";
		int pos=0,nt=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		ptnl::Pot::Name name;
		std::memcpy(&name,arr+pos,sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		if(name!=ptnl::Pot::Name::LJ_CUT) throw std::invalid_argument("serialize::unpack(PotLJCut&,const char*): Invalid name.");
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.f().data(),arr+pos,nt*nt*sizeof(int)); pos+=nt*nt*sizeof(int);//f_
			std::memcpy(obj.s().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//s_
			std::memcpy(obj.e().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//e_
		}
		obj.init();
		return pos;
	}
	
}