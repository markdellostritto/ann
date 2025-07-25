// math
#include "math/const.hpp"
// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// pot
#include "torch/pot_tcoul_long.hpp"

namespace ptnl{

//==== using statements ====

using math::constant::Rad2;
using math::constant::RadPI;

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotTCoulLong& pot){
	return out<<static_cast<const Pot&>(pot)<<" "<<pot.prec_<<" eps "<<pot.eps_;
}
	
//==== member functions ====

void PotTCoulLong::read(Token& token){
	//pot lj_long 6.0 1e-12
	static_cast<Pot&>(*this).read(token);
	prec_=std::atof(token.next().c_str());
	if(prec_<=0) throw std::invalid_argument("ptnl::PotTCoulLong::read(Token&): invalid precision.");
	coul_.prec()=prec_;
	coul_.rc()=rc_;
	if(!token.end()){
		eps_=std::atof(token.next().c_str());
		if(eps_<=0.0) throw std::invalid_argument("ptnl::PotGaussCut::read(Token&): Invalid epsilon.");
		coul_.eps()=eps_;
	}
}

void PotTCoulLong::coeff(Token& token){
	if(PTCL_PRINT_FUNC>0) std::cout<<"ptnl::PotTCoulLong::coeff(Token& token)\n";
	//coeff tcoul_long type alpha
	const int t=std::atof(token.next().c_str())-1;
	const double alpha=std::atof(token.next().c_str());
	
	if(t>=ntypes_) throw std::invalid_argument("ptnl::PotTCoulLong::coeff(Token& token): Invalid type.");
	
	int tmin=t,tmax=t;
	if(t<0){tmin=0;tmax=ntypes_-1;}
	for(int i=tmin; i<=tmax; ++i){
		f_[i]=1;
		alpha_[i]=alpha;
	}
}

void PotTCoulLong::resize(int ntypes){
	if(PTCL_PRINT_FUNC>0) std::cout<<"ptnl::PotTCoulLong::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotTCoulLong::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::VectorXi::Zero(ntypes_);
		alpha_=Eigen::VectorXd::Zero(ntypes_);
		aij_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotTCoulLong::init(){
	if(PTCL_PRINT_FUNC>0) std::cout<<"ptnl::PotTCoulLong::init():\n";
	for(int i=0; i<ntypes_; ++i){
		if(f_[i]==0) throw std::invalid_argument("ptnl::PotTCoulLong::init(): No radius set.");
		const double ai=alpha_(i);
		if(ai<=0) throw std::invalid_argument("ptnl::PotTCoulLong::init(): Invalid radius.");
		for(int j=i; j<ntypes_; ++j){
			const double aj=alpha_(j);
			aij_(i,j)=std::pow(ai*aj,1.0/6.0);
			aij_(j,i)=aij_(i,j);
		}
	}
	coul_.prec()=prec_;
	coul_.rc()=rc_;
}

double PotTCoulLong::energy(const Structure& struc, const NeighborList& nlist){
	if(PTCL_PRINT_FUNC>0) std::cout<<"ptnl::PotTCoulLong::energy(const Structure&,const NeighborList&):\n";
	const double ke=units::Consts::ke()*eps_;
	// k-space
	coul_.init(struc);
	const double a=coul_.alpha();
	const double energyK_=coul_.energy(struc);
	// r-space
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double& qj=struc.charge(jj);
			const double& dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const double du=a_*dr/aij_(ti,tj);
				energyR_+=qi*qj*(1.0-(1.0-0.5*du)*exp(-du))/dr;
			}
		}
	}
	energyR_*=0.5*ke;
	//total
	if(PTCL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR_<<"\n";
		std::cout<<"energyK = "<<energyK_<<"\n";
	}
	return energyR_+energyK_;
}

double PotTCoulLong::compute(Structure& struc, const NeighborList& nlist){
	if(PTCL_PRINT_FUNC>0) std::cout<<"ptnl::PotTCoulLong::compute(const Structure&,const NeighborList&):\n";
	const double ke=units::Consts::ke()*eps_;
	// k-space
	coul_.init(struc);
	const double energyK_=coul_.compute(struc,nlist);
	const double a=coul_.alpha();
	// r-space
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				/*const Eigen::Vector3d& r=nlist.neigh(i,j).r();
				const double b=1.0/aij_(ti,tj);
				const double erf1=erf(dr*b);
				const double erf2=erf(dr*a);
				const double exp1=exp(-dr*dr*b*b);
				const double exp2=exp(-dr*dr*a*a);
				energyR_+=qi*qj/dr*(erf1-erf2);
				struc.force(i).noalias()+=units::Consts::ke()*qi*qj/(dr*dr*dr)*r*(
					-(2.0*b*dr/RadPI*exp1-erf1)
					+(2.0*a*dr/RadPI*exp2-erf2)
				);*/
			}
		}
	}
	energyR_*=0.5*ke;
	//total
	if(PTCL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR_<<"\n";
		std::cout<<"energyK = "<<energyK_<<"\n";
	}
	return energyR_+energyK_;
}

Eigen::MatrixXd& PotTCoulLong::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	if(PTCL_PRINT_FUNC>0) std::cout<<"ptnl::PotTCoulLong::J(const Structure&,const NeighborList&,Eigen::MatrixXd&):\n";
	const double ke=units::Consts::ke()*eps_;
	coul_.init(struc);
	const double a=coul_.alpha();
	//k-space
	coul_.J(struc,J);
	//r-space
	const int nAtoms=struc.nAtoms();
	Eigen::MatrixXd Jr=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double& dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const double du=a_*dr/aij_(ti,tj);
				Jr(i,jj)+=(1.0-(1.0-0.5*du)*exp(-du))/dr;
			}
		}
	}
	//return matrix
	J.noalias()+=Jr*ke;
	return J;
}

double PotTCoulLong::energy(const Structure& struc, const verlet::List& vlist){
	if(PTCL_PRINT_FUNC>0) std::cout<<"ptnl::PotTCoulLong::energy(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	// k-space
	coul_.init(struc);
	const double a=coul_.alpha();
	const double energyK_=coul_.energy(struc);
	// r-space
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double& qj=struc.charge(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double du=a_*dr/aij_(ti,tj);
				energyR_+=qi*qj*(1.0-(1.0-0.5*du)*exp(-du))/dr;
			}
		}
	}
	energyR_*=0.5*ke;
	//total
	if(PTCL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR_<<"\n";
		std::cout<<"energyK = "<<energyK_<<"\n";
	}
	return energyR_+energyK_;
}

double PotTCoulLong::compute(Structure& struc, const verlet::List& vlist){
	if(PTCL_PRINT_FUNC>0) std::cout<<"ptnl::PotTCoulLong::compute(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	// k-space
	coul_.init(struc);
	const double energyK_=coul_.compute(struc,vlist);
	const double a=coul_.alpha();
	// r-space
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				/*const double dr=sqrt(dr2);
				const double b=1.0/aij_(ti,tj);
				const double erf1=erf(dr*b);
				const double erf2=erf(dr*a);
				const double exp1=exp(-dr*dr*b*b);
				const double exp2=exp(-dr*dr*a*a);
				energyR_+=qi*qj/dr*(erf1-erf2);
				struc.force(i).noalias()+=units::Consts::ke()*qi*qj/(dr*dr*dr)*drv*(
					-(2.0*b*dr/RadPI*exp1-erf1)
					+(2.0*a*dr/RadPI*exp2-erf2)
				);*/
			}
		}
	}
	energyR_*=0.5*ke;
	//total
	if(PTCL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR_<<"\n";
		std::cout<<"energyK = "<<energyK_<<"\n";
	}
	return energyR_+energyK_;
}

Eigen::MatrixXd& PotTCoulLong::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	if(PTCL_PRINT_FUNC>0) std::cout<<"ptnl::PotTCoulLong::J(const Structure&,const verlet::List&,Eigen::MatrixXd&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	coul_.init(struc);
	const double a=coul_.alpha();
	//k-space
	coul_.J(struc,J);
	//r-space
	const int nAtoms=struc.nAtoms();
	Eigen::MatrixXd Jr=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double du=a_*dr/aij_(ti,tj);
				Jr(i,jj)+=(1.0-(1.0-0.5*du)*exp(-du))/dr;
			}
		}
	}
	//return matrix
	J.noalias()+=Jr*ke;
	return J;
}

double PotTCoulLong::cQ(Structure& struc){
	coul_.init(struc);
	return units::Consts::ke()*0.5*math::constant::PI/(coul_.alpha()*struc.vol());
}

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotTCoulLong& obj){
		if(PTCL_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotTCoulLong&):\n";
		int size=0;
		const int nt=obj.ntypes();
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(int);//ntypes_
		size+=sizeof(double);//eps_
		size+=sizeof(double);//prec_
		size+=sizeof(int)*nt;//f
		size+=sizeof(double)*nt;//radius
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotTCoulLong& obj, char* arr){
		if(PTCL_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotTCoulLong&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(arr+pos,&obj.prec(),sizeof(double)); pos+=sizeof(double);//prec_
		const int nt=obj.ntypes();
		if(nt>0){
			std::memcpy(arr+pos,obj.f().data(),sizeof(int)*nt); pos+=sizeof(int)*nt;//f
			std::memcpy(arr+pos,obj.radius().data(),sizeof(double)*nt); pos+=sizeof(double)*nt;//radius
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotTCoulLong& obj, const char* arr){
		if(PTCL_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotTCoulLong&,const char*):\n";
		int pos=0,nt=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(&obj.prec(),arr+pos,sizeof(double)); pos+=sizeof(double);//prec_
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.f().data(),arr+pos,sizeof(int)*nt); pos+=sizeof(int)*nt;//f
			std::memcpy(obj.radius().data(),arr+pos,sizeof(double)*nt); pos+=sizeof(double)*nt;//radius
		}
		obj.init();
		return pos;
	}
	
}