// math
#include "math/const.hpp"
// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// pot
#include "torch/pot_qeq_gauss_long.hpp"

namespace ptnl{

#define PQGL_SELF_ENERGY

//==== using statements ====

using math::constant::Rad2;
using math::constant::RadPI;

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotQEQGaussLong& pot){
	return out<<static_cast<const Pot&>(pot)<<" "<<pot.prec_<<" eps "<<pot.eps_;
}
	
//==== member functions ====

void PotQEQGaussLong::read(Token& token){
	//pot lj_long 6.0 1e-12
	static_cast<Pot&>(*this).read(token);
	prec_=std::atof(token.next().c_str());
	if(prec_<=0) throw std::invalid_argument("ptnl::PotQEQGaussLong::read(Token&): invalid precision.");
	coul_.prec()=prec_;
	coul_.rc()=rc_;
	if(!token.end()){
		eps_=std::atof(token.next().c_str());
		if(eps_<=0.0) throw std::invalid_argument("ptnl::PotGaussCut::read(Token&): Invalid epsilon.");
		coul_.eps()=eps_;
	}
}

void PotQEQGaussLong::coeff(Token& token){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGaussLong::coeff(Token& token)\n";
	//coeff gauss_long type radius
	const int t=std::atof(token.next().c_str())-1;
	const double rad=std::atof(token.next().c_str());
	
	if(t>=ntypes_) throw std::invalid_argument("ptnl::PotQEQGaussLong::coeff(Token& token): Invalid type.");
	
	int tmin=t,tmax=t;
	if(t<0){tmin=0;tmax=ntypes_-1;}
	for(int i=tmin; i<=tmax; ++i){
		f_[i]=1;
		radius_[i]=rad;
	}
}

void PotQEQGaussLong::resize(int ntypes){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGaussLong::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotQEQGaussLong::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::VectorXi::Zero(ntypes_);
		radius_=Eigen::VectorXd::Zero(ntypes_);
		rij_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotQEQGaussLong::init(){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGaussLong::init():\n";
	for(int i=0; i<ntypes_; ++i){
		if(f_[i]==0) throw std::invalid_argument("ptnl::PotQEQGaussLong::init(): No radius set.");
		const double ri=radius_(i);
		if(ri<=0) throw std::invalid_argument("ptnl::PotQEQGaussLong::init(): Invalid radius.");
		rij_(i,i)=2.0*ri;
		for(int j=i+1; j<ntypes_; ++j){
			const double rj=radius_(j);
			rij_(i,j)=Rad2*sqrt(ri*ri+rj*rj);
			rij_(j,i)=rij_(i,j);
		}
	}
	coul_.prec()=prec_;
	coul_.rc()=rc_;
}

double PotQEQGaussLong::energy(const Structure& struc, const NeighborList& nlist){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGaussLong::energy(const Structure&,const NeighborList&):\n";
	const double ke=units::Consts::ke()*eps_;
	// resize
	J_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	A_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	b_.resize(struc.nAtoms()+1);
	x_.resize(struc.nAtoms()+1);
	//compute coulomb integrals
	this->J(struc,nlist,J_);
	//add coulomb integrals to A
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			A_(j,i)=J(j,i);
		}
		A_(struc.nAtoms(),i)=1.0;
	}
	for(int i=0; i<struc.nAtoms(); ++i) A_(i,struc.nAtoms())=1.0;
	A_(struc.nAtoms(),struc.nAtoms())=0.0;
	//add idempotential
	for(int i=0; i<struc.nAtoms(); ++i){
		A_(i,i)+=struc.eta(i);
	}
	//calculate the solution vector
	for(int i=0; i<struc.nAtoms(); ++i) b_[i]=-1.0*struc.chi(i);
	b_[struc.nAtoms()]=struc.qtot();
	//solve the linear equations
	x_.noalias()=A_.partialPivLu().solve(b_);
	//set the atomic charges
	for(int i=0; i<struc.nAtoms(); ++i) struc.charge(i)=x_[i];
	// compute the energy
	double energy=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			energy+=struc.charge(j)*J_(j,i)*struc.charge(i);
		}
	}
	energy*=0.5*ke;
	return energy;
}

double PotQEQGaussLong::compute(Structure& struc, const NeighborList& nlist){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGaussLong::compute(const Structure&,const NeighborList&):\n";
	const double ke=units::Consts::ke()*eps_;
	const double ke=units::Consts::ke()*eps_;
	// resize
	J_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	A_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	b_.resize(struc.nAtoms()+1);
	x_.resize(struc.nAtoms()+1);
	//compute coulomb integrals
	this->J(struc,nlist,J_);
	//add coulomb integrals to A
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			A_(j,i)=J(j,i);
		}
		A_(struc.nAtoms(),i)=1.0;
	}
	for(int i=0; i<struc.nAtoms(); ++i) A_(i,struc.nAtoms())=1.0;
	A_(struc.nAtoms(),struc.nAtoms())=0.0;
	//add idempotential
	for(int i=0; i<struc.nAtoms(); ++i){
		A_(i,i)+=struc.eta(i);
	}
	//calculate the solution vector
	for(int i=0; i<struc.nAtoms(); ++i) b_[i]=-1.0*struc.chi(i);
	b_[struc.nAtoms()]=struc.qtot();
	//solve the linear equations
	x_.noalias()=A_.partialPivLu().solve(b_);
	//set the atomic charges
	for(int i=0; i<struc.nAtoms(); ++i) struc.charge(i)=x_[i];
	
	// k-space
	const double energyK_=coul_.compute(struc,nlist);
	const double a=coul_.alpha();
	// r-space
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		const int ti=struc.type(i);
		#ifdef PGL_SELF_ENERGY
		energyR_+=qi*qi*2.0/(RadPI*rij_(ti,ti));
		#endif
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const Eigen::Vector3d& r=nlist.neigh(i,j).r();
				const double b=1.0/rij_(ti,tj);
				const double erf1=erf(dr*b);
				const double erf2=erf(dr*a);
				const double exp1=exp(-dr*dr*b*b);
				const double exp2=exp(-dr*dr*a*a);
				energyR_+=qi*qj/dr*(erf1-erf2);
				struc.force(i).noalias()+=units::Consts::ke()*qi*qj/(dr*dr*dr)*r*(
					-(2.0*b*dr/RadPI*exp1-erf1)
					+(2.0*a*dr/RadPI*exp2-erf2)
				);
			}
		}
	}
	energyR_*=0.5*ke;
	//total
	if(PGL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR_<<"\n";
		std::cout<<"energyK = "<<energyK_<<"\n";
	}
	return energyR_+energyK_;
}

Eigen::MatrixXd& PotQEQGaussLong::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGaussLong::J(const Structure&,const NeighborList&,Eigen::MatrixXd&):\n";
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
		#ifdef PGL_SELF_ENERGY
		Jr(i,i)+=2.0/(RadPI*rij_(ti,ti));
		#endif
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double& dr=nlist.neigh(i,j).dr();
			if(dr<rc_) Jr(i,jj)+=(erf(dr/rij_(ti,tj))-erf(a*dr))/dr;
		}
	}
	//return matrix
	J.noalias()+=Jr*ke;
	return J;
}

double PotQEQGaussLong::energy(const Structure& struc, const verlet::List& vlist){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGaussLong::energy(const Structure&,const verlet::List&):\n";
	const double ke=units::Consts::ke()*eps_;
	// resize
	J_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	A_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	b_.resize(struc.nAtoms()+1);
	x_.resize(struc.nAtoms()+1);
	//compute coulomb integrals
	this->J(struc,vlist,J_);
	//add coulomb integrals to A
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			A_(j,i)=J(j,i);
		}
		A_(struc.nAtoms(),i)=1.0;
	}
	for(int i=0; i<struc.nAtoms(); ++i) A_(i,struc.nAtoms())=1.0;
	A_(struc.nAtoms(),struc.nAtoms())=0.0;
	//add idempotential
	for(int i=0; i<struc.nAtoms(); ++i){
		A_(i,i)+=struc.eta(i);
	}
	//calculate the solution vector
	for(int i=0; i<struc.nAtoms(); ++i) b_[i]=-1.0*struc.chi(i);
	b_[struc.nAtoms()]=struc.qtot();
	//solve the linear equations
	x_.noalias()=A_.partialPivLu().solve(b_);
	//set the atomic charges
	for(int i=0; i<struc.nAtoms(); ++i) struc.charge(i)=x_[i];
	// compute the energy
	double energy=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			energy+=struc.charge(j)*J_(j,i)*struc.charge(i);
		}
	}
	energy*=0.5*ke;
	return energy;
}

double PotQEQGaussLong::compute(Structure& struc, const verlet::List& vlist){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGaussLong::compute(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	const double ke=units::Consts::ke()*eps_;
	// resize
	J_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	A_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	b_.resize(struc.nAtoms()+1);
	x_.resize(struc.nAtoms()+1);
	//compute coulomb integrals
	this->J(struc,vlist,J_);
	//add coulomb integrals to A
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			A_(j,i)=J(j,i);
		}
		A_(struc.nAtoms(),i)=1.0;
	}
	for(int i=0; i<struc.nAtoms(); ++i) A_(i,struc.nAtoms())=1.0;
	A_(struc.nAtoms(),struc.nAtoms())=0.0;
	//add idempotential
	for(int i=0; i<struc.nAtoms(); ++i){
		A_(i,i)+=struc.eta(i);
	}
	//calculate the solution vector
	for(int i=0; i<struc.nAtoms(); ++i) b_[i]=-1.0*struc.chi(i);
	b_[struc.nAtoms()]=struc.qtot();
	//solve the linear equations
	x_.noalias()=A_.partialPivLu().solve(b_);
	//set the atomic charges
	for(int i=0; i<struc.nAtoms(); ++i) struc.charge(i)=x_[i];
	
	// k-space
	const double energyK_=coul_.compute(struc,vlist);
	const double a=coul_.alpha();
	// r-space
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		const int ti=struc.type(i);
		#ifdef PGL_SELF_ENERGY
		energyR_+=qi*qi*2.0/(RadPI*rij_(ti,ti));
		#endif
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double b=1.0/rij_(ti,tj);
				const double erf1=erf(dr*b);
				const double erf2=erf(dr*a);
				const double exp1=exp(-dr*dr*b*b);
				const double exp2=exp(-dr*dr*a*a);
				energyR_+=qi*qj/dr*(erf1-erf2);
				struc.force(i).noalias()+=units::Consts::ke()*qi*qj/(dr*dr2)*drv*(
					-(2.0*b*dr/RadPI*exp1-erf1)
					+(2.0*a*dr/RadPI*exp2-erf2)
				);
			}
		}
	}
	energyR_*=0.5*ke;
	//total
	if(PGL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR_<<"\n";
		std::cout<<"energyK = "<<energyK_<<"\n";
	}
	return energyR_+energyK_;
}

Eigen::MatrixXd& PotQEQGaussLong::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGaussLong::J(const Structure&,const verlet::List&,Eigen::MatrixXd&):\n";
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
		#ifdef PGL_SELF_ENERGY
		Jr(i,i)+=2.0/(RadPI*rij_(ti,ti));
		#endif
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				Jr(i,jj)+=(erf(dr/rij_(ti,tj))-erf(a*dr))/dr;
			}
		}
	}
	//return matrix
	J.noalias()+=Jr*ke;
	return J;
}

double PotQEQGaussLong::cQ(Structure& struc){
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
	
	template <> int nbytes(const ptnl::PotQEQGaussLong& obj){
		if(PQGL_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotQEQGaussLong&):\n";
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
	
	template <> int pack(const ptnl::PotQEQGaussLong& obj, char* arr){
		if(PQGL_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotQEQGaussLong&,char*):\n";
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
	
	template <> int unpack(ptnl::PotQEQGaussLong& obj, const char* arr){
		if(PQGL_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotQEQGaussLong&,const char*):\n";
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