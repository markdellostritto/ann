// math
#include "math/const.hpp"
// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// pot
#include "torch/pot_qeq_gl.hpp"

namespace ptnl{

#define PGL_SELF_ENERGY

//==== using statements ====

using math::constant::Rad2;
using math::constant::RadPI;

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotQEQGL& pot){
	return out<<static_cast<const Pot&>(pot)<<" "<<pot.prec_<<" eps "<<pot.eps_;
}
	
//==== member functions ====

void PotQEQGL::read(Token& token){
	//pot lj_long 6.0 1e-12
	static_cast<Pot&>(*this).read(token);
	prec_=std::atof(token.next().c_str());
	qtot_=std::atof(token.next().c_str());
	if(prec_<=0) throw std::invalid_argument("ptnl::PotQEQGL::read(Token&): invalid precision.");
	coul_.prec()=prec_;
	coul_.rc()=rc_;
	if(!token.end()){
		eps_=std::atof(token.next().c_str());
		if(eps_<=0.0) throw std::invalid_argument("ptnl::PotGaussCut::read(Token&): Invalid epsilon.");
		coul_.eps()=eps_;
	}
}

void PotQEQGL::coeff(Token& token){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGL::coeff(Token& token)\n";
	//coeff gauss_long type radius
	const int t=std::atof(token.next().c_str())-1;
	const double rad=std::atof(token.next().c_str());
	
	if(t>=ntypes_) throw std::invalid_argument("ptnl::PotQEQGL::coeff(Token& token): Invalid type.");
	
	int tmin=t,tmax=t;
	if(t<0){tmin=0;tmax=ntypes_-1;}
	for(int i=tmin; i<=tmax; ++i){
		f_[i]=1;
		radius_[i]=rad;
	}
}

void PotQEQGL::resize(int ntypes){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGL::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotQEQGL::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::VectorXi::Zero(ntypes_);
		radius_=Eigen::VectorXd::Zero(ntypes_);
		rij_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotQEQGL::init(){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGL::init():\n";
	for(int i=0; i<ntypes_; ++i){
		if(f_[i]==0) throw std::invalid_argument("ptnl::PotQEQGL::init(): No radius set.");
		const double ri=radius_(i);
		if(ri<=0) throw std::invalid_argument("ptnl::PotQEQGL::init(): Invalid radius.");
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

double PotQEQGL::energy(const Structure& struc, const NeighborList& nlist){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGL::energy(const Structure&,const NeighborList&):\n";
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	// qeq
	A_.resize(nAtoms+1,nAtoms+1);
	b_.resize(nAtoms+1);
	x_.resize(nAtoms+1);
	Jr_=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	Jk_=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	// k-space
	coul_.init(struc);
	const double a=coul_.alpha();
	coul_.J(struc,Jk_);
	//r-space
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		#ifdef PGL_SELF_ENERGY
		Jr_(i,i)+=2.0/(RadPI*rij_(ti,ti));
		#endif
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double& dr=nlist.neigh(i,j).dr();
			if(dr<rc_) Jr_(i,jj)+=(erf(dr/rij_(ti,tj))-erf(a*dr))/dr;
		}
	}
	Jr_*=ke;
	//add coulomb integrals to A
	J_=Jk_+Jr_;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			A_(j,i)=J_(j,i);
		}
		A_(struc.nAtoms(),i)=1.0;
	}
	for(int i=0; i<struc.nAtoms(); ++i) A_(i,struc.nAtoms())=1.0;
	A_(struc.nAtoms(),struc.nAtoms())=qtot_;
	for(int i=0; i<struc.nAtoms(); ++i){
		A_(i,i)+=struc.eta(i);
	}
	//calculate the solution vector
	for(int i=0; i<struc.nAtoms(); ++i) b_[i]=-1.0*struc.chi(i);
	b_[struc.nAtoms()]=struc.qtot();
	//solve the linear equations
	x_.noalias()=A_.partialPivLu().solve(b_);
	//compute energy
	double energy=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			energy+=x_[i]*J_(j,i)*x_[j];
		}
	}
	energy*=0.5;
	return energy;
}

double PotQEQGL::compute(Structure& struc, const NeighborList& nlist){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGL::compute(const Structure&,const NeighborList&):\n";
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	// qeq
	A_.resize(nAtoms+1,nAtoms+1);
	b_.resize(nAtoms+1);
	x_.resize(nAtoms+1);
	Jr_=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	Jk_=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	// k-space
	coul_.init(struc);
	const double a=coul_.alpha();
	coul_.J(struc,Jk_);
	//r-space
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		#ifdef PGL_SELF_ENERGY
		Jr_(i,i)+=2.0/(RadPI*rij_(ti,ti));
		#endif
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double& dr=nlist.neigh(i,j).dr();
			if(dr<rc_) Jr_(i,jj)+=(erf(dr/rij_(ti,tj))-erf(a*dr))/dr;
		}
	}
	Jr_*=ke;
	//add coulomb integrals to A
	J_=Jk_+Jr_;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			A_(j,i)=J_(j,i);
		}
		A_(struc.nAtoms(),i)=1.0;
	}
	for(int i=0; i<struc.nAtoms(); ++i) A_(i,struc.nAtoms())=1.0;
	A_(struc.nAtoms(),struc.nAtoms())=qtot_;
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
	// force: k-space
	const double energyK_=coul_.compute(struc,nlist);
	// force: r-space
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
	if(PQGL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR_<<"\n";
		std::cout<<"energyK = "<<energyK_<<"\n";
	}
	return energyR_+energyK_;
}

Eigen::MatrixXd& PotQEQGL::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGL::J(const Structure&,const NeighborList&,Eigen::MatrixXd&):\n";
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

double PotQEQGL::energy(const Structure& struc, const verlet::List& vlist){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGL::energy(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	// qeq
	A_.resize(nAtoms+1,nAtoms+1);
	b_.resize(nAtoms+1);
	x_.resize(nAtoms+1);
	Jr_=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	Jk_=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	// k-space
	coul_.init(struc);
	const double a=coul_.alpha();
	coul_.J(struc,Jk_);
	//r-space
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		#ifdef PGL_SELF_ENERGY
		Jr_(i,i)+=2.0/(RadPI*rij_(ti,ti));
		#endif
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				Jr_(i,jj)+=(erf(dr/rij_(ti,tj))-erf(a*dr))/dr;
			}
		}
	}
	Jr_*=ke;
	//add coulomb integrals to A
	J_=Jk_+Jr_;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			A_(j,i)=J_(j,i);
		}
		A_(struc.nAtoms(),i)=1.0;
	}
	for(int i=0; i<struc.nAtoms(); ++i) A_(i,struc.nAtoms())=1.0;
	A_(struc.nAtoms(),struc.nAtoms())=qtot_;
	for(int i=0; i<struc.nAtoms(); ++i){
		A_(i,i)+=struc.eta(i);
	}
	//calculate the solution vector
	for(int i=0; i<struc.nAtoms(); ++i) b_[i]=-1.0*struc.chi(i);
	b_[struc.nAtoms()]=struc.qtot();
	//solve the linear equations
	x_.noalias()=A_.partialPivLu().solve(b_);
	//compute energy
	double energy=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			energy+=x_[i]*J_(j,i)*x_[j];
		}
	}
	energy*=0.5;
	return energy;
}

double PotQEQGL::compute(Structure& struc, const verlet::List& vlist){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGL::compute(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	// qeq
	A_.resize(nAtoms+1,nAtoms+1);
	b_.resize(nAtoms+1);
	x_.resize(nAtoms+1);
	Jr_=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	Jk_=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	// k-space
	coul_.init(struc);
	const double a=coul_.alpha();
	coul_.J(struc,Jk_);
	//r-space
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		#ifdef PGL_SELF_ENERGY
		Jr_(i,i)+=2.0/(RadPI*rij_(ti,ti));
		#endif
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				Jr_(i,jj)+=(erf(dr/rij_(ti,tj))-erf(a*dr))/dr;
			}
		}
	}
	Jr_*=ke;
	//add coulomb integrals to A
	J_=Jk_+Jr_;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			A_(j,i)=J_(j,i);
		}
		A_(struc.nAtoms(),i)=1.0;
	}
	for(int i=0; i<struc.nAtoms(); ++i) A_(i,struc.nAtoms())=1.0;
	A_(struc.nAtoms(),struc.nAtoms())=qtot_;
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
	// force: k-space
	const double energyK_=coul_.compute(struc,vlist);
	// force: r-space
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
				struc.force(i).noalias()+=units::Consts::ke()*qi*qj/(dr*dr*dr)*drv*(
					-(2.0*b*dr/RadPI*exp1-erf1)
					+(2.0*a*dr/RadPI*exp2-erf2)
				);
			}
		}
	}
	energyR_*=0.5*ke;
	//total
	if(PQGL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR_<<"\n";
		std::cout<<"energyK = "<<energyK_<<"\n";
	}
	return energyR_+energyK_;
}

Eigen::MatrixXd& PotQEQGL::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	if(PQGL_PRINT_FUNC>0) std::cout<<"ptnl::PotQEQGL::J(const Structure&,const verlet::List&,Eigen::MatrixXd&):\n";
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

double PotQEQGL::cQ(Structure& struc){
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
	
	template <> int nbytes(const ptnl::PotQEQGL& obj){
		if(PQGL_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotQEQGL&):\n";
		int size=0;
		const int nt=obj.ntypes();
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(int);//ntypes_
		size+=sizeof(double);//eps_
		size+=sizeof(double);//prec_
		size+=sizeof(double);//qtot_
		size+=sizeof(int)*nt;//f
		size+=sizeof(double)*nt;//radius
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotQEQGL& obj, char* arr){
		if(PQGL_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotQEQGL&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(arr+pos,&obj.prec(),sizeof(double)); pos+=sizeof(double);//prec_
		std::memcpy(arr+pos,&obj.qtot(),sizeof(double)); pos+=sizeof(double);//qtot_
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
	
	template <> int unpack(ptnl::PotQEQGL& obj, const char* arr){
		if(PQGL_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotQEQGL&,const char*):\n";
		int pos=0,nt=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(&obj.prec(),arr+pos,sizeof(double)); pos+=sizeof(double);//prec_
		std::memcpy(&obj.qtot(),arr+pos,sizeof(double)); pos+=sizeof(double);//qtot_
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.f().data(),arr+pos,sizeof(int)*nt); pos+=sizeof(int)*nt;//f
			std::memcpy(obj.radius().data(),arr+pos,sizeof(double)*nt); pos+=sizeof(double)*nt;//radius
		}
		obj.init();
		return pos;
	}
	
}