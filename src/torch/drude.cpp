// c++
#include <stdexcept>
// struc
#include "struc/verlet.hpp"
// torch
#include "torch/drude.hpp"
// math
#include "math/const.hpp"

using math::constant::RadPI;

void Drude::load(double rc, const Structure& struc){
	if(DRUDE_PRINT_FUNC>0) std::cout<<"Drude::load(double,const Structure&):\n";
	//set cutoff
	rc_=rc;
	rc2_=rc_*rc_;
	//find the number of atoms/drude particles
	nAtom_=struc.nAtoms();
	nDrude_=0;
	dIndex_.resize(nAtom_,-1);
	for(int i=0; i<nAtom_; ++i){
		if(struc.name(i)!="H"){
			dIndex_[i]=nAtom_+nDrude_;
			nDrude_++;
		}
	}
	if(DRUDE_PRINT_DATA>0) std::cout<<"nAtom = "<<nAtom_<<" nDrude = "<<nDrude_<<"\n";
	if(DRUDE_PRINT_DATA>1){
		for(int i=0; i<nAtom_; ++i){
			std::cout<<"dIndex["<<i<<"] = "<<dIndex_[i]<<"\n";
		}
	}
	//make structure
	struc_.resize(nAtom_+nDrude_,struc.atomType());
	static_cast<Cell&>(struc_).init(struc.R());
	//set nuclear charge
	ntypes_=0;
	for(int i=0; i<nAtom_; ++i){
		struc_.posn(i)=struc.posn(i);
		struc_.type(i)=struc.type(i);
		struc_.mass(i)=struc.mass(i);
		struc_.alpha(i)=struc.alpha(i);
		struc_.radius(i)=struc.radius(i);
		struc_.charge(i)=struc.charge(i);
		struc_.name(i)=struc.name(i);
		if(struc.type(i)>ntypes_) ntypes_=struc.type(i)+1;
	}
	//set drude charge/mass/posn
	for(int i=0; i<nAtom_; ++i){
		if(dIndex_[i]>0){
			const int ii=dIndex_[i];
			struc_.posn(ii)=struc_.posn(i)-Eigen::Vector3d::Random()*0.1;
			struc_.type(ii)=ntypes_+1;
			struc_.mass(ii)=0.1;
			struc_.alpha(ii)=0.0;
			struc_.charge(ii)=-1.0*struc_.charge(i)-2.0;
			//struc_.charge(ii)=-1.0*struc_.charge(i);
			struc_.radius(ii)=0.1;
			struc_.name(ii)="e";
		}
	}
	
	std::cout<<"name type mass charge posn\n";
	for(int i=0; i<struc_.nAtoms(); ++i){
		std::cout<<struc_.name(i)<<" "<<struc_.type(i)<<" "<<struc_.mass(i)<<" "<<struc_.charge(i)<<" "<<struc_.radius(i)<<" "<<struc_.posn(i).transpose()<<"\n";
	}
	
	//kspace
	//coul_.rc()=rc_;
	//coul_.prec()=prec_;
	//coul_.eps()=eps_;
	//coul_.init(struc_);
	//verlet list
	vlist_.rc()=rc_;
	vlist_.build(struc_);
}

void Drude::init(const Eigen::VectorXd& alpha){
	if(DRUDE_PRINT_FUNC>0) std::cout<<"Drude::init(const Eigen::VectorXd&):\n";
	if(alpha.size()!=ntypes_) throw std::invalid_argument("Drude::init(const Eigen::VectorXd&): Invalid number of parameters.");
	alpha_=alpha;
	aij_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	for(int i=0; i<ntypes_; ++i){
		const double ai=alpha_(i);
		for(int j=i; j<ntypes_; ++j){
			const double aj=alpha_(j);
			aij_(i,j)=std::pow(ai*aj,1.0/6.0);
			aij_(j,i)=aij_(i,j);
		}
	}
	k_=Eigen::VectorXd::Zero(ntypes_);
	for(int i=0; i<ntypes_; ++i){
		k_[i]=1.0/alpha_[i];
	}
}

double Drude::compute_sr(){
	if(DRUDE_PRINT_FUNC>0) std::cout<<"Drude::compute_sr():\n";
	// constants
	const double ke=units::Consts::ke()*eps_;
	// zero force
	for(int i=0; i<struc_.nAtoms(); ++i){
		struc_.force(i).setZero();
	}
	// coulomb - r-space
	if(DRUDE_PRINT_STATUS>0) std::cout<<"computing coulomb\n";
	energyR_=0;
	Eigen::Vector3d drv;
	for(int i=0; i<struc_.nAtoms(); ++i){
		const int ii=i;
		const int ti=struc_.type(ii);
		const double qi=struc_.charge(ii);
		const double ri=struc_.radius(ii);
		//energyR_+=qi*qi*2.0/(ri*RadPI);
		for(int j=0; j<vlist_.size(i); ++j){
			const int jj=vlist_.neigh(i,j).index();
			const int tj=struc_.type(jj);
			const double qj=struc_.charge(jj);
			const double rj=struc_.radius(jj);
			struc_.diff(struc_.posn(ii),struc_.posn(jj),drv);
			drv.noalias()-=struc_.R()*vlist_.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(jj!=dIndex_[ii]){
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double irij=1.0/sqrt(ri*rj);
				const double erff=erf(dr*irij);
				const double expf=exp(-dr*dr*irij*irij);
				//std::cout<<"energyR("<<struc_.name(ii)<<"("<<ii<<"),"<<struc_.name(jj)<<"("<<jj<<")) = "<<qi*qj*erff/dr<<"\n";
				energyR_+=qi*qj*erff/dr;
				struc_.force(i).noalias()-=ke*qi*qj*(2.0*irij*dr/RadPI*expf-erff)/(dr*dr*dr)*drv;
			}
			}
		}
	}
	energyR_*=0.5;
	if(DRUDE_PRINT_DATA>0) std::cout<<"energyR_ = "<<energyR_<<"\n";
	// harmonic
	if(DRUDE_PRINT_STATUS>0) std::cout<<"computing harmonic\n";
	energyS_=0;
	for(int i=0; i<nAtom_; ++i){
		if(dIndex_[i]>0){
			const int ii=i;
			const int jj=dIndex_[i];
			const int ti=struc_.type(ii);
			const int tj=struc_.type(jj);
			struc_.diff(struc_.posn(ii),struc_.posn(jj),drv);
			drv.noalias()-=struc_.R()*vlist_.neigh(ii,jj).cell();
			const double dr=drv.norm();
			const double k=struc_.charge(ii)*struc_.charge(ii)/struc_.alpha(ii);
			energyS_+=0.5*k*dr*dr;
			struc_.force(ii).noalias()-=k*drv;
			struc_.force(jj).noalias()+=k*drv;
		}
	}
	if(DRUDE_PRINT_DATA>0) std::cout<<"energyS_ = "<<energyS_<<"\n";
	//return energy
	return energyR_+energyS_;
}

double Drude::compute_lr(){
	if(DRUDE_PRINT_FUNC>0) std::cout<<"Drude::compute_lr():\n";
	//constants
	const double ke=units::Consts::ke()*eps_;
	// zero force
	for(int i=0; i<struc_.nAtoms(); ++i){
		struc_.force(i).setZero();
	}
	// coulomb - k-space
	energyK_=coul_.compute(struc_,vlist_);
	const double ac=coul_.alpha();
	// coulomb - r-space
	energyR_=0;
	Eigen::Vector3d drv;
	for(int i=0; i<struc_.nAtoms(); ++i){
		const int ii=i;
		const int ti=struc_.type(ii);
		const double qi=struc_.charge(ii);
		for(int j=0; j<vlist_.size(i); ++j){
			const int jj=vlist_.neigh(i,j).index();
			const int tj=struc_.type(jj);
			const double qj=struc_.charge(jj);
			if(std::abs(ii-jj)!=nAtom_){
				struc_.diff(struc_.posn(ii),struc_.posn(jj),drv);
				drv.noalias()-=struc_.R()*vlist_.neigh(i,j).cell();
				const double dr2=drv.squaredNorm();
				if(dr2<rc2_){
					const double dr=sqrt(dr2);
					const double erfcf=erfc(ac*dr);
					const double expf=exp(-ac*ac*dr2);
					energyR_+=qi*qj*erfcf/dr;
					struc_.force(i).noalias()+=ke*qi*qj*(erfcf+2.0*ac*dr/RadPI*expf)/(dr*dr2)*drv;
				}
			}
		}
	}
	// harmonic
	energyS_=0;
	for(int i=0; i<nAtom_; ++i){
		const int j=i+nAtom_;
		const int ti=struc_.type(i);
		const int tj=struc_.type(i);
		struc_.diff(struc_.posn(i),struc_.posn(j),drv);
		drv.noalias()-=struc_.R()*vlist_.neigh(i,j).cell();
		const double dr=drv.norm();
		const double k=1.0/struc_.alpha(i);
		energyS_-=0.5*k*dr*dr;
		struc_.force(i).noalias()+=k*drv;
		struc_.force(j).noalias()-=k*drv;
	}
	//return energy
	return energyK_+energyR_+energyS_;
}

void Drude::quickmin(){
	if(DRUDE_PRINT_FUNC>0) std::cout<<"Drude::quickmin():\n";
	//compute force
	struc_.pe()=compute_sr();
	//project velocity along force
	for(int i=nAtom_; i<nAtom_+nDrude_; ++i){
		//project velocity along force
		Eigen::Vector3d fhat=struc_.force(i);
		fhat/=struc_.force(i).norm();
		const double vdotf=struc_.vel(i).dot(fhat);
		struc_.vel(i)=fhat*vdotf;
		if(struc_.vel(i).dot(fhat)<0.0) struc_.vel(i).setZero();
		//euler step
		struc_.posn(i).noalias()+=struc_.vel(i)*dt_;
		struc_.vel(i).noalias()+=struc_.force(i)*dt_;
	}
	//compute KE, T
	struc_.ke()=0;
	for(int i=0; i<struc_.nAtoms(); ++i){
		struc_.ke()+=struc_.mass(i)*struc_.vel(i).squaredNorm();
	}
	struc_.ke()*=0.5;
	struc_.temp()=struc_.ke()*(2.0/3.0)/(struc_.nAtoms()*units::Consts::kb());
	//increment
	++struc_.t();
}