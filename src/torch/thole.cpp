// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"
// torch
#include "torch/thole.hpp"

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Thole& thole){
	return out<<"natoms "<<thole.nAtoms_<<" a "<<thole.a_;
}

//==== access ====

void Thole::resize(int nAtoms){
	if(nAtoms<0) throw std::invalid_argument("Invalid number of atoms.");
	nAtoms_=nAtoms;
	const int size=3*nAtoms_;
	ai_.resize(size);
	A_.resize(size,size);
	AI_.resize(size,size);
	T_.resize(size,size);
	S_.resize(size,3);
}

Eigen::Matrix3d& Thole::compute(const Structure& struc){
	const double ke=units::Consts::ke();
	
	//compute the inverse of the atomic polarizabilities
	for(int i=0; i<nAtoms_; ++i){
		const double ai=1.0/struc.alpha(i);
		ai_[3*i+0]=ai;
		ai_[3*i+1]=ai;
		ai_[3*i+2]=ai;
	}
	
	//compute the interaction matrices
	for(int i=0; i<nAtoms_; ++i){
		T_.block<3,3>(3*i,3*i).setZero();
		for(int j=i+1; j<nAtoms_; ++j){
			const Eigen::Vector3d r=struc.posn(i)-struc.posn(j);
			const double dr=r.norm();
			const double dr3=dr*dr*dr;
			const double b=a_*dr*1.0/sqrt(struc.radius(i)*struc.radius(j));
			const double fexp=std::exp(-b);
			T_.block<3,3>(j*3,i*3)=-1.0*ke*(r*r.transpose()*3.0*(1.0-((((1.0/6.0)*b+0.5)*b+1.0)*b+1.0)*fexp)/(dr3*dr*dr)
				-Eigen::Matrix3d::Identity()*(1.0-((0.5*b+1.0)*b+1.0)*fexp)/dr3);
			T_.block<3,3>(i*3,j*3)=T_.block<3,3>(j*3,i*3);
		}
	}
	
	//compute A and A^-1
	A_=ai_.asDiagonal();
	A_.noalias()+=T_;
	AI_=A_.inverse();
	
	//compute S
	for(int n=0; n<nAtoms_; ++n) S_.block<3,3>(3*n,0)=Eigen::Matrix3d::Identity();
	
	//compute total polarizability
	atot_.noalias()=S_.transpose()*AI_*S_;
	
	//return polarizability
	return atot_;
}

void Thole::gradient(const Structure& struc){
	Eigen::MatrixXd P=ai_.asDiagonal()*AI_*S_;
	
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(struc.atot());
	const Eigen::Vector3d eigenVecI=eigensolver.eigenvalues().cwiseInverse().array().sqrt()*1.0/math::constant::Rad3;
	const Eigen::Vector3d nvec=eigensolver.eigenvectors()*eigenVecI;
	
	for(int i=0; i<nAtoms_; ++i){
		Eigen::MatrixXd delta=Eigen::MatrixXd::Zero(3*nAtoms_,3*nAtoms_);
		delta.block<3,3>(i*3,i*3)=Eigen::Matrix3d::Identity();
		const Eigen::VectorXd gv=nvec.transpose()*(P.transpose()*delta*P)*nvec;
	}
}