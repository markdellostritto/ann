// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
// c++ libraries
#include <iostream>
// math
#include "math/const.hpp"
#include "math/special.hpp"
// chem
#include "chem/units.hpp"
// torch
#include "torch/kspace_london.hpp"

using math::constant::PI;
using math::constant::RadPI;

namespace KSpace{


//==== operators ====

std::ostream& operator<<(std::ostream& out, const London& l){
	return out<<static_cast<const Base&>(l);
}

//==== static functions ====

double London::fv(double pre, double a, double rc, double prec){
	const double b=a*rc;
	const double b2i=1.0/(b*b);
	const double a5=math::special::powint(a,5);
	return pre*a5*(b2i*(b2i+1.0)+0.5)*erfc(b)-prec;
}

double London::fd(double pre, double a, double rc, double prec){
	const double da=a*0.0001;
	const double fp=fv(pre,a+da,rc,prec);
	const double fm=fv(pre,a-da,rc,prec);
	return 0.5*(fp-fm)/da;
}

//==== member functions ====

void London::init(const Structure& struc, const Eigen::MatrixXd& b){
	if(KSPACEL_PRINT_FUNC>0) std::cout<<"KSpace::London::init(const Structure&,const Eigen::MatrixXd&):\n";
	if(prec_<=0) throw std::invalid_argument("KSpace::London::init(const Structure&): invalid precision.");
	if(rc_<=0) throw std::invalid_argument("KSpace::London::init(const Structure&): invalid rcut.");
	
	//==== summations over c6 ====
	b_=b;
	bs_.resize(b_.rows());
	for(int i=0; i<b_.rows(); ++i){
		bs_[i]=sqrt(b_(i,i));
	}
	double bsum=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		bsum+=b_(ti,ti);
		for(int j=i+1; j<struc.nAtoms(); ++j){
			const int tj=struc.type(j);
			bsum+=2.0*b_(ti,tj);
		}
	}
	
	//==== set structural data ====
	const Eigen::Matrix3d& R=struc.R();
	const Eigen::Matrix3d& K=struc.K();
	Eigen::Vector3d L; L<<R.col(0).norm(),R.col(1).norm(),R.col(2).norm();
	
	//==== set real space error and alpha ====
	alpha_=(1.35-0.15*log(prec_))/rc_;
	errEr_=0;
	{
		bool error=false;
		const double pre=RadPI*RadPI*RadPI*bsum/(struc.vol()*struc.nAtoms());
		const int max=10000;
		const double tol=1.0e-5;
		for(int i=0; i<max; ++i){
			const double da=fv(pre,alpha_,rc_,prec_)/fd(pre,alpha_,rc_,prec_);
			alpha_-=da;
			if(fabs(da) < tol) break;
			if(alpha_<0 || alpha_!=alpha_){
				error=true; break;
			}
		}
		if(error) alpha_=(1.35-0.15*log(prec_))/rc_;
		errEr_=fv(pre,alpha_,rc_,0.0);
	}
	a3_=alpha_*alpha_*alpha_;
	const double a6_=a3_*a3_;
	
	//==== set reciprocal space error and nk ====
	nk_.setZero();
	errEk_=0;
	for(int i=0; i<3; ++i){
		double errK=0;
		const double Kn=K.col(i).norm();
		const double pre=bsum/(6.0*RadPI*struc.nAtoms())*a6_;
		do{
			nk_[i]++;
			const double kv=nk_[i]*Kn;
			const double arg=0.5*kv/alpha_;
			errK=pre*(2.0*arg*exp(-arg*arg)+RadPI*erfc(arg));
		}while(errK>prec_);
		if(errK>errEk_) errEk_=errK;
	}
	
	//==== compute k-vecs and k-amps ====
	const int NK=(2*nk_+Eigen::Vector3i::Constant(1)).prod()-1;
	k_.resize(NK);
	int count=0;
	for(int ix=-nk_[0]; ix<=nk_[0]; ++ix){
		for(int iy=-nk_[1]; iy<=nk_[1]; ++iy){
			for(int iz=-nk_[2]; iz<=nk_[2]; ++iz){
				const Eigen::Vector3d Ktmp=ix*K.col(0)+iy*K.col(1)+iz*K.col(2);
				if(Ktmp.norm()>math::constant::ZERO) k_[count++]=Ktmp;
			}
		}
	}
	ka_.resize(NK);
	const double c=RadPI*RadPI*RadPI/(12.0*struc.vol());
	for(int i=0; i<NK; ++i){
		const double kn=k_[i].norm();
		const double b=0.5*kn/alpha_;
		ka_[i]=c*kn*kn*kn*(RadPI*erfc(b)+(1.0/(2.0*b*b*b)-1.0/b)*exp(-b*b));
	}
	
	if(KSPACEL_PRINT_STATUS>0){
		std::cout<<"alpha = "<<alpha_<<"\n";
		std::cout<<"errEr = "<<errEr_<<"\n";
		std::cout<<"errEk = "<<errEk_<<"\n";
		std::cout<<"nk    = "<<nk_.transpose()<<"\n";
		std::cout<<"NK    = "<<NK<<"\n";
	}
	
}

double London::energy(const Structure& struc)const{
	if(KSPACEL_PRINT_FUNC>0) std::cout<<"KSpace::London::energy(const Structure&)const:\n";
	double energy=0;
	const int natoms=struc.nAtoms();
	//kspace
	for(int n=0; n<k_.size(); ++n){
		std::complex<double> sf(0,0);
		for(int i=0; i<natoms; ++i){
			sf+=bs_[struc.type(i)]*exp(-I_*k_[n].dot(struc.posn(i)));
		}
		energy+=ka_[n]*std::norm(sf);
	}
	//dispersion
	double bijs=0,biis=0;
	for(int i=0; i<natoms; ++i){
		const int ti=struc.type(i);
		biis+=b_(ti,ti);
		for(int j=0; j<i; ++j){
			const int tj=struc.type(j);
			bijs+=b_(ti,tj);
		}
	}
	bijs*=2.0;
	bijs+=biis;
	//constant
	energy+=a3_/3.0*(RadPI*RadPI*RadPI/struc.vol()*bijs-a3_/2.0*biis);
	//return total
	return -0.5*energy;
}

double London::compute(Structure& struc)const{
	if(KSPACEL_PRINT_FUNC>0) std::cout<<"KSpace::London::compute(const Structure&)const:\n";
	double energy=0;
	const int natoms=struc.nAtoms();
	//kspace
	for(int n=0; n<k_.size(); ++n){
		std::complex<double> sf(0,0);
		for(int i=0; i<natoms; ++i){
			sf+=bs_[struc.type(i)]*exp(-I_*k_[n].dot(struc.posn(i)));
		}
		energy+=ka_[n]*std::norm(sf);
		for(int i=0; i<natoms; ++i){
			const std::complex<double> tmp=sf*exp(I_*k_[n].dot(struc.posn(i)));
			const Eigen::Vector3d force=bs_[struc.type(i)]*ka_[n]*k_[n]*tmp.imag();
			struc.force(i).noalias()-=force;
		}
	}
	//dispersion
	double bijs=0,biis=0;
	for(int i=0; i<natoms; ++i){
		const int ti=struc.type(i);
		biis+=b_(ti,ti);
		for(int j=0; j<i; ++j){
			const int tj=struc.type(j);
			bijs+=b_(ti,tj);
		}
	}
	bijs*=2.0;
	bijs+=biis;
	//constant
	energy+=a3_/3.0*(RadPI*RadPI*RadPI/struc.vol()*bijs-a3_/2.0*biis);
	//return total
	return -0.5*energy;
}

/*
Eigen::MatrixXd& London::J(const Structure& struc, Eigen::MatrixXd& J)const{
	if(KSPACEL_PRINT_STATUS>0) std::cout<<"KSpace::London::J(const Structure&,Eigen::MatrixXd&)const:\n";
	Eigen::Vector3d dr;
	const int nAtoms=struc.nAtoms();
	//constants
	const double cii=a3_/3.0*(RadPI*RadPI*RadPI/struc.vol()-a3_/2.0);
	const double cij=a3_*RadPI*RadPI*RadPI/(3.0*struc.vol());
	//resize
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	//real space
	for(int i=0; i<nAtoms; ++i){
		//constant term
		J(i,i)=cii;
		for(int j=i+1; j<nAtoms; ++j){
			J(j,i)=cij;
			//diff
			struc.diff(struc.posn(i),struc.posn(j),dr);
			//reciprocal space
			for(int n=0; n<k_.size(); ++n){
				J(j,i)+=ka_[n]*cos(k_[n].dot(dr));
			}
		}
	}
	//symmetrize
	J=J.selfadjointView<Eigen::Lower>();
	//return 
	J*=units::consts::ke();
	return J;
}
*/

}