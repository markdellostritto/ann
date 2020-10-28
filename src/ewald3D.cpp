// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
// c++ libraries
#include <iostream>
// ann - math
#include "math_const.hpp"
// ann - structure
#include "structure.hpp"
// ann - units
#include "units.hpp"
// ann - print
#include "print.hpp"
// ann - ewald
#include "ewald3D.hpp"

namespace Ewald3D{

//**********************************************************************************************************
//Utility Class
//**********************************************************************************************************

//operators

/**
* print class
* @param out - output stream
* @param u - object
*/
std::ostream& operator<<(std::ostream& out, const Utility& u){
	out<<"PREC    = "<<u.prec_<<"\n";
	out<<"ALPHA   = "<<u.alpha_<<"\n";
	out<<"WEIGHT  = "<<u.weight_<<"\n";
	out<<"R_MAX   = "<<u.rMax_<<"\n";
	out<<"K_MAX   = "<<u.kMax_<<"\n";
	out<<"EPSILON = "<<u.eps_;
	return out;
}

//member functions

/**
* set defaults
*/
void Utility::defaults(){
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Utility::defaults():\n";
	//calculation parameters
	prec_=1E-6;
	rMax_=0;
	kMax_=0;
	weight_=1;
	alpha_=0;
	//electrostatics
	eps_=1;//vacuum boundary conditions
}

/**
* initialize object
* @param prec - precision of ewald calculation
*/
void Utility::init(double prec){
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Utility::init(const Cell&, int, double):\n";
	//set the precision
	if(EWALD_PRINT_STATUS>1) std::cout<<"setting precision\n";
	if(prec-1>=math::constant::ZERO || prec<=0) throw std::invalid_argument("Utility::init(double): Precision must be in (0,1)");
	else prec_=prec;
}

//**********************************************************************************************************
//Coulomb Class
//**********************************************************************************************************

//operators

std::ostream& operator<<(std::ostream& out, const Coulomb& c){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("EWALD - COULOMB",str)<<"\n";
	out<<static_cast<const Utility&>(c)<<"\n";
	out<<print::title("EWALD - COULOMB",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//member functions

/**
* set defaults
*/
void Coulomb::defaults(){
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Coulomb::defaults():\n";
	Utility::defaults();
	vSelfR_=0;
	vSelfK_=0;
	vSelfC_=0;
}

/**
* initialiize
* @param struc - structure
* @param prec - precision of ewald calculation
*/
void Coulomb::init(const Structure& struc, double prec){
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Coulomb::init(const Cell&, int, double):\n";
	Utility::init(prec);
	//local function variables
	const double pi=math::constant::PI;
	
	//compute the limits
	if(EWALD_PRINT_STATUS>0) std::cout<<"computing the limits\n";
	alpha_=std::pow(struc.nAtoms()*weight_*pi*pi*pi/(struc.vol()*struc.vol()),1.0/6.0);
	rMax_=std::sqrt(-1.0*std::log(prec))/alpha_;
	kMax_=2.0*alpha_*std::sqrt(-1.0*std::log(prec));
	if(EWALD_PRINT_DATA>0) std::cout<<"ALPHA = "<<alpha_<<" R_MAX = "<<rMax_<<" K_MAX = "<<kMax_<<"\n";
	
	//find the lattice vectors
	if(EWALD_PRINT_STATUS>0) std::cout<<"finding the lattice vectors\n";
	const int rNMaxX=std::ceil(rMax_/struc.R().row(0).lpNorm<Eigen::Infinity>());//max of row - max abs x the lattice vectors
	const int rNMaxY=std::ceil(rMax_/struc.R().row(1).lpNorm<Eigen::Infinity>());//max of row - max abs y the lattice vectors
	const int rNMaxZ=std::ceil(rMax_/struc.R().row(2).lpNorm<Eigen::Infinity>());//max of row - max abs z the lattice vectors
	const int kNMaxX=std::ceil(kMax_/struc.K().row(0).lpNorm<Eigen::Infinity>());//max of row - max abs x the lattice vectors
	const int kNMaxY=std::ceil(kMax_/struc.K().row(1).lpNorm<Eigen::Infinity>());//max of row - max abs y the lattice vectors
	const int kNMaxZ=std::ceil(kMax_/struc.K().row(2).lpNorm<Eigen::Infinity>());//max of row - max abs z the lattice vectors
	if(EWALD_PRINT_DATA>0) std::cout<<"RN = ("<<rNMaxX<<","<<rNMaxY<<","<<rNMaxZ<<") - "<<(2*rNMaxX+1)*(2*rNMaxY+1)*(2*rNMaxZ+1)<<"\n";
	if(EWALD_PRINT_DATA>0) std::cout<<"KN = ("<<kNMaxX<<","<<kNMaxY<<","<<kNMaxZ<<") - "<<(2*kNMaxX+1)*(2*kNMaxY+1)*(2*kNMaxZ+1)<<"\n";
	const double f=1.05;
	R.resize((2*rNMaxX+1)*(2*rNMaxY+1)*(2*rNMaxZ+1));
	int rCount=0;
	for(int i=-rNMaxX; i<=rNMaxX; ++i){
		for(int j=-rNMaxY; j<=rNMaxY; ++j){
			for(int k=-rNMaxZ; k<=rNMaxZ; ++k){
				//note: don't skip R=0
				const Eigen::Vector3d vtemp=i*struc.R().col(0)+j*struc.R().col(1)+k*struc.R().col(2);
				if(vtemp.norm()<f*rMax_) R[rCount++].noalias()=vtemp;
			}
		}
	}
	R.resize(rCount);
	K.resize((2*kNMaxX+1)*(2*kNMaxY+1)*(2*kNMaxZ+1));
	int kCount=0;
	for(int i=-kNMaxX; i<=kNMaxX; ++i){
		for(int j=-kNMaxY; j<=kNMaxY; ++j){
			for(int k=-kNMaxZ; k<=kNMaxZ; ++k){
				//note: skip K=0
				const Eigen::Vector3d vtemp=i*struc.K().col(0)+j*struc.K().col(1)+k*struc.K().col(2);
				const double norm=vtemp.norm();
				if(norm>0 && norm<f*kMax_) K[kCount++].noalias()=vtemp;
			}
		}
	}
	K.resize(kCount);
	if(EWALD_PRINT_DATA>0) std::cout<<"rCount = "<<rCount<<", kCount = "<<kCount<<"\n";
	
	//compute the k sum amplitudes
	if(EWALD_PRINT_STATUS>0) std::cout<<"computing the K-amplitudes\n";
	kAmp.resize(kCount);
	for(int i=0; i<kCount; ++i){
		#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
		kAmp[i]=4.0*math::constant::PI/struc.vol()*std::exp(-K[i].dot(K[i])/(4.0*alpha_*alpha_))/(K[i].dot(K[i]));
		#elif defined __ICC || defined __INTEL_COMPILER
		kAmp[i]=4.0*math::constant::PI/struc.vol()*exp(-K[i].dot(K[i])/(4.0*alpha_*alpha_))/(K[i].dot(K[i]));
		#endif
	}
	
	//compute the self-interaction strength
	if(EWALD_PRINT_STATUS>0) std::cout<<"computing self-interaction strength\n";
	vSelfR_=0; vSelfK_=0;
	for(int i=0; i<R.size(); ++i){
		const double RN=R[i].norm();
		if(RN!=0) vSelfR_+=std::erfc(alpha_*RN)/RN;
	}
	for(int i=0; i<kAmp.size(); ++i){
		vSelfK_+=kAmp[i];
	}
	vSelfC_=2.0*alpha_/math::constant::RadPI;
}

void Coulomb::init_alpha(const Structure& struc, double prec){
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Coulomb::init_alpha(const Structure&,double):\n";
	if(prec>0) prec_=prec;
	alpha_=std::pow(struc.nAtoms()*weight_*math::constant::PI*math::constant::PI*math::constant::PI/(struc.vol()*struc.vol()),1.0/6.0);
	rMax_=std::sqrt(-1.0*std::log(prec_))/alpha_;
	kMax_=2.0*alpha_*std::sqrt(-1.0*std::log(prec_));
	if(EWALD_PRINT_DATA>0) std::cout<<"ALPHA = "<<alpha_<<" R_MAX = "<<rMax_<<" K_MAX = "<<kMax_<<"\n";
}

//calculation - energy

/**
* compute the energy of a structure
* @param struc - structure
* @return the coulombic energy of the structure
*/
double Coulomb::energy(const Structure& struc)const{
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Coulomb::energy(const Structure&,int,int*):\n";
	//local function variables
	double energyT=0,q2s=0;
	const std::complex<double> I(0,1);
	Eigen::Vector3d vec;
	if(EWALD_PRINT_DATA==0){
		//r-space
		for(int i=0; i<struc.nAtoms(); ++i){
			for(int j=0; j<i; ++j){
				Cell::diff(struc.posn(i),struc.posn(j),vec,struc.R(),struc.RInv());
				double energyS=0;
				for(int n=0; n<R.size(); ++n){
					const double dist=(vec+R[n]).norm();
					#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
					energyS+=std::erfc(alpha_*dist)/dist;
					#elif defined __ICC || defined __INTEL_COMPILER
					energyS+=erfc(alpha_*dist)/dist;
					#endif
				}
				energyT+=struc.charge(i)*struc.charge(j)*energyS;
			}
		}
		//k-space
		for(int n=0; n<K.size(); ++n){
			std::complex<double> sf(0,0);
			for(int i=0; i<struc.nAtoms(); ++i){
				#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
				sf+=struc.charge(i)*std::exp(-I*K[n].dot(struc.posn(i)));
				#elif defined __ICC || defined __INTEL_COMPILER
				sf+=struc.charge(i)*exp(-I*K[n].dot(struc.posn(i)));
				#endif
			}
			energyT+=0.5*kAmp[n]*std::norm(sf);
		}
		//total charge and dipole
		//vec.setZero();
		for(int i=0; i<struc.nAtoms(); ++i){
			q2s+=struc.charge(i)*struc.charge(i);
			//vec.noalias()+=struc.charge(i)*struc.posn(i);
		}
		//self-energy 
		energyT+=0.5*q2s*(vSelfR_-vSelfC_);
		//polarization energy
		//energyT+=2.0*math::constant::PI/(2.0*eps_+1.0)*vec.dot(vec)/struc.vol();
	} else {
		double energyR=0,energyK=0,energyS,energyP;
		//r-space
		for(int i=0; i<struc.nAtoms(); ++i){
			for(int j=0; j<i; ++j){
				Cell::diff(struc.posn(i),struc.posn(j),vec,struc.R(),struc.RInv());
				energyS=0;
				for(int n=0; n<R.size(); ++n){
					const double dist=(vec+R[n]).norm();
					#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
					energyS+=std::erfc(alpha_*dist)/dist;
					#elif defined __ICC || defined __INTEL_COMPILER
					energyS+=erfc(alpha_*dist)/dist;
					#endif
				}
				energyR+=struc.charge(i)*struc.charge(j)*energyS;
			}
		}
		//k-space
		for(int n=0; n<K.size(); ++n){
			std::complex<double> sf(0,0);
			for(int i=0; i<struc.nAtoms(); ++i){
				#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
				sf+=struc.charge(i)*std::exp(-I*K[n].dot(struc.posn(i)));
				#elif defined __ICC || defined __INTEL_COMPILER
				sf+=struc.charge(i)*exp(-I*K[n].dot(struc.posn(i)));
				#endif
			}
			energyK+=0.5*kAmp[n]*std::norm(sf);
		}
		//self-energy 
		for(int i=0; i<struc.nAtoms(); ++i) q2s+=struc.charge(i)*struc.charge(i);
		energyS=0.5*q2s*(vSelfR_-vSelfC_);
		//polarization energy
		vec.setZero();
		for(int i=0; i<struc.nAtoms(); ++i) vec.noalias()+=struc.charge(i)*struc.posn(i);
		energyP=2.0*math::constant::PI/(2.0*eps_+1.0)*vec.dot(vec)/struc.vol();
		
		std::cout<<"==============================\n";
		std::cout<<"dipole   = "<<vec.transpose()<<"\n";
		std::cout<<"ke       = "<<units::consts::ke()<<"\n";
		std::cout<<"energy-r = "<<energyR<<"\n";
		std::cout<<"energy-k = "<<energyK<<"\n";
		std::cout<<"energy-s = "<<energyS<<"\n";
		std::cout<<"energy-p = "<<energyP<<"\n";
		std::cout<<"energy-t = "<<energyR+energyK+energyS+energyP<<"\n";
		std::cout<<"==============================\n";
		//energyT=energyR+energyK+energyS+energyP;
		energyT=energyR+energyK+energyS;
	}
	return units::consts::ke()*energyT;
}

/**
* compute the energy of a structure - no initialization
* @param struc - structure
* @return the coulombic energy of the structure
*/
double Coulomb::energy_single(const Structure& struc){
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Coulomb::energy(const Structure&,int,int*):\n";
	//local function variables
	int shellx,shelly,shellz,count=0;
	double energyR=0,energyK=0,energyS,energyP,q2s=0;
	const std::complex<double> I(0,1);
	Eigen::Vector3d vec;
	
	//init alpha
	init_alpha(struc);
	
	//r-space
	shellx=std::ceil(rMax_/struc.R().row(0).lpNorm<Eigen::Infinity>());//max of row - max abs x the lattice vectors
	shelly=std::ceil(rMax_/struc.R().row(1).lpNorm<Eigen::Infinity>());//max of row - max abs y the lattice vectors
	shellz=std::ceil(rMax_/struc.R().row(2).lpNorm<Eigen::Infinity>());//max of row - max abs z the lattice vectors
	if(EWALD_PRINT_DATA>0) std::cout<<"R_SHELL = ("<<shellx<<","<<shelly<<","<<shellz<<")\n";
	R.resize((2*shellx+1)*(2*shelly+1)*(2*shellz+1));
	count=0;
	for(int i=-shellx; i<=shellx; ++i){
		for(int j=-shelly; j<=shelly; ++j){
			for(int k=-shellz; k<=shellz; ++k){
				//note: don't skip R=0
				const Eigen::Vector3d vtemp=i*struc.R().col(0)+j*struc.R().col(1)+k*struc.R().col(2);
				if(vtemp.norm()<1.05*rMax_) R[count++].noalias()=vtemp;
			}
		}
	}
	R.resize(count);
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<i; ++j){
			Cell::diff(struc.posn(i),struc.posn(j),vec,struc.R(),struc.RInv());
			energyS=0;
			for(int n=0; n<R.size(); ++n){
				double dist=(vec+R[n]).norm();
				#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
				energyS+=std::erfc(alpha_*dist)/dist;
				#elif defined __ICC || defined __INTEL_COMPILER
				energyS+=erfc(alpha_*dist)/dist;
				#endif
			}
			energyR+=struc.charge(i)*struc.charge(j)*energyS;
		}
	}
	
	//k-space
	shellx=std::ceil(kMax_/struc.K().row(0).lpNorm<Eigen::Infinity>());//max of row - max abs x the lattice vectors
	shelly=std::ceil(kMax_/struc.K().row(1).lpNorm<Eigen::Infinity>());//max of row - max abs y the lattice vectors
	shellz=std::ceil(kMax_/struc.K().row(2).lpNorm<Eigen::Infinity>());//max of row - max abs z the lattice vectors
	if(EWALD_PRINT_DATA>0) std::cout<<"K_SHELL = ("<<shellx<<","<<shelly<<","<<shellz<<")\n";
	K.resize((2*shellx+1)*(2*shelly+1)*(2*shellz+1));
	count=0;
	for(int i=-shellx; i<=shellx; ++i){
		for(int j=-shelly; j<=shelly; ++j){
			for(int k=-shellz; k<=shellz; ++k){
				const Eigen::Vector3d vtemp=i*struc.K().col(0)+j*struc.K().col(1)+k*struc.K().col(2);
				const double norm=vtemp.norm();//note: skip K=0
				if(norm>0 && norm<1.05*kMax_) K[count++].noalias()=vtemp;
			}
		}
	}
	K.resize(count);
	for(int n=0; n<K.size(); ++n){
		std::complex<double> sf(0,0);
		const double kdot=K[n].dot(K[n]);
		const double kamp=4.0*math::constant::PI/struc.vol()*std::exp(-kdot/(4.0*alpha_*alpha_))/kdot;
		for(int i=0; i<struc.nAtoms(); ++i){
			#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
			sf+=struc.charge(i)*std::exp(-I*K[n].dot(struc.posn(i)));
			#elif defined __ICC || defined __INTEL_COMPILER
			sf+=struc.charge(i)*exp(-I*K[n].dot(struc.posn(i)));
			#endif
		}
		energyK+=0.5*kamp*std::norm(sf);
	}
	
	//total charge, dipole
	vec.setZero();
	for(int i=0; i<struc.nAtoms(); ++i){
		q2s+=struc.charge(i)*struc.charge(i);
		vec.noalias()+=struc.charge(i)*struc.posn(i);
	}
	
	//self-energy
	double vSelfR=0;
	for(int i=0; i<R.size(); ++i){
		#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
		if(R[i].norm()!=0) vSelfR+=std::erfc(alpha_*R[i].norm())/R[i].norm();
		#elif defined __ICC || defined __INTEL_COMPILER
		if(R[i].norm()!=0) vSelfR+=erfc(alpha_*R[i].norm())/R[i].norm();
		#endif
	}
	energyS=0.5*q2s*(vSelfR-2.0*alpha_/math::constant::RadPI);
	
	//polarization energy
	if(EWALD_PRINT_DATA>0) std::cout<<"dipole = "<<vec.transpose()<<"\n";
	energyP=2.0*math::constant::PI/(2.0*eps_+1.0)*vec.dot(vec)/struc.vol();
	
	if(EWALD_PRINT_DATA>0){
		std::cout<<"==============================\n";
		std::cout<<"dipole   = "<<vec.transpose()<<"\n";
		std::cout<<"energy-r = "<<energyR<<"\n";
		std::cout<<"energy-k = "<<energyK<<"\n";
		std::cout<<"energy-s = "<<energyS<<"\n";
		std::cout<<"energy-p = "<<energyP<<"\n";
		std::cout<<"energy-t = "<<energyR+energyK+energyS+energyP<<"\n";
		std::cout<<"==============================\n";
	}
	
	//return units::consts::ke()*(energyR+energyK+energyS+energyP);
	return units::consts::ke()*(energyR+energyK+energyS);
}

/**
* compute the energy of a structure - brute force
* @param struc - structure
* @return the coulombic energy of the structure
*/
double Coulomb::energy_brute(const Structure& struc, int N)const{
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Coulomb::energy_brute(const Structure&,int,int*):\n";
	//local function variables
	double interEnergy=0,energySelf=0,chargeSum=0;
	Eigen::Vector3d dr;
	
	if(EWALD_PRINT_STATUS>0) std::cout<<"interaction energy\n";
	for(int n=0; n<struc.nAtoms(); ++n){
		for(int m=n+1; m<struc.nAtoms(); ++m){
			if(EWALD_PRINT_DATA>1) std::cout<<"Pair ("<<n<<","<<m<<")\n";
			//find the zeroth cell distance and energy
			Cell::diff(struc.posn(n),struc.posn(m),dr,struc.R(),struc.RInv());
			const double chgProd=struc.charge(n)*struc.charge(m);
			for(int i=-N; i<=N; ++i){
				for(int j=-N; j<=N; ++j){
					for(int k=-N; k<=N; ++k){
						interEnergy+=chgProd/(dr+i*struc.R().col(0)+j*struc.R().col(1)+k*struc.R().col(2)).norm();
					}
				}
			}
		}
	}
	
	if(EWALD_PRINT_STATUS>0) std::cout<<"self-energy\n";
	for(int n=0; n<struc.nAtoms(); ++n){
		chargeSum+=struc.charge(n)*struc.charge(n);
	}
	energySelf=0;
	for(int i=-N; i<=N; ++i){
		for(int j=-N; j<=N; ++j){
			for(int k=-N; k<=N; ++k){
				const double norm=(i*struc.R().col(0)+j*struc.R().col(1)+k*struc.R().col(2)).norm();
				energySelf+=(norm>0)?1.0/norm:0;
			}
		}
	}
	energySelf*=0.5*chargeSum;
	
	if(EWALD_PRINT_DATA>0){
		std::cout<<"energy-s = "<<energySelf<<"\n";
		std::cout<<"energy-i = "<<interEnergy<<"\n";
		std::cout<<"energy-t = "<<interEnergy+energySelf<<"\n";
	}
	
	return units::consts::ke()*(interEnergy+energySelf);
}

//calculation - potential

/**
* compute the electrostatic potential at a given point
* @param struc - structure
* @param r - position
* @return the electrostatic potential at r
*/
double Coulomb::phi(const Structure& struc, const Eigen::Vector3d& r)const{
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Coulomb::phi(const Structure&,const Eigen::Vector3d&)const:\n";
	double vTot=0,r2=0,chgsum=0;
	Eigen::Vector3d dr;
	if(EWALD_PRINT_DATA==0){
		for(int i=0; i<struc.nAtoms(); ++i){
			double vLoc=0;
			//compute difference
			Cell::diff(r,struc.posn(i),dr,struc.R(),struc.RInv());
			//real-space contribution
			for(int n=0; n<R.size(); ++n){
				const double dist=(dr+R[n]).norm();
				if(dist>math::constant::ZERO){
					#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
					vLoc+=std::erfc(alpha_*dist)/dist;
					#elif defined __ICC || defined __INTEL_COMPILER
					vLoc+=erfc(alpha_*dist)/dist;
					#endif
				}
			}
			//k-space contribution
			for(int n=0; n<K.size(); ++n){
				#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
				vLoc+=kAmp[n]*std::cos(K[n].dot(dr));
				#elif defined __ICC || defined __INTEL_COMPILER
				vLoc+=kAmp[n]*cos(K[n].dot(dr));
				#endif
			}
			vTot+=vLoc*struc.charge(i);
			r2+=dr.squaredNorm()*struc.charge(i);
			chgsum+=struc.charge(i);
		}
		//polarization contribution
		vTot+=-2.0*math::constant::PI/(2.0*eps_+1.0)*r2/struc.vol();
		//constant contribution
		vTot+=-math::constant::PI/(alpha_*alpha_*struc.vol())*chgsum;
	} else {
		double vR=0,vK=0,vP,vC;
		Eigen::Vector3d dr;
		for(int i=0; i<struc.nAtoms(); ++i){
			Cell::diff(r,struc.posn(i),dr,struc.R(),struc.RInv());
			for(int n=0; n<R.size(); ++n){
				const double dist=(dr+R[n]).norm();
				if(dist>math::constant::ZERO){
					#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
					vR+=std::erfc(alpha_*dist)/dist*struc.charge(i);
					#elif defined __ICC || defined __INTEL_COMPILER
					vR+=erfc(alpha_*dist)/dist*struc.charge(i);
					#endif
				}
			}
			for(int n=0; n<K.size(); ++n){
				#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
				vK+=kAmp[n]*std::cos(K[n].dot(dr))*struc.charge(i);
				#elif defined __ICC || defined __INTEL_COMPILER
				vK+=kAmp[n]*cos(K[n].dot(dr))*struc.charge(i);
				#endif
			}
			r2+=dr.squaredNorm()*struc.charge(i);
			chgsum+=struc.charge(i);
		}
		//polarization contribution
		vP=-2.0*math::constant::PI/(2.0*eps_+1.0)*r2/struc.vol();
		//constant contribution
		vC=-math::constant::PI/(alpha_*alpha_*struc.vol())*chgsum;
		
		std::cout<<"==============================\n";
		std::cout<<"chg-t = "<<chgsum<<"\n";
		std::cout<<"pot-r = "<<vR<<"\n";
		std::cout<<"pot-k = "<<vK<<"\n";
		std::cout<<"pot-p = "<<vP<<"\n";
		std::cout<<"pot-c = "<<vC<<"\n";
		std::cout<<"pot-t = "<<vR+vK+vP+vC<<"\n";
		std::cout<<"==============================\n";
		vTot=vR+vK+vP+vC;
	}
	
	return units::consts::ke()*vTot;
}

double Coulomb::phi(const Structure& struc, int nn)const{
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Coulomb::phi(const Structure&,int)const:\n";
	return phi(struc,struc.posn(nn))+units::consts::ke()*((vSelfR_-vSelfC_)*struc.charge(nn));
}

double Coulomb::phis()const{
	return units::consts::ke()*(vSelfR_+vSelfK_-vSelfC_);
}

double Coulomb::potentialBrute(const Structure& struc, int n, int N)const{
	if(EWALD_PRINT_FUNC>0) std::cout<<"Ewald3D::Coulomb::potentialBrute(const Structure&,int,const Eigen::Vector3d&)const:\n";
	double v=0;
	Eigen::Vector3d dr;
	
	for(int ii=0; ii<struc.nAtoms(); ++ii){
		if(ii==n) continue;
		Cell::diff(struc.posn(ii),struc.posn(n),dr,struc.R(),struc.RInv());
		for(int i=-N; i<=N; ++i){
			for(int j=-N; j<=N; ++j){
				for(int k=-N; k<=N; ++k){
					v+=1.0/(dr+i*struc.R().col(0)+j*struc.R().col(1)+k*struc.R().col(2)).norm()*struc.charge(ii);
				}
			}
		}
	}
	
	return units::consts::ke()*v;
}

//calculation - electric field

/**
* compute the electric field at a given atom
* @param struc - structure
* @param n - index of atom
* @param field - the electric field
* @return the electric field
*/
Eigen::Vector3d& Coulomb::efield(const Structure& struc, int n, Eigen::Vector3d& field)const{
	if(EWALD_PRINT_FUNC>0) std::cout<<"field(const Structure&,int,Eigen::Vector3d&)const:\n";
	const double aa=2.0*alpha_/std::sqrt(math::constant::PI);
	const double bb=4.0*math::constant::PI/(2.0*eps_+1.0);
	Eigen::Vector3d dr;
	
	//zero field 
	field.setZero();
	
	//compute real/reciprocal space contributions
	for(int i=0; i<struc.nAtoms(); ++i){
		if(i==n) continue;
		Cell::diff(struc.posn(i),struc.posn(n),dr,struc.R(),struc.RInv());
		field.noalias()+=bb*dr*struc.charge(i);
		for(int n=0; n<R.size(); ++n){
			const double dist=(dr+R[n]).norm();
			field.noalias()+=(dr+R[n])*(aa*std::exp(-alpha_*alpha_*dist*dist)
				+std::erfc(alpha_*dist)/dist)*struc.charge(i)/(dist*dist);
		}
		for(int n=0; n<K.size(); ++n){
			field.noalias()+=K[n]*kAmp[n]*std::sin(K[n].dot(dr))*struc.charge(i);
		}
	}
	
	return field;
}

/**
* compute the electric field at a given atom - brute force
* @param struc - structure
* @param n - index of atom
* @param field - the electric field
* @return the electric field
*/
Eigen::Vector3d& Coulomb::efieldBrute(const Structure& struc, int n, Eigen::Vector3d& field, int N)const{
	if(EWALD_PRINT_FUNC>0) std::cout<<"potentialBrute(const Structure&,int,Eigen::Vector3d&,int)const:\n";
	Eigen::Vector3d dr;
	
	//zero field
	field.setZero();
	
	for(int ii=0; ii<struc.nAtoms(); ++ii){
		Cell::diff(struc.posn(ii),struc.posn(n),dr,struc.R(),struc.RInv());
		for(int i=-N; i<=N; ++i){
			for(int j=-N; j<=N; ++j){
				for(int k=-N; k<=N; ++k){
					const double drn=(dr+i*struc.R().col(0)+j*struc.R().col(1)+k*struc.R().col(2)).norm();
					if(drn>math::constant::ZERO) field.noalias()+=dr/(drn*drn*drn)*struc.charge(ii);
				}
			}
		}
	}
	
	return field;
}

//potential

}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const Ewald3D::Utility& obj){
	int size=0;
	size+=sizeof(double);//prec_
	size+=sizeof(double);//weight_
	size+=sizeof(double);//eps_
	return size;
}
template <> int nbytes(const Ewald3D::Coulomb& obj){
	return nbytes(static_cast<const Ewald3D::Utility&>(obj));
}

//**********************************************
// packing
//**********************************************

template <> int pack(const Ewald3D::Utility& obj, char* arr){
	int pos=0;
	const double prec=obj.prec();
	std::memcpy(arr+pos,&prec,sizeof(double)); pos+=sizeof(double);//prec_
	std::memcpy(arr+pos,&obj.weight(),sizeof(double)); pos+=sizeof(double);//weight_
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);//eps_
	return pos;
}
template <> int pack(const Ewald3D::Coulomb& obj, char* arr){
	return pack(static_cast<const Ewald3D::Utility&>(obj),arr);
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(Ewald3D::Utility& obj, const char* arr){
	int pos=0;
	double prec=0;
	std::memcpy(&prec,arr+pos,sizeof(double)); pos+=sizeof(double);//prec_
	std::memcpy(&obj.weight(),arr+pos,sizeof(double)); pos+=sizeof(double);//weight_
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);//eps_
	obj.init(prec);
	return pos;
}
template <> int unpack(Ewald3D::Coulomb& obj, const char* arr){
	return unpack(static_cast<Ewald3D::Utility&>(obj),arr);
}
	
}
