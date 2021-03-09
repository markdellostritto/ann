// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
#include <cstring>
// c++ libaries
#include <ostream>
// ann - symm - angular - g3
#include "symm_angular_g3.hpp"

//*****************************************
// PHIA - G3 - Behler
//*****************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PhiA_G3& f){
	return out<<"G3 "<<f.eta<<" "<<f.zeta<<" "<<f.lambda;
}

/**
* compute equality of two symmetry functions
* @param phia1 - G3 - first
* @param phia2 - G3 - second
* @return equiality of phia1 and phia2
*/
bool operator==(const PhiA_G3& phia1, const PhiA_G3& phia2){
	if(static_cast<const PhiA&>(phia1)!=static_cast<const PhiA&>(phia2)) return false;
	else if(phia1.lambda!=phia2.lambda) return false;
	else if(phia1.eta!=phia2.eta) return false;
	else if(phia1.zeta!=phia2.zeta) return false;
	else return true;
}

//==== member functions ====

/**
* compute the value of the function
* @param cos - cosine of the angle between rij and rik
* @param r - distances rij, rik, rjk
* @param c - cutoff as a function of rij, rik, rjk
*/
double PhiA_G3::val(double cos, const double r[3], const double c[3])const noexcept{
	return pow(fabs(0.5*(1.0+lambda*cos)),zeta)*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))*c[0]*c[1]*c[2];
	//return angle(cos)*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk))*cij*cik*cjk;
}

/**
* compute the value of the distance-dependent part of the function
* @param r - distances rij, rik, rjk
* @param c - cutoff as a function of rij, rik, rjk
*/
double PhiA_G3::dist(const double r[3], const double c[3])const noexcept{
	return exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))*c[0]*c[1]*c[2];
	//return std::exp(-eta*(rij*rij+rik*rik+rjk*rjk))*cij*cik*cjk;
}

/**
* compute the value of the angular-dependent part of the function
* @param cos - cosine of the angle between rij and rik
*/
double PhiA_G3::angle(double cos)const noexcept{
	return pow(fabs(0.5*(1.0+lambda*cos)),zeta);
}

/**
* compute the gradient w.r.t. the angle
* @param cos - cosine of the angle between rij and rik
*/
double PhiA_G3::grad_angle(double cos)const noexcept{
	return 0.5*zeta*lambda*pow(fabs(0.5*(1.0+lambda*cos)),zeta-1.0);
}

/**
* compute the angular-dependent part of the function and gradient w.r.t. the angle
* @param cos - cosine of the angle between rij and rik
*/
void PhiA_G3::compute_angle(double cos, double& val, double& grad)const noexcept{
	cos=fabs(0.5*(1.0+lambda*cos));
	grad=pow(cos,zeta-1.0);
	val=cos*grad;
	grad*=0.5*zeta*lambda;
}

/**
* compute the gradient with respect to the first argument
* @param r - distances rij, rik, rjk
* @param c - cutoff as a function of rij, rik, rjk
* @param gij - gradient of cutoff w.r.t. rij
*/
double PhiA_G3::grad_dist_0(const double r[3], const double c[3], double gij)const noexcept{
	return (-2.0*eta*r[0]*c[0]+gij)*c[1]*c[2]*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	//return (-2.0*eta*rij*cij+gij)*cik*cjk*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

/**
* compute the gradient with respect to the second argument
* @param r - distances rij, rik, rjk
* @param c - cutoff as a function of rij, rik, rjk
* @param gjk - gradient of cutoff w.r.t. rik
*/
double PhiA_G3::grad_dist_1(const double r[3], const double c[3], double gik)const noexcept{
	return (-2.0*eta*r[1]*c[1]+gik)*c[0]*c[2]*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	//return (-2.0*eta*rik*cik+gik)*cij*cjk*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

/**
* compute the gradient with respect to the third argument
* @param r - distances rij, rik, rjk
* @param c - cutoff as a function of rij, rik, rjk
* @param gjk - gradient of cutoff w.r.t. rjk
*/
double PhiA_G3::grad_dist_2(const double r[3], const double c[3], double gjk)const noexcept{
	return (-2.0*eta*r[2]*c[2]+gjk)*c[0]*c[1]*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	//return (-2.0*eta*rjk*cjk+gjk)*cij*cik*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

/**
* compute the distance-dependent part of the function and the gradients
* @param r - distances rij, rik, rjk
* @param c - cutoff as a function of rij, rik, rjk
* @param gjk - gradient of cutoff w.r.t. rjk
*/
void PhiA_G3::compute_dist(const double r[3], const double c[3], const double g[3], double& dist, double* gradd)const noexcept{
	const double expf=exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	dist=expf*c[0]*c[1]*c[2];
	gradd[0]=(-2.0*eta*r[0]*c[0]+g[0])*c[1]*c[2]*expf;
	gradd[1]=(-2.0*eta*r[1]*c[1]+g[1])*c[0]*c[2]*expf;
	gradd[2]=(-2.0*eta*r[2]*c[2]+g[2])*c[0]*c[1]*expf;
}

//*****************************************
// PHIA - G3 - Behler - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiA_G3& obj){
		int N=0;
		N+=nbytes(static_cast<const PhiA&>(obj));
		N+=2*sizeof(double);
		N+=sizeof(int);
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiA_G3& obj, char* arr){
		int pos=0;
		pos+=pack(static_cast<const PhiA&>(obj),arr);
		std::memcpy(arr+pos,&obj.eta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.zeta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.lambda,sizeof(int));  pos+=sizeof(int);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiA_G3& obj, const char* arr){
		int pos=0;
		pos+=unpack(static_cast<PhiA&>(obj),arr);
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.zeta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.lambda,arr+pos,sizeof(int));  pos+=sizeof(int);
		return pos;
	}
	
}
