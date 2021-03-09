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
#include "symm_angular_g3.h"

//*****************************************
// PHIA - G3 - Behler
//*****************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PhiA_G3& f){
	return out<<static_cast<const PhiA&>(f)<<" G3 "<<f.eta<<" "<<f.zeta<<" "<<f.lambda;
}

bool operator==(const PhiA_G3& phia1, const PhiA_G3& phia2){
	if(static_cast<const PhiA&>(phia1)!=static_cast<const PhiA&>(phia2)) return false;
	else if(phia1.lambda!=phia2.lambda) return false;
	else if(phia1.eta!=phia2.eta) return false;
	else if(phia1.zeta!=phia2.zeta) return false;
	else return true;
}

//==== member functions ====

double PhiA_G3::val(double cos, const double r[3], const double c[3])const{
	return pow(fabs(0.5*(1.0+lambda*cos)),zeta)*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))*c[0]*c[1]*c[2];
	//return angle(cos)*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk))*cij*cik*cjk;
}

double PhiA_G3::dist(const double r[3], const double c[3])const{
	return exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))*c[0]*c[1]*c[2];
	//return std::exp(-eta*(rij*rij+rik*rik+rjk*rjk))*cij*cik*cjk;
}

double PhiA_G3::angle(double cos)const{
	return pow(fabs(0.5*(1.0+lambda*cos)),zeta);
}

double PhiA_G3::grad_angle(double cos)const{
	return 0.5*zeta*lambda*pow(fabs(0.5*(1.0+lambda*cos)),zeta-1.0);
}

void PhiA_G3::compute_angle(double cos, double& val, double& grad)const{
	cos=fabs(0.5*(1.0+lambda*cos));
	grad=pow(cos,zeta-1.0);
	val=cos*grad;
	grad*=0.5*zeta*lambda;
}

double PhiA_G3::grad_dist_0(const double r[3], const double c[3], double gij)const{
	return (-2.0*eta*r[0]*c[0]+gij)*c[1]*c[2]*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	//return (-2.0*eta*rij*cij+gij)*cik*cjk*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

double PhiA_G3::grad_dist_1(const double r[3], const double c[3], double gik)const{
	return (-2.0*eta*r[1]*c[1]+gik)*c[0]*c[2]*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	//return (-2.0*eta*rik*cik+gik)*cij*cjk*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

double PhiA_G3::grad_dist_2(const double r[3], const double c[3], double gjk)const{
	return (-2.0*eta*r[2]*c[2]+gjk)*c[0]*c[1]*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	//return (-2.0*eta*rjk*cjk+gjk)*cij*cik*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

void PhiA_G3::compute_dist(const double r[3], const double c[3], const double g[3], double& dist, double* gradd)const{
	const double expf=exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	dist=expf*c[0]*c[1]*c[2];
	gradd[0]=(-2.0*eta*r[0]*c[0]+g[0])*c[1]*c[2]*expf;
	gradd[1]=(-2.0*eta*r[1]*c[1]+g[1])*c[0]*c[2]*expf;
	gradd[2]=(-2.0*eta*r[2]*c[2]+g[2])*c[0]*c[1]*expf;
}

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
		std::memcpy(arr+pos,&obj.lambda,sizeof(int)); pos+=sizeof(int);
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
		std::memcpy(&obj.lambda,arr+pos,sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
}
