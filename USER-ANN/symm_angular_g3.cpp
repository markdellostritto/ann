// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
// c++ libaries
#include <ostream>
// ann - symm - angular - g3
#include "symm_angular_g3.h"

//Behler G3

//operators

std::ostream& operator<<(std::ostream& out, const PhiA_G3& f){
	return out<<"G3 "<<f.eta<<" "<<f.zeta<<" "<<f.lambda;
}

bool operator==(const PhiA_G3& phia1, const PhiA_G3& phia2){
	if(static_cast<const PhiA&>(phia1)!=static_cast<const PhiA&>(phia2)) return false;
	else if(phia1.lambda!=phia2.lambda) return false;
	else if(phia1.eta!=phia2.eta) return false;
	else if(phia1.zeta!=phia2.zeta) return false;
	else return true;
}

//member functions

double PhiA_G3::val(double cos, const double r[3], const double c[3])const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return angle(cos)*std::exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))*c[0]*c[1]*c[2];
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return angle(cos)*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))*c[0]*c[1]*c[2];
	#endif
	//return angle(cos)*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk))*cij*cik*cjk;
}

double PhiA_G3::dist(const double r[3], const double c[3])const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return std::exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))*c[0]*c[1]*c[2];
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]))*c[0]*c[1]*c[2];
	#endif
	//return std::exp(-eta*(rij*rij+rik*rik+rjk*rjk))*cij*cik*cjk;
}

double PhiA_G3::angle(double cos)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return std::pow(std::fabs(0.5*(1.0+lambda*cos)),zeta);
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return pow(fabs(0.5*(1.0+lambda*cos)),zeta);
	#endif
}

double PhiA_G3::grad_angle(double cos)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return 0.5*zeta*lambda*std::pow(std::fabs(0.5*(1.0+lambda*cos)),zeta-1.0);
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return 0.5*zeta*lambda*pow(fabs(0.5*(1.0+lambda*cos)),zeta-1.0);
	#endif
}

void PhiA_G3::compute_angle(double cos, double& val, double& grad)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	cos=std::fabs(0.5*(1.0+lambda*cos));
	grad=std::pow(cos,zeta-1.0);
	#elif (defined __ICC || defined __INTEL_COMPILER)
	cos=fabs(0.5*(1.0+lambda*cos));
	grad=pow(cos,zeta-1.0);
	#endif
	val=cos*grad;
	grad*=0.5*zeta*lambda;
}

double PhiA_G3::grad_dist_0(const double r[3], const double c[3], double gij)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return (-2.0*eta*r[0]*c[0]+gij)*c[1]*c[2]*std::exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return (-2.0*eta*r[0]*c[0]+gij)*c[1]*c[2]*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	#endif
	//return (-2.0*eta*rij*cij+gij)*cik*cjk*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

double PhiA_G3::grad_dist_1(const double r[3], const double c[3], double gik)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return (-2.0*eta*r[1]*c[1]+gik)*c[0]*c[2]*std::exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return (-2.0*eta*r[1]*c[1]+gik)*c[0]*c[2]*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	#endif
	//return (-2.0*eta*rik*cik+gik)*cij*cjk*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

double PhiA_G3::grad_dist_2(const double r[3], const double c[3], double gjk)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return (-2.0*eta*r[2]*c[2]+gjk)*c[0]*c[1]*std::exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return (-2.0*eta*r[2]*c[2]+gjk)*c[0]*c[1]*exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	#endif
	//return (-2.0*eta*rjk*cjk+gjk)*cij*cik*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

void PhiA_G3::compute_dist(const double r[3], const double c[3], const double g[3], double& dist, double* gradd)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	const double expf=std::exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	#elif (defined __ICC || defined __INTEL_COMPILER)
	const double expf=exp(-eta*(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]));
	#endif
	dist=expf*c[0]*c[1]*c[2];
	gradd[0]=(-2.0*eta*r[0]*c[0]+g[0])*c[1]*c[2]*expf;
	gradd[1]=(-2.0*eta*r[1]*c[1]+g[1])*c[0]*c[2]*expf;
	gradd[2]=(-2.0*eta*r[2]*c[2]+g[2])*c[0]*c[1]*expf;
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA_G3& obj){
		unsigned int N=0;
		N+=nbytes(static_cast<const PhiA&>(obj));
		N+=2*sizeof(double);
		N+=sizeof(int);
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const PhiA_G3& obj, char* arr){
		unsigned int pos=0;
		pack(static_cast<const PhiA&>(obj),arr); pos+=nbytes(static_cast<const PhiA&>(obj));
		std::memcpy(arr+pos,&obj.eta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.zeta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.lambda,sizeof(int));  pos+=sizeof(int);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(PhiA_G3& obj, const char* arr){
		unsigned int pos=0;
		unpack(static_cast<PhiA&>(obj),arr); pos+=nbytes(static_cast<const PhiA&>(obj));
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.zeta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.lambda,arr+pos,sizeof(int));  pos+=sizeof(int);
		return pos;
	}
	
}
