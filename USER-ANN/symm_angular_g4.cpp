// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
#include <cstring>
// c++ libaries
#include <ostream>
// ann - symm - angular - g4
#include "symm_angular_g4.h"

//*****************************************
// PHIA - G4 - Behler
//*****************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PhiA_G4& f){
	return out<<static_cast<const PhiA&>(f)<<" G4 "<<f.eta<<" "<<f.zeta<<" "<<f.lambda;
}
bool operator==(const PhiA_G4& phia1, const PhiA_G4& phia2){
	if(static_cast<const PhiA&>(phia1)!=static_cast<const PhiA&>(phia2)) return false;
	else if(phia1.lambda!=phia2.lambda) return false;
	else if(phia1.eta!=phia2.eta) return false;
	else if(phia1.zeta!=phia2.zeta) return false;
	else return true;
}

//==== member functions - evaluation ====

double PhiA_G4::val(double cos, const double r[3], const double c[3])const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return std::pow(std::fabs(0.5*(1.0+lambda*cos)),zeta)*std::exp(-eta*(r[0]*r[0]+r[1]*r[1]))*c[0]*c[1];
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return pow(fabs(0.5*(1.0+lambda*cos)),zeta)*exp(-eta*(r[0]*r[0]+r[1]*r[1]))*c[0]*c[1];
	#endif
	//return angle(cos)*std::exp(-eta*(rij*rij+rik*rik))*cij*cik;
}

double PhiA_G4::dist(const double r[3], const double c[3])const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return std::exp(-eta*(r[0]*r[0]+r[1]*r[1]))*c[0]*c[1];
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return exp(-eta*(r[0]*r[0]+r[1]*r[1]))*c[0]*c[1];
	#endif
	//return std::exp(-eta*(rij*rij+rik*rik))*cij*cik;
}

double PhiA_G4::angle(double cos)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return std::pow(std::fabs(0.5*(1.0+lambda*cos)),zeta);
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return pow(fabs(0.5*(1.0+lambda*cos)),zeta);
	#endif
}

double PhiA_G4::grad_angle(double cos)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return 0.5*zeta*lambda*std::pow(std::fabs(0.5*(1.0+lambda*cos)),zeta-1.0);
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return 0.5*zeta*lambda*pow(fabs(0.5*(1.0+lambda*cos)),zeta-1.0);
	#endif
}

void PhiA_G4::compute_angle(double cos, double& val, double& grad)const{
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

double PhiA_G4::grad_dist_0(const double r[3], const double c[3], double gij)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return (-2.0*eta*r[0]*c[0]+gij)*c[1]*std::exp(-eta*(r[0]*r[0]+r[1]*r[1]));
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return (-2.0*eta*r[0]*c[0]+gij)*c[1]*exp(-eta*(r[0]*r[0]+r[1]*r[1]));
	#endif
	//return (-2.0*eta*rij*cij+gij)*cik*std::exp(-eta*(rij*rij+rik*rik));
}

double PhiA_G4::grad_dist_1(const double r[3], const double c[3], double gik)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return (-2.0*eta*r[1]*c[1]+gik)*c[0]*std::exp(-eta*(r[0]*r[0]+r[1]*r[1]));
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return (-2.0*eta*r[1]*c[1]+gik)*c[0]*exp(-eta*(r[0]*r[0]+r[1]*r[1]));
	#endif
	//return (-2.0*eta*rik*cik+gik)*cij*std::exp(-eta*(rij*rij+rik*rik));
}

double PhiA_G4::grad_dist_2(const double r[3], const double c[3], double gjk)const{
	return 0.0;
}

void PhiA_G4::compute_dist(const double r[3], const double c[3], const double g[3], double& dist, double* gradd)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	const double expf=std::exp(-eta*(r[0]*r[0]+r[1]*r[1]));
	#elif (defined __ICC || defined __INTEL_COMPILER)
	const double expf=exp(-eta*(r[0]*r[0]+r[1]*r[1]));
	#endif
	dist=expf*c[0]*c[1];
	gradd[0]=(-2.0*eta*r[0]*c[0]+g[0])*c[1]*expf;
	gradd[1]=(-2.0*eta*r[1]*c[1]+g[1])*c[0]*expf;
	gradd[2]=0.0;
}

//*****************************************
// PHIA - G4 - Behler - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiA_G4& obj){
		int N=0;
		N+=nbytes(static_cast<const PhiA&>(obj));
		N+=2*sizeof(double);
		N+=sizeof(int);
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiA_G4& obj, char* arr){
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
	
	template <> int unpack(PhiA_G4& obj, const char* arr){
		int pos=0;
		pos+=unpack(static_cast<PhiA&>(obj),arr);
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.zeta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.lambda,arr+pos,sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
}
