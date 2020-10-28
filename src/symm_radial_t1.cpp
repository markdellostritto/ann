// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
#include <cstring>
// c++ libaries
#include <ostream>
// ann - symm - radial - t1
#include "symm_radial_t1.hpp"

//*****************************************
// PHIR - T1 - DelloStritto
//*****************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PhiR_T1& f){
	return out<<"T1 "<<f.rs<<" "<<f.eta<<" ";
}

/**
* compute equality of two symmetry functions
* @param phi1 - G2 - first
* @param phi2 - G2 - second
* @return equiality of phi1 and phi2
*/
bool operator==(const PhiR_T1& phi1, const PhiR_T1& phi2){
	if(static_cast<const PhiR&>(phi1)!=static_cast<const PhiR&>(phi2)) return false;
	else if(phi1.eta!=phi2.eta) return false;
	else if(phi1.rs!=phi2.rs) return false;
	else return true;
}

//==== member functions - evaluation ====

/**
* compute the value of the function
* @param r - distance rij
* @param c - cutoff as a function of rij
*/
double PhiR_T1::val(double r, double cut)const noexcept{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return 0.5*(std::tanh(-eta*(r-rs))+1.0)*cut;
	#elif defined __ICC || defined __INTEL_COMPILER
	return 0.5*(tanh(-eta*(r-rs))+1.0)*cut;
	#endif
}

/**
* compute the value of the function
* @param r - distance rij
* @param cut - cutoff as a function of rij
* @param gcut - gradient of cutoff as a function of rij
*/
double PhiR_T1::grad(double r, double cut, double gcut)const noexcept{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	const double tanhf=tanh(-eta*(r-rs));
	#elif defined __ICC || defined __INTEL_COMPILER
	const double tanhf=tanh(-eta*(r-rs));
	#endif
	return 0.5*(-eta*(1.0-tanhf*tanhf)*cut+(1.0+tanhf)*gcut);
}

//*****************************************
// PHIR - T1 - DelloStritto - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiR_T1& obj){
		int N=0;
		N+=nbytes(static_cast<const PhiR&>(obj));
		N+=sizeof(double);//eta_
		N+=sizeof(double);//rs_
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiR_T1& obj, char* arr){
		int pos=0;
		pos+=pack(static_cast<const PhiR&>(obj),arr);
		std::memcpy(arr+pos,&obj.eta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.rs,sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiR_T1& obj, const char* arr){
		int pos=0;
		pos+=unpack(static_cast<PhiR&>(obj),arr);
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.rs,arr+pos,sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
}