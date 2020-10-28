// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
#include <cstring>
// c++ libaries
#include <ostream>
// ann - symm - radial - g2
#include "symm_radial_g2.hpp"

//*****************************************
// PHIR - G2 - Behler
//*****************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PhiR_G2& f){
	return out<<"G2 "<<f.rs<<" "<<f.eta<<" ";
}

/**
* compute equality of two symmetry functions
* @param phi1 - G2 - first
* @param phi2 - G2 - second
* @return equiality of phi1 and phi2
*/
bool operator==(const PhiR_G2& phi1, const PhiR_G2& phi2){
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
double PhiR_G2::val(double r, double cut)const noexcept{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return std::exp(-eta*(r-rs)*(r-rs))*cut;
	#elif defined __ICC || defined __INTEL_COMPILER
	return exp(-eta*(r-rs)*(r-rs))*cut;
	#endif
}

/**
* compute the value of the function
* @param r - distance rij
* @param cut - cutoff as a function of rij
* @param gcut - gradient of cutoff as a function of rij
*/
double PhiR_G2::grad(double r, double cut, double gcut)const noexcept{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return std::exp(-eta*(r-rs)*(r-rs))*(-2.0*eta*(r-rs)*cut+gcut);
	#elif defined __ICC || defined __INTEL_COMPILER
	return exp(-eta*(r-rs)*(r-rs))*(-2.0*eta*(r-rs)*cut+gcut);
	#endif
}

//*****************************************
// PHIR - G2 - Behler - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiR_G2& obj){
		int N=0;
		N+=nbytes(static_cast<const PhiR&>(obj));
		N+=2*sizeof(double);
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiR_G2& obj, char* arr){
		int pos=0;
		pos+=pack(static_cast<const PhiR&>(obj),arr);
		std::memcpy(arr+pos,&obj.eta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.rs,sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiR_G2& obj, const char* arr){
		int pos=0;
		pos+=unpack(static_cast<PhiR&>(obj),arr);
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.rs,arr+pos,sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
}