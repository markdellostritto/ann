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
#include "symm_radial_t1.h"

//*****************************************
// PHIR - T1 - DelloStritto
//*****************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PhiR_T1& f){
	return out<<"T1 "<<f.rs<<" "<<f.eta<<" ";
}

bool operator==(const PhiR_T1& phi1, const PhiR_T1& phi2){
	if(static_cast<const PhiR&>(phi1)!=static_cast<const PhiR&>(phi2)) return false;
	else if(phi1.eta!=phi2.eta) return false;
	else if(phi1.rs!=phi2.rs) return false;
	else return true;
}

//==== member functions - evaluation ====

double PhiR_T1::val(double r, double cut)const{
	return 0.5*(tanh(-eta*(r-rs))+1.0)*cut;
}

double PhiR_T1::grad(double r, double cut, double gcut)const{
	const double tanhf=tanh(-eta*(r-rs));
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