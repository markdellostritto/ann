#include "symm_radial_t1.hpp"

//T1
std::ostream& operator<<(std::ostream& out, const PhiR_T1& f){
	return out<<"T1 "<<f.rs<<" "<<f.eta<<" ";
}

bool operator==(const PhiR_T1& phi1, const PhiR_T1& phi2){
	if(static_cast<const PhiR&>(phi1)!=static_cast<const PhiR&>(phi2)) return false;
	else if(phi1.eta!=phi2.eta) return false;
	else if(phi1.rs!=phi2.rs) return false;
	else return true;
}

double PhiR_T1::val(double r, double cut)const noexcept{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return 0.5*(std::tanh(-eta*(r-rs))+1.0)*cut;
	#elif defined __ICC || defined __INTEL_COMPILER
	return 0.5*(tanh(-eta*(r-rs))+1.0)*cut;
	#endif
}

double PhiR_T1::grad(double r, double cut, double gcut)const noexcept{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	const double tanhf=tanh(-eta*(r-rs));
	#elif defined __ICC || defined __INTEL_COMPILER
	const double tanhf=tanh(-eta*(r-rs));
	#endif
	return 0.5*(-eta*(1.0-tanhf*tanhf)*cut+(1.0+tanhf)*gcut);
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR_T1& obj){
		unsigned int N=0;
		N+=nbytes(static_cast<const PhiR&>(obj));
		N+=2*sizeof(double);
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR_T1& obj, char* arr){
		unsigned int pos=0;
		pack(static_cast<const PhiR&>(obj),arr); pos+=nbytes(static_cast<const PhiR&>(obj));
		std::memcpy(arr+pos,&obj.eta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.rs,sizeof(double));
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR_T1& obj, const char* arr){
		unsigned int pos=0;
		unpack(static_cast<PhiR&>(obj),arr); pos+=nbytes(static_cast<const PhiR&>(obj));
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.rs,arr+pos,sizeof(double));
	}
	
}