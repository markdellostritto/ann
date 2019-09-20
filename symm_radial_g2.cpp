#include "symm_radial_g2.hpp"

//Behler G2
std::ostream& operator<<(std::ostream& out, const PhiR_G2& f){
	return out<<"G2 "<<f.rs<<" "<<f.eta<<" ";
}

bool operator==(const PhiR_G2& phi1, const PhiR_G2& phi2){
	if(static_cast<const PhiR&>(phi1)!=static_cast<const PhiR&>(phi2)) return false;
	else if(phi1.eta!=phi2.eta) return false;
	else if(phi1.rs!=phi2.rs) return false;
	else return true;
}

double PhiR_G2::val(double r, double cut)const noexcept{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return std::exp(-eta*(r-rs)*(r-rs))*cut;
	#elif defined __ICC || defined __INTEL_COMPILER
	return exp(-eta*(r-rs)*(r-rs))*cut;
	#endif
}

double PhiR_G2::grad(double r, double cut, double gcut)const noexcept{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return std::exp(-eta*(r-rs)*(r-rs))*(-2.0*eta*(r-rs)*cut+gcut);
	#elif defined __ICC || defined __INTEL_COMPILER
	return exp(-eta*(r-rs)*(r-rs))*(-2.0*eta*(r-rs)*cut+gcut);
	#endif
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR_G2& obj){
		unsigned int N=0;
		N+=nbytes(static_cast<const PhiR&>(obj));
		N+=2*sizeof(double);
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR_G2& obj, char* arr){
		unsigned int pos=0;
		pack(static_cast<const PhiR&>(obj),arr); pos+=nbytes(static_cast<const PhiR&>(obj));
		std::memcpy(arr+pos,&obj.eta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.rs,sizeof(double));
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR_G2& obj, const char* arr){
		unsigned int pos=0;
		unpack(static_cast<PhiR&>(obj),arr); pos+=nbytes(static_cast<const PhiR&>(obj));
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.rs,arr+pos,sizeof(double));
	}
	
}