#include "symm_radial_g1.hpp"

//Behler G1
double PhiR_G1::operator()(double r)const noexcept{
	return CutoffF::funcs[tcut](r,rc);
}
std::ostream& operator<<(std::ostream& out, const PhiR_G1& f){
	return out<<static_cast<const PhiR&>(f)<<" G1";
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR_G1& obj){
		return nbytes(static_cast<const PhiR&>(obj));
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR_G1& obj, char* arr){
		pack(static_cast<const PhiR&>(obj),arr);
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR_G1& obj, const char* arr){
		unpack(static_cast<PhiR&>(obj),arr);
	}
	
}