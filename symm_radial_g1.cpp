// c++ libaries
#include <ostream>
// ann - symm - radial - g1
#include "symm_radial_g1.hpp"

//*****************************************
// PHIR - G1 - Behler
//*****************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PhiR_G1& f){
	return out<<"G1";
}

bool operator==(const PhiR_G1& phir1, const PhiR_G1& phir2){
	return static_cast<const PhiR&>(phir1)==static_cast<const PhiR&>(phir2);
}

//*****************************************
// PHIR - G1 - Behler - serialization
//*****************************************

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
	
	template <> unsigned int pack(const PhiR_G1& obj, char* arr){
		pack(static_cast<const PhiR&>(obj),arr);
		return nbytes(static_cast<const PhiR&>(obj));
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(PhiR_G1& obj, const char* arr){
		unpack(static_cast<PhiR&>(obj),arr);
		return nbytes(static_cast<PhiR&>(obj));
	}
	
}