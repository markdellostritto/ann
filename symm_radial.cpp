#include "symm_radial.hpp"

//*****************************************
// PhiRN - radial function names
//*****************************************

//loading/printing

PhiRN::type PhiRN::load(const char* str){
	if(std::strcmp(str,"G1")==0) return PhiRN::G1;
	else if(std::strcmp(str,"G2")==0) return PhiRN::G2;
	else if(std::strcmp(str,"T1")==0) return PhiRN::T1;
	else return PhiRN::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, const PhiRN::type& t){
	if(t==PhiRN::G1) out<<"G1";
	else if(t==PhiRN::G2) out<<"G2";
	else if(t==PhiRN::T1) out<<"T1";
	else out<<"UNKNOWN";
	return out;
}

//PhiR
std::ostream& operator<<(std::ostream& out, const PhiR& f){
	return out;
}

bool operator==(const PhiR& phir1, const PhiR& phir2){
	return true;
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR& obj){
		return 0;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR& obj, char* arr){
		
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR& obj, const char* arr){
		
	}
	
}