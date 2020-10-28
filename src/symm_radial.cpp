// c libraries
#include <cstring>
// c++ libraries
#include <ostream>
// ann - symm - radial
#include "symm_radial.hpp"

//*****************************************
// PhiRN - radial function names
//*****************************************

PhiRN::type PhiRN::read(const char* str){
	if(std::strcmp(str,"G1")==0) return PhiRN::G1;
	else if(std::strcmp(str,"G2")==0) return PhiRN::G2;
	else if(std::strcmp(str,"T1")==0) return PhiRN::T1;
	else return PhiRN::UNKNOWN;
}

const char* PhiRN::name(const PhiRN::type& t){
	switch(t){
		case PhiRN::G1: return "G1";
		case PhiRN::G2: return "G2";
		case PhiRN::T1: return "T1";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const PhiRN::type& t){
	switch(t){
		case PhiRN::G1: out<<"G1"; break;
		case PhiRN::G2: out<<"G2"; break;
		case PhiRN::T1: out<<"T1"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//*****************************************
// PhiR - radial function base class
//*****************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PhiR& f){
	return out;
}

bool operator==(const PhiR& phir1, const PhiR& phir2){
	return true;
}

//*****************************************
// PhiR - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiR& obj){
		return 0;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiR& obj, char* arr){
		return 0;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiR& obj, const char* arr){
		return 0;
	}
	
}