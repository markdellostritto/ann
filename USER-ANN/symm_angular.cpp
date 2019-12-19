// c libraries
#include <cstring>
// ann - symm - angular
#include "symm_angular.h"

//*****************************************
// PhiAN - angular function names
//*****************************************

//loading/printing

PhiAN::type PhiAN::load(const char* str){
	if(std::strcmp(str,"G3")==0) return PhiAN::G3;
	else if(std::strcmp(str,"G4")==0) return PhiAN::G4;
	else return PhiAN::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, const PhiAN::type& t){
	if(t==PhiAN::G3) out<<"G3";
	else if(t==PhiAN::G4) out<<"G4";
	else out<<"UNKNOWN";
	return out;
}

bool operator==(const PhiA& phia1, const PhiA& phia2){
	return true;
}

//PhiA
std::ostream& operator<<(std::ostream& out, const PhiA& f){
	return out;
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA& obj){
		return 0;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const PhiA& obj, char* arr){
		return 0;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(PhiA& obj, const char* arr){
		return 0;
	}
	
}