#include "symm_angular.hpp"

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

//PhiA
std::ostream& operator<<(std::ostream& out, const PhiA& f){
	return out<<"PhiA "<<f.tcut<<" "<<f.rc;
}
