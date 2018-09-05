#include "symm_radial.hpp"

//loading/printing

PhiRN::type PhiRN::load(const char* str){
	if(std::strcmp(str,"G1")==0) return PhiRN::G1;
	else if(std::strcmp(str,"G2")==0) return PhiRN::G2;
	else return PhiRN::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, const PhiRN::type& t){
	if(t==PhiRN::G1) out<<"G1";
	else if(t==PhiRN::G2) out<<"G2";
	else out<<"UNKNOWN";
	return out;
}

//PhiR
std::ostream& operator<<(std::ostream& out, const PhiR& f){
	return out<<"PhiR "<<f.tcut<<" "<<f.rc;
}
