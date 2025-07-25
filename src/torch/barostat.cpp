#include "torch/barostat.hpp"

//****************************************************************************
// Barostat
//****************************************************************************

Barostat Barostat::read(const char* str){
	if(std::strcmp(str,"NONE")==0) return Barostat::NONE;
	else if(std::strcmp(str,"ISO")==0) return Barostat::ISO;
	else if(std::strcmp(str,"ANISO")==0) return Barostat::ANISO;
	else return Barostat::UNKNOWN;
}

const char* Barostat::name(const Barostat& t){
	switch(t){
		case Barostat::NONE: return "NONE";
		case Barostat::ISO: return "ISO";
		case Barostat::ANISO: return "ANISO";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Barostat& t){
	switch(t){
		case Barostat::NONE: out<<"NONE"; break;
		case Barostat::ISO: out<<"ISO"; break;
		case Barostat::ANISO: out<<"ANISO"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}
