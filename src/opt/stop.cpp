// c
#include <cstring>
// c++
#include <iostream>
// opt
#include "opt/stop.hpp"

//***************************************************
// stopping criterion
//***************************************************

namespace opt{
	
std::ostream& operator<<(std::ostream& out, const Stop& stop){
	switch(stop){
		case Stop::FABS: out<<"FABS"; break;
		case Stop::FREL: out<<"FREL"; break;
		case Stop::XABS: out<<"XABS"; break;
		case Stop::XREL: out<<"XREL"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Stop::name(const Stop& stop){
	switch(stop){
		case Stop::FABS: return "FABS";
		case Stop::FREL: return "FREL";
		case Stop::XABS: return "XABS";
		case Stop::XREL: return "XREL";
		default: return "UNKNOWN";
	}
}

Stop Stop::read(const char* str){
	if(std::strcmp(str,"XREL")==0) return Stop::XREL;
	else if(std::strcmp(str,"XABS")==0) return Stop::XABS;
	else if(std::strcmp(str,"FREL")==0) return Stop::FREL;
	else if(std::strcmp(str,"FABS")==0) return Stop::FABS;
	else return Stop::UNKNOWN;
}

}