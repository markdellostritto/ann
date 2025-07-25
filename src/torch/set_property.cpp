#include "torch/set_property.hpp"

namespace property{

//****************************************************************************
// Name
//****************************************************************************

Name Name::read(const char* str){
	if(std::strcmp(str,"MASS")==0) return Name::MASS;
	else if(std::strcmp(str,"TYPE")==0) return Name::TYPE;
	else if(std::strcmp(str,"CHARGE")==0) return Name::CHARGE;
	else if(std::strcmp(str,"VELOCITY")==0) return Name::VELOCITY;
	else if(std::strcmp(str,"TEMP")==0) return Name::TEMP;
	else return Name::UNKNOWN;
}

const char* Name::name(const Name& t){
	switch(t){
		case Name::MASS: return "MASS";
		case Name::TYPE: return "TYPE";
		case Name::CHARGE: return "CHARGE";
		case Name::VELOCITY: return "VELOCITY";
		case Name::TEMP: return "TEMP";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Name& t){
	switch(t){
		case Name::MASS: out<<"MASS"; break;
		case Name::TYPE: out<<"TYPE"; break;
		case Name::CHARGE: out<<"CHARGE"; break;
		case Name::VELOCITY: out<<"VELOCITY"; break;
		case Name::TEMP: out<<"TEMP"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//****************************************************************************
// Base
//****************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Base& base){
	return out<<base.name_<<" "<<base.group_;
}
	
}
