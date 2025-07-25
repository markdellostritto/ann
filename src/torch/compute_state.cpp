#include "torch/compute_state.hpp"

namespace compute{

namespace state{

//****************************************************************************
// Name
//****************************************************************************

Name Name::read(const char* str){
	if(std::strcmp(str,"KE")==0) return Name::KE;
	else if(std::strcmp(str,"PE")==0) return Name::PE;
	else if(std::strcmp(str,"TE")==0) return Name::TE;
	else if(std::strcmp(str,"TEMP")==0) return Name::TEMP;
	else return Name::UNKNOWN;
}

const char* Name::name(const Name& t){
	switch(t){
		case Name::KE: return "KE";
		case Name::PE: return "PE";
		case Name::TE: return "TE";
		case Name::TEMP: return "TEMP";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Name& t){
	switch(t){
		case Name::KE: out<<"KE"; break;
		case Name::PE: out<<"PE"; break;
		case Name::TE: out<<"TE"; break;
		case Name::TEMP: out<<"TEMP"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//****************************************************************************
// Base
//****************************************************************************

std::ostream& operator<<(std::ostream& out, const Base& base){
	return out <<base.name_<<" ";
}

}

}