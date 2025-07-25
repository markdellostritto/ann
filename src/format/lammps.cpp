// format
#include "format/lammps.hpp"

namespace LAMMPS{

//*****************************************************
//STYLE_ATOM struct
//*****************************************************

STYLE_ATOM::type STYLE_ATOM::read(const char* str){
	if(std::strcmp(str,"FULL")==0) return STYLE_ATOM::FULL;
	else if(std::strcmp(str,"BOND")==0) return STYLE_ATOM::BOND;
	else if(std::strcmp(str,"ATOMIC")==0) return STYLE_ATOM::ATOMIC;
	else if(std::strcmp(str,"CHARGE")==0) return STYLE_ATOM::CHARGE;
	else return STYLE_ATOM::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, STYLE_ATOM::type& t){
	if(t==STYLE_ATOM::FULL) out<<"FULL";
	else if(t==STYLE_ATOM::BOND) out<<"BOND";
	else if(t==STYLE_ATOM::ATOMIC) out<<"ATOMIC";
	else if(t==STYLE_ATOM::CHARGE) out<<"CHARGE";
	else out<<"UNKNOWN";
	return out;
}

//*****************************************************
//FORMAT_ATOM struct
//*****************************************************

std::ostream& operator<<(std::ostream& out, FORMAT_ATOM& f){
	out<<"index = "<<f.index<<"\n";
	out<<"mol   = "<<f.mol<<"\n";
	out<<"type  = "<<f.type<<"\n";
	out<<"x     = "<<f.x<<"\n";
	out<<"y     = "<<f.y<<"\n";
	out<<"z     = "<<f.z<<"\n";
	out<<"xu    = "<<f.xu<<"\n";
	out<<"yu    = "<<f.yu<<"\n";
	out<<"zu    = "<<f.zu<<"\n";
	out<<"q     = "<<f.q<<"\n";
	out<<"fx    = "<<f.fx<<"\n";
	out<<"fy    = "<<f.fy<<"\n";
	out<<"fz    = "<<f.fz;
	return out;
}

}