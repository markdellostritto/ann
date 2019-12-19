// c libraries
#include <cstring>
// c++ libraries
#include <ostream> 
// ann - cutoff
#include "cutoff.hpp"

//************************************************************
// CUTOFF NAMES
//************************************************************

CutoffN::type CutoffN::read(const char* str){
	if(std::strcmp(str,"COS")==0) return CutoffN::COS;
	else if(std::strcmp(str,"TANH")==0) return CutoffN::TANH;
	else return CutoffN::UNKNOWN;
}

//************************************************************
// CUTOFF FUNCTIONS
//************************************************************

std::ostream& operator<<(std::ostream& out, const CutoffN::type& cutt){
	if(cutt==CutoffN::COS) out<<"COS";
	else if(cutt==CutoffN::TANH) out<<"TANH";
	else out<<"UNKNOWN";
	return out;
}

const FCutT CutoffF::funcs[N_CUT_F]={
	&CutoffF::cut_cos,
	&CutoffF::cut_tanh
};

const FCutT CutoffFD::funcs[N_CUT_F]={
	&CutoffFD::cut_cos,
	&CutoffFD::cut_tanh
};
