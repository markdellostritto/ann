#include "torch/job.hpp"

//c
#include <cstring>
//c++
#include <iostream>

//****************************************************************************
// Job
//****************************************************************************

Job Job::read(const char* str){
	if(std::strcmp(str,"SP")==0) return Job::SP;
	else if(std::strcmp(str,"MC")==0) return Job::MC;
	else if(std::strcmp(str,"MD")==0) return Job::MD;
	else return Job::UNKNOWN;
}

const char* Job::name(const Job& t){
	switch(t){
		case Job::SP: return "SP";
		case Job::MC: return "MC";
		case Job::MD: return "MD";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Job& t){
	switch(t){
		case Job::SP: out<<"SP"; break;
		case Job::MC: out<<"MC"; break;
		case Job::MD: out<<"MD"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}
