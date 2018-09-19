#include "symm_angular.h"

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

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA& obj){
		unsigned int N=0;
		N+=sizeof(double);//rc
		N+=sizeof(obj.tcut);//cutoff type
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiA& obj, char* arr){
		std::memcpy(arr,&obj.rc,sizeof(double));
		std::memcpy(arr+sizeof(double),&obj.tcut,sizeof(obj.tcut));
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiA& obj, const char* arr){
		std::memcpy(&obj.rc,arr,sizeof(double));
		std::memcpy(&obj.tcut,arr+sizeof(double),sizeof(obj.tcut));
	}
	
}
