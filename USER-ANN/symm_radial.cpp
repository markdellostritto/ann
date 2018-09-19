#include "symm_radial.h"

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

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR& obj){
		unsigned int N=0;
		N+=sizeof(double);//rc
		N+=sizeof(obj.tcut);//cutoff type
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR& obj, char* arr){
		std::memcpy(arr,&obj.rc,sizeof(double));
		std::memcpy(arr+sizeof(double),&obj.tcut,sizeof(obj.tcut));
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR& obj, const char* arr){
		std::memcpy(&obj.rc,arr,sizeof(double));
		std::memcpy(&obj.tcut,arr+sizeof(double),sizeof(obj.tcut));
	}
	
}
