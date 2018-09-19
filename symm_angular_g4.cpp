#include "symm_angular_g4.hpp"

//Behler G4

double PhiA_G4::operator()(double cos, double rij, double rik, double rjk)const noexcept{
	return angle(cos)*dist(rij,rik,rjk);
}
std::ostream& operator<<(std::ostream& out, const PhiA_G4& f){
	return out<<static_cast<const PhiA&>(f)<<" G4 "<<f.eta<<" "<<f.zeta<<" "<<f.lambda;
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA_G4& obj){
		unsigned int N=0;
		N+=nbytes(static_cast<const PhiA&>(obj));
		N+=2*sizeof(double);
		N+=sizeof(int);
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiA_G4& obj, char* arr){
		unsigned int pos=0;
		pack(static_cast<const PhiA&>(obj),arr); pos+=nbytes(static_cast<const PhiA&>(obj));
		std::memcpy(arr+pos,&obj.eta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.zeta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.lambda,sizeof(int));
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiA_G4& obj, const char* arr){
		unsigned int pos=0;
		unpack(static_cast<PhiA&>(obj),arr); pos+=nbytes(static_cast<const PhiA&>(obj));
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.zeta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.lambda,arr+pos,sizeof(int));
	}
	
}