#include "symm_radial_g2.h"

//Behler G2
double PhiR_G2::operator()(double r)const{
	return std::exp(-eta*(r-rs)*(r-rs))*CutoffF::funcs[tcut](r,rc);
}
std::ostream& operator<<(std::ostream& out, const PhiR_G2& f){
	return out<<static_cast<const PhiR&>(f)<<" G2 "<<f.rs<<" "<<f.eta<<" ";
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR_G2& obj){
		unsigned int N=0;
		N+=nbytes(static_cast<const PhiR&>(obj));
		N+=2*sizeof(double);
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR_G2& obj, char* arr){
		unsigned int pos=0;
		pack(static_cast<const PhiR&>(obj),arr); pos+=nbytes(static_cast<const PhiR&>(obj));
		std::memcpy(arr+pos,&obj.eta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.rs,sizeof(double));
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR_G2& obj, const char* arr){
		unsigned int pos=0;
		unpack(static_cast<PhiR&>(obj),arr); pos+=nbytes(static_cast<const PhiR&>(obj));
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.rs,arr+pos,sizeof(double));
	}
	
}