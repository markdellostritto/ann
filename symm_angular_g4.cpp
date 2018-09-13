#include "symm_angular_g4.hpp"

//Behler G4

double PhiA_G4::operator()(double cos, double rij, double rik, double rjk)const noexcept{
	return angle(cos)*dist(rij,rik,rjk);
}

double PhiA_G4::val(double cos, double rij, double rik, double rjk)const noexcept{
	return angle(cos)*dist(rij,rik,rjk);
}

double PhiA_G4::angle(double cos)const noexcept{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	else return 2.0*std::pow(0.5*(1.0+lambda*cos),zeta);
}

double PhiA_G4::dist(double rij, double rik, double rjk)const noexcept{
	return std::exp(-eta*(rij*rij+rik*rik))
		*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc);
}

double PhiA_G4::grad_angle(double cos)const noexcept{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	else return zeta*lambda*std::pow(0.5*(1.0+lambda*cos),zeta-1.0);
}

double PhiA_G4::grad_dist(double rij, double rik, double rjk, unsigned int gindex)const{
	switch(gindex){
		case 0: return (-2.0*eta*rij*CutoffF::funcs[tcut](rij,rc)+CutoffFD::funcs[tcut](rij,rc))
			*CutoffF::funcs[tcut](rik,rc)*std::exp(-eta*(rij*rij+rik*rik));
		case 1: return (-2.0*eta*rik*CutoffF::funcs[tcut](rik,rc)+CutoffFD::funcs[tcut](rik,rc))
			*CutoffF::funcs[tcut](rij,rc)*std::exp(-eta*(rij*rij+rik*rik));
		case 2: return 0.0;
		default: throw std::invalid_argument("Invalid gradient index.");
	}
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