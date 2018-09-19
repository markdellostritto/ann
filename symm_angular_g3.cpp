#include "symm_angular_g3.hpp"

//Behler G3

double PhiA_G3::operator()(double cos, double rij, double rik, double rjk)const noexcept{
	return angle(cos)*dist(rij,rik,rjk);
}

double PhiA_G3::val(double cos, double rij, double rik, double rjk)const noexcept{
	return angle(cos)*dist(rij,rik,rjk);
}

double PhiA_G3::angle(double cos)const noexcept{
	return (std::fabs(cos+1)<num_const::ZERO)?0:2.0*std::pow(0.5*(1.0+lambda*cos),zeta);
}

double PhiA_G3::dist(double rij, double rik, double rjk)const noexcept{
	return std::exp(-eta*(rij*rij+rik*rik+rjk*rjk))
		*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc)*CutoffF::funcs[tcut](rjk,rc);
}

double PhiA_G3::grad_angle(double cos)const noexcept{
	return (std::fabs(cos+1)<num_const::ZERO)?0:zeta*std::pow(0.5*(1.0+lambda*cos),zeta-1.0);
}

double PhiA_G3::grad_dist_0(double rij, double rik, double rjk)const noexcept{
	return (-2.0*eta*rij*CutoffF::funcs[tcut](rij,rc)+CutoffFD::funcs[tcut](rij,rc))
		*CutoffF::funcs[tcut](rik,rc)*CutoffF::funcs[tcut](rjk,rc)*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

double PhiA_G3::grad_dist_1(double rij, double rik, double rjk)const noexcept{
	return (-2.0*eta*rik*CutoffF::funcs[tcut](rik,rc)+CutoffFD::funcs[tcut](rik,rc))
		*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rjk,rc)*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

double PhiA_G3::grad_dist_2(double rij, double rik, double rjk)const noexcept{
	return (-2.0*eta*rjk*CutoffF::funcs[tcut](rjk,rc)+CutoffFD::funcs[tcut](rjk,rc))
		*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc)*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

std::ostream& operator<<(std::ostream& out, const PhiA_G3& f){
	return out<<static_cast<const PhiA&>(f)<<" G3 "<<f.eta<<" "<<f.zeta<<" "<<f.lambda;
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA_G3& obj){
		unsigned int N=0;
		N+=nbytes(static_cast<const PhiA&>(obj));
		N+=2*sizeof(double);
		N+=sizeof(int);
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiA_G3& obj, char* arr){
		unsigned int pos=0;
		pack(static_cast<const PhiA&>(obj),arr); pos+=nbytes(static_cast<const PhiA&>(obj));
		std::memcpy(arr+pos,&obj.eta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.zeta,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.lambda,sizeof(int));
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiA_G3& obj, const char* arr){
		unsigned int pos=0;
		unpack(static_cast<PhiA&>(obj),arr); pos+=nbytes(static_cast<const PhiA&>(obj));
		std::memcpy(&obj.eta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.zeta,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.lambda,arr+pos,sizeof(int));
	}
	
}
