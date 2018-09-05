#include "symm_angular_g3.hpp"

//Behler G3

double PhiA_G3::operator()(double cos, double rij, double rik, double rjk)const noexcept{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	else return 2.0*std::pow(0.5*(1.0+lambda*cos),zeta)*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk))
		*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc)*CutoffF::funcs[tcut](rjk,rc);
}

double PhiA_G3::val(double cos, double rij, double rik, double rjk)const noexcept{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	else return 2.0*std::pow(0.5*(1.0+lambda*cos),zeta)*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk))
		*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc)*CutoffF::funcs[tcut](rjk,rc);
}

double PhiA_G3::amp(double cos, double rij, double rik, double rjk)const noexcept{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	else return 2.0*std::pow(0.5*(1.0+lambda*cos),zeta)*std::exp(-eta*(rij*rij+rik*rik+rjk*rjk));
}

double PhiA_G3::cut(double rij, double rik, double rjk)const noexcept{
	return CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc)*CutoffF::funcs[tcut](rjk,rc);
}

double PhiA_G3::grad(double cos, double rij, double rik, double rjk, unsigned int gindex)const{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	switch(gindex){
		case 0: return (zeta*lambda*(1.0/rik-cos/rij)/(1.0+lambda*cos)-2.0*eta*rij
			+CutoffFD::funcs[tcut](rij,rc)/CutoffF::funcs[tcut](rik,rc))*PhiA_G3::val(cos,rij,rik,rjk);
		case 1: return (zeta*lambda*(1.0/rij-cos/rik)/(1.0+lambda*cos)-2.0*eta*rik
			+CutoffFD::funcs[tcut](rik,rc)/CutoffF::funcs[tcut](rik,rc))*PhiA_G3::val(cos,rij,rik,rjk);
		case 2: return (-2.0*eta*rjk
			+CutoffFD::funcs[tcut](rjk,rc)/CutoffF::funcs[tcut](rjk,rc))*PhiA_G3::val(cos,rij,rik,rjk);
		default: throw std::invalid_argument("Invalid gradient index.");
	}
}

double PhiA_G3::grad_pre(double cos, double rij, double rik, double rjk, unsigned int gindex)const{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	switch(gindex){
		case 0: return (zeta*lambda*(1.0/rik-cos/rij)/(1.0+lambda*cos)-2.0*eta*rij
			+CutoffFD::funcs[tcut](rij,rc)/CutoffF::funcs[tcut](rik,rc));
		case 1: return (zeta*lambda*(1.0/rij-cos/rik)/(1.0+lambda*cos)-2.0*eta*rik
			+CutoffFD::funcs[tcut](rik,rc)/CutoffF::funcs[tcut](rik,rc));
		case 2: return (-2.0*eta*rjk
			+CutoffFD::funcs[tcut](rjk,rc)/CutoffF::funcs[tcut](rjk,rc));
		default: throw std::invalid_argument("Invalid gradient index.");
	}
}

std::ostream& operator<<(std::ostream& out, const PhiA_G3& f){
	return out<<static_cast<const PhiA&>(f)<<" G3 "<<f.eta<<" "<<f.zeta<<" "<<f.lambda;
}
