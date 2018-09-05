#include "symm_angular_g4.hpp"

//Behler G4

double PhiA_G4::operator()(double cos, double rij, double rik, double rjk)const noexcept{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	else return 2.0*std::pow(0.5*(1.0+lambda*cos),zeta)*std::exp(-eta*(rij*rij+rik*rik))
		*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc);
}

double PhiA_G4::val(double cos, double rij, double rik, double rjk)const noexcept{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	else return 2.0*std::pow(0.5*(1.0+lambda*cos),zeta)*std::exp(-eta*(rij*rij+rik*rik))
		*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc);
}

double PhiA_G4::amp(double cos, double rij, double rik, double rjk)const noexcept{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	else return 2.0*std::pow(0.5*(1.0+lambda*cos),zeta)*std::exp(-eta*(rij*rij+rjk*rjk));
}

double PhiA_G4::cut(double rij, double rik, double rjk)const noexcept{
	return CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc);
}

double PhiA_G4::grad(double cos, double rij, double rik, double rjk, unsigned int gindex)const{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	switch(gindex){
		case 0: return (zeta*lambda*(1.0/rik-cos/rij)*2.0/(1.0+lambda*cos)-2.0*eta*rij
			+CutoffFD::funcs[tcut](rij,rc)/CutoffF::funcs[tcut](rij,rc))*PhiA_G4::val(cos,rij,rik,rjk);
		case 1: return (zeta*lambda*(1.0/rij-cos/rik)*2.0/(1.0+lambda*cos)-2.0*eta*rik
			+CutoffFD::funcs[tcut](rik,rc)/CutoffF::funcs[tcut](rik,rc))*PhiA_G4::val(cos,rij,rik,rjk);
		case 2: return 0;
		default: throw std::invalid_argument("Invalid gradient index.");
	}
}

double PhiA_G4::grad_pre(double cos, double rij, double rik, double rjk, unsigned int gindex)const{
	if(std::fabs(cos+1)<num_const::ZERO) return 0;
	switch(gindex){
		case 0: return (zeta*lambda*(1.0/rik-cos/rij)*2.0/(1.0+lambda*cos)-2.0*eta*rij
			+CutoffFD::funcs[tcut](rij,rc)/CutoffF::funcs[tcut](rij,rc));
		case 1: return (zeta*lambda*(1.0/rij-cos/rik)*2.0/(1.0+lambda*cos)-2.0*eta*rik
			+CutoffFD::funcs[tcut](rik,rc)/CutoffF::funcs[tcut](rik,rc));
		case 2: return 0;
		default: throw std::invalid_argument("Invalid gradient index.");
	}
}

std::ostream& operator<<(std::ostream& out, const PhiA_G4& f){
	return out<<static_cast<const PhiA&>(f)<<" G4 "<<f.eta<<" "<<f.zeta<<" "<<f.lambda;
}
