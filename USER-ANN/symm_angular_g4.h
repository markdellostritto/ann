#ifndef SYMM_ANGULAR_G4_HPP
#define SYMM_ANGULAR_G4_HPP

// c libraries
#include <cmath>
// c++ libaries
#include <iostream>
// local libraries
#include "symm_angular.h"

//Behler G4
struct PhiA_G4: public PhiA{
	int lambda;//sign of cosine window
	double eta;//radial exponential width
	double zeta;//angular exponential width
	PhiA_G4():PhiA(),eta(0.0),zeta(0.0),lambda(0.0){};
	PhiA_G4(CutoffN::type tcut_, double rc_, double eta_, double zeta_, int lambda_):PhiA(tcut_,rc_),eta(eta_),zeta(zeta_),lambda(lambda_){};
	double operator()(double cos, double ri, double rj, double rij)const;
	//inline double val(double cos, double rij, double rik, double rjk)const{return angle(cos)*dist(rij,rik,rjk);};
	inline double val(double cos, double rij, double rik, double rjk)const{
		return (std::fabs(cos+1)<num_const::ZERO)?0:2.0*std::pow(0.5*(1.0+lambda*cos),zeta)
			*std::exp(-eta*(rij*rij+rik*rik))*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc);
	};
	inline double angle(double cos)const{return (std::fabs(cos+1)<num_const::ZERO)?0:2.0*std::pow(0.5*(1.0+lambda*cos),zeta);};
	inline double dist(double rij, double rik, double rjk)const{return std::exp(-eta*(rij*rij+rik*rik))*CutoffF::funcs[tcut](rij,rc)*CutoffF::funcs[tcut](rik,rc);};
	inline double grad_angle(double cos)const{return (std::fabs(cos+1)<num_const::ZERO)?0:zeta*lambda*std::pow(0.5*(1.0+lambda*cos),zeta-1.0);};
	inline double grad_dist_0(double rij, double rik, double rjk)const
		{return (-2.0*eta*rij*CutoffF::funcs[tcut](rij,rc)+CutoffFD::funcs[tcut](rij,rc))*CutoffF::funcs[tcut](rik,rc)*std::exp(-eta*(rij*rij+rik*rik));};
	inline double grad_dist_1(double rij, double rik, double rjk)const
		{return (-2.0*eta*rik*CutoffF::funcs[tcut](rik,rc)+CutoffFD::funcs[tcut](rik,rc))*CutoffF::funcs[tcut](rij,rc)*std::exp(-eta*(rij*rij+rik*rik));};
	inline double grad_dist_2(double rij, double rik, double rjk)const{return 0.0;};
};
std::ostream& operator<<(std::ostream& out, const PhiA_G4& f);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA_G4& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiA_G4& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiA_G4& obj, const char* arr);
	
}

/* References:
Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050.
*/

#endif
