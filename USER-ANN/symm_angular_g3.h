#ifndef SYMM_ANGULAR_G3_HPP
#define SYMM_ANGULAR_G3_HPP

// c libraries
#include <cmath>
// c++ libaries
#include <iostream>
// local libraries
#include "symm_angular.h"

//Behler G3
struct PhiA_G3: public PhiA{
	int lambda;//sign of cosine window
	double eta;//radial exponential width
	double zeta;//angular exponential width
	PhiA_G3():PhiA(),eta(0.0),zeta(0.0),lambda(0.0){};
	PhiA_G3(CutoffN::type tcut_, double rc_, double eta_, double zeta_, int lambda_):PhiA(tcut_,rc_),eta(eta_),zeta(zeta_),lambda(lambda_){};
	double operator()(double cos, double ri, double rj, double rij)const;
	double val(double cos, double ri, double rj, double rij)const;
	double angle(double cos)const;
	double dist(double ri, double rj, double rij)const;
	double grad_angle(double cos)const;
	double grad_dist_0(double rij, double rik, double rjk)const;
	double grad_dist_1(double rij, double rik, double rjk)const;
	double grad_dist_2(double rij, double rik, double rjk)const;
};
std::ostream& operator<<(std::ostream& out, const PhiA_G3& f);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA_G3& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiA_G3& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiA_G3& obj, const char* arr);
	
}

/* References:
Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050.
*/

#endif
