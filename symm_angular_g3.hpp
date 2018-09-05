#ifndef SYMM_ANGULAR_G3_HPP
#define SYMM_ANGULAR_G3_HPP

// c libraries
#include <cmath>
// c++ libaries
#include <iostream>
// local libraries
#include "symm_angular.hpp"

//Behler G3
struct PhiA_G3: public PhiA{
	double eta;//radial exponential width
	double zeta;//angular exponential width
	int lambda;//sign of cosine window
	PhiA_G3(CutoffN::type tcut_, double rc_, double eta_, double zeta_, int lambda_):PhiA(tcut_,rc_),eta(eta_),zeta(zeta_),lambda(lambda_){};
	double operator()(double cos, double ri, double rj, double rij)const noexcept;
	double val(double cos, double ri, double rj, double rij)const noexcept;
	double amp(double cos, double ri, double rj, double rij)const noexcept;
	double cut(double ri, double rj, double rij)const noexcept;
	double grad(double cos, double ri, double rj, double rij, unsigned int gindex)const;
	double grad_pre(double cos, double ri, double rj, double rij, unsigned int gindex)const;
};
std::ostream& operator<<(std::ostream& out, const PhiA_G3& f);

#endif