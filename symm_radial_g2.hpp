#ifndef SYMM_RADIAL_G2_HPP
#define SYMM_RADIAL_G2_HPP

// c libraries
#include <cmath>
// c++ libaries
#include <iostream>
// local libraries
#include "symm_radial.hpp"

//Behler G2
struct PhiR_G2: public PhiR{
	double eta;//radial exponential width 
	double rs;//center of radial window
	PhiR_G2(CutoffN::type tcut_, double rc_, double eta_, double rs_):PhiR(tcut_,rc_),eta(eta_),rs(rs_){};
	double operator()(double r)const noexcept;
	double val(double r)const noexcept;
	double amp(double r)const noexcept;
	double cut(double r)const noexcept;
	double grad(double r)const noexcept;
};
std::ostream& operator<<(std::ostream& out, const PhiR_G2& f);

#endif