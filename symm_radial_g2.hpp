#ifndef SYMM_RADIAL_G2_HPP
#define SYMM_RADIAL_G2_HPP

// c libraries
#include <cmath>
// c++ libaries
#include <iostream>
// local libraries
#include "symm_radial.hpp"
// local libraries - serialization
#include "serialize.hpp"

//Behler G2
struct PhiR_G2: public PhiR{
	double eta;//radial exponential width 
	double rs;//center of radial window
	PhiR_G2():PhiR(),eta(0.0),rs(0.0){};
	PhiR_G2(CutoffN::type tcut_, double rc_, double rs_, double eta_):PhiR(tcut_,rc_),rs(rs_),eta(eta_){};
	double operator()(double r)const noexcept final;
	inline double val(double r)const noexcept final{return std::exp(-eta*(r-rs)*(r-rs))*CutoffF::funcs[tcut](r,rc);};
	inline double grad(double r)const noexcept final{return std::exp(-eta*(r-rs)*(r-rs))*(-2.0*eta*(r-rs)*CutoffF::funcs[tcut](r,rc)+CutoffFD::funcs[tcut](r,rc));};
};
std::ostream& operator<<(std::ostream& out, const PhiR_G2& f);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR_G2& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR_G2& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR_G2& obj, const char* arr);
	
}

/* References:
Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050.
*/

#endif