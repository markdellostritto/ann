#ifndef SYMM_RADIAL_G1_HPP
#define SYMM_RADIAL_G1_HPP

// c libraries
#include <cmath>
// c++ libaries
#include <iostream>
// local libraries
#include "symm_radial.h"
// local libraries - serialization
#include "serialize.h"

//Behler G1
struct PhiR_G1: public PhiR{
	PhiR_G1():PhiR(){};
	PhiR_G1(CutoffN::type tcut_, double rc_):PhiR(tcut_,rc_){};
	double operator()(double r)const;
	inline double val(double r)const{return CutoffF::funcs[tcut](r,rc);};
	inline double grad(double r)const{return CutoffFD::funcs[tcut](r,rc);};
};
std::ostream& operator<<(std::ostream& out, const PhiR_G1& f);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR_G1& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR_G1& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR_G1& obj, const char* arr);
	
}

/* References:
Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050.
*/

#endif
