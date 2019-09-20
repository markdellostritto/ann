#pragma once
#ifndef SYMM_RADIAL_G1_HPP
#define SYMM_RADIAL_G1_HPP

// c libraries
#include <cmath>
// c++ libaries
#include <iostream>
// local libraries
#include "symm_radial.hpp"
// local libraries - serialization
#include "serialize.hpp"

//Behler G1
struct PhiR_G1: public PhiR{
	PhiR_G1():PhiR(){};
	inline double val(double r, double cut)const noexcept final{return cut;}
	inline double grad(double r, double cut, double gcut)const noexcept final{return gcut;}
};
std::ostream& operator<<(std::ostream& out, const PhiR_G1& f);
bool operator==(const PhiR_G1& phi1, const PhiR_G1& phi2);
inline bool operator!=(const PhiR_G1& phi1, const PhiR_G1& phi2){return !(phi1==phi2);};

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