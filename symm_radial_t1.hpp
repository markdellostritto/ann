#pragma once
#ifndef SYMM_RADIAL_TANH_HPP
#define SYMM_RADIAL_TANH_HPP

// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
// c++ libaries
#include <iostream>
// local libraries
#include "symm_radial.hpp"
// local libraries - serialization
#include "serialize.hpp"

//tanh
struct PhiR_T1: public PhiR{
	double eta;//radial exponential width 
	double rs;//center of radial window
	PhiR_T1():PhiR(),eta(0.0),rs(0.0){};
	PhiR_T1(double rs_, double eta_):PhiR(),eta(eta_),rs(rs_){};
	double val(double r, double cut)const noexcept final;
	double grad(double r, double cut, double gcut)const noexcept final;
};
std::ostream& operator<<(std::ostream& out, const PhiR_T1& f);
bool operator==(const PhiR_T1& phir1, const PhiR_T1& phir2);
inline bool operator!=(const PhiR_T1& phir1, const PhiR_T1& phir2){return !(phir1==phir2);};

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR_T1& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR_T1& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR_T1& obj, const char* arr);
	
}

/* References:
Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050.
*/

#endif