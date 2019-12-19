#pragma once
#ifndef SYMM_RADIAL_G1_HPP
#define SYMM_RADIAL_G1_HPP

// c++ libaries
#include <iosfwd>
// ann - symm - radial 
#include "symm_radial.h"
// ann - serialization
#include "serialize.h"

//Behler G1
struct PhiR_G1: public PhiR{
	PhiR_G1():PhiR(){}
	inline double val(double r, double cut)const{return cut;}
	inline double grad(double r, double cut, double gcut)const{return gcut;}
};
std::ostream& operator<<(std::ostream& out, const PhiR_G1& f);
bool operator==(const PhiR_G1& phi1, const PhiR_G1& phi2);
inline bool operator!=(const PhiR_G1& phi1, const PhiR_G1& phi2){return !(phi1==phi2);}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR_G1& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const PhiR_G1& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(PhiR_G1& obj, const char* arr);
	
}

/* References:
Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050.
*/

#endif