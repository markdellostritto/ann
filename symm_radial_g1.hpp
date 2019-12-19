#pragma once
#ifndef SYMM_RADIAL_G1_HPP
#define SYMM_RADIAL_G1_HPP

// c++ libaries
#include <iosfwd>
// ann - symm - radial
#include "symm_radial.hpp"
// ann - serialization
#include "serialize.hpp"

//*****************************************
// PHIR - G1 - Behler
//*****************************************

struct PhiR_G1: public PhiR{
	//==== constructors/destructors ====
	PhiR_G1():PhiR(){}
	//==== member functions - evaluation ====
	inline double val(double r, double cut)const noexcept final{return cut;}
	inline double grad(double r, double cut, double gcut)const noexcept final{return gcut;}
};
//==== operators ====
std::ostream& operator<<(std::ostream& out, const PhiR_G1& f);
bool operator==(const PhiR_G1& phi1, const PhiR_G1& phi2);
inline bool operator!=(const PhiR_G1& phi1, const PhiR_G1& phi2){return !(phi1==phi2);}

//*****************************************
// PHIR - G1 - Behler - serialization
//*****************************************

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