#pragma once
#ifndef SYMM_RADIAL_G2_HPP
#define SYMM_RADIAL_G2_HPP

// c++ libaries
#include <iosfwd>
// ann - symm - radial
#include "symm_radial.hpp"
// ann - serialization
#include "serialize.hpp"

//*****************************************
// PHIR - G2 - Behler
//*****************************************

struct PhiR_G2: public PhiR{
	//==== function parameters ====
	double eta;//radial exponential width 
	double rs;//center of radial window
	//==== constructors/destructors ====
	PhiR_G2():PhiR(),eta(0.0),rs(0.0){}
	PhiR_G2(double rs_, double eta_):PhiR(),eta(eta_),rs(rs_){}
	//==== member functions - evaluation ====
	double val(double r, double cut)const noexcept final;
	double grad(double r, double cut, double gcut)const noexcept final;
};
//==== operators ====
std::ostream& operator<<(std::ostream& out, const PhiR_G2& f);
bool operator==(const PhiR_G2& phir1, const PhiR_G2& phir2);
inline bool operator!=(const PhiR_G2& phir1, const PhiR_G2& phir2){return !(phir1==phir2);}

//*****************************************
// PHIR - G2 - Behler - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR_G2& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const PhiR_G2& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(PhiR_G2& obj, const char* arr);
	
}

/* References:
Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050.
*/

#endif