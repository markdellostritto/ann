#ifndef BASIS_RADIAL_HPP
#define BASIS_RADIAL_HPP

// c libraries
#include <cstring>
// c++ libraries
#include <iostream>
#include <memory>
#include <vector>
// symmetry functions
#include "cutoff.hpp"
#include "symm_radial_g1.hpp"
#include "symm_radial_g2.hpp"
//string
#include "string.hpp"
// local libraries - serialization
#include "serialize.hpp"

#ifndef BASIS_RADIAL_PRINT_FUNC
#define BASIS_RADIAL_PRINT_FUNC 0
#endif

struct BasisR{
	//member variables
	static const double V_CUT;//value cutoff
	PhiRN::type phiRN;//type of radial functions
	std::vector<std::shared_ptr<PhiR> > fR;//radial functions
	//initialization
	static void init_G1(BasisR& basis, CutoffN::type tcut, double rcut, double rmin);
	static void init_G2(BasisR& basis, unsigned int nR, CutoffN::type tcut, double rcut, double rmin);
	//loading/printing
	static void write(FILE* writer,const BasisR& basis);
	static void read(FILE* writer, BasisR& basis);
};
std::ostream& operator<<(std::ostream& out, const BasisR& basisR);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const BasisR& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const BasisR& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(BasisR& obj, const char* arr);
	
}

#endif