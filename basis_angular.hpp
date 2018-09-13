#ifndef BASIS_ANGULAR_HPP
#define BASIS_ANGULAR_HPP

// c libraries
#include <cstring>
// c++ libraries
#include <iostream>
#include <memory>
#include <vector>
// symmetry functions
#include "cutoff.hpp"
#include "symm_angular_g3.hpp"
#include "symm_angular_g4.hpp"
// string
#include "string.hpp"

#ifndef BASIS_ANGULAR_PRINT_FUNC
#define BASIS_ANGULAR_PRINT_FUNC 0
#endif

struct BasisA{
public:
	//member variables
	static const double V_CUT;//value cutoff
	PhiAN::type phiAN;//type of angular functions
	std::vector<std::shared_ptr<PhiA> > fA;//angular functions
	//initialization
	static void init_G3(BasisA& basis, unsigned int nA, CutoffN::type tcut, double rcut);
	static void init_G4(BasisA& basis, unsigned int nA, CutoffN::type tcut, double rcut);
	//loading/printing
	static void write(FILE* writer,const BasisA& basis);
	static void read(FILE* writer, BasisA& basis);
};
std::ostream& operator<<(std::ostream& out, const BasisA& basisA);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const BasisA& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const BasisA& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(BasisA& obj, const char* arr);
	
}

#endif