#ifndef BASIS_ANGULAR_HPP
#define BASIS_ANGULAR_HPP

// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
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
private:
	static const double V_CUT;//value cutoff
	PhiAN::type phiAN_;//type of angular functions
	unsigned int nfA_;
	PhiA** fA_;//angular functions
public:
	//constructors/destructors
	BasisA():nfA_(0),phiAN_(PhiAN::UNKNOWN),fA_(NULL){};
	BasisA(const BasisA& basisA);
	~BasisA();
	//operators
	BasisA& operator=(const BasisA& basisA);
	friend std::ostream& operator<<(std::ostream& out, const BasisA& basisA);
	//initialization
	void init_G3(unsigned int nA, CutoffN::type tcut, double rcut);
	void init_G4(unsigned int nA, CutoffN::type tcut, double rcut);
	//loading/printing
	static void write(FILE* writer,const BasisA& basis);
	static void read(FILE* writer, BasisA& basis);
	//member access
	const unsigned int& nfA()const{return nfA_;};
	PhiAN::type& phiAN(){return phiAN_;};
	const PhiAN::type& phiAN()const{return phiAN_;};
	PhiA& fA(unsigned int i){return *fA_[i];};
	const PhiA& fA(unsigned int i)const{return *fA_[i];};
	//member functions
	void clear();
};

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