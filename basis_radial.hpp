#ifndef BASIS_RADIAL_HPP
#define BASIS_RADIAL_HPP

// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
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
private:
	static const double V_CUT;//value cutoff
	unsigned int nfR_;//number of radial functions
	PhiRN::type phiRN_;//type of radial functions
	PhiR** fR_;//radial functions
public:
	//constructors/destructors
	BasisR():nfR_(0),phiRN_(PhiRN::UNKNOWN),fR_(NULL){};
	BasisR(const BasisR& basisR);
	~BasisR();
	//operators
	BasisR& operator=(const BasisR& basis);
	friend std::ostream& operator<<(std::ostream& out, const BasisR& basisR);
	//initialization
	void init_G1(CutoffN::type tcut, double rcut, double rmin);
	void init_G2(unsigned int nR, CutoffN::type tcut, double rcut, double rmin);
	//loading/printing
	static void write(FILE* writer,const BasisR& basis);
	static void read(FILE* writer, BasisR& basis);
	//member access
	const unsigned int& nfR()const{return nfR_;};
	PhiRN::type& phiRN(){return phiRN_;};
	const PhiRN::type& phiRN()const{return phiRN_;};
	PhiR& fR(unsigned int i){return *fR_[i];};
	const PhiR& fR(unsigned int i)const{return *fR_[i];};
	//member functions
	void clear();
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