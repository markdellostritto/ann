#ifndef SYMM_RADIAL_HPP
#define SYMM_RADIAL_HPP

// c libraries
#include <cstring>
// c++ libraries
#include <iostream>
// local libraries - cutoff
#include "cutoff.h"
// local libraries - serialization
#include "serialize.h"

//radial function names
struct PhiRN{
	enum type{
		UNKNOWN=-1,
		G1=0,//Behler G1
		G2=1//Behler G2
	};
	static type load(const char* str);
};
std::ostream& operator<<(std::ostream& out, const PhiRN::type& t);

//PhiR
struct PhiR{
	double rc;//cutoff radius
	CutoffN::type tcut;//cutoff function type
	PhiR():rc(0),tcut(CutoffN::COS){};
	PhiR(CutoffN::type tcut_, double rc_):tcut(tcut_),rc(rc_){};
	virtual ~PhiR(){};
	virtual double operator()(double r)const=0;
	virtual double val(double r)const=0;
	virtual double grad(double r)const=0;
};
std::ostream& operator<<(std::ostream& out, const PhiR& f);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiR& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiR& obj, const char* arr);
	
}

#endif
