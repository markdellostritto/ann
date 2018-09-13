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
	PhiR_G1(CutoffN::type tcut_, double rc_):PhiR(tcut_,rc_){};
	double operator()(double r)const noexcept;
	double val(double r)const noexcept;
	double amp(double r)const noexcept;
	double cut(double r)const noexcept;
	double grad(double r)const noexcept;
};
std::ostream& operator<<(std::ostream& out, const PhiR_G1& f);

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

#endif