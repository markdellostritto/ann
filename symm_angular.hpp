#ifndef SYMM_ANGULAR_HPP
#define SYMM_ANGULAR_HPP

// c libraries
#include <cstring>
// c++ libraries
#include <iostream>
// local libraries
#include "cutoff.hpp"
// local libraries - serialization
#include "serialize.hpp"

//angular function names
struct PhiAN{
	enum type{
		UNKNOWN=-1,
		G3=0,//Behler G3
		G4=1//Behler G4
	};
	static type load(const char* str);
};
std::ostream& operator<<(std::ostream& out, const PhiAN::type& t);

//PhiA
struct PhiA{
	double rc;//cutoff radius
	CutoffN::type tcut;//cutoff function type
	PhiA():tcut(CutoffN::COS),rc(0){};
	PhiA(CutoffN::type tcut_, double rc_):tcut(tcut_),rc(rc_){};
	virtual ~PhiA(){};
	virtual double operator()(double cos, double ri, double rj, double rij)const noexcept=0;
	virtual double val(double cos, double ri, double rj, double rij)const noexcept=0;
	virtual double angle(double cos)const noexcept=0;
	virtual double dist(double ri, double rj, double rij)const noexcept=0;
	virtual double grad_angle(double cos)const noexcept=0;
	virtual double grad_dist(double rij, double rik, double rjk, unsigned int gindex)const=0;
};
std::ostream& operator<<(std::ostream& out, const PhiA& f);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiA& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiA& obj, const char* arr);
	
}

#endif