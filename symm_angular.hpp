#ifndef SYMM_ANGULAR_HPP
#define SYMM_ANGULAR_HPP

// c libraries
#include <cstring>
// c++ libraries
#include <iostream>
// local libraries
#include "cutoff.hpp"

//angular function names
struct PhiAN{
	enum type{
		UNKNOWN,
		G3,//Behler G3
		G4//Behler G4
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
	virtual double amp(double cos, double ri, double rj, double rij)const noexcept=0;
	virtual double cut(double ri, double rj, double rij)const noexcept=0;
	virtual double grad(double cos, double ri, double rj, double rij, unsigned int gindex)const=0;
	virtual double grad_pre(double cos, double ri, double rj, double rij, unsigned int gindex)const=0;
};
std::ostream& operator<<(std::ostream& out, const PhiA& f);

#endif