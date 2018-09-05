#ifndef SYMM_RADIAL_HPP
#define SYMM_RADIAL_HPP

// c libraries
#include <cstring>
// c++ libraries
#include <iostream>
// local libraries
#include "cutoff.hpp"

//radial function names
struct PhiRN{
	enum type{
		UNKNOWN,
		G1,//Behler G1
		G2//Behler G2
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
	virtual double operator()(double r)const noexcept=0;
	virtual double val(double r)const noexcept=0;
	virtual double amp(double r)const noexcept=0;
	virtual double cut(double r)const noexcept=0;
	virtual double grad(double r)const noexcept=0;
};
std::ostream& operator<<(std::ostream& out, const PhiR& f);

#endif