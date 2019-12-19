#pragma once
#ifndef SYMM_RADIAL_HPP
#define SYMM_RADIAL_HPP

// c++ libraries
#include <iosfwd>
// ann - serialization
#include "serialize.h"

//*****************************************
// PhiRN - radial function names
//*****************************************

struct PhiRN{
	enum type{
		UNKNOWN=-1,
		G1=0,//Behler G1
		G2=1,//Behler G2
		T1=2//tanh
	};
	static type load(const char* str);
};
std::ostream& operator<<(std::ostream& out, const PhiRN::type& t);

//*****************************************
// PhiR - radial function base class
//*****************************************

struct PhiR{
	virtual ~PhiR(){};
	virtual double val(double r, double cut)const=0;
	virtual double grad(double r, double cut, double gcut)const=0;
};
std::ostream& operator<<(std::ostream& out, const PhiR& f);
bool operator==(const PhiR& phir1, const PhiR& phir2);
inline bool operator!=(const PhiR& phir1, const PhiR& phir2){return !(phir1==phir2);}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiR& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const PhiR& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(PhiR& obj, const char* arr);
	
}

#endif