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
		UNKNOWN=0,
		G1=1,//Behler G1
		G2=2,//Behler G2
		T1=3//tanh
	};
	static type read(const char* str);
	static const char* name(const PhiRN::type& t);
};
std::ostream& operator<<(std::ostream& out, const PhiRN::type& t);

//*****************************************
// PhiR - radial function base class
//*****************************************

struct PhiR{
	//==== constructors/destructors ====
	virtual ~PhiR(){}
	//==== member functions - evaluation ====
	virtual double val(double r, double cut)const=0;
	virtual double grad(double r, double cut, double gcut)const=0;
};
//==== operators ====
std::ostream& operator<<(std::ostream& out, const PhiR& f);
bool operator==(const PhiR& phir1, const PhiR& phir2);
inline bool operator!=(const PhiR& phir1, const PhiR& phir2){return !(phir1==phir2);}

//*****************************************
// PhiR - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiR& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiR& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiR& obj, const char* arr);
	
}

#endif