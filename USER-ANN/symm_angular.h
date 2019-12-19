#pragma once
#ifndef SYMM_ANGULAR_HPP
#define SYMM_ANGULAR_HPP

// c++ libraries
#include <iosfwd>
// ann - serialization
#include "serialize.h"

//*****************************************
// PhiAN - angular function names
//*****************************************

struct PhiAN{
	enum type{
		UNKNOWN=-1,
		G3=0,//Behler G3
		G4=1//Behler G4
	};
	static type load(const char* str);
};
std::ostream& operator<<(std::ostream& out, const PhiAN::type& t);

//*****************************************
// PhiA - angular function base class
//*****************************************

struct PhiA{
	virtual ~PhiA(){};
	virtual double val(double cos, const double r[3], const double c[3])const=0;
	virtual double dist(const double r[3], const double c[3])const=0;
	virtual double angle(double cos)const=0;
	virtual double grad_angle(double cos)const=0;
	virtual double grad_dist_0(const double r[3], const double c[3], double gij)const=0;
	virtual double grad_dist_1(const double r[3], const double c[3], double gik)const=0;
	virtual double grad_dist_2(const double r[3], const double c[3], double gjk)const=0;
	virtual void compute_angle(double cos, double& val, double& grad)const=0;
	virtual void compute_dist(const double r[3], const double c[3], const double g[3], double& dist, double* gradd)const=0;
};
std::ostream& operator<<(std::ostream& out, const PhiA& f);
bool operator==(const PhiA& phia1, const PhiA& phia2);
inline bool operator!=(const PhiA& phia1, const PhiA& phia2){return !(phia1==phia2);};

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const PhiA& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(PhiA& obj, const char* arr);
	
}

#endif