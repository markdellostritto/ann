#pragma once
#ifndef REDUCE_HPP
#define REDUCE_HPP

//  c
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// mem
#include "mem/serialize.hpp"

//***************************************************************************
// Reduction - General
//***************************************************************************

template <int D> 
class Reduce{};

//***************************************************************************
// Reduction - 1D
//***************************************************************************

template <>
class Reduce<1>{
private:
	int N_;//number of data points
	double min_;//min
	double max_;//max
	double avg_;//avg
	double m2_;//sum of squares of differences from average
public:
	//==== contructors/destructors ====
	Reduce<1>(){defaults();}
	~Reduce<1>(){}
	
	//==== operators ====
	Reduce<1>& operator+=(const Reduce<1>& r);
	
	//==== access ====
	int N()const{return N_;}
	double min()const{return min_;}
	double max()const{return max_;}
	double avg()const{return avg_;}
	double m2()const{return m2_;}
	double var()const{return m2_/(N_-1);}
	double dev()const{return sqrt(m2_/(N_-1));}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
	void reset(){defaults();}
	void push(double x);
	
	void setN(int x){N_=x;}
	void setmin(double x){min_=x;}
	void setmax(double x){max_=x;}
	void setavg(double x){avg_=x;}
	void setm2(double x){m2_=x;}
	
};
Reduce<1> operator+(const Reduce<1>& r1, const Reduce<1>& r2);

//***************************************************************************
// Reduction - 2D
//***************************************************************************

template <>
class Reduce<2>{
private:
	int N_;
	double avgX_,avgY_;
	double m2X_,m2Y_;
	double covar_;
public:
	//==== contructors/destructors ====
	Reduce<2>(){defaults();}
	~Reduce<2>(){}
	
	//==== operators ====
	Reduce<2>& operator+=(const Reduce<2>& r);
	
	//==== access ====
	int N()const{return N_;}
	double avgX()const{return avgX_;}
	double avgY()const{return avgY_;}
	double m2X()const{return m2X_;}
	double m2Y()const{return m2Y_;}
	double covar()const{return covar_/(N_-1);}
	double m()const{return (m2X_>0)?covar_/m2X_:0;}
	double b()const{return avgY_-m()*avgX_;}
	double r2()const{return (m2X_*m2Y_>0)?covar_*covar_/(m2X_*m2Y_):0;}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
	void push(double x, double y);
	
	void setN(int x){N_=x;}
	void setavgX(double x){avgX_=x;}
	void setavgY(double x){avgY_=x;}
	void setm2X(double x){m2X_=x;}
	void setm2Y(double x){m2Y_=x;}
	void setcovar(double x){covar_=x;}
};

//***************************************************************************
// serialization
//***************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Reduce<1>& obj);
	template <> int nbytes(const Reduce<2>& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Reduce<1>& obj, char* arr);
	template <> int pack(const Reduce<2>& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Reduce<1>& obj, const char* arr);
	template <> int unpack(Reduce<2>& obj, const char* arr);

}

#endif