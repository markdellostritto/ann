#pragma once
#ifndef MATH_SPECIAL_HPP
#define MATH_SPECIAL_HPP

// c libaries
#include <cstdio>
#include <cstdint>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// c++ libraries
#include <iosfwd>
#include <vector>
#include <array>
#include <complex>
// ann - math
#include "math/const.hpp"
#include "math/poly.hpp"

#ifndef DEBUG_MATH_SPECIAL
#define DEBUG_MATH_SPECIAL 0
#endif 

namespace math{

namespace special{
	
	using math::constant::PI;
	using math::constant::Rad2;
	
	static const double prec=1E-8;
	
	//**************************************************************
	// comparison functions
	//**************************************************************
	
	template <class T> inline T max(T x1, T x2)noexcept{return (x1>x2)?x1:x2;}
	template <class T> inline T min(T x1, T x2)noexcept{return (x1>x2)?x2:x1;}
	
	//**************************************************************
	// modulus functions
	//**************************************************************
	
	template <class T> inline T mod(T n, T z)noexcept{return (n%z+z)%z;}
	template <> inline int mod<int>(int n, int z)noexcept{return (n%z+z)%z;}
	template <> inline unsigned int mod<unsigned int>(unsigned int n, unsigned int z)noexcept{return (n%z+z)%z;}
	template <> inline float mod<float>(float n, float z)noexcept{return fmod(fmod(n,z)+z,z);}
	template <> inline double mod<double>(double n, double z)noexcept{return fmod(fmod(n,z)+z,z);}
	template <class T> inline T mod(T n, T lLim, T uLim)noexcept{return mod<T>(n-lLim,uLim-lLim)+lLim;}
	
	//**************************************************************
	// sign function
	//**************************************************************

	template <typename T> int sgn(T val){return (T(0) < val) - (val < T(0));}
	
	//**************************************************************
	//trig (fdlibm)
	//**************************************************************
	
	static const double cos_const[6]={
		4.16666666666666019037e-02,
		-1.38888888888741095749e-03,
		2.48015872894767294178e-05,
		-2.75573143513906633035e-07,
		2.08757232129817482790e-09,
		-1.13596475577881948265e-11
	};
	//cosine function
	double cos(double x)noexcept;
	double coscut(double x)noexcept;
	
	static const double sin_const[6]={
		-1.66666666666666324348e-01,
		8.33333333332248946124e-03,
		-1.98412698298579493134e-04,
		2.75573137070700676789e-06,
		-2.50507602534068634195e-08,
		1.58969099521155010221e-10
	};
	//sine function
	double sin(double x)noexcept;
	double sincut(double x)noexcept;
	
	//**************************************************************
	//Hypberbolic Functions
	//**************************************************************
	
	double sinh(double x);
	double cosh(double x);
	double tanh(double x);
	double csch(double x);
	double sech(double x);
	double coth(double x);
	void tanhsech(double x, double& ftanh, double& fsech);
	double sech2(double x);
	
	//**************************************************************
	//Power
	//**************************************************************
	
	double powint(double x, int p);
	
	double sqrta(double x);
	
	//**************************************************************
	//Sigmoid
	//**************************************************************
	
	double sigmoid(double x);
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	double softplus(double x)noexcept;
	double softplus2(double x)noexcept;
	double logcosh(double x)noexcept;
	
	//**************************************************************
	//Exponential
	//**************************************************************
	
	double exp2_x86(double x);
	double fmexp(double x);
	double expn10(double x);
	double gauss10(double x);

	//**************************************************************
	//Error function
	//**************************************************************

	double erfa1(double x);
	
	//**************************************************************
	//Beta Function
	//**************************************************************
	
	double beta(double z, double w);
	double fratio(double num, double den);
	
	//**************************************************************
	//Kummer's (confluent hypergeometric) function 
	//**************************************************************
	double M(double a, double b, double z, double prec=1e-8);
	
	//**************************************************************
	//spherical harmonics - real
	//**************************************************************
	
	/*class YLMR{
	private:
		int l_;//angular momentum
		double nl_;//normalization constant l
		std::vector<double> nm_;//normalization constant m
		std::vector<double> pl_;//legendre polynomials
		std::vector<double> val_;
	public:
		//==== constructors/destructors ====
		YLMR(){init(0);}
		YLMR(int l){init(l);}
		~YLMR(){}
		
		//==== operators ====
		friend std::ostream& operator<<(std::ostream& out, const YLMR yrlm);
		
		//==== member access ====
		const int& l()const{return l_;}
		const double& nl()const{return nl_;}
		
		//==== member functions ====
		void clear();
		void init(int l);
		void compute(double theta, double phi);
		//void compute(Eigen::Vector3d& r);
		double val(int m);
	};*/
	
}

}

#endif
