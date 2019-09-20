#pragma once
#ifndef MATH_SPECIAL_HPP
#define MATH_SPECIAL_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include "math_const.hpp"
#include "math_func.hpp"

#ifndef DEBUG_MATH_SPECIAL
#define DEBUG_MATH_SPECIAL 0
#endif 

namespace special{
	
	static const double prec=1E-8;
	
	//**************************************************************
	//Sign functions
	//**************************************************************

	template <class T> inline int sign(T x)noexcept{return (x>0)-(x<0);}
	
	//**************************************************************
	//Modulus functions
	//**************************************************************
	
	template <class T> inline T mod(T n, T z)noexcept{return (n%z+z)%z;}
	template <> inline int mod<int>(int n, int z)noexcept{return (n%z+z)%z;}
	template <> inline uint mod<uint>(uint n, uint z)noexcept{return (n%z+z)%z;}
	template <> inline float mod<float>(float n, float z)noexcept{return fmod(fmod(n,z)+z,z);}
	template <> inline double mod<double>(double n, double z)noexcept{return fmod(fmod(n,z)+z,z);}
	template <class T> inline T mod(T n, T lLim, T uLim)noexcept{return mod<T>(n-lLim,uLim-lLim)+lLim;}
	
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
	
	//**************************************************************
	//Exponential
	//**************************************************************
	
	template <unsigned int N> inline double expl(double x)noexcept{
		x=1.0+x/std::pow(2.0,N);
		for(unsigned int i=0; i<N; ++i) x*=x;
		return x;
	}
	template <> inline double expl<4>(double x)noexcept{
		x=1.0+x/16.0;
		x*=x; x*=x; x*=x; x*=x;
		return x;
	};
	template <> inline double expl<6>(double x)noexcept{
		x=1.0+x/64.0;
		x*=x; x*=x; x*=x; x*=x; x*=x; x*=x;
		return x;
	};
	template <> inline double expl<8>(double x)noexcept{
		x=1.0+x/256.0;
		x*=x; x*=x; x*=x; x*=x;
		x*=x; x*=x; x*=x; x*=x;
		return x;
	};
	static union{
		double d;
		struct{
			#ifdef LITTLE_ENDIAN
				int j,i;
			#else
				int i,j;
			#endif
		} n;
	} eco;
	static const double EXPA=1048576.0/num_const::LOG2;
	static const double EXPB=1072693248.0;
	static const double EXPC=60801.0;
	inline double expb(const double x)noexcept{return (eco.n.i=EXPA*x+(EXPB-EXPC),eco.d);};
	
	//**************************************************************
	//Logarithm
	//**************************************************************
	
	static const double logp1c[6]={1.0,-1.0/2.0,1.0/3.0,-1.0/4.0,1.0/5.0,-1.0/6.0};
	double logp1(double x)noexcept;
	
	//**************************************************************
	//Sigmoid
	//**************************************************************
	
	inline double sigmoid(double x){return 1.0/(1.0+std::exp(-x));}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	static const double sfpc[5]={1.0,1.0/3.0,1.0/5.0,1.0/7.0,1.0/9.0};
	double softplus(double x)noexcept;
	
	//**************************************************************
	//Error Function - Approximations
	//**************************************************************
	struct erfa_const{
		static const double a1[5];
		static const double a2[5];
		static const double a3[7];
		static const double a4[7];
	};
	double erfa1(double x);//max error: 5e-4
	double erfa2(double x);//max error: 2.5e-5
	double erfa3(double x);//max error: 3e-7
	double erfa4(double x);//max error: 1.5e-7
	
	//**************************************************************
	//Kummer's (confluent hypergeometric) function 
	//**************************************************************
	double M(double a, double b, double z, double prec=1e-8);
	
	//**************************************************************
	//Legendre Poylnomials
	//**************************************************************
	std::vector<double>& legendre(unsigned int n, std::vector<double>& c);
	
	//**************************************************************
	//Chebyshev Polynomials
	//**************************************************************
	double chebyshev1r(unsigned int n, double x);//chebyshev polynomial of the first kind - recursive
	double chebyshev1l(unsigned int n, double x);//chebyshev polynomial of the first kind - loop
	std::vector<double>& chebyshev1l(unsigned int n, double x, std::vector<double>& r);//polynomial coefficients
	double chebyshev2r(unsigned int n, double x);//chebyshev polynomial of the second kind - recursive
	double chebyshev2l(unsigned int n, double x);//chebyshev polynomial of the second kind - loop
	std::vector<double>& chebyshev2l(unsigned int n, double x, std::vector<double>& r);//polynomial coefficients
	std::vector<double>& chebyshev1_root(unsigned int n, std::vector<double>& r);//polynomial roots
	std::vector<double>& chebyshev2_root(unsigned int n, std::vector<double>& r);//polynomial roots
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	double jacobi(unsigned int n, double a, double b, double x);
	std::vector<double>& jacobi(unsigned int n, double a, double b, std::vector<double>& c);
	
	//**************************************************************
	//Laguerre Polynomials
	//**************************************************************
	std::vector<double>& laguerre(unsigned int n, std::vector<double>& c);
}

#endif
