#pragma once
#ifndef MATH_SPECIAL_ANN_HPP
#define MATH_SPECIAL_ANN_HPP

//c++ libraries
#include <iosfwd>
//ann - math
#include "math_const_ann.h"

#ifndef DEBUG_MATH_SPECIAL
#define DEBUG_MATH_SPECIAL 0
#endif 

namespace special{
	
	static const double prec=1E-8;
	
	//**************************************************************
	//Sign functions
	//**************************************************************

	template <class T> inline int sign(T x){return (x>0)-(x<0);}
	
	//**************************************************************
	//Modulus functions
	//**************************************************************
	
	template <class T> inline T mod(T n, T z){return (n%z+z)%z;}
	template <> inline int mod<int>(int n, int z){return (n%z+z)%z;}
	template <> inline unsigned int mod<unsigned int>(unsigned int n, unsigned int z){return (n%z+z)%z;}
	template <> inline float mod<float>(float n, float z){return fmod(fmod(n,z)+z,z);}
	template <> inline double mod<double>(double n, double z){return fmod(fmod(n,z)+z,z);}
	template <class T> inline T mod(T n, T lLim, T uLim){return mod<T>(n-lLim,uLim-lLim)+lLim;}
	
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
	double cos(double x);
	
	static const double sin_const[6]={
		-1.66666666666666324348e-01,
		8.33333333332248946124e-03,
		-1.98412698298579493134e-04,
		2.75573137070700676789e-06,
		-2.50507602534068634195e-08,
		1.58969099521155010221e-10
	};
	//sine function
	double sin(double x);
	
	//**************************************************************
	//Exponential
	//**************************************************************
	
	template <unsigned int N> inline double expl(double x){
		x=1.0+x/std::pow(2.0,N);
		for(unsigned int i=0; i<N; ++i) x*=x;
		return x;
	}
	template <> inline double expl<4>(double x){
		x=1.0+x/16.0;
		x*=x; x*=x; x*=x; x*=x;
		return x;
	}
	template <> inline double expl<6>(double x){
		x=1.0+x/64.0;
		x*=x; x*=x; x*=x; x*=x; x*=x; x*=x;
		return x;
	}
	template <> inline double expl<8>(double x){
		x=1.0+x/256.0;
		x*=x; x*=x; x*=x; x*=x;
		x*=x; x*=x; x*=x; x*=x;
		return x;
	}
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
	inline double expb(const double x){return (eco.n.i=EXPA*x+(EXPB-EXPC),eco.d);}
	
	//**************************************************************
	//Logarithm
	//**************************************************************
	
	static const double logp1c[5]={1.0,1.0/3.0,1.0/5.0,1.0/7.0,1.0/9.0};
	double logp1(double x);
	
	//**************************************************************
	//Sigmoid
	//**************************************************************
	
	inline double sigmoid(double x){return 1.0/(1.0+std::exp(-x));}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	static const double sfpc[5]={1.0,1.0/3.0,1.0/5.0,1.0/7.0,1.0/9.0};
	double softplus(double x);
	
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
	
}

#endif