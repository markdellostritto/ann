#pragma once
#ifndef MATH_FUNC_HPP
#define MATH_FUNC_HPP

#include <cmath>
#include <vector>

#ifndef DEBUG_MATH_SPECIAL
#define DEBUG_MATH_SPECIAL 0
#endif 

namespace math{

namespace func{
	
	//**************************************************************
	// sign function
	//**************************************************************

	template <class T> inline int sign(T x)noexcept{return (x>0)-(x<0);}
	
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
	//polynomial evaluation
	//**************************************************************
	
	double poly(double x, const std::vector<double>& a);
	double poly(double x, const double* a, unsigned int s);
	template <unsigned int N> double poly(double x, const double* a)noexcept{
		unsigned int s=N;
		double result=a[--s];
		while(s>0) result=x*result+a[--s];
		return result;
	}
	template <> inline double poly<0>(double x, const double* a)noexcept{
		return a[0];
	}
	template <> inline double poly<1>(double x, const double* a)noexcept{
		return x*a[1]+a[0];
	}
	template <> inline double poly<2>(double x, const double* a)noexcept{
		return x*(x*a[2]+a[1])+a[0];
	}
	template <> inline double poly<3>(double x, const double* a)noexcept{
		return x*(x*(x*a[3]+a[2])+a[1])+a[0];
	}
	template <> inline double poly<4>(double x, const double* a)noexcept{
		return x*(x*(x*(x*a[4]+a[3])+a[2])+a[1])+a[0];
	}
	template <> inline double poly<5>(double x, const double* a)noexcept{
		return x*(x*(x*(x*(x*a[5]+a[4])+a[3])+a[2])+a[1])+a[0];
	}
	template <> inline double poly<6>(double x, const double* a)noexcept{
		return x*(x*(x*(x*(x*(x*a[6]+a[5])+a[4])+a[3])+a[2])+a[1])+a[0];
	}
	
	//**************************************************************
	//power evaluation
	//**************************************************************
	
	template<unsigned int N> double power(double x){
		double result=1;
		for(int i=N; i>0; --i) result*=x;
		return result;
	}
	template<> inline double power<0>(double x){
		return 1.0;
	}
	template<> inline double power<2>(double x){
		return x*x;
	}
	template<> inline double power<3>(double x){
		return x*x*x;
	}
	template<> inline double power<4>(double x){
		x*=x;
		return x*x;
	}
	template<> inline double power<5>(double x){
		x*=x; x*=x; x*=x; x*=x;
		return x;
	}
	template<> inline double power<6>(double x){
		x*=x; x*=x; x*=x; x*=x; x*=x;
		return x;
	}
	template<> inline double power<7>(double x){
		x*=x; x*=x; x*=x; x*=x; x*=x; x*=x;
		return x;
	}
	template<> inline double power<8>(double x){
		x*=x; x*=x; x*=x; x*=x; x*=x; x*=x; x*=x;
		return x;
	}
}

}

#endif