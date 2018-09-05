#ifndef MATH_FUNCTION_HPP
#define MATH_FUNCTION_HPP

#include <cmath>
#include <vector>

#ifndef DEBUG_MATH_SPECIAL
#define DEBUG_MATH_SPECIAL 0
#endif 

namespace function{
	
	//======== modulo ========
	
	template <class T> inline T mod(T n, T z){return (n%z+z)%z;}
	template <class T> inline T mod(T n, T lLim, T uLim){return mod<T>(n-lLim,uLim-lLim)+lLim;}
	
	template<> inline int mod<int>(int n, int z){return (n%z+z)%z;}
	template<> inline int mod<int>(int n, int lLim, int uLim){return mod<int>(n-lLim,uLim-lLim)+lLim;}
	
	template<> inline double mod<double>(double n, double z){return fmod(fmod(n,z)+z,z);}
	template<> inline double mod<double>(double n, double lLim, double uLim){return mod<double>(n-lLim,uLim-lLim)+lLim;}
	
	//======== sign ========
	
	template <class T> inline int sign(T x){return (x>0)-(x<0);}
	
	//======== round ========
	template <class T> int round(T x){return (std::fabs(x-(int)x)>=0.5) ? (int)x+1*sign(x) : (int)x;}
	
	//======== polynomial ========
	
	double poly(double x, const std::vector<double>& a);
	double poly(double x, const double* a, unsigned int s);

}

#endif