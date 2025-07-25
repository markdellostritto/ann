#pragma once
#ifndef MATH_POLY_HPP
#define MATH_POLY_HPP

// c libaries
#include <cstdio>
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
// ann - math
#include "math/const.hpp"
#include "math/special.hpp"

#include <iostream>

#ifndef DEBUG_MATH_POLY
#define DEBUG_MATH_POLY 0
#endif 

namespace math{

namespace poly{

using math::constant::PI;

double alegendre(int l, int m, double x){
	if(m<0 || m>l) throw std::invalid_argument("math::poly::legendre(int,int,double): Invalid integers.");
	//if(fabs(x)>1.0) throw std::invalid_argument("math::poly::legendre(int,int,double): Invalid argument.");
	double pmm=1.0;
	if(m>0){
		const double somx2=sqrt((1.0-x)*(1.0+x));
		double fact=1.0;
		for (int i=1; i<=m; ++i) {
			pmm *= -fact*somx2;
			fact += 2.0;
		}
	}
	if(l==m){
		return pmm;
	} else { 
		double pmmp1=x*(2*m+1)*pmm;
		if(l==m+1){
			return pmmp1;
		} else { 
			double pll=0;
			for(int ll=m+2; ll<=l; ++ll){
				pll=(x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m);
				pmm=pmmp1;
				pmmp1=pll;
			}
			return pll;
		}
	}
}

double legendre(int n, double x){
	if(n<0) throw std::runtime_error("legendre(int,double): invalid order");
	else if(n==0) return 1.0;
	else {
		double rm2=1.0,rm1=x,r=x;
		for(int i=2; i<=n; ++i){
			r=((2.0*n-1.0)*x*rm1-(n-1.0)*rm2)/i;
			rm2=rm1; rm1=r;
		}
		return r;
	}
}

double plegendre(int l, int m, double x){
	if(abs(m)>l || abs(x)>1.0) throw std::invalid_argument("plegendre(int,int,double): invalid arguments.");
	double sfac=1.0;
	if(m<0){
		m*=-1;
		if(m%2!=0) sfac*=-1;
	}
	double rval=0.0;
	double pmm=1.0;
	if(m>0){
		double fact=1.0;
		const double omx2=(1.0-x)*(1.0+x);
		for(int i=1; i<=m; i++){
			pmm*=omx2*fact/(fact+1.0);
			fact+=2.0;
		}
	}
	pmm=sqrt((2.0*m+1.0)*pmm/(4.0*PI));
	if(m%2!=0) pmm=-pmm;
	if(l==m) rval=pmm;
	else{
		double pmmp1=x*sqrt(2.0*m+3.0)*pmm;
		if(l==(m+1)) rval=pmmp1;
		else{
			double pll;
			double oldfact=sqrt(2.0*m+3.0);
			for(int ll=m+2; ll<=l; ll++){
				const double fact=sqrt((4.0*ll*ll-1.0)/(ll*ll-m*m));
				pll=(x*pmmp1-pmm/oldfact)*fact;
				oldfact=fact;
				pmm=pmmp1;
				pmmp1=pll;
			}
			rval=pll;
		}
	}
	return sfac*rval;
}

double chebyshev(int n, double x){
	if(n<0 || abs(x)>1.0) throw std::invalid_argument("chebyshev(int,double): invalid arguments.");
	if(n==0) return 1.0;
	else if(n==1) return x;
	else {
		double fnm2=1.0;
		double fnm1=x;
		double fnm0;
		for(int i=2; i<=n; ++i){
			fnm0=2.0*x*fnm1-fnm2;
			fnm2=fnm1;
			fnm1=fnm0;
		}
		return fnm0;
	}
}

std::vector<double>& chebyshev(int n, double x, std::vector<double>& f){
	if(n<0 || abs(x)>1.0) throw std::invalid_argument("chebyshev(int,double): invalid arguments.");
	f.resize(n+1,0.0);
	if(n>=0) f[0]=1.0;
	if(n>=1) f[1]=x;
	if(n>=2){
		for(int i=2; i<=n; ++i){
			f[i]=2.0*x*f[i-1]-f[i-2];
		}
	}
	return f;
}

}//end namespace poly

}//end namespace math

#endif