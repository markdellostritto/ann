#ifndef MATH_SPECIAL_HPP
#define MATH_SPECIAL_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "math_const.hpp"

#ifndef DEBUG_MATH_SPECIAL
#define DEBUG_MATH_SPECIAL 0
#endif 

namespace special{
	
	static const double prec=1E-8;
	
	//**************************************************************
	//Sigmoid function
	//**************************************************************
	
	inline double sigmoid(double x){return 1.0/(1.0+std::exp(-x));}
	
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