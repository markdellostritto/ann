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

#ifndef DEBUG_MATH_POLY
#define DEBUG_MATH_POLY 0
#endif 

namespace math{

namespace poly{

double alegendre(int l, int m, double x);
double legendre(int n, double x);
double plegendre(int l, int m, double x);
double chebyshev(int n, double x);
std::vector<double>& chebyshev(int n, double x, std::vector<double>& f);

}//end namespace poly

}//end namespace math

#endif