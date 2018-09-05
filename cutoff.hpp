#ifndef CUTOFF_HPP
#define CUTOFF_HPP

// c libraries
#include <cmath>
#include <cstring>
// c ++libraries
#include <iostream> 
#include <functional> 
// local libraries
#include "math_const.hpp"

//************************************************************
// CUTOFF NAMES
//************************************************************

//cutoff names
struct CutoffN{
	enum type{
		UNKNOWN=-1,
		COS=0,
		TANH=1
	};
	static type load(const char* str);
};
std::ostream& operator<<(std::ostream& out, const CutoffN::type& cutt);

//typedefs
typedef std::function<double(double, double)> FCutT;

//************************************************************
// CUTOFF FUNCTIONS
//************************************************************

//cutoff functions
struct CutoffF{
	static const unsigned int N_CUT_F=2;
	static double cut_cos(double r, double rc)noexcept{return 0.5*(std::cos(num_const::PI*r/rc)+1.0)*(r<=rc);};
	static double cut_tanh(double r, double rc)noexcept{double f=std::tanh(1.0-r/rc); return f*f*f*(r<=rc);};
	static FCutT funcs[N_CUT_F];
};

//cutoff functions
struct CutoffFD{
	static const unsigned int N_CUT_F=2;
	static double cut_cos(double r, double rc)noexcept{return -0.5*num_const::PI/rc*std::sin(num_const::PI*r/rc)*(r<=rc);};
	static double cut_tanh(double r, double rc)noexcept{
		double cosh=std::cosh(1.0-r/rc);
		double tanh=std::tanh(1.0-r/rc);
		return -3.0*tanh*tanh/(rc*cosh*cosh)*(r<=rc);
	};
	static FCutT funcs[N_CUT_F];
};

#endif