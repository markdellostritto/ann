#pragma once
#ifndef CUTOFF_HPP
#define CUTOFF_HPP

// c libraries
#include <cmath>
// c ++libraries
#include <iosfwd>
// ann - math
#include "math_const_ann.h"
#include "math_special_ann.h"

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
	static type read(const char* str);
};
std::ostream& operator<<(std::ostream& out, const CutoffN::type& cutt);

//typedefs
typedef double (*FCutT)(double,double);

//************************************************************
// CUTOFF FUNCTIONS
//************************************************************

//cutoff functions
struct CutoffF{
	static const unsigned int N_CUT_F=2;
	static inline double cut_cos(double r, double rc){return (r>rc)?0:0.5*(special::cos(num_const::PI*r/rc)+1.0);}
	static inline double cut_tanh(double r, double rc){const double f=(r>rc)?0:std::tanh(1.0-r/rc); return f*f*f;}
	static const FCutT funcs[N_CUT_F];
};

//cutoff functions
struct CutoffFD{
	static const unsigned int N_CUT_F=2;
	static inline double cut_cos(double r, double rc){return (r>rc)?0:-0.5*num_const::PI/rc*special::sin(num_const::PI*r/rc);}
	static inline double cut_tanh(double r, double rc){
		if(r<=rc){
			const double tanh=std::tanh(1.0-r/rc);
			return -3.0*tanh*tanh*(1.0-tanh*tanh)/rc;
		} else return 0;
	};
	static const FCutT funcs[N_CUT_F];
};

#endif
