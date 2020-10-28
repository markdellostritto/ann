#ifndef CUTOFF_HPP
#define CUTOFF_HPP

// c libraries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
// c++ libraries
#include <iosfwd> 
// ann - math
#include "math_const_ann.h"
#include "math_special_ann.h"

namespace cutoff{

//==== using statements ====

using math::constant::PI;

//************************************************************
// CUTOFF NAMES
//************************************************************

//cutoff names
struct Name{
	enum type{
		UNKNOWN=-1,
		COS=0,
		TANH=1
	};
	static type read(const char* str);
	static const char* name(const Name::type& cutt);
};
std::ostream& operator<<(std::ostream& out, const Name::type& cutt);

//************************************************************
// CUTOFF FUNCTIONS
//************************************************************

//==== Func ====

class Func{
protected:
	Name::type name_;
	double rc_,rci_;
public:
	//constructors/destructors
	Func():rc_(0),rci_(0),name_(Name::UNKNOWN){}
	Func(double rc):rc_(rc),rci_(1.0/rc),name_(Name::UNKNOWN){}
	virtual ~Func(){}
	//access
	const Name::type& name()const{return name_;}
	const double& rc()const{return rc_;}
	const double& rci()const{return rci_;}
	//member functions
	virtual double val(double r)const=0;
	virtual double grad(double r)const=0;
	virtual void compute(double r, double& v, double& g)const=0;
};

//==== Cos ====

class Cos: public Func{
private:
	double prci_;
public:
	//constructors/destructors
	Cos():Func(){name_=Name::COS;}
	Cos(double rc):Func(rc),prci_(rci_*PI){name_=Name::COS;}
	virtual ~Cos(){}
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Cos& obj);
	//member functions
	double val(double r)const;
	double grad(double r)const;
	void compute(double r, double& v, double& g)const;
};

//==== Tanh ====

class Tanh: public Func{
public:
	//constructors/destructors
	Tanh():Func(){name_=Name::TANH;}
	Tanh(double rc):Func(rc){name_=Name::TANH;}
	virtual ~Tanh(){}
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Tanh& obj);
	//member functions
	double val(double r)const;
	double grad(double r)const;
	void compute(double r, double& v, double& g)const;
};

}

#endif
