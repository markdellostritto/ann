#pragma once
#ifndef WINDOW_HPP
#define WINDOW_HPP

//c libraries
#include <cmath>
//c++ libraries
#include <iosfwd>
//math
#include "math/const.hpp"

namespace window{

struct NAME{
	enum type{
		IDENTITY,
		GAUSSIAN,
		BLACKMANHARRIS,
		UNKNOWN
	};
	static NAME::type read(const char* str);
};
std::ostream& operator<<(std::ostream& out, const NAME::type& t);

class Base{
protected:
	int N_;
	NAME::type type_;
public:
	//constructors/destructors
	Base():N_(0),type_(NAME::UNKNOWN){}
	Base(int N):N_(N),type_(NAME::UNKNOWN){}
	virtual ~Base(){}
	//access
	int& N(){return N_;}
	const int& N()const{return N_;}
	const NAME::type& type()const{return type_;}
	//operators
	virtual double operator()(int t)const=0;
};

class Identity: public Base{
public:
	//constructors/destructors
	Identity():Base(){type_=NAME::IDENTITY;}
	//operators
	double operator()(int t)const{return 1.0;}
};

class BlackmanHarris: public Base{
public:
	//constructors/destructors
	BlackmanHarris(int n):Base(n){type_=NAME::BLACKMANHARRIS;}
	//operators
	double operator()(int t)const{
		const double p=2.0*math::constant::PI/(N_-1.0);
		return 0.35875-0.48829*std::cos(t*p)+0.14128*std::cos(t*p*2.0)-0.01168*std::cos(t*p*3.0);
	}
};

class Gaussian: public Base{
private:
	double s_;
public:
	//constructors/destructors
	Gaussian(int n, double s):Base(n),s_(s){type_=NAME::GAUSSIAN;}
	//access
	double& s(){return s_;}
	const double& s()const{return s_;}
	//operators
	double operator()(int t)const{
		const double x=(t-0.5*(N_-1.0))/(s_*0.5*(N_-1.0));
		return std::exp(-0.5*x*x);
	}
};

}

#endif
