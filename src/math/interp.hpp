#ifndef INTERP_HPP
#define INTERP_HPP

//c++
#include <vector>
#include <iostream>
// Eigen
#include <Eigen/Dense>
//math
#include "math/rbf.hpp"
//string
#include "str/token.hpp"

namespace math{
namespace interp{

//***************************************************
// Name
//***************************************************

class Name{
public:
	enum Type{
		UNKNOWN,
		CONST,
		LINEAR,
		AKIMA,
		RBFI
	};
	//constructor
	Name():t_(Type::UNKNOWN){}
	Name(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Name read(const char* str);
	static const char* name(const Name& name);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Name& name);

//***************************************************
// Base
//***************************************************

class Base{
protected:
	Name name_;
	int size_;
	Eigen::VectorXd x_;
	Eigen::VectorXd y_;
	Eigen::VectorXd d_;
public:
	//==== constructors/destructors ====
	Base():name_(Name::UNKNOWN){}
	Base(Name name):name_(name){}
	Base(Name name, int s):name_(name){resize(s);}
	virtual ~Base(){}
	
	//==== access ====
	const Name& name()const{return name_;}
	double& x(int i){return x_[i];}
	const double& x(int i)const{return x_[i];}
	double& y(int i){return y_[i];}
	const double& y(int i)const{return y_[i];}
	double& d(int i){return d_[i];}
	const double& d(int i)const{return d_[i];}
	
	//==== member functions ====
	void clear();
	virtual void resize(int s);
	int search(double x);
	virtual void init(){};
	virtual void read(Token& token){};
	virtual double eval(double x)=0;
};

//***************************************************
// Const
//***************************************************

class Const: public Base{
public:
	//==== constructors/destructors ====
	Const():Base(Name::CONST){}
	Const(int s):Base(Name::CONST){resize(s);}
	~Const(){}
	
	//=== member functions ====
	double eval(double x);
};

//***************************************************
// Linear
//***************************************************

class Linear: public Base{
public:
	//==== constructors/destructors ====
	Linear():Base(Name::LINEAR){}
	Linear(int s):Base(Name::LINEAR){resize(s);}
	~Linear(){}
	
	//=== member functions ====
	double eval(double x);
};

//***************************************************
// Akima
//***************************************************

class Akima: public Base{
protected:
	std::vector<double> s_;
	std::vector<double> m_;
public:
	//==== constructors/destructors ====
	Akima():Base(Name::AKIMA){}
	Akima(int s):Base(Name::AKIMA){resize(s);}
	~Akima(){}
	
	//=== member functions ====
	void resize(int s);
	void init();
	double eval(double x);
};

//***************************************************
// RBFI
//***************************************************

class RBFI: public Base{
protected:
	RBF rbf_;//radial basis function
	Eigen::MatrixXd m_;//rbf matrix
	Eigen::VectorXd z_;//result
public:
	//==== constructors/destructors ====
	RBFI():Base(Name::RBFI){}
	RBFI(int s):Base(Name::RBFI){resize(s);}
	~RBFI(){}
	
	//=== access ====
	RBF& rbf(){return rbf_;}
	const RBF& rbf()const{return rbf_;}
	
	//=== member functions ====
	void resize(int s);
	void init();
	double eval(double x);
};

}
}

#endif