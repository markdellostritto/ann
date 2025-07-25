#pragma once
#ifndef RBF_HPP
#define RBF_HPP

// c++ libraries
#include <iostream>
#include <vector>
#include <memory>
// Eigen
#include <Eigen/Dense>

class RBF{
public:
	class Name{
	public:
		enum Type{
			GAUSSIAN,
			MULTIQUADRIC,
			IMQUADRIC,
			IQUADRATIC,
			UNKNOWN
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
	};
private:
	Name name_;
	double eps_;
	double ieps2_;
public:
	//==== constructors/destructors ====
	RBF():name_(Name::UNKNOWN){}
	RBF(const Name& name, double eps):eps_(eps),ieps2_(1.0/(eps*eps)),name_(name){}
	~RBF(){}
	
	//==== access ====
	const double& eps()const{return eps_;}
	const Name& name()const{return name_;}
	
	//==== operators ====
	double operator()(double r)const;
	friend std::ostream& operator<<(std::ostream& out, const RBF& b){return out<<b.name_<<" "<<b.eps_;}
	
	//==== member functions ====
	double val(double r)const;
	double val2(double r2)const;
};
std::ostream& operator<<(std::ostream& out, const RBF::Name& name);

#endif