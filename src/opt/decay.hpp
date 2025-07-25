#pragma once
#ifndef DECAY_HPP
#define DECAY_HPP

// c++
#include <iosfwd>
#include <memory>
// str
#include "str/token.hpp"
// opt
#include "opt/objective.hpp"
#include "opt/iter.hpp"

#ifndef OPT_DECAY_PRINT_FUNC
#define OPT_DECAY_PRINT_FUNC 0
#endif

namespace opt{

//***************************************************
// decay
//***************************************************

class Decay{
public:
	class Name{
	public:
		enum Type{
			UNKNOWN,
			CONST,
			EXP,
			GAUSS,
			SQRT,
			INV,
			STEP
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
private:
	Name name_;
	int period_;
	double alpha_;
public:
	//==== constructors/destructors ====
	Decay():name_(Name::UNKNOWN),alpha_(0.0),period_(0){}
	Decay(const Name& name, double alpha):name_(name),alpha_(alpha),period_(0){}
	~Decay(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Decay& obj);
	
	//==== member access ====
	double& alpha(){return alpha_;}
	const double& alpha()const{return alpha_;}
	int& period(){return period_;}
	const int& period()const{return period_;}
	Name& name(){return name_;}
	const Name& name()const{return name_;}
	
	//==== member functions ====
	double step(double gamma, const Iterator& iter)const;
	void read(Token& token);
	
};
std::ostream& operator<<(std::ostream& out, const Decay::Name& name);
	
}

//**********************************************
// serialization
//**********************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const opt::Decay& obj);

//**********************************************
// packing
//**********************************************

template <> int pack(const opt::Decay& obj, char* arr);

//**********************************************
// unpacking
//**********************************************

template <> int unpack(opt::Decay& obj, const char* arr);
	
}

#endif