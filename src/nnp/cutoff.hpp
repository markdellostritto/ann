#pragma once
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
// math
#include "math/const.hpp"
// mem
#include "mem/serialize.hpp"

class Cutoff{
public:
	class Name{
	public:
		enum Type{
			NONE,
			STEP,
			COS,
			TANH,
			POLY3
		};
		//constructor
		Name():t_(Type::NONE){}
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
protected:
	Name name_;//cutoff name
	double rc_;//cutoff radius
	double rci_;//cutoff radius inverse
	double pirci_;//PI*rci_
public:
	//==== constructors/destructors ====
	Cutoff():name_(Name::NONE),rc_(0),rci_(0),pirci_(0){}
	Cutoff(Name name, double rc);
	~Cutoff(){}
	
	//==== member access ====
	const Name& name()const{return name_;}
	const double& rc()const{return rc_;}
	
	//==== member functions ====
	double cutf(double dr)const;
	double cutg(double dr)const;
	void compute(double dr, double& v, double& g)const;
};
std::ostream& operator<<(std::ostream& out, const Cutoff::Name& name);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Cutoff& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Cutoff& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Cutoff& obj, const char* arr);
	
}

#endif
