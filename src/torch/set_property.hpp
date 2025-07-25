#pragma once
#ifndef SET_PROPERTY_HPP
#define SET_PROPERTY_HPP

//c++
#include <iosfwd>
//string
#include "str/token.hpp"
//structure
#include "struc/structure.hpp"
#include "analysis/group.hpp"

namespace property{

//****************************************************************************
// Name
//****************************************************************************

class Name{
public:
	enum Type{
		TYPE,
		MASS,
		CHARGE,
		VELOCITY,
		TEMP,
		UNKNOWN
	};
	//constructor
	Name():t_(Type::UNKNOWN){}
	Name(Type t):t_(t){}
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Name& name);
	operator Type()const{return t_;}
	//member functions
	static Name read(const char* str);
	static const char* name(const Name& name);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};

//****************************************************************************
// Base
//****************************************************************************

class Base{
protected:
	Name name_;
	Group group_;
public:
	//==== constructors/destructors ====
	Base():name_(Name::UNKNOWN){}
	Base(const Name& name):name_(name){}
	virtual ~Base(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Base& base);
	
	//==== access ====
	Name& name(){return name_;}
	const Name& name()const{return name_;}
	Group& group(){return group_;}
	const Group& group()const{return group_;}
	
	//==== member functions ====
	void read(Token& token){}
	virtual void set(Structure& struc)=0;
};

}

#endif