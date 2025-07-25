#pragma once
#ifndef COMPUTE_STATE_HPP
#define COMPUTE_STATE_HPP

namespace compute{

namespace state{

//****************************************************************************
// Name
//****************************************************************************

class Name{
public:
	enum Type{
		KE,
		PE,
		TE,
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
public:
	//==== constructors/destructors ====
	Base():name_(Name::UNKNOWN){}
	Base(const Name& name):name_(name){}
	virtual ~Base(){}

	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const Base& base);
	
	//==== access ====
	Name& name(){return name_;}
	const Name& name()const{return name_;}
	
	//==== member functions ====
	void read(Token& token){}
	void double compute(const Struc& struc)=0;
};

}

}

#endif