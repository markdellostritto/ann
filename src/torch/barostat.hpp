#pragma once
#ifndef BAROSTAT_HPP
#define BAROSTAT_HPP

//****************************************************************************
// Barostat
//****************************************************************************

class Barostat{
public:
	enum Type{
		NONE,
		ISO,
		ANISO,
		NONE
	};
	//constructor
	Barostat():t_(Type::NONE){}
	Barostat(Type t):t_(t){}
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Barostat& b);
	operator Type()const{return t_;}
	//member functions
	static Barostat read(const char* str);
	static const char* name(const Barostat& b);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};

#endif