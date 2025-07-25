#pragma once
#ifndef JOB_HPP
#define JOB_HPP

#ifndef ENGINE_PRINT_FUNC
#define ENGINE_PRINT_FUNC 0
#endif

//c++
#include <ostream>

//****************************************************************************
// Job
//****************************************************************************

class Job{
public:
	enum Type{
		SP,
		MC,
		MD,
		UNKNOWN
	};
	//constructor
	Job():t_(Type::UNKNOWN){}
	Job(Type t):t_(t){}
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Job& sys);
	operator Type()const{return t_;}
	//member functions
	static Job read(const char* str);
	static const char* name(const Job& job);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};

#endif