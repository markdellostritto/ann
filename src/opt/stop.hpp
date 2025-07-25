#pragma once
#ifndef STOP_HPP
#define STOP_HPP

// c++
#include <iosfwd>

//***************************************************
// stopping criterion
//***************************************************

namespace opt{
	
class Stop{
public:
	enum Type{
		UNKNOWN=0,
		FABS=1,
		FREL=2,
		XABS=3,
		XREL=4
	};
	//constructor
	Stop():t_(Type::UNKNOWN){}
	Stop(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Stop read(const char* str);
	static const char* name(const Stop& stop);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Stop& stop);

}

#endif