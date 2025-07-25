#pragma once
#ifndef SET_PROPERTY_TYPE_HPP
#define SET_PROPERTY_TYPE_HPP

//c++
#include <iosfwd>
#include <vector>
//string
#include "str/token.hpp"
//structure
#include "struc/structure.hpp"
//torch
#include "torch/set_property.hpp"

namespace property{
	
class Type: public Base{
private:
	int type_;
public:
	//==== constructors/destructors ====
	Type():Base(Name::CHARGE){}
	~Type(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Type& type);
	
	//==== access ====
	int& type(){return type_;}
	const int& type()const{return type_;}
	
	//==== member functions ====
	void read(Token& token);
	void set(Structure& struc);
};
	
}

#endif