#pragma once
#ifndef SET_PROPERTY_TEMP_HPP
#define SET_PROPERTY_TEMP_HPP

//c++
#include <iosfwd>
//string
#include "str/token.hpp"
//structure
#include "struc/structure.hpp"
//torch
#include "torch/set_property.hpp"

namespace property{

class Temp: public Base{
private:
	double temp_;
public:
	//==== constructors/destructors ====
	Temp():Base(Name::TEMP){}
	~Temp(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Temp& temp);
	
	//==== access ====
	const double& temp(){return temp_;}
	
	//==== member functions ====
	void read(Token& token);
	void set(Structure& struc);
};
	
}

#endif