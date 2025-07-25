#pragma once
#ifndef SET_PROPERTY_CHARGE_HPP
#define SET_PROPERTY_CHARGE_HPP

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
	
class Charge: public Base{
private:
	double charge_;
public:
	//==== constructors/destructors ====
	Charge():Base(Name::CHARGE){}
	~Charge(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Charge& chg);
	
	//==== access ====
	double& charge(){return charge_;}
	const double& charge()const{return charge_;}
	
	//==== member functions ====
	void read(Token& token);
	void set(Structure& struc);
};
	
}

#endif