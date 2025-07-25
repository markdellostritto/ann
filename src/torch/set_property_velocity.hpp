#pragma once
#ifndef SET_PROPERTY_VEL_HPP
#define SET_PROPERTY_VEL_HPP

//c++
#include <iosfwd>
//structure
#include "struc/structure.hpp"
//torch
#include "torch/set_property.hpp"
//eigen
#include <Eigen/Dense>

namespace property{
	
class Velocity: public Base{
private:
	Eigen::Vector3d vel_;
public:
	//==== constructors/destructors ====
	Velocity():Base(Name::VELOCITY){}
	~Velocity(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Velocity& vel);
	
	//==== access ====
	const Eigen::Vector3d& vel(){return vel_;}
	
	//==== member functions ====
	void read(Token& token);
	void set(Structure& struc);
};
	
}

#endif