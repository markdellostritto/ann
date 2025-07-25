//c++
#include <iostream>
//torch
#include "torch/set_property_velocity.hpp"

namespace property{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Velocity& vel){
	return out<<static_cast<const Base&>(vel)<<" "<<vel.vel_.transpose();
}

//==== member functions ====

void Velocity::read(Token& token){
	//property velocity ${group} ${vx} ${vy} ${vz}
	group_.label()=token.next();
	vel_[0]=std::atof(token.next().c_str());
	vel_[1]=std::atof(token.next().c_str());
	vel_[2]=std::atof(token.next().c_str());
}

void Velocity::set(Structure& struc){
	for(int i=0; i<group_.size(); ++i){
		struc.vel(group_.atom(i))=vel_;
	}
}

}