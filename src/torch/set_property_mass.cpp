//c++
#include <iostream>
//torch
#include "torch/set_property_mass.hpp"

namespace property{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Mass& mass){
	return out<<static_cast<const Base&>(mass)<<" "<<mass.mass_;
}

//==== member functions ====
void Mass::read(Token& token){
	//property mass ${group} ${mass}
	group_.label()=token.next();
	mass_=std::atof(token.next().c_str());
}

void Mass::set(Structure& struc){
	for(int i=0; i<group_.size(); ++i){
		struc.mass(group_.atom(i))=mass_;
	}
}

}