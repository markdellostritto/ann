//c++
#include <iostream>
//torch
#include "torch/set_property_charge.hpp"

namespace property{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Charge& chg){
	return out<<static_cast<const Base&>(chg)<<" "<<chg.charge_;
}

//==== member functions ====
void Charge::read(Token& token){
	//property charge ${group} ${charge}
	group_.label()=token.next();
	charge_=std::atof(token.next().c_str());
}

void Charge::set(Structure& struc){
	for(int i=0; i<group_.size(); ++i){
		struc.charge(group_.atom(i))=charge_;
	}
}

}