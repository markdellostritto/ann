//c++
#include <iostream>
//torch
#include "torch/set_property_type.hpp"

namespace property{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Type& type){
	return out<<static_cast<const Base&>(type)<<" "<<type.type_;
}

//==== member functions ====
void Type::read(Token& token){
	//property type ${group} ${type}
	group_.label()=token.next();
	type_=std::atoi(token.next().c_str());
}

void Type::set(Structure& struc){
	for(int i=0; i<group_.size(); ++i){
		struc.type(group_.atom(i))=type_;
	}
}

}