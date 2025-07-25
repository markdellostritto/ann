//c++
#include <algorithm>
//struc
#include "struc/pair.hpp"
// analysis
#include "analysis/type.hpp"

//==== operators ====

std::ostream& operator<<(std::ostream& out, Type& type){
	out<<type.name()<<"("<<type.an()<<") ";
	for(int i=0; i<type.bondn().size(); ++i){
		out<<type.bondn()[i]<<"("<<type.bonda()[i]<<") ";
	}
	return out;
}

//==== static functions ====

static bool val(std::pair<std::string,int>& p1,std::pair<std::string,int>& p2){
	return p1.second<p2.second;
}

Type& Type::read(Token& token, Type& type){	
	atom_.first=token.next();
	atom_.second=ptable::an(name_.c_str());
	bonds_.clear();
	while(!token.end()){
		std::string name=token.next();
		bonds_.push_back(std::pair<std::string,int>(name,ptable::an(name.c_str())));
	}
	std::sort(bonds_.begin(),bonds_.end(),Type:val);
}

void Type::make(Structure& struc, Type& type){
	//compute the bonds 
}