// c++
#include <iostream>
// chem
#include "chem/alias.hpp"

std::ostream& operator<<(std::ostream& out, const Alias& alias){
	out<<alias.alias()<<" ";
	for(int i=0; i<alias.labels().size(); ++i){
		out<<alias.labels()[i]<<" ";
	}
	return out;
}

Alias& Alias::read(Token& token, Alias& alias){
	alias.clear();
	alias.alias()=token.next();
	while(!token.end()){
		alias.labels().push_back(token.next());
	}
	return alias;
}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const Alias& obj){
	int size=0;
	size+=sizeof(int);//size
	size+=nbytes(obj.alias());
	for(int i=0; i<obj.labels().size(); ++i){
		size+=nbytes(obj.labels()[i]);
	}
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const Alias& obj, char* arr){
	int pos=0;
	const int size=obj.labels().size();
	std::memcpy(arr+pos,&size,sizeof(int)); pos+=sizeof(int);
	pos+=pack(obj.alias(),arr+pos);
	for(int i=0; i<obj.labels().size(); ++i){
		pos+=pack(obj.labels()[i],arr+pos);
	}
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(Alias& obj, const char* arr){
	int pos=0;
	int size=0;
	std::memcpy(&size,arr+pos,sizeof(int)); pos+=sizeof(int);
	if(size>0){
		obj.labels().resize(size);
		pos+=unpack(obj.alias(),arr+pos);
		for(int i=0; i<obj.labels().size(); ++i){
			pos+=unpack(obj.labels()[i],arr+pos);
		}	
	}
	return pos;
}
	
}
