#include "map.hpp"

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const Map<std::string,unsigned int>& obj){
		unsigned int size=0;
		size+=sizeof(unsigned int);//size
		size+=obj.size()*sizeof(unsigned int);//value
		for(unsigned int i=0; i<obj.size(); ++i) size+=nbytes(obj.key(i));//key
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const Map<std::string,unsigned int>& obj, char* arr){
		unsigned int pos=0;
		unsigned int size=obj.size();
		std::memcpy(arr+pos,&size,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		for(unsigned int i=0; i<size; ++i){
			pack(obj.key(i),arr+pos); pos+=nbytes(obj.key(i));
		}
		for(unsigned int i=0; i<size; ++i){
			std::memcpy(arr+pos,&obj.val(i),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(Map<std::string,unsigned int>& obj, const char* arr){
		unsigned int pos=0;
		obj.clear();
		unsigned int size=0;
		std::memcpy(&size,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		std::vector<std::string> key(size);
		std::vector<unsigned int> val(size);
		for(unsigned int i=0; i<size; ++i){
			unpack(key[i],arr+pos); pos+=nbytes(key[i]);
		}
		for(unsigned int i=0; i<size; ++i){
			std::memcpy(&val[i],arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		for(unsigned int i=0; i<size; ++i){
			obj.add(key[i],val[i]);
		}
	}
	
}