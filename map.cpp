// c libraries
#include <cstring>
// c++ libraries
#include <string>
#include <ostream>
// ann - map
#include "map.hpp"

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const Map<std::string,unsigned int>& obj){
		unsigned int size=0;
		size+=sizeof(unsigned int);//size
		for(unsigned int i=0; i<obj.size(); ++i) size+=nbytes(obj.key(i));//key
		size+=obj.size()*sizeof(unsigned int);//value
		return size;
	}
	template <> unsigned int nbytes(const Map<unsigned int,unsigned int>& obj){
		unsigned int size=0;
		size+=sizeof(unsigned int);//size
		size+=obj.size()*sizeof(unsigned int);//key
		size+=obj.size()*sizeof(unsigned int);//value
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const Map<std::string,unsigned int>& obj, char* arr){
		unsigned int pos=0;
		//size
		const unsigned int size=obj.size();
		std::memcpy(arr+pos,&size,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		//key
		for(unsigned int i=0; i<size; ++i){
			pack(obj.key(i),arr+pos); pos+=nbytes(obj.key(i));
		}
		//val
		for(unsigned int i=0; i<size; ++i){
			std::memcpy(arr+pos,&obj.val(i),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//return bytes written
		return pos;
	}
	template <> unsigned int pack(const Map<unsigned int,unsigned int>& obj, char* arr){
		unsigned int pos=0;
		//size
		const unsigned int size=obj.size();
		std::memcpy(arr+pos,&size,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		//key
		for(unsigned int i=0; i<size; ++i){
			std::memcpy(arr+pos,&obj.key(i),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//val
		for(unsigned int i=0; i<size; ++i){
			std::memcpy(arr+pos,&obj.val(i),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//return bytes written
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(Map<std::string,unsigned int>& obj, const char* arr){
		unsigned int pos=0;
		obj.clear();
		//size
		unsigned int size=0;
		std::memcpy(&size,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		std::vector<std::string> key(size);
		std::vector<unsigned int> val(size);
		//key
		for(unsigned int i=0; i<size; ++i){
			unpack(key[i],arr+pos); pos+=nbytes(key[i]);
		}
		//val
		for(unsigned int i=0; i<size; ++i){
			std::memcpy(&val[i],arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//assign
		for(unsigned int i=0; i<size; ++i){
			obj.add(key[i],val[i]);
		}
		//return bytes read
		return pos;
	}
	template <> unsigned int unpack(Map<unsigned int,unsigned int>& obj, const char* arr){
		unsigned int pos=0;
		obj.clear();
		//size
		unsigned int size=0;
		std::memcpy(&size,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		std::vector<unsigned int> key(size);
		std::vector<unsigned int> val(size);
		//key
		for(unsigned int i=0; i<size; ++i){
			std::memcpy(&key[i],arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//val
		for(unsigned int i=0; i<size; ++i){
			std::memcpy(&val[i],arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//assign
		for(unsigned int i=0; i<size; ++i){
			obj.add(key[i],val[i]);
		}
		//return bytes read
		return pos;
	}
	
}