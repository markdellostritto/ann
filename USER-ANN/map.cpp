// c libraries
#include <cstring>
// c++ libraries
#include <string>
#include <ostream>
// ann - map
#include "map.h"

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Map<std::string,int>& obj){
		int size=0;
		size+=sizeof(int);//size
		for(int i=0; i<obj.size(); ++i) size+=nbytes(obj.key(i));//key
		size+=obj.size()*sizeof(int);//value
		return size;
	}
	template <> int nbytes(const Map<int,int>& obj){
		int size=0;
		size+=sizeof(int);//size
		size+=obj.size()*sizeof(int);//key
		size+=obj.size()*sizeof(int);//value
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Map<std::string,int>& obj, char* arr){
		int pos=0;
		//size
		const int size=obj.size();
		std::memcpy(arr+pos,&size,sizeof(int)); pos+=sizeof(int);
		//key
		for(int i=0; i<size; ++i){
			pos+=pack(obj.key(i),arr+pos);
		}
		//val
		for(int i=0; i<size; ++i){
			std::memcpy(arr+pos,&obj.val(i),sizeof(int)); pos+=sizeof(int);
		}
		//return bytes written
		return pos;
	}
	template <> int pack(const Map<int,int>& obj, char* arr){
		int pos=0;
		//size
		const int size=obj.size();
		std::memcpy(arr+pos,&size,sizeof(int)); pos+=sizeof(int);
		//key
		for(int i=0; i<size; ++i){
			std::memcpy(arr+pos,&obj.key(i),sizeof(int)); pos+=sizeof(int);
		}
		//val
		for(int i=0; i<size; ++i){
			std::memcpy(arr+pos,&obj.val(i),sizeof(int)); pos+=sizeof(int);
		}
		//return bytes written
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Map<std::string,int>& obj, const char* arr){
		int pos=0;
		obj.clear();
		//size
		int size=0;
		std::memcpy(&size,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::vector<std::string> key(size);
		std::vector<int> val(size);
		//key
		for(int i=0; i<size; ++i){
			pos+=unpack(key[i],arr+pos);
		}
		//val
		for(int i=0; i<size; ++i){
			std::memcpy(&val[i],arr+pos,sizeof(int)); pos+=sizeof(int);
		}
		//assign
		for(int i=0; i<size; ++i){
			obj.add(key[i],val[i]);
		}
		//return bytes read
		return pos;
	}
	template <> int unpack(Map<int,int>& obj, const char* arr){
		int pos=0;
		obj.clear();
		//size
		int size=0;
		std::memcpy(&size,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::vector<int> key(size);
		std::vector<int> val(size);
		//key
		for(int i=0; i<size; ++i){
			std::memcpy(&key[i],arr+pos,sizeof(int)); pos+=sizeof(int);
		}
		//val
		for(int i=0; i<size; ++i){
			std::memcpy(&val[i],arr+pos,sizeof(int)); pos+=sizeof(int);
		}
		//assign
		for(int i=0; i<size; ++i){
			obj.add(key[i],val[i]);
		}
		//return bytes read
		return pos;
	}
	
}