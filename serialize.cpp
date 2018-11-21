#include "serialize.hpp"

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const bool& obj){return sizeof(obj);};
template <> unsigned int nbytes(const char& obj){return sizeof(obj);};
template <> unsigned int nbytes(const unsigned char& obj){return sizeof(obj);};
template <> unsigned int nbytes(const short& obj){return sizeof(obj);};
template <> unsigned int nbytes(const unsigned short& obj){return sizeof(obj);};
template <> unsigned int nbytes(const int& obj){return sizeof(obj);};
template <> unsigned int nbytes(const unsigned int& obj){return sizeof(obj);};
template <> unsigned int nbytes(const float& obj){return sizeof(obj);};
template <> unsigned int nbytes(const double& obj){return sizeof(obj);};
template <> unsigned int nbytes(const std::string& str){return str.length()+sizeof(unsigned int);}//size + length
	
//**********************************************
// packing
//**********************************************

template <> void pack(const bool& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj));};
template <> void pack(const char& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj));};
template <> void pack(const unsigned char& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj));};
template <> void pack(const short& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj));};
template <> void pack(const unsigned short& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj));};
template <> void pack(const int& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj));};
template <> void pack(const unsigned int& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj));};
template <> void pack(const float& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj));};
template <> void pack(const double& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj));};
template <> void pack(const std::string& str, char* arr){
	unsigned int pos=0;
	unsigned int length=str.length();
	std::memcpy(arr+pos,&length,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(arr+pos,str.c_str(),sizeof(char)*str.length());
}

//**********************************************
// unpacking
//**********************************************

template <> void unpack(bool& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj));};
template <> void unpack(char& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj));};
template <> void unpack(unsigned char& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj));};
template <> void unpack(short& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj));};
template <> void unpack(unsigned short& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj));};
template <> void unpack(int& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj));};
template <> void unpack(unsigned int& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj));};
template <> void unpack(float& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj));};
template <> void unpack(double& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj));};
template <> void unpack(std::string& str, const char* arr){
	unsigned int pos=0;
	unsigned int length=0;
	std::memcpy(&length,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	str.resize(length);
	for(unsigned int i=0; i<length; ++i){
		std::memcpy(&(str[i]),arr+pos,sizeof(char)); pos+=sizeof(char);
	}
}

}