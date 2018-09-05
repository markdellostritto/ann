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

}