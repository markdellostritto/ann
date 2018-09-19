#ifndef SERIALIZE_HPP
#define SERIALIZE_HPP

#include <cstring>
#include <stdexcept>

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <class T>
unsigned int nbytes(const T& obj){
	throw std::runtime_error("No byte measure defined.");
}

template <> unsigned int nbytes(const bool& obj);
template <> unsigned int nbytes(const char& obj);
template <> unsigned int nbytes(const unsigned char& obj);
template <> unsigned int nbytes(const short& obj);
template <> unsigned int nbytes(const unsigned short& obj);
template <> unsigned int nbytes(const int& obj);
template <> unsigned int nbytes(const unsigned int& obj);
template <> unsigned int nbytes(const float& obj);
template <> unsigned int nbytes(const double& obj);

//**********************************************
// packing
//**********************************************

template <class T>
void pack(const T& obj, char* arr){
	throw std::runtime_error("No serialization method defined.");
}

template <> void pack(const bool& obj, char* arr);
template <> void pack(const char& obj, char* arr);
template <> void pack(const unsigned char& obj, char* arr);
template <> void pack(const short& obj, char* arr);
template <> void pack(const unsigned short& obj, char* arr);
template <> void pack(const int& obj, char* arr);
template <> void pack(const unsigned int& obj, char* arr);
template <> void pack(const float& obj, char* arr);
template <> void pack(const double& obj, char* arr);

//**********************************************
// unpacking
//**********************************************

template <class T>
void unpack(T& obj, const char* arr){
	throw std::runtime_error("No serialization method defined.");
}

template <> void unpack(bool& obj, const char* arr);
template <> void unpack(char& obj, const char* arr);
template <> void unpack(unsigned char& obj, const char* arr);
template <> void unpack(short& obj, const char* arr);
template <> void unpack(unsigned short& obj, const char* arr);
template <> void unpack(int& obj, const char* arr);
template <> void unpack(unsigned int& obj, const char* arr);
template <> void unpack(float& obj, const char* arr);
template <> void unpack(double& obj, const char* arr);

}

#endif