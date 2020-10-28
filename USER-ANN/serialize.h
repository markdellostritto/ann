#pragma once
#ifndef SERIALIZE_HPP
#define SERIALIZE_HPP

// c++ libraries
#include <stdexcept>
#include <string>
#include <vector>

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <class T>
int nbytes(const T& obj){
	throw std::runtime_error("No byte measure defined.");
}

template <> int nbytes(const bool& obj);
template <> int nbytes(const char& obj);
template <> int nbytes(const unsigned char& obj);
template <> int nbytes(const short& obj);
template <> int nbytes(const unsigned short& obj);
template <> int nbytes(const int& obj);
template <> int nbytes(const unsigned int& obj);
template <> int nbytes(const float& obj);
template <> int nbytes(const double& obj);
template <> int nbytes(const std::string& str);
template <> int nbytes(const std::vector<std::string>& strlist);
template <> int nbytes(const std::vector<int>& vec);
template <> int nbytes(const std::vector<unsigned int>& vec);
template <> int nbytes(const std::vector<double>& vec);

//**********************************************
// packing
//**********************************************

template <class T>
int pack(const T& obj, char* arr){
	throw std::runtime_error("No serialization method defined.");
}

template <> int pack(const bool& obj, char* arr);
template <> int pack(const char& obj, char* arr);
template <> int pack(const unsigned char& obj, char* arr);
template <> int pack(const short& obj, char* arr);
template <> int pack(const unsigned short& obj, char* arr);
template <> int pack(const int& obj, char* arr);
template <> int pack(const unsigned int& obj, char* arr);
template <> int pack(const float& obj, char* arr);
template <> int pack(const double& obj, char* arr);
template <> int pack(const std::string& str, char* arr);
template <> int pack(const std::vector<std::string>& strlist, char* arr);
template <> int pack(const std::vector<int>& str, char* arr);
template <> int pack(const std::vector<unsigned int>& str, char* arr);
template <> int pack(const std::vector<double>& str, char* arr);

//**********************************************
// unpacking
//**********************************************

template <class T>
int unpack(T& obj, const char* arr){
	throw std::runtime_error("No serialization method defined.");
}

template <> int unpack(bool& obj, const char* arr);
template <> int unpack(char& obj, const char* arr);
template <> int unpack(unsigned char& obj, const char* arr);
template <> int unpack(short& obj, const char* arr);
template <> int unpack(unsigned short& obj, const char* arr);
template <> int unpack(int& obj, const char* arr);
template <> int unpack(unsigned int& obj, const char* arr);
template <> int unpack(float& obj, const char* arr);
template <> int unpack(double& obj, const char* arr);
template <> int unpack(std::string& str, const char* arr);
template <> int unpack(std::vector<std::string>& strlist, const char* arr);
template <> int unpack(std::vector<int>& str, const char* arr);
template <> int unpack(std::vector<unsigned int>& str, const char* arr);
template <> int unpack(std::vector<double>& str, const char* arr);

}

#endif