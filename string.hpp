#pragma once
#ifndef STRING_HPP
#define STRING_HPP

//c++ libraries
#include <string>
#include <vector>

namespace string{
	
	//******************************************************
	//Standard Strings
	//******************************************************
	
	static const unsigned int M=250;
	static const char* WS=" \r\t\n";
	static const char* DIGITS="1234567890";
	const char* const COMMENT="#";
	
	//******************************************************
	//Hash
	//******************************************************
	
	unsigned int hash(const char* str);
	unsigned int hash(const std::string& str);
	unsigned short hash_s(const char* str);
	unsigned short hash_s(const std::string& str);
	
	//******************************************************
	//Case
	//******************************************************
	
	char* to_upper(char* str);
	char* to_lower(char* str);
	std::string& to_upper(std::string& str);
	std::string& to_lower(std::string& str);
	
	//******************************************************
	//Trimming
	//******************************************************
	
	char* trim(char* str);
	char* trim_left(char* str);
	char* trim_right(char* str);
	char* trim_all(char* str);
	char* trim_left(char* str, const char* delim);
	char* trim_right(char* str, const char* delim);
	char* trim(char* str, const char* delim);
	
	//******************************************************
	//Modifying
	//******************************************************
	
	char* replace(char* str, char c1, char c2);
	
	//******************************************************
	//Copying
	//******************************************************
	
	char* copy_left(char* str1, const char* str2, const char* delim);
	char* copy_right(char* str1, const char* str2, const char* delim);
	
	//******************************************************
	//String Info
	//******************************************************
	
	bool empty(const char* str);
	bool boolean(const char* str);
	unsigned int substrN(const char* str, const char* delim);
	
	//******************************************************
	//Splitting string
	//******************************************************
	
	unsigned int split(const char* str, const char* delim, std::vector<std::string>& strlist);
	
}	

#endif
