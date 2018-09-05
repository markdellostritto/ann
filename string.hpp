#ifndef STRING_OPS_H
#define STRING_OPS_H

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <iostream>
#include <stdexcept>

namespace string{
	
	//******************************************************
	//Standard Strings
	//******************************************************
	
	static const unsigned int M=250;
	static const char* WS=" \r\t\n";
	static const char* DIGITS="1234567890";
	const char* const COMMENT="#";
	
	//******************************************************
	//Case
	//******************************************************
	char* to_upper(char* str);
	char* to_lower(char* str);
	
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
	//Parser class
	//******************************************************
	/*class Parser{
	private:
		static const int M=500;
		char str_[M];//stores the string we are parsing
		char substr_[M];//stores the substrings of the string we are parsing
		char delim_[M];//stores the delimeters
		int strLen_;//length of str
		int delimLen_;//length of delim
		int posn_;//the current position in the string
	public:
		Parser():posn_(0),strLen_(0),delimLen_(0){};
		~Parser(){};
		
		int parse(const char* str, const char* delim);
		const char* next();
		const char* next(int n);
	};*/
}	

#endif
