#pragma once
#ifndef PRINT_HPP
#define PRINT_HPP

namespace print{
	
	static const char char_buf='=';
	static const char char_title='*';
	static const unsigned int len_buf=61;
	
	char* buf(char* str);
	char* buf(char* str, char c);
	
	char* title(const char* t, char* str);
	char* title(const char* t, char* str, char c);
	
}

#endif