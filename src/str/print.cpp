// c libraries
#include <cstring>
// ann - print
#include "str/print.hpp"

namespace print{
	
	char* buf(char* str){
		for(int i=0; i<len_buf-1; ++i) str[i]=char_buf;
		str[len_buf-1]='\0';
		return str;
	}
	
	char* buf(char* str, char c){
		for(int i=0; i<len_buf-1; ++i) str[i]=c;
		str[len_buf-1]='\0';
		return str;
	}
	
	char* title(const char* t, char* str){
		const int strl=std::strlen(t);
		const int len=len_buf-1;
		const int buf_f=(len-(strl+2))/2;
		const int buf_r=buf_f+strl%2;
		int count=0;
		for(int i=0; i<buf_f; ++i) str[count++]=char_title;
		str[count++]=' ';
		for(int i=0; i<strl; ++i) str[count++]=t[i];
		str[count++]=' ';
		for(int i=0; i<buf_r; ++i) str[count++]=char_title;
		str[count++]='\0';
		return str;
	}
	
	char* title(const char* t, char* str, char c){
		const int strl=std::strlen(t);
		const int len=len_buf-1;
		const int buf_f=(len-(strl+2))/2;
		const int buf_r=buf_f+strl%2;
		int count=0;
		for(int i=0; i<buf_f; ++i) str[count++]=c;
		str[count++]=' ';
		for(int i=0; i<strl; ++i) str[count++]=t[i];
		str[count++]=' ';
		for(int i=0; i<buf_r; ++i) str[count++]=c;
		str[count++]='\0';
		return str;
	}
	
}
