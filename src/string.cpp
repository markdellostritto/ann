// c libraries
#include <cstring>
#include <cctype>
#include <cstdlib>
// c++ libraries
#include <ostream>
#include <string>
// ann - string
#include "string.hpp"

namespace string{

//******************************************************
//Hash
//******************************************************

int hash(const char* str){
	const int p=31;
	const int m=2147483647;//mersenne prime 31
	int val=0;
	int pow=1;
	while(*str){
		val=(val+(*str-'a'+1)*pow)%m;
		pow=(pow*p)%m;
		++str;
	}
	return val;
}

int hash(const std::string& name){
	return hash(name.c_str());
}

//******************************************************
//Case
//******************************************************

char* to_upper(char* str){
	char* s=str;
	while(*s){
		*s=std::toupper(*s);
		++s;
	}
	return str;
}

char* to_lower(char* str){
	char* s=str;
	while(*s){
		*s=std::tolower(*s);
		++s;
	}
	return str;
}

std::string& to_upper(std::string& str){
	for(int i=str.size()-1; i>=0; --i) str[i]=std::toupper(str[i]);
	return str;
}

std::string& to_lower(std::string& str){
	for(int i=str.size()-1; i>=0; --i) str[i]=std::tolower(str[i]);
	return str;
}

//******************************************************
//Trimming
//******************************************************
	
char* trim(char* str){
	char *sb=str,*se=str+std::strlen(str)-1;
	while(isspace(*sb)) ++sb;
	while(isspace(*se)) --se;
	if(sb<se){
		std::memmove(str,sb,(se-sb+1)*sizeof(char));
		*(str+(se-sb+1))='\0';
	} else *str='\0';
	return str;
}

char* trim_left(char* str){
	char* s=str;
	while(std::isspace(*s)) ++s;
	std::memmove(str,s,std::strlen(s)*sizeof(char));
	return str;
}

char* trim_right(char* str){
	char* s=str+std::strlen(str)-1;
	while(std::isspace(*s)) --s;
	std::memmove(str,str,(s-str+1)*sizeof(char));
	*++s='\0';
	return str;
}

char* trim_all(char* str){
	char* temp=new char[std::strlen(str)];
	int count=0;
	for(int i=0; i<std::strlen(str); ++i){
		if(!std::isspace(str[i])) temp[count++]=str[i];
	}
	temp[count]='\0';
	std::strcpy(str,temp);
	delete[] temp;
	return str;
}

char* trim_left(char* str, const char* delim){
	char* s=std::strpbrk(str,delim);
	if(s!=NULL) std::memmove(str,s+1,(std::strlen(s))*sizeof(char));
	return str;
}

char* trim_right(char* str, const char* delim){
	char* s=std::strpbrk(str,delim);
	if(s!=NULL) *s='\0';
	return str;
}

char* trim(char* str, const char* delim){
	trim_left(str,delim);
	trim_right(str,delim);
	return str;
}

//******************************************************
//Modifying
//******************************************************

char* replace(char* str, char c1, char c2){
	while(*str){
		if(*str==c1) *str=c2;
		++str;
	}
	return str;
}

//******************************************************
//Copying
//******************************************************

char* copy_left(char* str1, const char* str2, const char* delim){
	std::strcpy(str1,str2);
	trim_right(str1,delim);
	return str1;
}

char* copy_right(char* str1, const char* str2, const char* delim){
	std::strcpy(str1,str2);
	trim_left(str1,delim);
	return str1;
}

char* copy(char* dest, char* src){
	while(*src) *(dest++)=*(src++);
	return dest;
}

char* copy(char* dest, char* src, int n){
	for(int i=0; i<n; ++i){
		dest[i]=src[i];
		if(src[i]=='\0') break;
	}
	dest[n]='\0';
	return dest;
}

//******************************************************
//String Info
//******************************************************

bool empty(const char* str){
	while(*str) if(!std::isspace(*str++)) return false;
	return true;
}

bool boolean(const char* str){
	if(
		strcmp("T", str)==0 || 
		strcmp("t", str)==0 || 
		strcmp("TRUE", str)==0 || 
		strcmp("true", str)==0 || 
		strcmp("True", str)==0 ||
		strcmp("1", str)==0
	) return true;
	else return false;
}

int substrN(const char* str, const char* delim){
	int n=0;
	while(std::strpbrk(str,delim)){
		const char* stemp=std::strpbrk(str,delim);
		if(stemp-str>0) ++n;
		str=std::strpbrk(str,delim)+1;
	}
	if(std::strlen(str)>0) ++n;
	return n;
}

//******************************************************
//Splitting string
//******************************************************

int split(const char* str, const char* delim, std::vector<std::string>& strlist){
	int n=substrN(str,delim);
	strlist.resize(n);
	n=0;
	while(std::strpbrk(str,delim)){
		const char* stemp=std::strpbrk(str,delim);
		if(stemp-str>0) strlist[n++]=std::string(str,stemp-str);
		str=std::strpbrk(str,delim)+1;
	}
	if(std::strlen(str)>0) strlist[n++]=std::string(str,std::strlen(str));
	return strlist.size();
}


}
