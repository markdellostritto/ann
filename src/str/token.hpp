#ifndef TOKEN_HPP
#define TOKEN_HPP

// c++
#include <string>

class Token{
private:
	int pos_;
	std::string str_;
	std::string delim_;
	std::string token_;
public:
	//==== constructors/destructors ====
	Token():pos_(0){}
	Token(const std::string& str, const std::string& delim){read(str,delim);}
	Token(const char* str, const char* delim){read(str,delim);}
	~Token(){}
	
	//==== access ====
	const std::string& str()const{return str_;}
	const std::string& delim()const{return delim_;}
	const std::string& token()const{return token_;}
	
	//==== member functions ====
	Token& read(const std::string& str, const std::string& delim);
	Token& read(const char* str, const char* delim);
	std::string& next();
	std::string& next(int n);
	bool end()const;
};

#endif