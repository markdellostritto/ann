// c++
#include <iostream>
#include <stdexcept>
// str
#include "str/token.hpp"

//==== member functions ====

Token& Token::read(const std::string& str, const std::string& delim){
	str_=str;
	delim_=delim;
	pos_=str_.find_first_not_of(delim_);
	token_.clear();
	return *this;
}

Token& Token::read(const char* str, const char* delim){
	const std::string str1(str);
	const std::string str2(delim);
	read(str1,str2);
	return *this;
}

std::string& Token::next(){
	if(end()) throw std::runtime_error("Token::next(): no tokens left");
	
	const int end_=str_.find_first_of(delim_,pos_);
	if(end_==std::string::npos){
		token_=str_.substr(pos_);
		pos_=end_;
	} else {
		token_=str_.substr(pos_,end_-pos_);
		pos_=str_.find_first_not_of(delim_,end_+1);
	}
	
	return token_;
}

std::string& Token::next(int n){
	for(int i=0; i<n; ++i){
		if(end()) throw std::runtime_error("Token::next(): no tokens left");
		const int end_=str_.find_first_of(delim_,pos_);
		if(end_==std::string::npos){
			token_=str_.substr(pos_);
			pos_=end_;
		} else {
			token_=str_.substr(pos_,end_-pos_);
			pos_=str_.find_first_not_of(delim_,end_+1);
		}
	}
	
	return token_;
}

bool Token::end()const{
    return pos_ == std::string::npos;
}
