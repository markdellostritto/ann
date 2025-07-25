#pragma once
#ifndef PARSE_HPP
#define PARSE_HPP

//c++ libraries
#include <string>
#include <vector>

namespace input{
	
class Arg{
private:
	std::string key_;
	std::vector<std::string> vals_;
public:
	//==== constructors/destsructors ====
	Arg(){}
	~Arg(){}
	
	//==== access ====
	std::string& key(){return key_;}
	const std::string& key()const{return key_;}
	int nvals()const{return vals_.size();}
	const std::vector<std::string>& vals()const{return vals_;}
	std::string& val(int i){return vals_.at(i);}
	const std::string& val(int i)const{return vals_.at(i);}
	std::vector<std::string>& vals(){return vals_;}
	
	//==== member functions ====
	void clear();
};
	
void parse(int argc, char* argv[], std::vector<Arg>& args);

}

#endif