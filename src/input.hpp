#pragma once
#ifndef INPUT_HPP
#define INPUT_HPP

//c++ libraries
#include <vector>
#include <string>

class Input{
private:
	std::vector<std::string> strlist_;
public:
	//==== constructors/destructors ====
	Input(){}
	Input(int argc, char* argv[]){parse(argc,argv);}
	~Input(){}
	
	//==== access ====
	const std::vector<std::string>& strlist()const{return strlist_;}
	
	//==== member functions ====
	//parsing
	const std::vector<std::string>& parse(int argc, char* argv[]);
	//checking for flags
	bool find(const char* flag)const;
	bool find(const std::string& flag)const{return find(flag.c_str());}
	//extracting parameter values
	std::string get(const char* flag)const;
	std::string get(const std::string& flag)const{return get(flag.c_str());}
};

#endif