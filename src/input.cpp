#include "input.hpp"

//==== member functions ====

//parsing

/**
* parse an array of strings and store them in a vector of strings
* @param argc - the number of strings
* @param argv - the array of strings
* @return the vector of strings storing the array of strings
*/
const std::vector<std::string>& Input::parse(int argc, char* argv[]){
	strlist_.resize(argc);
	for(int i=0; i<argc; ++i){
		strlist_[i]=argv[i];
	}
}

//checking for flags

/**
* search the internal list of arguments for a given flag
* @param flag - the flag being searched for
* @return whether the flag is included in the list of arguments
*/
bool Input::find(const char* flag)const{
	for(int i=0; i<strlist_.size(); ++i){
		if(strlist_[i]==flag) return true;
	}
	return false;
}

//extracting parameter values

/**
* get the argument following a flag
* @param flag - the flag being searched for
* @return the string argument following the flag
*/
std::string Input::get(const char* flag)const{
	std::string str;
	for(int i=0; i<strlist_.size(); ++i){
		if(strlist_[i]==flag && i+1<strlist_.size()) str=strlist_[i+1];
	}
	return str;
}
