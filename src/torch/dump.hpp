#pragma once
#ifndef DUMP_HPP
#define DUMP_HPP

// string
#include "str/token.hpp"

class Dump{
private:
	int nprint_;//nprint
	int nwrite_;//nwrite
	std::string file_;
public:
	//==== constructors/destructors ====
	Dump():nprint_(-1),nwrite_(-1){}
	~Dump(){}
	
	//=== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Dump& dump);
	
	//==== access ====
	int& nprint(){return nprint_;}
	const int& nprint()const{return nprint_;}
	int& nwrite(){return nwrite_;}
	const int& nwrite()const{return nwrite_;}
	std::string& file(){return file_;}
	const std::string& file()const{return file_;}
	
	//==== member functions ====
	void read(Token& token);
	
	//==== static functions ====
	static void write(Structure& struc, FILE* writer);
};

#endif