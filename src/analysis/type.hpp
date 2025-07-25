#pragma once
#ifndef TYPE_HPP
#define TYPE_HPP

// c++ libraries
#include <iosfwd>
#include <string>
#include <vector>
//str
#include "str/token.hpp"
// struc
#include "struc/structure_fwd.hpp"

class Type{
private:
	std::pair<std::string,int> atom_;
	std::vector<std::pair<std::string,int> > bonds_;
public:
	//==== constructors/destructors ====
	Type(){}
	~Type(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, Type& type);
	
	//==== member access ====
	std::pair<std::string,int>& atom(){return atom_;}
	const std::pair<std::string,int>& atom()const{return atom_;}
	std::vector<std::pair<std::string,int> >& bonds(){return bonds_;}
	const std::vector<std::pair<std::string,int> >& bonds()const{return bonds_;}
	
	//==== static functions ====
	static Type& read(Token& token, Type& type);
	static void make(Structure& struc, Type& type);
	static bool val(std::pair<std::string,int>& p1,std::pair<std::string,int>& p2);
};

#endif