#pragma once
#ifndef MOLKIT2_HPP
#define MOLKIT2_HPP

// c++ libraries
#include <ostream>
#include <vector>
#include <string>
// structure
#include "struc/structure.hpp"
// string
#include "str/token.hpp"
// math
#include "math/graph.hpp"

namespace molkit{

//***************************************************************
// Type
//***************************************************************

class Type{
private:
	int depth_;
	std::string name_;
	std::string lsmiles_;
public:
	//==== constructors/destructors ====
	Type(){}
	Type(const std::string& name):name_(name),depth_(-1){}
	Type(const std::string& name, const std::string& lsmiles):name_(name),lsmiles_(lsmiles){}
	~Type(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Type& type);
	
	//==== member access ====
	int& depth(){return depth_;}
	const int& depth()const{return depth_;}
	std::string& name(){return name_;}
	const std::string& name()const{return name_;}
	std::string& lsmiles(){return lsmiles_;}
	const std::string& lsmiles()const{return lsmiles_;}
	
	//==== static functions ====
	static Type& read(Token& token, Type& type);
};
bool operator==(const Type& type1, const Type& type2);

//***************************************************************
// Label
//***************************************************************

class Label{
private:
	int index_;
	std::vector<Type> types_;
public:
	//==== constructors/destructors ====
	Label():index_(-1){}
	~Label(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Label& label);
	
	//==== member access ====
	int& index(){return index_;}
	const int& index()const{return index_;}
	//access - vector
	std::vector<Type>& types(){return types_;}
	const std::vector<Type>& types()const{return types_;}
	//access - element
	Type& type(int i){return types_[i];}
	const Type& type(int i)const{return types_[i];}
	
	//==== member functions ====
	void clear();
	void resize(int size);
};

//***************************************************************
// Coeff
//***************************************************************

class Coeff{
private:
	int index_;
	std::vector<double> params_;
public:
	//==== constructors/destructors ====
	Coeff():index_(-1){}
	~Coeff(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Coeff& coeff);
	
	//==== member access ====
	int& index(){return index_;}
	const int& index()const{return index_;}
	//access - vector
	std::vector<double>& params(){return params_;}
	const std::vector<double>& params()const{return params_;}
	//access - element
	double& param(int i){return params_[i];}
	const double& param(int i)const{return params_[i];}
	
	//==== member functions ====
	void clear();
	void resize(int size);
};

//***************************************************************
// Molecule Functions
//***************************************************************

Structure& remove_H(const Structure& struc1, Structure& struc2);
Graph& make_graph(const Structure& struc, Graph& graph);
std::vector<int>& make_lsmiles(const Structure& struc, Graph& graph, int atom, int depth, std::vector<int>& lsmiles);
	
};

#endif