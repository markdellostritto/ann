#pragma once
#ifndef MOLKIT_HPP
#define MOLKIT_HPP

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
	int an_;
	int lsmiles_hash_;
	double charge_;
	double radius_;
	std::string label_;
	std::string element_;
	std::string lsmiles_name_;
public:
	//==== constructors/destructors ====
	Type():an_(0),charge_(0.0),lsmiles_hash_(0){}
	~Type(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Type& type);
	
	//==== member access ====
	int& an(){return an_;}
	const int& an()const{return an_;}
	int& lsmiles_hash(){return lsmiles_hash_;}
	const int& lsmiles_hash()const{return lsmiles_hash_;}
	double& charge(){return charge_;}
	const double& charge()const{return charge_;}
	double& radius(){return radius_;}
	const double& radius()const{return radius_;}
	std::string& label(){return label_;}
	const std::string& label()const{return label_;}
	std::string& element(){return element_;}
	const std::string& element()const{return element_;}
	std::string& lsmiles_name(){return lsmiles_name_;}
	const std::string& lsmiles_name()const{return lsmiles_name_;}
	
	//==== static functions ====
	static Type& read(Token& token, Type& type);
};


//***************************************************************
// Alias
//***************************************************************

class Alias{
private:
	std::string alias_;
	std::vector<std::string> labels_;
public:
	//==== constructors/destructors ====
	Alias(){}
	~Alias(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Alias& alias);
	
	//==== member access ====
	std::string& alias(){return alias_;}
	const std::string& alias()const{return alias_;}
	std::vector<std::string>& labels(){return labels_;}
	const std::vector<std::string>& labels()const{return labels_;}
	
	//==== member functions ====
	void clear(){alias_.clear(); labels_.clear();}
	
	//==== static functions ====
	static Alias& read(Token& token, Alias& alias);
};

//***************************************************************
// Coeff
//***************************************************************

class Coeff{
private:
	int type_;
	std::vector<double> params_;
public:
	Coeff():type_(0){}
	~Coeff(){}
	
	friend std::ostream& operator<<(std::ostream& out, const Coeff& coeff);
	
	int& type(){return type_;}
	const int& type()const{return type_;}
	std::vector<double>& params(){return params_;}
	const std::vector<double>& params()const{return params_;}
	
	void clear(){type_=0; params_.clear();}
	
	static Coeff& read(Token& token, Coeff& coeff);
};

//***************************************************************
// Link
//***************************************************************

class Link{
private:
	int type_;
	std::vector<std::string> labels_;
public:
	Link():type_(0){}
	~Link(){}
	
	friend std::ostream& operator<<(std::ostream& out, const Link& link);
	
	int& type(){return type_;}
	const int& type()const{return type_;}
	std::vector<std::string>& labels(){return labels_;}
	const std::vector<std::string>& labels()const{return labels_;}
	
	void clear(){type_=0; labels_.clear();}
	
	static Link& read(Token& token, Link& link);
};

//***************************************************************
// Molecule Functions
//***************************************************************

Structure& remove_H(const Structure& struc1, Structure& struc2);
Graph& make_graph(const Structure& struc, Graph& graph);
std::vector<int>& make_lsmiles(const Structure& struc, Graph& graph, int atom, int depth, std::vector<int>& lsmiles);
	
};

#endif