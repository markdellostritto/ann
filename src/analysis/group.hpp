#pragma once
#ifndef GROUP_HPP
#define GROUP_HPP

// structure
#include "struc/structure.hpp"
// str
#include "str/string.hpp"
#include "str/token.hpp"

class Group{
public:
	struct Style{
	public:
		//enum
		enum Type{ID,TYPE,NAME,UNKNOWN};
		//constructor
		Style():t_(Type::UNKNOWN){}
		Style(Type t):t_(t){}
		//operators
		operator Type()const{return t_;}
		//member functions
		static Style read(const char* str);
		static const char* name(const Style& style);
	private:
		Type t_;
	};
private:
	int id_;
	Style style_;
	std::string label_;
	std::vector<int> atoms_;
	std::vector<int> types_;
	std::vector<std::string> names_;
	std::vector<std::array<int,3> > limits_;
public:
	//==== constructors/destructors ====
	Group(){clear();}
	~Group(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Group& group);
	
	//==== access ====
	//label
	int& id(){return id_;}
	const int& id()const{return id_;}
	std::string& label(){return label_;}
	const std::string& label()const{return label_;}
	Style& style(){return style_;}
	const Style& style()const{return style_;}
	//atoms
	const std::vector<int>& atoms()const{return atoms_;}
	int& atom(int i){return atoms_[i];}
	const int& atom(int i)const{return atoms_[i];}
	int size()const{return atoms_.size();}
	//types
	const std::vector<int>& types()const{return types_;}
	int& type(int i){return types_[i];}
	const int& type(int i)const{return types_[i];}
	//names
	const std::vector<std::string>& names()const{return names_;}
	std::string& name(int i){return names_[i];}
	const std::string& name(int i)const{return names_[i];}
	
	//==== member functions ====
	//modification
	void clear();
	void resize(int natoms);
	void label(const char* str);
	void init(const std::string& label, const std::vector<int>& atoms);
	//atoms
	bool contains(int atom);
	int find(int atom);
	//reading/writing
	void read(Token& token);
	//building
	void build(const Structure& struc);
};

//==== operators ====

inline bool operator==(const Group& g1, const Group& g2){return g1.id()==g2.id();}
inline bool operator!=(const Group& g1, const Group& g2){return g1.id()!=g2.id();}

#endif