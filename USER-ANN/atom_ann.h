#pragma once
#ifndef ATOM_ANN_HPP
#define ATOM_ANN_HPP

//c++ libraries
#include <iosfwd>
#include <string>
// ann - serialize
#include "serialize.h"

#ifndef ATOM_PRINT_FUNC
#define ATOM_PRINT_FUNC 0
#endif

//************************************************************
// ATOM
//************************************************************

class AtomANN{
private:
	double mass_,energy_,charge_;
	unsigned int id_;
	std::string name_;
public:
	//constructors/destructors
	AtomANN(){clear();}
	~AtomANN(){}
	//access
	double& mass(){return mass_;}
	const double& mass()const{return mass_;}
	double& energy(){return energy_;}
	const double& energy()const{return energy_;}
	double& charge(){return charge_;}
	const double& charge()const{return charge_;}
	unsigned int& id(){return id_;}
	const unsigned int& id()const{return id_;}
	std::string& name(){return name_;}
	const std::string& name()const{return name_;}
	//member functions
	void clear();
	//static functions
	static AtomANN& read(const char* str, AtomANN& atom);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const AtomANN& atom);
};
bool operator==(const AtomANN& atom1, const AtomANN& atom2);
inline bool operator!=(const AtomANN& atom1, const AtomANN& atom2){return !(atom1==atom2);}
bool operator<(const AtomANN& atom1, const AtomANN& atom2);
bool operator>(const AtomANN& atom1, const AtomANN& atom2);
bool operator<=(const AtomANN& atom1, const AtomANN& atom2);
bool operator>=(const AtomANN& atom1, const AtomANN& atom2);

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const AtomANN& obj);

//**********************************************
// packing
//**********************************************

template <> unsigned int pack(const AtomANN& obj, char* arr);

//**********************************************
// unpacking
//**********************************************

template <> unsigned int unpack(AtomANN& obj, const char* arr);

}

#endif
