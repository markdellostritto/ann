// c libraries
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
// c++ libraries
#include <iostream>
// ann - string
#include "string_ann.h"
// ann - atom
#include "atom_ann.h"

void AtomANN::clear(){
	if(ATOM_PRINT_FUNC>0) std::cout<<"AtomANN::clear():\n";
	name_=std::string("NULL");
	id_=string::hash(name_);
	mass_=0;
	energy_=0;
}

AtomANN& AtomANN::read(const char* str, AtomANN& atom){
	if(ATOM_PRINT_FUNC>0) std::cout<<"AtomANN::read(const char*,AtomANN&):\n";
	char* name=new char[10];
	std::sscanf(str,"%s %lf %lf %lf",name,&atom.mass(),&atom.energy(),&atom.charge());
	atom.name()=name;
	atom.id()=string::hash(atom.name());
	delete[] name;
	return atom;
}

std::ostream& operator<<(std::ostream& out, const AtomANN& atom){
	return out<<atom.name_<<" "<<atom.id_<<" "<<atom.mass_<<" "<<atom.energy_<<" "<<atom.charge_;
}

bool operator==(const AtomANN& atom1, const AtomANN& atom2){
	return (
		atom1.id()==atom2.id() &&
		std::fabs(atom1.mass()-atom2.mass())<1e-6 &&
		std::fabs(atom1.energy()-atom2.energy())<1e-6 &&
		std::fabs(atom1.charge()-atom2.charge())<1e-6
	);
}

bool operator!=(const AtomANN& atom1, const AtomANN& atom2){
	return atom1.id()!=atom2.id();
}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const AtomANN& obj){
	if(ATOM_PRINT_FUNC>0) std::cout<<"nbytes(const AtomANN&):\n";
	int size=0;
	size+=sizeof(int);//id
	size+=sizeof(double);//mass
	size+=sizeof(double);//energy
	size+=sizeof(double);//charge
	size+=nbytes(obj.name());//name
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const AtomANN& obj, char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"pack(const AtomANN&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.id(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.mass(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.energy(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.charge(),sizeof(double)); pos+=sizeof(double);
	pack(obj.name(),arr+pos); pos+=nbytes(obj.name());
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(AtomANN& obj, const char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"unpack(AtomANN&,const char*):\n";
	int pos=0;
	std::memcpy(&obj.id(),arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&obj.mass(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.energy(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.charge(),arr+pos,sizeof(double)); pos+=sizeof(double);
	unpack(obj.name(),arr+pos); pos+=nbytes(obj.name());
	obj.id()=string::hash(obj.name());
	return pos;
}
	
}