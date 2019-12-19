// c libraries
#include <cmath>
// c++ libraries
#include <iostream>
#include <string>
#include <vector>
// ann - math
#include "math_const_ann.h"
// ann - string
#include "string_ann.h"
// ann - atom
#include "atom_ann.h"

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const AtomANN& obj){
	if(ATOM_PRINT_FUNC>0) std::cout<<"nbytes(const AtomANN&):\n";
	unsigned int size=0;
	size+=sizeof(double);//mass
	size+=sizeof(double);//energy
	size+=sizeof(double);//charge
	size+=sizeof(unsigned int);//id
	size+=nbytes(obj.name());//name
	return size;
}

//**********************************************
// packing
//**********************************************

template <> unsigned int pack(const AtomANN& obj, char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"pack(const AtomANN&,char*):\n";
	unsigned int pos=0;
	std::memcpy(arr+pos,&obj.mass(),sizeof(double)); pos+=sizeof(double);//mass
	std::memcpy(arr+pos,&obj.energy(),sizeof(double)); pos+=sizeof(double);//energy
	std::memcpy(arr+pos,&obj.charge(),sizeof(double)); pos+=sizeof(double);//charge
	std::memcpy(arr+pos,&obj.id(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//id
	pack(obj.name(),arr+pos); pos+=nbytes(obj.name());//name
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> unsigned int unpack(AtomANN& obj, const char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"unpack(AtomANN&,const char*):\n";
	unsigned int pos=0;
	std::memcpy(&obj.mass(),arr+pos,sizeof(double)); pos+=sizeof(double);//mass
	std::memcpy(&obj.energy(),arr+pos,sizeof(double)); pos+=sizeof(double);//energy
	std::memcpy(&obj.charge(),arr+pos,sizeof(double)); pos+=sizeof(double);//charge
	std::memcpy(&obj.id(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//id
	unpack(obj.name(),arr+pos); pos+=nbytes(obj.name());//name
	return pos;
}
	
}
	
void AtomANN::clear(){
	if(ATOM_PRINT_FUNC>0) std::cout<<"AtomANN::clear():\n";
	name_=std::string("NULL");
	id_=string::hash(name_);
	mass_=0;
	energy_=0;
	charge_=0;
}

AtomANN& AtomANN::read(const char* str, AtomANN& atom){
	if(ATOM_PRINT_FUNC>0) std::cout<<"AtomANN::read(const char*,AtomANN&):\n";
	std::vector<std::string> strlist;
	string::split(str,string::WS,strlist);
	if(strlist.size()!=3 && strlist.size()!=4) throw std::invalid_argument("Invalid atom string.");
	atom.name()=strlist[0];
	atom.id()=string::hash(atom.name());
	atom.mass()=std::atof(strlist[1].c_str());
	atom.energy()=std::atof(strlist[2].c_str());
	if(strlist.size()==4) atom.charge()=std::atof(strlist[3].c_str());
	return atom;
}

std::ostream& operator<<(std::ostream& out, const AtomANN& atom){
	return out<<atom.name_<<" "<<atom.mass_<<" "<<atom.energy_<<" "<<atom.charge_;
}

bool operator==(const AtomANN& atom1, const AtomANN& atom2){
	return atom1.id()==atom2.id();
}

bool operator<(const AtomANN& atom1, const AtomANN& atom2){
	return atom1.id()<atom2.id();
}

bool operator>(const AtomANN& atom1, const AtomANN& atom2){
	return atom1.id()>atom2.id();
}

bool operator<=(const AtomANN& atom1, const AtomANN& atom2){
	return atom1.id()<=atom2.id();
}

bool operator>=(const AtomANN& atom1, const AtomANN& atom2){
	return atom1.id()>=atom2.id();
}
