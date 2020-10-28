// c libraries
#include <cstdlib>
#include <cstring>
#include <cmath>
// c++ libraries
#include <iostream>
// ann - string
#include "string.hpp"
// ann - atom
#include "atom.hpp"
	
void Atom::clear(){
	if(ATOM_PRINT_FUNC>0) std::cout<<"Atom::clear():\n";
	name_=std::string("NULL");
	id_=string::hash(name_);
	mass_=0;
	energy_=0;
	charge_=0;
}

Atom& Atom::read(const char* str, Atom& atom){
	if(ATOM_PRINT_FUNC>0) std::cout<<"Atom::read(const char*,Atom&):\n";
	char* name=new char[10];
	std::sscanf(str,"%s %lf %lf %lf",name,&atom.mass(),&atom.energy(),&atom.charge());
	atom.name()=name;
	atom.id()=string::hash(atom.name());
	delete[] name;
	return atom;
}

std::ostream& operator<<(std::ostream& out, const Atom& atom){
	return out<<atom.name_<<" "<<atom.mass_<<" "<<atom.energy_<<" "<<atom.charge_;
}

void Atom::print(FILE* out, const Atom& atom){
	fprintf(out,"%s %f %f %f\n",atom.name().c_str(),atom.mass(),atom.energy(),atom.charge());
}

bool operator==(const Atom& atom1, const Atom& atom2){
	return (
		atom1.id()==atom2.id() &&
		std::fabs(atom1.mass()-atom2.mass())<1e-6 &&
		std::fabs(atom1.energy()-atom2.energy())<1e-6 &&
		std::fabs(atom1.charge()-atom2.charge())<1e-6
	);
}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const Atom& obj){
	if(ATOM_PRINT_FUNC>0) std::cout<<"nbytes(const Atom&):\n";
	int size=0;
	size+=sizeof(double);//mass
	size+=sizeof(double);//energy
	size+=sizeof(double);//charge
	size+=sizeof(int);//id
	size+=nbytes(obj.name());//name
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const Atom& obj, char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"pack(const Atom&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.mass(),sizeof(double)); pos+=sizeof(double);//mass
	std::memcpy(arr+pos,&obj.energy(),sizeof(double)); pos+=sizeof(double);//energy
	std::memcpy(arr+pos,&obj.charge(),sizeof(double)); pos+=sizeof(double);//charge
	std::memcpy(arr+pos,&obj.id(),sizeof(int)); pos+=sizeof(int);//id
	pos+=pack(obj.name(),arr+pos);//name
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(Atom& obj, const char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"unpack(Atom&,const char*):\n";
	int pos=0;
	std::memcpy(&obj.mass(),arr+pos,sizeof(double)); pos+=sizeof(double);//mass
	std::memcpy(&obj.energy(),arr+pos,sizeof(double)); pos+=sizeof(double);//energy
	std::memcpy(&obj.charge(),arr+pos,sizeof(double)); pos+=sizeof(double);//charge
	std::memcpy(&obj.id(),arr+pos,sizeof(int)); pos+=sizeof(int);//id
	pos+=unpack(obj.name(),arr+pos);//name
	return pos;
}
	
}
