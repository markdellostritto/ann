#include "atom.hpp"

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const Atom& obj){
	if(ATOM_PRINT_FUNC>0) std::cout<<"nbytes(const Atom&):\n";
	unsigned int size=0;
	size+=sizeof(unsigned int);//id
	size+=sizeof(double);//mass
	size+=sizeof(double);//energy
	size+=sizeof(double);//charge
	size+=nbytes(obj.name());//name
	return size;
}

//**********************************************
// packing
//**********************************************

template <> void pack(const Atom& obj, char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"pack(const Atom&,char*):\n";
	unsigned int pos=0;
	std::memcpy(arr+pos,&obj.id(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//id
	std::memcpy(arr+pos,&obj.mass(),sizeof(double)); pos+=sizeof(double);//mass
	std::memcpy(arr+pos,&obj.energy(),sizeof(double)); pos+=sizeof(double);//energy
	std::memcpy(arr+pos,&obj.charge(),sizeof(double)); pos+=sizeof(double);//charge
	pack(obj.name(),arr+pos); pos+=nbytes(obj.name());//name
}

//**********************************************
// unpacking
//**********************************************

template <> void unpack(Atom& obj, const char* arr){
	if(ATOM_PRINT_FUNC>0) std::cout<<"unpack(Atom&,const char*):\n";
	unsigned int pos=0;
	std::memcpy(&obj.id(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//id
	std::memcpy(&obj.mass(),arr+pos,sizeof(double)); pos+=sizeof(double);//mass
	std::memcpy(&obj.energy(),arr+pos,sizeof(double)); pos+=sizeof(double);//energy
	std::memcpy(&obj.charge(),arr+pos,sizeof(double)); pos+=sizeof(double);//charge
	unpack(obj.name(),arr+pos); pos+=nbytes(obj.name());//name
}
	
}
	
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

std::ostream& operator<<(std::ostream& out, const Atom& atom){
	return out<<atom.name_<<" "<<atom.mass_<<" "<<atom.energy_<<" "<<atom.charge_;
}

bool operator==(const Atom& atom1, const Atom& atom2){
	return atom1.id()==atom2.id();
}

bool operator<(const Atom& atom1, const Atom& atom2){
	return atom1.id()<atom2.id();
}

bool operator>(const Atom& atom1, const Atom& atom2){
	return atom1.id()>atom2.id();
}

bool operator<=(const Atom& atom1, const Atom& atom2){
	return atom1.id()<=atom2.id();
}

bool operator>=(const Atom& atom1, const Atom& atom2){
	return atom1.id()>=atom2.id();
}
