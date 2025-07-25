// c libraries
#include <cstdlib>
#include <cstring>
// c++ libraries
#include <iostream>
// str
#include "str/string.hpp"
// ann - serialize
#include "mem/serialize.hpp"
// type
#include "nnp/type.hpp"
	
//************************************************************
// TYPE
//************************************************************

void Type::clear(){
	if(TYPE_PRINT_FUNC>0) std::cout<<"Type::clear():\n";
	name_=std::string("NULL");
	id_=string::hash(name_);
	mass_.clear();
	energy_.clear();
	charge_.clear();
	chi_.clear();
	eta_.clear();
	js_.clear();
	c6_.clear();
	rvdw_.clear();
	rcov_.clear();
	z_.clear();
	alpha_.clear();
	weight_.clear();
}

Type& Type::read(const char* str, Type& type){
	if(TYPE_PRINT_FUNC>0) std::cout<<"Type::read(const char*,Type&):\n";
	Token token(str,string::WSC);
	while(!token.end()){
		const std::string tag=string::to_upper(token.next());
		if(tag=="NAME"){
			type.name()=token.next();
		} else if(tag=="MASS"){
			type.mass().flag()=true;
			type.mass().val()=std::atof(token.next().c_str());
		} else if(tag=="CHARGE"){
			type.charge().flag()=true;
			type.charge().val()=std::atof(token.next().c_str());
		} else if(tag=="ENERGY"){
			type.energy().flag()=true;
			type.energy().val()=std::atof(token.next().c_str());
		} else if(tag=="CHI"){
			type.chi().flag()=true;
			type.chi().val()=std::atof(token.next().c_str());
		} else if(tag=="ETA"){
			type.eta().flag()=true;
			type.eta().val()=std::atof(token.next().c_str());
		} else if(tag=="JS"){
			type.js().flag()=true;
			type.js().val()=std::atof(token.next().c_str());
		} else if(tag=="C6"){
			type.c6().flag()=true;
			type.c6().val()=std::atof(token.next().c_str());
		} else if(tag=="RVDW"){
			type.rvdw().flag()=true;
			type.rvdw().val()=std::atof(token.next().c_str());
		} else if(tag=="RCOV"){
			type.rcov().flag()=true;
			type.rcov().val()=std::atof(token.next().c_str());
		} else if(tag=="Z"){
			type.z().flag()=true;
			type.z().val()=std::atof(token.next().c_str());
		} else if(tag=="ALPHA"){
			type.alpha().flag()=true;
			type.alpha().val()=std::atof(token.next().c_str());
		} else if(tag=="WEIGHT"){
			type.weight().flag()=true;
			type.weight().val()=std::atof(token.next().c_str());
		}
	}
	type.id()=string::hash(type.name());
	return type;
}

Type& Type::read(Type& type, Token& token){
	if(TYPE_PRINT_FUNC>0) std::cout<<"Type::read(const char*,Type&):\n";
	while(!token.end()){
		const std::string tag=string::to_upper(token.next());
		if(tag=="NAME"){
			type.name()=token.next();
		} else if(tag=="MASS"){
			type.mass().flag()=true;
			type.mass().val()=std::atof(token.next().c_str());
		} else if(tag=="CHARGE"){
			type.charge().flag()=true;
			type.charge().val()=std::atof(token.next().c_str());
		} else if(tag=="ENERGY"){
			type.energy().flag()=true;
			type.energy().val()=std::atof(token.next().c_str());
		} else if(tag=="CHI"){
			type.chi().flag()=true;
			type.chi().val()=std::atof(token.next().c_str());
		} else if(tag=="ETA"){
			type.eta().flag()=true;
			type.eta().val()=std::atof(token.next().c_str());
		} else if(tag=="JS"){
			type.js().flag()=true;
			type.js().val()=std::atof(token.next().c_str());
		} else if(tag=="C6"){
			type.c6().flag()=true;
			type.c6().val()=std::atof(token.next().c_str());
		} else if(tag=="RVDW"){
			type.rvdw().flag()=true;
			type.rvdw().val()=std::atof(token.next().c_str());
		} else if(tag=="RCOV"){
			type.rcov().flag()=true;
			type.rcov().val()=std::atof(token.next().c_str());
		} else if(tag=="Z"){
			type.z().flag()=true;
			type.z().val()=std::atof(token.next().c_str());
		} else if(tag=="ALPHA"){
			type.alpha().flag()=true;
			type.alpha().val()=std::atof(token.next().c_str());
		} else if(tag=="WEIGHT"){
			type.weight().flag()=true;
			type.weight().val()=std::atof(token.next().c_str());
		}
	}
	type.id()=string::hash(type.name());
	return type;
}

std::ostream& operator<<(std::ostream& out, const Type& type){
	out<<"name "<<type.name_<<" ";
	if(type.mass().flag()) out<<"mass "<<type.mass().val()<<" ";
	if(type.energy().flag()) out<<"energy "<<type.energy().val()<<" ";
	if(type.charge().flag()) out<<"chg "<<type.charge().val()<<" ";
	if(type.chi().flag()) out<<"chi "<<type.chi().val()<<" ";
	if(type.eta().flag()) out<<"eta "<<type.eta().val()<<" ";
	if(type.js().flag()) out<<"js "<<type.js().val()<<" ";
	if(type.c6().flag()) out<<"c6 "<<type.c6().val()<<" ";
	if(type.rvdw().flag()) out<<"rvdw "<<type.rvdw().val()<<" ";
	if(type.rcov().flag()) out<<"rcov "<<type.rcov().val()<<" ";
	if(type.z().flag()) out<<"z "<<type.z().val()<<" ";
	if(type.alpha().flag()) out<<"alpha "<<type.alpha().val()<<" ";
	if(type.weight().flag()) out<<"weight "<<type.weight().val()<<" ";
	return out;
}

void Type::write(FILE* out, const Type& type){
	fprintf(out,"name %s ",type.name().c_str());
	if(type.mass().flag()) fprintf(out,"mass %f ",type.mass().val());
	if(type.energy().flag()) fprintf(out,"energy %f ",type.energy().val());
	if(type.charge().flag()) fprintf(out,"charge %f ",type.charge().val());
	if(type.chi().flag()) fprintf(out,"chi %f ",type.chi().val());
	if(type.eta().flag()) fprintf(out,"eta %f ",type.eta().val());
	if(type.js().flag()) fprintf(out,"js %f ",type.js().val());
	if(type.c6().flag()) fprintf(out,"c6 %f ",type.c6().val());
	if(type.rvdw().flag()) fprintf(out,"rvdw %f ",type.rvdw().val());
	if(type.rcov().flag()) fprintf(out,"rcov %f ",type.rcov().val());
	if(type.z().flag()) fprintf(out,"z %f ",type.z().val());
	if(type.alpha().flag()) fprintf(out,"alpha %f ",type.alpha().val());
	if(type.weight().flag()) fprintf(out,"weight %f ",type.weight().val());
	fprintf(out,"\n");
}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const Type::Data<double>& obj){
	if(TYPE_PRINT_FUNC>0) std::cout<<"nbytes(const Type::Data<double>&):\n";
	int size=0;
	size+=sizeof(bool);//flag
	size+=sizeof(double);//val
	return size;
}
template <> int nbytes(const Type& obj){
	if(TYPE_PRINT_FUNC>0) std::cout<<"nbytes(const Type&):\n";
	int size=0;
	size+=nbytes(obj.mass());
	size+=nbytes(obj.energy());
	size+=nbytes(obj.charge());
	size+=nbytes(obj.chi());
	size+=nbytes(obj.eta());
	size+=nbytes(obj.js());
	size+=nbytes(obj.c6());
	size+=nbytes(obj.rvdw());
	size+=nbytes(obj.rcov());
	size+=nbytes(obj.z());
	size+=nbytes(obj.alpha());
	size+=nbytes(obj.weight());
	size+=nbytes(obj.name());//name
	size+=sizeof(int);//id
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const Type::Data<double>& obj, char* arr){
	if(TYPE_PRINT_FUNC>0) std::cout<<"pack(const Type::Data<double>&,char*):\n";
	int pos=0;
	pos+=pack(obj.flag(),arr+pos);
	pos+=pack(obj.val(),arr+pos);
	return pos;
}
template <> int pack(const Type& obj, char* arr){
	if(TYPE_PRINT_FUNC>0) std::cout<<"pack(const Type&,char*):\n";
	int pos=0;
	pos+=pack(obj.mass(),arr+pos);
	pos+=pack(obj.energy(),arr+pos);
	pos+=pack(obj.charge(),arr+pos);
	pos+=pack(obj.chi(),arr+pos);
	pos+=pack(obj.eta(),arr+pos);
	pos+=pack(obj.js(),arr+pos);
	pos+=pack(obj.c6(),arr+pos);
	pos+=pack(obj.rvdw(),arr+pos);
	pos+=pack(obj.rcov(),arr+pos);
	pos+=pack(obj.z(),arr+pos);
	pos+=pack(obj.alpha(),arr+pos);
	pos+=pack(obj.weight(),arr+pos);
	pos+=pack(obj.name(),arr+pos);
	std::memcpy(arr+pos,&obj.id(),sizeof(int)); pos+=sizeof(int);
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(Type::Data<double>& obj, const char* arr){
	if(TYPE_PRINT_FUNC>0) std::cout<<"unpack(Type::Data<double>&,const char*):\n";
	int pos=0;
	pos+=unpack(obj.flag(),arr+pos);
	pos+=unpack(obj.val(),arr+pos);
	return pos;
}
template <> int unpack(Type& obj, const char* arr){
	if(TYPE_PRINT_FUNC>0) std::cout<<"unpack(Type&,const char*):\n";
	int pos=0;
	pos+=unpack(obj.mass(),arr+pos);
	pos+=unpack(obj.energy(),arr+pos);
	pos+=unpack(obj.charge(),arr+pos);
	pos+=unpack(obj.chi(),arr+pos);
	pos+=unpack(obj.eta(),arr+pos);
	pos+=unpack(obj.js(),arr+pos);
	pos+=unpack(obj.c6(),arr+pos);
	pos+=unpack(obj.rvdw(),arr+pos);
	pos+=unpack(obj.rcov(),arr+pos);
	pos+=unpack(obj.z(),arr+pos);
	pos+=unpack(obj.alpha(),arr+pos);
	pos+=unpack(obj.weight(),arr+pos);
	pos+=unpack(obj.name(),arr+pos);
	std::memcpy(&obj.id(),arr+pos,sizeof(int)); pos+=sizeof(int);//id
	return pos;
}
	
}
