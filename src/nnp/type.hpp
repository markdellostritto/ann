#pragma once
#ifndef TYPE_HPP
#define TYPE_HPP

//c++ libraries
#include <iosfwd>
#include <string>
// str
#include "str/token.hpp"
#include "str/string.hpp"
// mem
#include "mem/serialize.hpp"

#ifndef TYPE_PRINT_FUNC
#define TYPE_PRINT_FUNC 0
#endif

//************************************************************
// TYPE
//************************************************************

class Type{
public:
	template<typename T> class Data{
	private:
		bool flag_;
		T val_;
	public:
		//==== constructors/destructors ====
		Data():flag_(false),val_(0){}
		Data(bool flag, const T& val):flag_(flag),val_(val){}
		~Data(){}
		
		//==== access ====
		bool& flag(){return flag_;}
		const bool& flag()const{return flag_;}
		T& val(){return val_;}
		const T& val()const{return val_;}
		
		//==== member functions ====
		void clear(){flag_=false; val_=0;}
	};
private:
	Data<double> mass_;
	Data<double> energy_;
	Data<double> charge_;
	Data<double> chi_;
	Data<double> eta_;
	Data<double> js_;
	Data<double> c6_;
	Data<double> rvdw_;
	Data<double> rcov_;
	Data<double> z_;
	Data<double> alpha_;
	Data<double> weight_;
	std::string name_;
	int id_;
public:
	//==== constructors/destructors ====
	Type(){clear();}
	Type(const std::string& name){clear();name_=name;}
	~Type(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Type& type);
	
	//==== access ====
	Data<double>& mass(){return mass_;}
	const Data<double>& mass()const{return mass_;}
	Data<double>& energy(){return energy_;}
	const Data<double>& energy()const{return energy_;}
	Data<double>& charge(){return charge_;}
	const Data<double>& charge()const{return charge_;}
	Data<double>& chi(){return chi_;}
	const Data<double>& chi()const{return chi_;}
	Data<double>& eta(){return eta_;}
	const Data<double>& eta()const{return eta_;}
	Data<double>& js(){return js_;}
	const Data<double>& js()const{return js_;}
	Data<double>& c6(){return c6_;}
	const Data<double>& c6()const{return c6_;}
	Data<double>& rvdw(){return rvdw_;}
	const Data<double>& rvdw()const{return rvdw_;}
	Data<double>& rcov(){return rcov_;}
	const Data<double>& rcov()const{return rcov_;}
	Data<double>& z(){return z_;}
	const Data<double>& z()const{return z_;}
	Data<double>& alpha(){return alpha_;}
	const Data<double>& alpha()const{return alpha_;}
	Data<double>& weight(){return weight_;}
	const Data<double>& weight()const{return weight_;}
	std::string& name(){return name_;}
	const std::string& name()const{return name_;}
	int& id(){return id_;}
	const int& id()const{return id_;}
	
	//==== member functions ====
	void clear();
	
	//==== static functions ====
	static Type& read(const char* str, Type& type);
	static Type& read(Type& type, Token& token);
	static void write(FILE* out, const Type& type);
};

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const Type::Data<double>& obj);
template <> int nbytes(const Type& obj);

//**********************************************
// packing
//**********************************************

template <> int pack(const Type::Data<double>& obj, char* arr);
template <> int pack(const Type& obj, char* arr);

//**********************************************
// unpacking
//**********************************************

template <> int unpack(Type::Data<double>& obj, const char* arr);
template <> int unpack(Type& obj, const char* arr);

}

#endif
