#pragma once
#ifndef BASIS_RADIAL_HPP
#define BASIS_RADIAL_HPP

// c++ libraries
#include <iosfwd>
// eigen
#include <Eigen/Dense>
// symmetry functions
#include "nnp/cutoff.hpp"
#include "nnp/basis.hpp"
// ann - serialization
#include "mem/serialize.hpp"

#ifndef BASIS_RADIAL_PRINT_FUNC
#define BASIS_RADIAL_PRINT_FUNC 0
#endif

//*****************************************
// BasisR - radial basis
//*****************************************

class BasisR: public Basis{
public:
	class Name{
	public:
		enum Type{
			NONE,
			GAUSSIAN,
			SECH,
			LOGISTIC,
			TANH,
			LOGCOSH,
			LOGCOSH2
		};
		//constructor
		Name():t_(Type::NONE){}
		Name(Type t):t_(t){}
		//operators
		operator Type()const{return t_;}
		//member functions
		static Name read(const char* str);
		static const char* name(const Name& name);
	private:
		Type t_;
		//prevent automatic conversion for other built-in types
		//template<typename T> operator T() const;
	};
private:
Name name_;//type of radial functions
	std::vector<double> rs_;//center
	std::vector<double> eta_;//width
public:
	//==== constructors/destructors ====
	BasisR():Basis(),name_(Name::NONE){}
	BasisR(double rc, Cutoff::Name cutname, int nf, Name name);
	~BasisR();
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const BasisR& basisR);
	
	//==== reading/writing ====
	static void write(FILE* writer, const BasisR& basis);
	static void read(FILE* writer, BasisR& basis);
	
	//==== member access ====
	Name& name(){return name_;}
	const Name& name()const{return name_;}
	double& rs(int i){return rs_[i];}
	const double& rs(int i)const{return rs_[i];}
	double& eta(int i){return eta_[i];}
	const double& eta(int i)const{return eta_[i];}
	std::vector<double>& rs(){return rs_;}
	const std::vector<double>& rs()const{return rs_;}
	std::vector<double>& eta(){return eta_;}
	const std::vector<double>& eta()const{return eta_;}
	
	Eigen::VectorXd& symm(){return symm_;}
	const Eigen::VectorXd& symm()const{return symm_;}
	
	//==== member functions ====
	void clear();
	void resize(int size);
	void init();
	double symmf(double dr, double eta, double rs)const;
	double symmd(double dr, double eta, double rs)const;
	void symm(const std::vector<double>& dr);
	double force(const std::vector<double>& dr, const double* dEdG)const;
};
std::ostream& operator<<(std::ostream& out, const BasisR& basisR);
std::ostream& operator<<(std::ostream& out, const BasisR::Name& name);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisR& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisR& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisR& obj, const char* arr);
	
}

#endif