#pragma once
#ifndef POT_HPP
#define POT_HPP

// c++
#include <memory>
#include <iostream>
// struc
#include "struc/structure.hpp"
#include "struc/neighbor.hpp"
#include "struc/verlet.hpp"
// str
#include "str/token.hpp"
//mem
#include "mem/serialize.hpp"
// thread
#include "thread/dist.hpp"

#ifndef POT_PRINT_FUNC
#define POT_PRINT_FUNC 0
#endif

#ifndef LDAMP_A
#define LDAMP_A 6 //pot_ldamp_cut/pot_ldamp_long
#endif

namespace ptnl{

class Pot{
public:
	class Name{
	public:
		enum Type{
			PAULI,
			LJ_CUT,
			LJ_LONG,
			LJ_SM,
			LDAMP_CUT,
			LDAMP_DSF,
			LDAMP_LONG,
			COUL_CUT,
			COUL_WOLF,
			COUL_DSF,
			COUL_LONG,
			GAUSS_CUT,
			GAUSS_DSF,
			GAUSS_LONG,
			QEQ_GL,
			SPIN_EX,
			NNPE,
			UNKNOWN
		};
		//constructor
		Name():t_(Type::UNKNOWN){}
		Name(Type t):t_(t){}
		//operators
		operator Type()const{return t_;}
		friend std::ostream& operator<<(std::ostream& out, const Name& sys);
		//member functions
		static Name read(const char* str);
		static const char* name(const Name& name);
	private:
		Type t_;
		//prevent automatic conversion for other built-in types
		//template<typename T> operator T() const;
	};
protected:
	Pot::Name name_;//name
	int ntypes_;//number of types
	double rc_;//cutoff radius
	double rc2_;//cutoff radius squared
public:
	//==== contructors/destructors ====
	Pot():rc_(0.0),rc2_(0.0),ntypes_(0),name_(Name::UNKNOWN){}
	Pot(const Name& name):rc_(0.0),rc2_(0.0),ntypes_(0),name_(name){}
	~Pot(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const Pot& pot);
	
	//==== access ====
	double& rc(){return rc_;}
	const double& rc()const{return rc_;}
	double& rc2(){return rc2_;}
	const double& rc2()const{return rc2_;}
	const int& ntypes()const{return ntypes_;}
	const Pot::Name& name()const{return name_;}
	
	//==== member functions ====
	void read(Token& token);
	virtual void init(){}
	virtual void coeff(Token& token){}
	virtual void resize(int ntypes){}
	virtual double energy(const Structure& struc, const NeighborList& nlist)=0;
	virtual double energy(const Structure& struc, const NeighborList& nlist, int i){return 0.0;}
	virtual double compute(Structure& struc, const NeighborList& nlist){return 0.0;}
	virtual double compute(Structure& struc, const NeighborList& nlist, int i){return 0.0;}
	virtual Eigen::MatrixXd& J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
		throw std::runtime_error("ptnl::Pot::J(const Structure&,const Neighbor&,Eigen::MatrixXd&): No function defined for computing J.");
	}
	virtual double energy(const Structure& struc, const verlet::List& vlist){return 0.0;}
	virtual double energy(const Structure& struc, const verlet::List& vlist, int i){return 0.0;}
	virtual double compute(Structure& struc, const verlet::List& vlist){return 0.0;}
	virtual double compute(Structure& struc, const verlet::List& vlist, int i){return 0.0;}
	virtual Eigen::MatrixXd& J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
		throw std::runtime_error("ptnl::Pot::J(const Structure&,const Neighbor&,Eigen::MatrixXd&): No function defined for computing J.");
	}
	virtual double cQ(Structure& struc){return 0;}
};

//==== operator ====

double operator-(const Pot& pot1, const Pot& pot2);

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::Pot& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::Pot& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::Pot& obj, const char* arr);
	
}

#endif
