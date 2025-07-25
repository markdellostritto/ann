#pragma once
#ifndef POT_LJ_SM_HPP
#define POT_LJ_SM_HPP

// torch
#include "torch/pot.hpp"
// thread
#include "thread/dist.hpp"

#ifndef POT_LJ_SM_PRINT_FUNC
#define POT_LJ_SM_PRINT_FUNC 0
#endif

namespace ptnl{
	
class PotLJSm: public Pot{
private:
	Eigen::MatrixXi f_;
	Eigen::MatrixXd s_;
	Eigen::MatrixXd s6_;
	Eigen::MatrixXd e_;
	Eigen::MatrixXd phirc_;
	Eigen::MatrixXd dphirc_;
public:
	//==== constructors/destructors ====
	PotLJSm():Pot(Pot::Name::LJ_CUT){}
	~PotLJSm(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotLJSm& pot);
	
	//==== access ====
	Eigen::MatrixXi& f(){return f_;}
	const Eigen::MatrixXi& f()const{return f_;}
	Eigen::MatrixXd& s(){return s_;}
	const Eigen::MatrixXd& s()const{return s_;}
	Eigen::MatrixXd& s6(){return s6_;}
	const Eigen::MatrixXd& s6()const{return s6_;}
	Eigen::MatrixXd& e(){return e_;}
	const Eigen::MatrixXd& e()const{return e_;}
	
	//==== member functions ====
	void read(Token& token);
	void coeff(Token& token);
	void resize(int ntypes);
	void init();
	virtual double energy(const Structure& struc, const NeighborList& nlist);
	virtual double energy(const Structure& struc, const NeighborList& nlist, int i);
	virtual double compute(Structure& struc, const NeighborList& nlist);
	virtual double compute(Structure& struc, const NeighborList& nlist, int i);
	virtual double energy(const Structure& struc, const verlet::List& vlist);
	virtual double energy(const Structure& struc, const verlet::List& vlist, int i);
	virtual double compute(Structure& struc, const verlet::List& vlist);
	virtual double compute(Structure& struc, const verlet::List& vlist, int i);
};

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotLJSm& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLJSm& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotLJSm& obj, const char* arr);
	
}

#endif