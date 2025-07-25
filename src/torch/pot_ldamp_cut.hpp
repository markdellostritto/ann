#pragma once
#ifndef POT_LDAMP_CUT_HPP
#define POT_LDAMP_CUT_HPP

// torch
#include "torch/pot.hpp"
// thread
#include "thread/dist.hpp"

#ifndef POT_LDAMP_CUT_PRINT_FUNC
#define POT_LDAMP_CUT_PRINT_FUNC 0
#endif

namespace ptnl{

class PotLDampCut: public Pot{
private:
	Eigen::MatrixXi f_;
	Eigen::MatrixXd rvdw_;
	Eigen::MatrixXd rvdw6_;
	Eigen::MatrixXd c6_;
public:
	//==== constructors/destructors ====
	PotLDampCut():Pot(Pot::Name::LDAMP_CUT){}
	~PotLDampCut(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotLDampCut& pot);
	
	//==== access ====
	Eigen::MatrixXd& rvdw(){return rvdw_;}
	const Eigen::MatrixXd& rvdw()const{return rvdw_;}
	Eigen::MatrixXd& c6(){return c6_;}
	const Eigen::MatrixXd& c6()const{return c6_;}
	
	//==== member functions ====
	void init();
	void resize(int ntypes);
	void read(Token& token);
	void coeff(Token& token);
	double energy(const Structure& struc, const NeighborList& nlist);
	double compute(Structure& struc, const NeighborList& nlist);
	Eigen::MatrixXd& J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J);
	double energy(const Structure& struc, const verlet::List& vlist);
	double compute(Structure& struc, const verlet::List& vlist);
	Eigen::MatrixXd& J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J);
};

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotLDampCut& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLDampCut& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotLDampCut& obj, const char* arr);
	
}

#endif