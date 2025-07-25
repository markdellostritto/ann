#pragma once
#ifndef POT_COUL_CUT_HPP
#define POT_COUL_CUT_HPP

// torch
#include "torch/pot.hpp"

#ifndef PCC_PRINT_FUNC
#define PCC_PRINT_FUNC 0
#endif

namespace ptnl{

class PotCoulCut: public Pot{
private:
	double eps_;
public:
	//==== constructors/destructors ====
	PotCoulCut():Pot(Pot::Name::COUL_CUT),eps_(1.0){}
	~PotCoulCut(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotCoulCut& pot);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	
	//==== member functions ====
	void read(Token& token);
	double energy(const Structure& struc, const NeighborList& nlist);
	double compute(Structure& struc, const NeighborList& nlist);
	Eigen::MatrixXd& J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& Jm);
	double energy(const Structure& struc, const verlet::List& vlist);
	double compute(Structure& struc, const verlet::List& vlist);
	Eigen::MatrixXd& J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& Jm);
};

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotCoulCut& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotCoulCut& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotCoulCut& obj, const char* arr);
	
}

#endif