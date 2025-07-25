#pragma once
#ifndef POT_COUL_WOLF_HPP
#define POT_COUL_WOLF_HPP

// torch
#include "torch/pot.hpp"

#ifndef PCW_PRINT_FUNC
#define PCW_PRINT_FUNC 0
#endif

namespace ptnl{

class PotCoulWolf: public Pot{
private:
	double eps_;
	double alpha_;
public:
	//==== constructors/destructors ====
	PotCoulWolf():Pot(Pot::Name::COUL_WOLF),alpha_(0.0),eps_(1.0){}
	~PotCoulWolf(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotCoulWolf& pot);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& alpha(){return alpha_;}
	const double& alpha()const{return alpha_;}
	
	//==== member functions ====
	void read(Token& token);
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
	
	template <> int nbytes(const ptnl::PotCoulWolf& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotCoulWolf& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotCoulWolf& obj, const char* arr);
	
}

#endif