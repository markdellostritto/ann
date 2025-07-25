#pragma once
#ifndef POT_COUL_DSF_HPP
#define POT_COUL_DSF_HPP

// torch
#include "torch/pot.hpp"

#ifndef PCDSF_PRINT_FUNC
#define PCDSF_PRINT_FUNC 0
#endif

namespace ptnl{

class PotCoulDSF: public Pot{
private:
	double eps_;
	double alpha_;
public:
	//==== constructors/destructors ====
	PotCoulDSF():Pot(Pot::Name::COUL_DSF),eps_(1.0),alpha_(0.0){}
	~PotCoulDSF(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotCoulDSF& pot);
	
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
	
	template <> int nbytes(const ptnl::PotCoulDSF& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotCoulDSF& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotCoulDSF& obj, const char* arr);
	
}

#endif