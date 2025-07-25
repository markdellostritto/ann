#pragma once
#ifndef POT_COUL_LONG_HPP
#define POT_COUL_LONG_HPP

// torch
#include "torch/pot.hpp"
#include "torch/kspace_coul.hpp"

#ifndef PCL_PRINT_FUNC
#define PCL_PRINT_FUNC 0
#endif

#ifndef PCL_PRINT_STATUS
#define PCL_PRINT_STATUS 0
#endif

#ifndef PCL_PRINT_DATA
#define PCL_PRINT_DATA 0
#endif

namespace ptnl{

class PotCoulLong: public Pot{
private:
	double eps_;
	double prec_;
	KSpace::Coul coul_;
public:
	//==== constructors/destructors ====
	PotCoulLong():Pot(Pot::Name::COUL_LONG),eps_(1.0){}
	~PotCoulLong(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotCoulLong& pot);
	
	//==== access ====
	double& prec(){return prec_;}
	const double& prec()const{return prec_;}
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	KSpace::Coul& coul(){return coul_;}
	const KSpace::Coul& coul()const{return coul_;}
	
	//==== member functions ====
	void read(Token& token);
	void init();
	double energy(const Structure& struc, const NeighborList& nlist);
	double compute(Structure& struc, const NeighborList& nlist);
	Eigen::MatrixXd& J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& Jm);
	double energy(const Structure& struc, const verlet::List& vlist);
	double compute(Structure& struc, const verlet::List& vlist);
	Eigen::MatrixXd& J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& Jm);
	double cQ(Structure& struc);
};

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotCoulLong& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotCoulLong& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotCoulLong& obj, const char* arr);
	
}

#endif