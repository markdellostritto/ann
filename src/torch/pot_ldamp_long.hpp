#pragma once
#ifndef POT_LDAMP_LONG_HPP
#define POT_LDAMP_LONG_HPP

// torch
#include "torch/pot.hpp"
#include "torch/kspace_london.hpp"
// thread
#include "thread/dist.hpp"

#ifndef POT_LDAMP_LONG_PRINT_FUNC
#define POT_LDAMP_LONG_PRINT_FUNC 0
#endif

namespace ptnl{

class PotLDampLong: public Pot{
private:
	Eigen::MatrixXi f_;
	Eigen::MatrixXd rvdw_;
	Eigen::MatrixXd rvdw6_;
	Eigen::MatrixXd c6_;
	KSpace::London ksl_;
	double prec_;
public:
	//==== constructors/destructors ====
	PotLDampLong():Pot(Pot::Name::LDAMP_LONG){}
	~PotLDampLong(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotLDampLong& pot);
	
	//==== access ====
	Eigen::MatrixXd& rvdw(){return rvdw_;}
	const Eigen::MatrixXd& rvdw()const{return rvdw_;}
	Eigen::MatrixXd& rvdw6(){return rvdw6_;}
	const Eigen::MatrixXd& rvdw6()const{return rvdw6_;}
	Eigen::MatrixXd& c6(){return c6_;}
	const Eigen::MatrixXd& c6()const{return c6_;}
	KSpace::London& ksl(){return ksl_;}
	const KSpace::London& ksl()const{return ksl_;}
	double& prec(){return prec_;}
	const double& prec()const{return prec_;}
	
	//==== member functions ====
	void resize(int ntypes);
	void init();
	void read(Token& token);
	void coeff(Token& token);
	double energy(const Structure& struc, const NeighborList& nlist);
	double compute(Structure& struc, const NeighborList& nlist);
	//Eigen::MatrixXd& J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J);
	double energy(const Structure& struc, const verlet::List& vlist);
	double compute(Structure& struc, const verlet::List& vlist);
	//Eigen::MatrixXd& J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J);
};

//==== operator ====

double operator-(const PotLDampLong& pot1, const PotLDampLong& pot2);

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotLDampLong& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLDampLong& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotLDampLong& obj, const char* arr);
	
}

#endif