#pragma once
#ifndef POT_LDAMP_DSF_HPP
#define POT_LDAMP_DSF_HPP

// torch
#include "torch/pot.hpp"
// thread
#include "thread/dist.hpp"

#ifndef POT_LDAMP_DSF_PRINT_FUNC
#define POT_LDAMP_DSF_PRINT_FUNC 0
#endif

namespace ptnl{

class PotLDampDSF: public Pot{
private:
	double a_,a2_;
	double rc6_;
	Eigen::MatrixXi f_;
	Eigen::MatrixXd rvdw_;
	Eigen::MatrixXd rvdw6_;
	Eigen::MatrixXd c6_;
	Eigen::MatrixXd potfRc_;
	Eigen::MatrixXd potgRc_;
public:
	//==== constructors/destructors ====
	PotLDampDSF():Pot(Pot::Name::LDAMP_DSF){}
	~PotLDampDSF(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotLDampDSF& pot);
	
	//==== access ====
	double& a(){return a_;}
	const double& a()const{return a_;}
	Eigen::MatrixXd& rvdw(){return rvdw_;}
	const Eigen::MatrixXd& rvdw()const{return rvdw_;}
	Eigen::MatrixXd& rvdw6(){return rvdw6_;}
	const Eigen::MatrixXd& rvdw6()const{return rvdw6_;}
	Eigen::MatrixXd& c6(){return c6_;}
	const Eigen::MatrixXd& c6()const{return c6_;}
	Eigen::MatrixXd& potfRc(){return potfRc_;}
	const Eigen::MatrixXd& potfRc()const{return potfRc_;}
	Eigen::MatrixXd& potgRc(){return potgRc_;}
	const Eigen::MatrixXd& potgRc()const{return potgRc_;}
	
	//==== member functions ====
	void resize(int ntypes);
	void init();
	void read(Token& token);
	void coeff(Token& token);
	double energy(const Structure& struc, const NeighborList& nlist);
	double compute(Structure& struc, const NeighborList& nlist);
	double energy(const Structure& struc, const verlet::List& vlist);
	double compute(Structure& struc, const verlet::List& vlist);
};

//==== operator ====

double operator-(const PotLDampDSF& pot1, const PotLDampDSF& pot2);

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotLDampDSF& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLDampDSF& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotLDampDSF& obj, const char* arr);
	
}

#endif