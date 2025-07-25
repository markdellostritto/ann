#pragma once
#ifndef POT_TCOUL_LONG_HPP
#define POT_TCOUL_LONG_HPP

//mem
#include "mem/serialize.hpp"
// torch
#include "torch/pot.hpp"
#include "torch/kspace_coul.hpp"

#ifndef PTCL_PRINT_FUNC
#define PTCL_PRINT_FUNC 0
#endif

#ifndef PTCL_PRINT_DATA
#define PTCL_PRINT_DATA 0
#endif

namespace ptnl{
	
class PotTCoulLong: public Pot{
private:
	//parameters - global
	double eps_;
	double prec_;
	double a_;
	//parameters - atomic
	Eigen::VectorXi f_;
	Eigen::VectorXd alpha_;
	Eigen::MatrixXd aij_;
	//kspace
	KSpace::Coul coul_;
public:
	//==== constructors/destructors ====
	PotTCoulLong():Pot(Pot::Name::TCOUL_LONG),eps_(1.0){}
	~PotTCoulLong(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotTCoulLong& pot);
	
	//==== access ====
	//parameters - global
	double& prec(){return prec_;}
	const double& prec()const{return prec_;}
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& a(){return a_;}
	const double& a()const{return a_;}
	//parameters - atomic
	int& f(int i){return f_[i];}
	const int& f(int i)const{return f_[i];}
	Eigen::VectorXd& alpha(){return alpha_;}
	const Eigen::VectorXd& alpha()const{return alpha_;}
	double& alpha(int i){return alpha_[i];}
	const double& radius(int i)const{return alpha_[i];}
	const Eigen::MatrixXd& aij()const{return aij_;}
	const double& aij(int i, int j)const{return aij_(i,j);}
	//kspace
	KSpace::Coul& coul(){return coul_;}
	const KSpace::Coul& coul()const{return coul_;}
	Eigen::VectorXi& f(){return f_;}
	const Eigen::VectorXi& f()const{return f_;}
	
	//==== member functions ====
	void read(Token& token);
	void coeff(Token& token);
	void resize(int);
	void init();
	double energy(const Structure& struc, const NeighborList& nlist);
	double compute(Structure& struc, const NeighborList& nlist);
	Eigen::MatrixXd& J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J);
	double energy(const Structure& struc, const verlet::List& vlist);
	double compute(Structure& struc, const verlet::List& vlist);
	Eigen::MatrixXd& J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J);
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
	
	template <> int nbytes(const ptnl::PotTCoulLong& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotTCoulLong& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotTCoulLong& obj, const char* arr);
	
}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotTCoulLong& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotTCoulLong& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotTCoulLong& obj, const char* arr);
	
}

#endif