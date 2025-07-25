#pragma once
#ifndef POT_GAUSS_DSF_HPP
#define POT_GAUSS_DSF_HPP

// torch
#include "torch/pot.hpp"

#ifndef PGDSF_PRINT_FUNC
#define PGDSF_PRINT_FUNC 0
#endif

namespace ptnl{

class PotGaussDSF: public Pot{
private:
	double eps_;
	double alpha_;
	Eigen::VectorXi f_;
	Eigen::VectorXd radius_;
	Eigen::MatrixXd rij_;
	Eigen::MatrixXd erfij_;
	Eigen::MatrixXd expij_;
public:
	//==== constructors/destructors ====
	PotGaussDSF():Pot(Pot::Name::GAUSS_DSF),eps_(1.0),alpha_(0.0){}
	~PotGaussDSF(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotGaussDSF& pot);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& alpha(){return alpha_;}
	const double& alpha()const{return alpha_;}
	Eigen::VectorXi& f(){return f_;}
	const Eigen::VectorXi& f()const{return f_;}
	int& f(int i){return f_[i];}
	const int& f(int i)const{return f_[i];}
	Eigen::VectorXd& radius(){return radius_;}
	const Eigen::VectorXd& radius()const{return radius_;}
	double& radius(int i){return radius_[i];}
	const double& radius(int i)const{return radius_[i];}
	const Eigen::MatrixXd& rij()const{return rij_;}
	double& rij(int i, int j){return rij_(i,j);}
	const double& rij(int i, int j)const{return rij_(i,j);}
	
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
};

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotGaussDSF& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotGaussDSF& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotGaussDSF& obj, const char* arr);
	
}

#endif