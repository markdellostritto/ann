#pragma once
#ifndef POT_QEQ_GL_HPP
#define POT_QEQ_GL_HPP

//mem
#include "mem/serialize.hpp"
// torch
#include "torch/pot.hpp"
#include "torch/kspace_coul.hpp"
#include "torch/qeq.hpp"

#ifndef PQGL_PRINT_FUNC
#define PQGL_PRINT_FUNC 0
#endif

#ifndef PQGL_PRINT_DATA
#define PQGL_PRINT_DATA 0
#endif

namespace ptnl{
	
class PotQEQGL: public Pot{
private:
	//parameters - global
	double eps_;
	double prec_;
	//parameters - atomic
	Eigen::VectorXi f_;
	Eigen::VectorXd radius_;
	Eigen::MatrixXd rij_;
	//kspace
	KSpace::Coul coul_;
	//qeq
	double qtot_;
	Eigen::MatrixXd A_;//coulomb matrix
	Eigen::VectorXd b_;//constant vector
	Eigen::VectorXd x_;//solution vector
	Eigen::MatrixXd Jk_;
	Eigen::MatrixXd Jr_;
	Eigen::MatrixXd J_;
public:
	//==== constructors/destructors ====
	PotQEQGL():Pot(Pot::Name::QEQ_GL),eps_(1.0){}
	~PotQEQGL(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotQEQGL& pot);
	
	//==== access ====
	double& prec(){return prec_;}
	const double& prec()const{return prec_;}
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& qtot(){return qtot_;}
	const double& qtot()const{return qtot_;}
	KSpace::Coul& coul(){return coul_;}
	const KSpace::Coul& coul()const{return coul_;}
	Eigen::VectorXi& f(){return f_;}
	const Eigen::VectorXi& f()const{return f_;}
	int& f(int i){return f_[i];}
	const int& f(int i)const{return f_[i];}
	Eigen::VectorXd& radius(){return radius_;}
	const Eigen::VectorXd& radius()const{return radius_;}
	double& radius(int i){return radius_[i];}
	const double& radius(int i)const{return radius_[i];}
	const Eigen::MatrixXd& rij()const{return rij_;}
	const double& rij(int i, int j)const{return rij_(i,j);}
	const Eigen::MatrixXd& A()const{return A_;}
	const Eigen::VectorXd& b()const{return b_;}
	const Eigen::VectorXd& x()const{return x_;}
	
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
	
	template <> int nbytes(const ptnl::PotQEQGL& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotQEQGL& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotQEQGL& obj, const char* arr);
	
}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotQEQGL& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotQEQGL& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotQEQGL& obj, const char* arr);
	
}

#endif