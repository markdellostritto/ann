#pragma once
#ifndef POT_PSLATER_HPP
#define POT_PSLATER_HPP

// torch
#include "torch/pot.hpp"
// thread
#include "thread/dist.hpp"

#ifndef POT_PSLATER_PRINT_FUNC
#define POT_PSLATER_PRINT_FUNC 0
#endif

namespace ptnl{
	
class PotPSlater: public Pot{
private:
	Eigen::VectorXi f_;//parameter flag
	Eigen::VectorXd r_;//gaussian radius (exp(-(x/r)^2))
	Eigen::VectorXd z_;//effective nuclear charge
	Eigen::VectorXd a_;
	Eigen::MatrixXd x_;
	Eigen::MatrixXd c6_;
	Eigen::MatrixXd z2_;
public:
	static const double p;

	//==== constructors/destructors ====
	PotPSlater():Pot(Pot::Name::PAULI){}
	~PotPSlater(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotPSlater& pot);
	
	//==== access ====
	Eigen::VectorXi& f(){return f_;}
	const Eigen::VectorXi& f()const{return f_;}
	Eigen::VectorXd& r(){return r_;}
	const Eigen::VectorXd& r()const{return r_;}
	Eigen::VectorXd& z(){return z_;}
	const Eigen::VectorXd& z()const{return z_;}
	
	//==== member functions ====
	void read(Token& token);
	void coeff(Token& token);
	void resize(int ntypes);
	void init();
	virtual double energy(const Structure& struc, const NeighborList& nlist);
	virtual double energy(const Structure& struc, const NeighborList& nlist, int i);
	virtual double compute(Structure& struc, const NeighborList& nlist);
	virtual double compute(Structure& struc, const NeighborList& nlist, int i);
	virtual double energy(const Structure& struc, const verlet::List& vlist);
	virtual double energy(const Structure& struc, const verlet::List& vlist, int i);
	virtual double compute(Structure& struc, const verlet::List& vlist);
	virtual double compute(Structure& struc, const verlet::List& vlist, int i);
};

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotPSlater& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotPSlater& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotPSlater& obj, const char* arr);
	
}

#endif