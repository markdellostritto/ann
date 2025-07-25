#pragma once
#ifndef POT_PAULI_HPP
#define POT_PAULI_HPP

// torch
#include "torch/pot.hpp"
// thread
#include "thread/dist.hpp"

#ifndef POT_PAULI_PRINT_FUNC
#define POT_PAULI_PRINT_FUNC 0
#endif

namespace ptnl{
	
class PotPauli: public Pot{
private:
	Eigen::VectorXi f_;//parameter flag
	Eigen::VectorXd r_;//covalent radius
	Eigen::VectorXd z_;//effective nuclear charge
	Eigen::MatrixXd zz_;
	Eigen::MatrixXd rr_;
public:
	static const double CC;

	//==== constructors/destructors ====
	PotPauli():Pot(Pot::Name::PAULI){}
	~PotPauli(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotPauli& pot);
	
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
	
	template <> int nbytes(const ptnl::PotPauli& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotPauli& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotPauli& obj, const char* arr);
	
}

#endif