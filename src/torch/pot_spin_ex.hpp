#pragma once
#ifndef POT_SPIN_EX_HPP
#define POT_SPIN_EX_HPP

// torch
#include "torch/pot.hpp"

#ifndef POT_SPINEX_CUT_PRINT_FUNC
#define POT_SPINEX_CUT_PRINT_FUNC 0
#endif

namespace ptnl{

class PotSpinEx: public Pot{
private:
	Eigen::MatrixXi f_;
	Eigen::MatrixXd rad_;
	Eigen::MatrixXd alpha_;
public:
	//==== constructors/destructors ====
	PotSpinEx():Pot(Pot::Name::SPIN_EX){}
	~PotSpinEx(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotSpinEx& pot);
	
	//==== member functions ====
	void read(Token& token);
	void coeff(Token& token);
	void resize(int ntypes);
	void init();
	double energy(const Structure& struc, const NeighborList& nlist);
	Eigen::MatrixXd& J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J);
	double energy(const Structure& struc, const verlet::List& vlist);
	Eigen::MatrixXd& J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J);
};

} // namespace ptnl

#endif