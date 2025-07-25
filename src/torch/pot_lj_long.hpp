#pragma once
#ifndef POT_LJ_LONG_HPP
#define POT_LJ_LONG_HPP

// torch
#include "torch/pot.hpp"
#include "torch/kspace_london.hpp"
// thread
#include "thread/dist.hpp"

#ifndef POT_LJ_LONG_PRINT_FUNC
#define POT_LJ_LONG_PRINT_FUNC -1
#endif

namespace ptnl{

class PotLJLong: public Pot{
private:
	Eigen::MatrixXi f_;
	Eigen::MatrixXd s_;
	Eigen::MatrixXd s6_;
	Eigen::MatrixXd e_;
	KSpace::London ksl_;
	double prec_;
public:
	//==== constructors/destructors ====
	PotLJLong():Pot(Pot::Name::LJ_LONG){}
	~PotLJLong(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotLJLong& pot);
	
	//==== access ====
	Eigen::MatrixXi& f(){return f_;}
	const Eigen::MatrixXi& f()const{return f_;}
	Eigen::MatrixXd& s(){return s_;}
	const Eigen::MatrixXd& s()const{return s_;}
	Eigen::MatrixXd& s6(){return s6_;}
	const Eigen::MatrixXd& s6()const{return s6_;}
	Eigen::MatrixXd& e(){return e_;}
	const Eigen::MatrixXd& e()const{return e_;}
	KSpace::London& ksl(){return ksl_;}
	const KSpace::London& ksl()const{return ksl_;}
	
	//==== member functions ====
	void read(Token& token);
	void init();
	void coeff(Token& token);
	void resize(int ntypes);
	double energy(const Structure& struc, const NeighborList& nlist);
	double compute(Structure& struc, const NeighborList& nlist);
	double energy(const Structure& struc, const verlet::List& vlist);
	double compute(Structure& struc, const verlet::List& vlist);
};

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotLJLong& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLJLong& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotLJLong& obj, const char* arr);
	
}

#endif