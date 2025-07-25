#pragma once
#ifndef KSPACE_COUL_HPP
#define KSPACE_COUL_HPP

//mem
#include "mem/serialize.hpp"
// structure
#include "struc/neighbor.hpp"
#include "struc/verlet.hpp"
#include "struc/structure.hpp"
// torch
#include "torch/kspace.hpp"

#ifndef KSPACEC_PRINT_FUNC
#define KSPACEC_PRINT_FUNC 0
#endif

#ifndef KSPACEC_PRINT_STATUS
#define KSPACEC_PRINT_STATUS 0
#endif

#ifndef KSPACEC_PRINT_DATA
#define KSPACEC_PRINT_DATA 0
#endif

namespace KSpace{

class Coul: public Base{
private:
	double eps_;
	double q2_;//sum of squares of charges
	double vc_;//constant term
public:
	//==== constructors/destructors ====
	Coul():eps_(1.0){}
	~Coul(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Coul& c);
	
	//==== access ====
	const double& vc()const{return vc_;}
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	
	//==== member functions ====
	void init(const Structure& struc);
	double energy(const Structure& struc)const;
	double compute(Structure& struc, const NeighborList& nlist)const;
	Eigen::MatrixXd& J(const Structure& struc, Eigen::MatrixXd& J)const;
	double compute(Structure& struc, const verlet::List& vlist)const;
};

}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const KSpace::Coul& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const KSpace::Coul& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(KSpace::Coul& obj, const char* arr);
	
}

#endif