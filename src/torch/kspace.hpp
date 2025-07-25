#pragma once
#ifndef KSPACE_HPP
#define KSPACE_HPP

// c++
#include <iostream>
#include <complex>
// eigen
#include <Eigen/Dense>
//mem
#include "mem/serialize.hpp"

#ifndef KSPACE_PRINT_FUNC
#define KSPACE_PRINT_FUNC 0
#endif

namespace KSpace{

class Base{
protected:
	static const std::complex<double> I_;
	double rc_;//real-space cutoff
	double prec_;//precision
	double alpha_;//convergence parameter
	double errEr_;//error - energy/atom - real space
	double errEk_;//error - energy/atom - reciprocal space
	Eigen::Vector3i nk_;//number of k-points
	std::vector<double> ka_;//k-amps
	std::vector<Eigen::Vector3d> k_;//k-vecs
public:
	//==== constructors/destructors ====
	Base(){}
	~Base(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Base& base);
	
	//==== access ====
	double& rc(){return rc_;}
	const double& rc()const{return rc_;}
	double& rcut(){return rc_;}
	const double& rcut()const{return rc_;}
	double& prec(){return prec_;}
	const double& prec()const{return prec_;}
	const double& alpha()const{return alpha_;}
	const Eigen::Vector3i& nk()const{return nk_;}
	const double& errEr()const{return errEr_;}
	const double& errEk()const{return errEk_;}
	
};

//==== operators ====

double operator-(const Base& ks1, const Base& ks2);

}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const KSpace::Base& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const KSpace::Base& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(KSpace::Base& obj, const char* arr);
	
}


#endif