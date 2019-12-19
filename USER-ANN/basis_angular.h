#pragma once
#ifndef BASIS_ANGULAR_HPP
#define BASIS_ANGULAR_HPP

// c++ libraries
#include <iosfwd>
//eigen
#include <Eigen/Dense>
// ann - symmetry functions
#include "cutoff.h"
#include "symm_angular.h"
// ann - serialize
#include "serialize.h"

#ifndef BASIS_ANGULAR_PRINT_FUNC
#define BASIS_ANGULAR_PRINT_FUNC 0
#endif

struct BasisA{
private:
	static const double V_CUT;//value cutoff
	double rc_;//cutoff radius
	double norm_;//normalization factor
	CutoffN::type tcut_;//cutoff function type
	PhiAN::type phiAN_;//type of angular functions
	unsigned int nfA_;//number of angular functions
	PhiA** fA_;//angular functions
	Eigen::VectorXd symm_;//symmetry function
public:
	//constructors/destructors
	BasisA():phiAN_(PhiAN::UNKNOWN),nfA_(0),fA_(NULL){}
	BasisA(const BasisA& basisA);
	~BasisA();
	//operators
	BasisA& operator=(const BasisA& basisA);
	friend std::ostream& operator<<(std::ostream& out, const BasisA& basisA);
	//initialization
	void init_G3(unsigned int nA, CutoffN::type tcut, double rcut);
	void init_G4(unsigned int nA, CutoffN::type tcut, double rcut);
	//reading/writing
	static void write(const char* file,const BasisA& basis);
	static void read(const char* file, BasisA& basis);
	static void write(FILE* writer,const BasisA& basis);
	static void read(FILE* writer, BasisA& basis);
	//member access
	const CutoffN::type& tcut()const{return tcut_;}
	CutoffN::type& tcut(){return tcut_;}
	const double& rc()const{return rc_;}
	double& rc(){return rc_;}
	const unsigned int& nfA()const{return nfA_;};
	PhiAN::type& phiAN(){return phiAN_;};
	const PhiAN::type& phiAN()const{return phiAN_;};
	PhiA& fA(unsigned int i){return *fA_[i];};
	const PhiA& fA(unsigned int i)const{return *fA_[i];}
	const Eigen::VectorXd& symm()const{return symm_;}
	Eigen::VectorXd& symm(){return symm_;}
	//member functions
	void clear();
	void symm(double cos, const double d[3]);
	void force(double* fij, double* fik, double cos, const double d[3], const double* dEdG)const;
};

bool operator==(const BasisA& basis1, const BasisA& basis2);
inline bool operator!=(const BasisA& basis1, const BasisA& basis2){return !(basis1==basis2);}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const BasisA& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const BasisA& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(BasisA& obj, const char* arr);
	
}

#endif
