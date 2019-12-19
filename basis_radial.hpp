#pragma once
#ifndef BASIS_RADIAL_HPP
#define BASIS_RADIAL_HPP

// c++ libraries
#include <iosfwd>
// eigen
#include <Eigen/Dense>
// symmetry functions
#include "cutoff.hpp"
#include "symm_radial.hpp"
// ann - serialization
#include "serialize.hpp"

#ifndef BASIS_RADIAL_PRINT_FUNC
#define BASIS_RADIAL_PRINT_FUNC 0
#endif

struct BasisR{
private:
	static const double V_CUT;//value cutoff
	double rc_;//cutoff radius
	double norm_;//normalization factor
	CutoffN::type tcut_;//cutoff function type
	PhiRN::type phiRN_;//type of radial functions
	unsigned int nfR_;//number of radial functions
	PhiR** fR_;//radial functions
	Eigen::VectorXd symm_;//symmetry function
public:
	//constructors/destructors
	BasisR():phiRN_(PhiRN::UNKNOWN),nfR_(0),fR_(NULL){}
	BasisR(const BasisR& basisR);
	~BasisR();
	//operators
	BasisR& operator=(const BasisR& basis);
	friend std::ostream& operator<<(std::ostream& out, const BasisR& basisR);
	//initialization
	void init_G1(unsigned int nR, CutoffN::type tcut, double rcut);
	void init_G2(unsigned int nR, CutoffN::type tcut, double rcut);
	void init_T1(unsigned int nR, CutoffN::type tcut, double rcut);
	//loading/printing
	static void write(const char* filename, const BasisR& basis);
	static void read(const char* filename, BasisR& basis);
	static void write(FILE* writer, const BasisR& basis);
	static void read(FILE* reader, BasisR& basis);
	//member access
	const CutoffN::type& tcut()const{return tcut_;}
	CutoffN::type& tcut(){return tcut_;}
	const double& rc()const{return rc_;}
	double& rc(){return rc_;}
	const unsigned int& nfR()const{return nfR_;}
	PhiRN::type& phiRN(){return phiRN_;}
	const PhiRN::type& phiRN()const{return phiRN_;}
	PhiR& fR(unsigned int i){return *fR_[i];}
	const PhiR& fR(unsigned int i)const{return *fR_[i];}
	const Eigen::VectorXd& symm()const{return symm_;}
	Eigen::VectorXd& symm(){return symm_;}
	//member functions
	void clear();
	void symm(double dr);
	double force(double dr, const double* dEdG)const;
};
std::ostream& operator<<(std::ostream& out, const BasisR& basisR);

bool operator==(const BasisR& basis1, const BasisR& basis2);
inline bool operator!=(const BasisR& basis1, const BasisR& basis2){return !(basis1==basis2);}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const BasisR& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const BasisR& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(BasisR& obj, const char* arr);
	
}

#endif