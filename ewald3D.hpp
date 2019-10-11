#pragma once
#ifndef EWALD_3D_HPP
#define EWALD_3D_HPP

//no bounds checking in Eigen
#define EIGEN_NO_DEBUG

//c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
#include <cstdlib>
//Eigen
#include <Eigen/Dense>
//ame
#include "math_const.hpp"
//structure
#include "cell.hpp"
#include "structure.hpp"
//serialization
#include "serialize.hpp"
//units
#include "units.hpp"

#ifndef EWALD_PRINT_FUNC
#define EWALD_PRINT_FUNC 0
#endif

#ifndef EWALD_PRINT_STATUS
#define EWALD_PRINT_STATUS 0
#endif

#ifndef EWALD_PRINT_DATA
#define EWALD_PRINT_DATA 0
#endif

namespace Ewald3D{

//**********************************************************************************************************
//Utility Class
//**********************************************************************************************************

class Utility{
protected:
	//calculation parameters
	double prec_;//precision for the calculation
	double rMax_;//the maximum length of lattice vector included in the real-space sum
	double kMax_;//the maximum length of the lattice vector included in the reciprocal-space sum
	double alpha_;//the integral cutoff separating the real- and reciprocal-space sums
	double weight_;//weighting of real space calculations
	
	//unit cell
	std::vector<Eigen::Vector3d> R;//the lattice vectors to sum over
	std::vector<Eigen::Vector3d> K;//the reciprocal lattice vectors to sum over
	
	//electrostatics
	double eps_;//the dielectric constant of the outer material
public:
	Utility(){defaults();}
	Utility(double prec){init(prec);}
	Utility(const Utility& u);
	~Utility(){}
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Utility& u);
	
	//access
	double prec()const{return prec_;}
	double rMax()const{return rMax_;}
	double kMax()const{return kMax_;}
	double alpha()const{return alpha_;}
	const double& eps()const{return eps_;}
	double& eps(){return eps_;}
	const double& weight()const{return weight_;}
	double& weight(){return weight_;}
	
	//member functions
	void defaults();
	void clear(){defaults();}
	void init(double prec);
};

//**********************************************************************************************************
//Coulomb Class
//**********************************************************************************************************

class Coulomb: public Utility{
private:
	//calculation parameters
	double vSelfR_,vSelfK_,vSelfC_;//the self-interaction strength for an ion in a periodic lattice
	
	//unit cell
	std::vector<double> kAmp;//the reciprocal space sum amplitudes
public:
	Coulomb(){defaults();}
	Coulomb(Structure& struc, double prec){init(struc,prec);}
	~Coulomb(){}
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Coulomb& c);
	
	//access
	double phisR()const{return vSelfR_;}
	double phisK()const{return vSelfK_;}
	double phisC()const{return vSelfC_;}
	
	//member functions
	void defaults();
	void clear(){defaults();}
	void init(const Structure& struc, double prec);
	void init_alpha(const Structure& struc, double prec=0);
	
	//calculation - energy
	double energy(const Structure& struc)const;
	double energy_single(const Structure& struc);
	double energy_brute(const Structure& struc, int N)const;
	
	//calculation - potential
	double phi(const Structure& struc, const Eigen::Vector3d& dr)const;
	double phi(const Structure& struc, unsigned int nn)const;
	double phis()const;
	double potentialBrute(const Structure& struc, unsigned int n, int N)const;
	
	//calculation - electric field
	Eigen::Vector3d& efield(const Structure& struc, unsigned int n, Eigen::Vector3d&)const;
	Eigen::Vector3d& efieldBrute(const Structure& struc, unsigned int n, Eigen::Vector3d&, int N)const;
};

}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const Ewald3D::Utility& obj);
template <> unsigned int nbytes(const Ewald3D::Coulomb& obj);

//**********************************************
// packing
//**********************************************

template <> void pack(const Ewald3D::Utility& obj, char* arr);
template <> void pack(const Ewald3D::Coulomb& obj, char* arr);

//**********************************************
// unpacking
//**********************************************

template <> void unpack(Ewald3D::Utility& obj, const char* arr);
template <> void unpack(Ewald3D::Coulomb& obj, const char* arr);
	
}

#endif