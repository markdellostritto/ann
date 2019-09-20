#pragma once
#ifndef SYMM_ANGULAR_G4_HPP
#define SYMM_ANGULAR_G4_HPP

// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif defined __ICC || defined __INTEL_COMPILER
#include <mathimf.h> //intel math library
#endif
// c++ libaries
#include <iostream>
// local libraries
#include "symm_angular.hpp"
// local libraries - serialization
#include "serialize.hpp"

//Behler G4
struct PhiA_G4: public PhiA{
	double eta;//radial exponential width
	double zeta;//angular exponential width
	int lambda;//sign of cosine window
	PhiA_G4():PhiA(),eta(0.0),zeta(0.0),lambda(0.0){};
	PhiA_G4(double eta_, double zeta_, int lambda_):PhiA(),eta(eta_),zeta(zeta_),lambda(lambda_){};
	double val(double cos, const double r[3], const double c[3])const noexcept final;
	double dist(const double r[3], const double c[3])const noexcept final;
	double angle(double cos)const noexcept final;
	double grad_angle(double cos)const noexcept final;
	double grad_dist_0(const double r[3], const double c[3], double gij)const noexcept final;
	double grad_dist_1(const double r[3], const double c[3], double gik)const noexcept final;
	double grad_dist_2(const double r[3], const double c[3], double gjk)const noexcept final;
	void compute_angle(double cos, double& val, double& grad)const noexcept final;
	void compute_dist(const double r[3], const double c[3], const double g[3], double& dist, double* gradd)const noexcept final;
};
std::ostream& operator<<(std::ostream& out, const PhiA_G4& f);
bool operator==(const PhiA_G4& phia1, const PhiA_G4& phia2);
inline bool operator!=(const PhiA_G4& phia1, const PhiA_G4& phia2){return !(phia1==phia2);};

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const PhiA_G4& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const PhiA_G4& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(PhiA_G4& obj, const char* arr);
	
}

/* References:
Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050.
*/

#endif