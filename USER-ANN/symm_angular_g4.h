#pragma once
#ifndef SYMM_ANGULAR_G4_HPP
#define SYMM_ANGULAR_G4_HPP

// c++ libaries
#include <iosfwd>
// ann - symm - angular
#include "symm_angular.h"
// ann - serialization
#include "serialize.h"

//*****************************************
// PHIA - G4 - Behler
//*****************************************

struct PhiA_G4: public PhiA{
	//==== function parameters ====
	double eta;//radial exponential width
	double zeta;//angular exponential width
	int lambda;//sign of cosine window
	//==== constructors/destructors ====
	PhiA_G4():PhiA(),eta(0.0),zeta(0.0),lambda(0.0){}
	PhiA_G4(double eta_, double zeta_, int lambda_):PhiA(),eta(eta_),zeta(zeta_),lambda(lambda_){}
	//==== member functions - evaluation ====
	double val(double cos, const double r[3], const double c[3])const;
	double dist(const double r[3], const double c[3])const;
	double angle(double cos)const;
	double grad_angle(double cos)const;
	double grad_dist_0(const double r[3], const double c[3], double gij)const;
	double grad_dist_1(const double r[3], const double c[3], double gik)const;
	double grad_dist_2(const double r[3], const double c[3], double gjk)const;
	void compute_angle(double cos, double& val, double& grad)const;
	void compute_dist(const double r[3], const double c[3], const double g[3], double& dist, double* gradd)const;
};
//==== operators ====
std::ostream& operator<<(std::ostream& out, const PhiA_G4& f);
bool operator==(const PhiA_G4& phia1, const PhiA_G4& phia2);
inline bool operator!=(const PhiA_G4& phia1, const PhiA_G4& phia2){return !(phia1==phia2);}

//*****************************************
// PHIA - G4 - Behler - serialization
//*****************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const PhiA_G4& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const PhiA_G4& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(PhiA_G4& obj, const char* arr);
	
}

/* References:
Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050.
*/

#endif
