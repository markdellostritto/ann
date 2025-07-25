#ifndef DRUDE_HPP
#define DRUDE_HPP

//c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
//c++ libraries
#include <vector>
// eigen
#include <Eigen/Dense>
// struc
#include "struc/structure.hpp"
#include "struc/verlet.hpp"
// torch
#include "torch/kspace_coul.hpp"
// chem
#include "chem/units.hpp"

#ifndef DRUDE_PRINT_FUNC
#define DRUDE_PRINT_FUNC 0
#endif

#ifndef DRUDE_PRINT_STATUS
#define DRUDE_PRINT_STATUS 0
#endif

#ifndef DRUDE_PRINT_DATA
#define DRUDE_PRINT_DATA 0
#endif

class Drude{
private:
	//parameters - global
	double a_;
	double eps_;
	double prec_;
	double rc_,rc2_;
	double dt_;
	//parameters - atomic
	int ntypes_;
	Eigen::VectorXd alpha_;
	Eigen::VectorXd k_;
	Eigen::MatrixXd aij_;
	//energy
	double energyK_;
	double energyR_;
	double energyS_;
	//kspace
	KSpace::Coul coul_;
	//drude particles
	int nAtom_;//number of normal atoms
	int nDrude_;//number of drude particles
	std::vector<int> dIndex_;
	verlet::List vlist_;
	Structure struc_;
public:
	//==== constructors/destructors ====
	Drude():eps_(1.0),prec_(1.0e-6){}
	~Drude(){}
	
	//==== operators ====
	//parameters - global
	double& a(){return a_;}
	const double& a()const{return a_;}
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& prec(){return prec_;}
	const double& prec()const{return prec_;}
	double& dt(){return dt_;}
	const double& dt()const{return dt_;}
	//parameters - atomic
	double& alpha(int i){return alpha_[i];}
	const double& alpha(int i)const{return alpha_[i];}
	double& k(int i){return k_[i];}
	const double& k(int i)const{return k_[i];}
	const double& aij(int i, int j)const{return aij_(i,j);}
	//energy
	const double& energyS()const{return energyS_;}
	const double& energyR()const{return energyR_;}
	const double& energyK()const{return energyK_;}
	//kspace
	KSpace::Coul& coul(){return coul_;}
	const KSpace::Coul& coul()const{return coul_;}
	//drude particles
	const int& nAtom()const{return nAtom_;}
	const int& nDrude()const{return nDrude_;}
	const Structure& struc()const{return struc_;}
	
	//==== access ====
	void load(double rc, const Structure& struc);
	void init(const Eigen::VectorXd& alpha);
	
	//==== member functions ====
	double compute_lr();
	double compute_sr();
	void quickmin();
};

#endif