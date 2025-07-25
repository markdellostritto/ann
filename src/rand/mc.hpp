#ifndef MC_HPP
#define MC_HPP

// c++ libraries
#include <iostream>
// Eigen
#include <Eigen/Dense>
// rand
#include "rand/rng.hpp"

namespace mc{
	
//****************************************************
// Atom
//****************************************************

/**
* Class Atom
* This is a single "particle" in a Monte-Carlo ensemble.
* This can also be thought of a "walker" in a random walk.
* This class stores the position of the walker and any other
* necessary attributes for a given sampling approach.
*/
class Atom{
private:
	Eigen::VectorXd x_;//position
public:
	//==== constructor/destructor ====
	Atom(){}
	Atom(int d):x_(Eigen::VectorXd::Zero(d)){}
	~Atom(){}
	
	//==== access ====
	//dimension
	int d()const{return x_.size();}
	int dim()const{return x_.size();}
	//position
	Eigen::VectorXd& x(){return x_;}
	const Eigen::VectorXd& x()const{return x_;}
	double& x(int i){return x_[i];}
	const double& x(int i)const{return x_[i];}
};

//****************************************************
// Ensemble
//****************************************************

/**
* Class Ensemble
* Stores N d-dimensional "atoms" or "walkers" used in a random
* walk method for sampling from a probability distribution.
*/
class Ensemble{
private:
	int d_;//dimension
	std::vector<Atom> atoms_;//atoms
public:
	//==== constructor/destructor ====
	Ensemble(){}
	Ensemble(int n, int d){resize(n,d);}
	~Ensemble(){}
	
	//==== access ====
	const int& d()const{return d_;}
	const int& dim()const{return d_;}
	int size()const{return atoms_.size();}
	const std::vector<Atom>& atoms()const{return atoms_;}
	Atom& atom(int i){return atoms_[i];}
	const Atom& atom(int i)const{return atoms_[i];}
	
	//==== member functions ====
	void resize(int n, int d);
};

//****************************************************
// Metropolis
//****************************************************

/**
* Class Metropolis
* Implements the metroplis method for using a random walk
* with importance sampling to generate random variates from
* an arbitrary probability distribution.
* Given an ensemble with N walkers, any number of steps may
* be taken to initialize and then sample from a given distribution.
*/
class Metropolis{
private:
	//random walk
	int nstep_;//number of steps taken
	int nattempt_;//number of steps attemted
	int naccept_;//number of steps accepted
	double delta_;//step size
	// uniform variate generation
	int seed_;//random number seed
	std::mt19937 rng_;//random integer generator
	rng::dist::Uniform urng_;//uniform real number generator
	//target probability distribution
	std::function<double(const Eigen::VectorXd&)> rho_;//probability density
public:
	//==== constructor/destructor ====
	Metropolis():delta_(0.0){reset();}
	~Metropolis(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Metropolis& intg);
	
	//==== access ====
	//random walk
	const int& nstep()const{return nstep_;}
	const int& nattempt()const{return nattempt_;}
	const int& naccept()const{return naccept_;}
	double& delta(){return delta_;}
	const double& delta()const{return delta_;}
	// uniform variate generation
	int& seed(){return seed_;}
	const int& seed()const{return seed_;}
	//target probability distribution
	std::function<double(const Eigen::VectorXd&)>& rho(){return rho_;}
	const std::function<double(const Eigen::VectorXd&)>& rho()const{return rho_;}
	
	//==== member functions ====
	void reset();
	void init();
	void step(Ensemble& ensemble);
};

}

#endif