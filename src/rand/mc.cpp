// c++ libraries
#include <stdexcept>
#include <algorithm>
#include <chrono>
// rand
#include "rand/mc.hpp"

namespace mc{

//****************************************************
// Ensemble
//****************************************************

void Ensemble::resize(int n, int d){
	if(d<=0) throw std::invalid_argument("mc::Ensemble::resize(int,int): Invalid dimension.");
	if(n<0) throw std::invalid_argument("mc::Ensemble::resize(int,int): Invalid ensemble size.");
	else if(n==0) atoms_.clear();
	else {
		atoms_.resize(n,Atom(d));
	}
	d_=d;
}

//****************************************************
// Metropolis
//****************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Metropolis& metr){
	return out<<"seed "<<metr.seed_<<" delta "<<metr.delta_;
}

//==== member functions ====

/**
* reset the step count and random seed
*/
void Metropolis::reset(){
	//random walk
	nstep_=0;
	naccept_=0;
	nattempt_=0;
	// uniform variate generation
	seed_=-1;
}

/**
* initialiaze the uniform variate generation
*/
void Metropolis::init(){
	int seed=seed_;
	if(seed_<0) seed=std::chrono::system_clock::now().time_since_epoch().count();
	rng_=std::mt19937(seed);
	urng_=rng::dist::Uniform(0.0,1.0);
}

/**
* step the ensemble forward
* @param ensemble - the random walk ensemble
* Step the ensemble forward one atom at a time by generating a new position 
* and use importance sampling to determine whether the step should be accepted.
* If the step is not accepted, the atom remains at its current position.
*/
void Metropolis::step(Ensemble& ensemble){
	const int d=ensemble.d();
	Eigen::VectorXd y=Eigen::VectorXd::Zero(d);
	
	for(int i=0; i<ensemble.size(); ++i){
		const Eigen::VectorXd& x=ensemble.atom(i).x();
		y=x; y.noalias()+=delta_*Eigen::VectorXd::Random(d);
		const double a=std::min(rho_(y)/rho_(x),1.0);
		const double r=urng_.rand(rng_);
		if(a>=r){
			ensemble.atom(i).x()=y;
			naccept_++;
		}
		nattempt_++;
	}
	nstep_++;
	
}

}