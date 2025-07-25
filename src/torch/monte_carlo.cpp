// torch
#include "torch/monte_carlo.hpp"

//*****************************************************************
//MonteCarlo Class
//*****************************************************************

std::ostream& operator<<(std::ostream& out, const MonteCarlo& obj){
	return out<<"nstep "<<obj.nStep_<<" nacc "<<obj.nAccept_;
}

//*****************************************************************
//Demon Class
//*****************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Demon& obj){
	return out<<static_cast<const MonteCarlo&>(obj)<<" ener "<<obj.energy_;
}

//==== member functions ====

void Demon::clear(){
	static_cast<MonteCarlo*>(this)->clear();
	energy_=0;
	accum_.clear();
}

bool Demon::step(double dE){
	bool accept=false;
	if(dE<=energy_){
		energy_-=dE;
		accum_.push(energy_);
		nStep_++;
		nAccept_++;
		accept=true;
	} else {
		nStep_++;
		accum_.push(energy_);
		accept=false;
	}
	return accept;
}

void Demon::stepT(double dE){
	energy_-=dE;
	accum_.push(energy_);
	nStep_++;
	nAccept_++;
}

void Demon::stepF(){
	nAccept_++;
	accum_.push(energy_);
}

//*****************************************************************
//Metropolis Class
//*****************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Metropolis& obj){
	return out<<static_cast<const MonteCarlo&>(obj)<<" k "<<obj.k_<<" T "<<obj.T_;
}

//==== member functions ====

void Metropolis::clear(){
	static_cast<MonteCarlo*>(this)->clear();
	k_=-1;
	T_=-1;
}

bool Metropolis::step(double dE){
	nStep_++;
	bool accept=false;
	if(dE<0.0){
		nAccept_++;
		accept=true;
	} else {
		const double AR=std::min(1.0,std::exp(-dE/(k_*T_)));
		const double u=((double)std::rand())/RAND_MAX;
		if(u<AR){
			nAccept_++;
			accept=true;
		}
	}
	return accept;
}