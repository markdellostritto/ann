#ifndef MONTE_CARLO_HPP
#define MONTE_CARLO_HPP

//c++ libraries
#include <iostream>
//c libraries
#include <cmath>
//math
#include "math/reduce.hpp"

/** MonteCarlo
Parent class for Monte Carlo algorithms
*/
class MonteCarlo{
protected:
	int nStep_;//the number of MC steps
	int nAccept_;//the number of accepted MC steps
public:
	//==== constructors/destructors ====
	MonteCarlo():nStep_(0),nAccept_(0){}
	~MonteCarlo(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const MonteCarlo& obj);
	
	//==== access ====
	int& nStep(){return nStep_;}
	const int& nStep()const{return nStep_;}
	int& nAccept(){return nAccept_;}
	const int& nAccept()const{return nAccept_;}
	
	//==== member functions ====
	void clear(){nStep_=0;nAccept_=0;}
	void reset(){nStep_=0;nAccept_=0;}
};

/** Demon
Class for Monte Carlo simulations in the microcanonical ensemble, where the energy is controlled.
This class is called "Demon" because the energy of the system is controlled by a external degree of 
freedom often referred to as "demon" which stores energy and transfers it to and from the physical system.
*/
class Demon: public MonteCarlo{
private:
	double energy_;//the energy of the demon
	Reduce<1> accum_;//accumulates the energy
public:
	//==== constructors/destructors ====
	Demon():energy_(0){}
	Demon(const Demon& d):energy_(d.energy()){}
	~Demon(){};
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Demon& obj);
	
	//==== access ====
	double& energy(){return energy_;}
	const double& energy()const{return energy_;}
	Reduce<1>& accum(){return accum_;}
	const Reduce<1>& accum()const{return accum_;}
	
	//==== member functions ====
	void clear();
	bool step(double dE);
	void stepT(double dE);
	void stepF();
	void resetAccumulator(int nBins, int cacheSize);
};

/** Metropolis
Class for Monte Carlo simulations in the macrocanonical ensemble, where the temperature is controlled.
This class implements the Metropolis-Hastings algorithm.
*/
class Metropolis: public MonteCarlo{
private:
	double k_;//boltzmann's constant
	double T_;//reservoir temperature
public:
	//==== constructors/destructors ====
	Metropolis():k_(-1),T_(-1){std::srand(time(NULL));}
	Metropolis(double k, double T):k_(k),T_(T){std::srand(time(NULL));}
	~Metropolis(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Metropolis& m);
	
	//access
	double& k(){return k_;}
	const double& k()const{return k_;}
	double& T(){return T_;}
	const double& T()const{return T_;}
	
	//member functions
	void clear();
	void stepT(){nAccept_++;nStep_++;}
	void stepF(){nStep_++;}
	bool step(double dE);
};

#endif