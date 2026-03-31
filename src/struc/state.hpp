#pragma once
#ifndef STATE_HPP
#define STATE_HPP

// c++ libraries
#include <iosfwd>
// Eigen
#include <Eigen/Dense>
// mem
#include "mem/serialize.hpp"

//**********************************************************************************************
//State
//**********************************************************************************************

class State{
protected:
	//energy
	double etot_;//energy - total
	double ecoul_;//energy - coulomb
	double evdw_;//energy - vdw
	double erep_;//energy - repulsive
	double pe_;//energy - potential
	double ke_;//energy - kinetic
	//state
	double temp_;//temperature
	double press_;//pressure
	//charge
	double qtot_;//total charge
	//stress
	Eigen::Matrix3d stress_;
	//dipole
	Eigen::Vector3d mutot_;
	Eigen::Matrix3d atot_;
	//time
	double dt_;//timestep
	int t_;//time
public:
	//==== constructors/destructors ====
	State():
		etot_(0.0),ecoul_(0.0),evdw_(0.0),erep_(0.0),ke_(0.0),pe_(0.0),
		temp_(0.0),press_(0.0),qtot_(0.0),t_(0),dt_(0),
		stress_(Eigen::Matrix3d::Zero()),
		mutot_(Eigen::Vector3d::Zero()),atot_(Eigen::Matrix3d::Zero()){}
	~State(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const State& obj);
	
	//==== access ====
	//energy
	double& etot(){return etot_;}
	const double& etot()const{return etot_;}
	double& ecoul(){return ecoul_;}
	const double& ecoul()const{return ecoul_;}
	double& evdw(){return evdw_;}
	const double& evdw()const{return evdw_;}
	double& erep(){return erep_;}
	const double& erep()const{return erep_;}
	double& pe(){return pe_;}
	const double& pe()const{return pe_;}
	double& ke(){return ke_;}
	const double& ke()const{return ke_;}
	//state
	double& temp(){return temp_;}
	const double& temp()const{return temp_;}
	double& press(){return press_;}
	const double& press()const{return press_;}
	//charge
	double& qtot(){return qtot_;}
	const double& qtot()const{return qtot_;}
	//stress
	Eigen::Matrix3d& stress(){return stress_;}
	const Eigen::Matrix3d& stress()const{return stress_;}
	//polarizability
	Eigen::Vector3d& mutot(){return mutot_;}
	const Eigen::Vector3d& mutot()const{return mutot_;}
	Eigen::Matrix3d& atot(){return atot_;}
	const Eigen::Matrix3d& atot()const{return atot_;}
	//time
	double& dt(){return dt_;}
	const double& dt()const{return dt_;}
	int& t(){return t_;}
	const int& t()const{return t_;}
	
	//==== member functions ====
	void clear();
};

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const State& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const State& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(State& obj, const char* arr);
	
}

#endif