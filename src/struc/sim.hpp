#pragma once
#ifndef SIM_HPP
#define SIM_HPP

//no bounds checking in Eigen
#define EIGEN_NO_DEBUG

//c++ libraries
#include <iosfwd>
//Eigen
#include <Eigen/Dense>
// struc
#include "struc/structure.hpp"
#include "struc/interval.hpp"
// mem 
#include "mem/serialize.hpp"
// string
#include "str/string.hpp"

#ifndef SIM_PRINT_FUNC
#define SIM_PRINT_FUNC 0
#endif

//**********************************************
// Simulation
//**********************************************

class Simulation{
private:
	std::string name_;
	double timestep_;
	int timesteps_;
	AtomType atomT_;
	std::vector<Structure> frames_;
public:
	//==== constructors/destructors ====
	Simulation(){defaults();}
	~Simulation(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Simulation& sim);
	
	//==== access ====
	std::string& name(){return name_;}
	const std::string& name()const{return name_;}
	double& timestep(){return timestep_;}
	const double& timestep()const{return timestep_;}
	int& timesteps(){return timesteps_;}
	const int& timesteps()const{return timesteps_;}
	AtomType& atomT(){return atomT_;}
	const AtomType& atomT()const{return atomT_;}
	Structure& frame(int i){return frames_[i];}
	const Structure& frame(int i)const{return frames_[i];}
	
	//==== member functions ====
	void defaults();
	void clear();
	void resize(int ts, int nAtoms, const AtomType& atomT);
	void resize(int ts);
	
	//==== static functions ====
	static void set_image(Simulation & sim);
	static void unwrap(Simulation & sim);
};


//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Simulation& sim);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Simulation& sim, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Simulation& sim, const char* arr);
	
}

#endif