#pragma once
#ifndef SIM_HPP
#define SIM_HPP

//no bounds checking in Eigen
#define EIGEN_NO_DEBUG

//c++ libraries
#include <iosfwd>
//Eigen
#include <Eigen/Dense>
// ann - cell
#include "structure.hpp"
// ann - serialize
#include "serialize.hpp"
// ann - string
#include "string.hpp"

#ifndef DEBUG_SIM
#define DEBUG_SIM 0
#endif

//**********************************************************************************************
//Interval
//**********************************************************************************************

struct Interval{
	int beg,end,stride;
	Interval(int b,int e,int s):beg(b),end(e),stride(s){}
	Interval():beg(1),end(-1),stride(1){}
	~Interval(){}
	friend std::ostream& operator<<(std::ostream& out, const Interval& i);
	static Interval read(const char* str);
	static Interval split(const Interval& interval, int rank, int nproc);
};

//**********************************************
// Simulation
//**********************************************

class Simulation{
private:
	std::string name_;
	double timestep_;
	int timesteps_;
	int beg_,end_,stride_;
	bool cell_fixed_;
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
	int& beg(){return beg_;}
	const int& beg()const{return beg_;}
	int& end(){return end_;}
	const int& end()const{return end_;}
	int& stride(){return stride_;}
	const int& stride()const{return stride_;}
	bool& cell_fixed(){return cell_fixed_;}
	const bool& cell_fixed()const{return cell_fixed_;}
	AtomType& atomT(){return atomT_;}
	const AtomType& atomT()const{return atomT_;}
	Structure& frame(int i){return frames_[i];}
	const Structure& frame(int i)const{return frames_[i];}
	
	//==== member functions ====
	void defaults();
	void clear();
	void resize(int ts, int nAtoms, const AtomType& atomT);
	void resize(int ts);
};


//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Interval& obj);
	template <> int nbytes(const Simulation& sim);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Interval& obj, char* arr);
	template <> int pack(const Simulation& sim, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Interval& obj, const char* arr);
	template <> int unpack(Simulation& sim, const char* arr);
	
}

#endif