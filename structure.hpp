#pragma once
#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

//no bounds checking in Eigen
#define EIGEN_NO_DEBUG

//c libraries
#include <cstdlib>
#include <stdexcept>
//c++ libraries
#include <iostream>
#include <string>
//Eigen
#include <Eigen/Dense>
// ann library - typedefs
#include "typedef.hpp"
// ann library - cell
#include "cell.hpp"
// ann library - strings
#include "string.hpp"
// ann library - chemistry
#include "ptable.hpp"

#ifndef DEBUG_STRUCTURE
#define DEBUG_STRUCTURE 0
#endif

//**********************************************************************************************
//AtomType
//**********************************************************************************************

struct AtomType{
	//==== data ====
	bool name;
	bool an;
	bool type;
	bool index;
	bool mass;
	bool charge;
	bool posn;
	bool force;
	bool symm;
	bool frac;
	//==== constructors/destructors ====
	AtomType(){defaults();}
	~AtomType(){}
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AtomType& atomT);
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
	unsigned int nbytes()const;
};

//**********************************************************************************************
//FILE_FORMAT struct
//**********************************************************************************************

struct FILE_FORMAT{
	enum type{
		UNKNOWN,//Unknown format
		XDATCAR,//VASP xdatcar file
		POSCAR,//VASP poscar file
		OUTCAR,//VASP outcar file
		VASP_XML,//VASP XML file
		XYZ,//XYZ file
		LAMMPS,//LAMMPS input,data,dump files
		QE,//quantum espresso output files
		AME//ame format
	};
	static FILE_FORMAT::type read(const std::string& str);
};

std::ostream& operator<<(std::ostream& out, FILE_FORMAT::type& format);

//**********************************************************************************************
//Interval
//**********************************************************************************************

struct Interval{
	int beg,end,stride;
	Interval(int b,int e,int s):beg(b),end(e),stride(s){}
	Interval():beg(0),end(0),stride(1){}
	~Interval(){}
	friend std::ostream& operator<<(std::ostream& out, const Interval& i);
	static Interval read(const char* str);
};

//**********************************************************************************************
//AtomList
//**********************************************************************************************

class AtomList{
protected:
	//==== atomic data ====
	uint nSpecies_;//number of atomic species
	uint nAtomsT_;//total number of atoms
	std::vector<uint> nAtoms_;//the number of atoms of each species
	std::vector<uint> offsets_;//the offsets for each species
	std::vector<std::string> species_;//the names of each species
public:
	//==== constructors/destructors ====
	AtomList(){defaults();}
	~AtomList(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AtomList& sim);
	
	//==== access ====
	uint nSpecies()const{return nSpecies_;}
	uint nAtoms()const{return nAtomsT_;}
	const std::vector<uint>& nAtomsV()const{return nAtoms_;}
	uint nAtoms(uint i)const{return nAtoms_[i];}
	const std::vector<uint>& offsets()const{return offsets_;}
	uint offsets(uint i)const{return offsets_[i];}
	const std::vector<std::string>& speciesV()const{return species_;}
	const std::string& species(uint i)const{return species_[i];}
	const std::vector<std::string>& species()const{return species_;}
	
	//==== member functions ====
	void clear(){defaults();}
	void defaults();
	void resize(const std::vector<uint>& nAtoms, const std::vector<std::string>& atomNames);
	int speciesIndex(const std::string& str)const{return speciesIndex(str,species_);}
	int speciesIndex(const char* str)const{return speciesIndex(str,species_);}
	int id(uint specie, unsigned index)const{return offsets_[specie]+index;}
	
	//==== static functions ====
	static int speciesIndex(const std::string& str, const std::vector<std::string>& atomNames);
	static int speciesIndex(const char* str, const std::vector<std::string>& atomNames);
	static std::vector<uint>& read_atoms(const char* str, std::vector<uint>& ids, const AtomList& strucI);
	static uint read_natoms(const char* str);
	static std::vector<int>& read_indices(const char* str, std::vector<int>& indices);
	static std::vector<std::string>& read_names(const char* str, std::vector<std::string>& names);
};

//**********************************************************************************************
//Thermo
//**********************************************************************************************

class Thermo{
protected:
	double energy_;//energy
	double temp_;//temperature
public:
	//==== constructors/destructors ====
	Thermo():energy_(0.0),temp_(0.0){}
	~Thermo(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Thermo& obj);
	
	//==== access ====
	double& energy(){return energy_;}
	const double& energy()const{return energy_;}
	double& temp(){return temp_;}
	const double& temp()const{return temp_;}
};

//**********************************************************************************************
//AtomData
//**********************************************************************************************

class AtomData{
protected:
	//atom properties
	std::vector<std::string> name_;//name
	std::vector<ushort> an_;//atomic_number
	std::vector<ushort> type_;//type
	std::vector<uint>   index_;//index
	std::vector<double> mass_;//mass
	std::vector<double> charge_;//charge
	std::vector<vec3d>	posn_;//position
	std::vector<vec3d>	force_;//force
	std::vector<vecXd>	symm_;//symmetry function
public:
	//==== constructors/destructors ====
	AtomData(){}
	~AtomData(){}
	
	//==== access - vectors ====
	const std::vector<std::string>& name()const{return name_;}
	const std::vector<ushort>& an()const{return an_;}
	const std::vector<ushort>& type()const{return type_;}
	const std::vector<uint>& index()const{return index_;}
	const std::vector<double>& mass()const{return mass_;}
	const std::vector<double>& charge()const{return charge_;}
	const std::vector<vec3d>& posn()const{return posn_;}
	const std::vector<vec3d>& force()const{return force_;}
	const std::vector<vecXd>& symm()const{return symm_;}
		
	//==== member functions ====
	void clear();
	void resize(uint nAtoms, const AtomType& atomT);
	AtomType atomType()const;
};

//**********************************************************************************************
//Structure
//**********************************************************************************************

class Structure: public AtomList, public Cell, public AtomData, public Thermo{
public:
	//==== constructors/destructors ====
	Structure(){}
	Structure(const std::vector<uint>& nAtoms, const std::vector<std::string>& atomNames, const AtomType& atomT){resize(nAtoms,atomNames,atomT);}
	~Structure(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Structure& sim);
	
	//==== member functions ====
	void clear();
	void resize(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& atomNames, const AtomType& atomT);
	
	//==== access - atoms ====
	std::string& name(uint i){return name_[i];}
	const std::string& name(uint i)const{return name_[i];}
	ushort& an(uint i){return an_[i];}
	const ushort& an(uint i)const{return an_[i];}
	ushort& type(uint i){return type_[i];}
	const ushort& type(uint i)const{return type_[i];}
	uint& index(uint i){return index_[i];}
	const uint& index(uint i)const{return index_[i];}
	double& mass(uint i){return mass_[i];}
	const double& mass(uint i)const{return mass_[i];}
	double& charge(uint i){return charge_[i];}
	const double& charge(uint i)const{return charge_[i];}
	vec3d& posn(uint i){return posn_[i];}
	const vec3d& posn(uint i)const{return posn_[i];}
	vec3d& force(uint i){return force_[i];}
	const vec3d& force(uint i)const{return force_[i];}
	vecXd& symm(uint i){return symm_[i];}
	const vecXd& symm(uint i)const{return symm_[i];}
	
	//==== access - properties ====
	std::string& name(uint i, uint j){return name_[offsets_[i]+j];}
	const std::string& name(uint i, uint j)const{return name_[offsets_[i]+j];}
	ushort& an(uint i, uint j){return an_[offsets_[i]+j];}
	const ushort& an(uint i, uint j)const{return an_[offsets_[i]+j];}
	ushort& type(uint i, uint j){return type_[offsets_[i]+j];}
	const ushort& type(uint i, uint j)const{return type_[offsets_[i]+j];}
	uint& index(uint i, uint j){return index_[offsets_[i]+j];}
	const uint& index(uint i, uint j)const{return index_[offsets_[i]+j];}
	double& mass(uint i, uint j){return mass_[offsets_[i]+j];}
	const double& mass(uint i, uint j)const{return mass_[offsets_[i]+j];}
	double& charge(uint i, uint j){return charge_[offsets_[i]+j];}
	const double& charge(uint i, uint j)const{return charge_[offsets_[i]+j];}
	vec3d& posn(uint i, uint j){return posn_[offsets_[i]+j];}
	const vec3d& posn(uint i, uint j)const{return posn_[offsets_[i]+j];}
	vec3d& force(uint i, uint j){return force_[offsets_[i]+j];}
	const vec3d& force(uint i, uint j)const{return force_[offsets_[i]+j];}
	vecXd& symm(uint i, uint j){return symm_[offsets_[i]+j];}
	const vecXd& symm(uint i, uint j)const{return symm_[offsets_[i]+j];}
};

//**********************************************
// Simulation
//**********************************************

class Simulation{
private:
	std::string name_;
	double timestep_;
	uint timesteps_;
	uint beg_,end_,stride_;
	bool cell_fixed_;
	AtomType atomT_;
	std::vector<Structure> frames_;
public:
	//==== constructors/destructors ====
	Simulation(){defaults();}
	~Simulation(){};
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Simulation& sim);
	
	//==== access ====
	std::string& name(){return name_;}
	const std::string& name()const{return name_;}
	double& timestep(){return timestep_;}
	const double& timestep()const{return timestep_;}
	uint& timesteps(){return timesteps_;}
	const uint& timesteps()const{return timesteps_;}
	uint& beg(){return beg_;}
	const uint& beg()const{return beg_;}
	uint& end(){return end_;}
	const uint& end()const{return end_;}
	uint& stride(){return stride_;}
	const uint& stride()const{return stride_;}
	bool& cell_fixed(){return cell_fixed_;}
	const bool& cell_fixed()const{return cell_fixed_;}
	AtomType& atomT(){return atomT_;}
	const AtomType& atomT()const{return atomT_;}
	Structure& frame(uint i){return frames_[i];}
	const Structure& frame(uint i)const{return frames_[i];}
	
	//==== member functions ====
	void defaults();
	void clear();
	void resize(uint ts, const std::vector<uint>& nAtoms, const std::vector<std::string>& names, const AtomType& atomT);
	void resize(uint ts);
};

#endif