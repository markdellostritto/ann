#pragma once
#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

//no bounds checking in Eigen
#define EIGEN_NO_DEBUG

//c++ libraries
#include <iosfwd>
//Eigen
#include <Eigen/Dense>
// ann - typedefs
#include "typedef.hpp"
// ann - cell
#include "cell.hpp"
// ann - serialize
#include "serialize.hpp"

#ifndef STRUC_PRINT_FUNC
#define STRUC_PRINT_FUNC 0
#endif

#ifndef STRUC_PRINT_STATUS
#define STRUC_PRINT_STATUS 0
#endif

//**********************************************************************************************
//AtomType
//**********************************************************************************************

struct AtomType{
	//==== data ====
	//coordinates
	bool frac;
	//basic properties
	bool name;
	bool an;
	bool type;
	bool index;
	//serial properties
	bool mass;
	bool charge;
	bool spin;
	//vector properties
	bool posn;
	bool vel;
	bool force;
	//nnp
	bool symm;
	//==== constructors/destructors ====
	AtomType(){defaults();}
	~AtomType(){}
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AtomType& atomT);
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
};

//**********************************************************************************************
//Thermo
//**********************************************************************************************

class Thermo{
protected:
	double energy_;//energy
	double ewald_;//ewald energy
	double temp_;//temperature
	double press_;//pressure
public:
	//==== constructors/destructors ====
	Thermo():energy_(0.0),ewald_(0.0),temp_(0.0),press_(0.0){}
	~Thermo(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Thermo& obj);
	
	//==== access ====
	double& energy(){return energy_;}
	const double& energy()const{return energy_;}
	double& ewald(){return ewald_;}
	const double& ewald()const{return ewald_;}
	double& temp(){return temp_;}
	const double& temp()const{return temp_;}
	double& press(){return press_;}
	const double& press()const{return press_;}
	
	//==== member functions ====
	void clear();
	
	//==== static functions ====
	static Thermo& make_super(const Eigen::Vector3i& s, const Thermo& thermo1, Thermo& thermo2);
};

//**********************************************************************************************
//AtomData
//**********************************************************************************************

class AtomData{
protected:
	//atom type
	AtomType atomType_;
	//number of atoms
	int nAtoms_;
	//basic properties
	std::vector<std::string> name_;//name
	std::vector<int>	an_;//atomic_number
	std::vector<int>	type_;//type
	std::vector<int>	index_;//index
	//serial properties
	std::vector<double>	mass_;//mass
	std::vector<double>	charge_;//charge
	std::vector<double>	spin_;//spin
	//vector properties
	std::vector<vec3d>	posn_;//position
	std::vector<vec3d>	vel_;//velocity
	std::vector<vec3d>	force_;//force
	//nnp
	std::vector<vecXd>	symm_;//symmetry function
public:
	//==== constructors/destructors ====
	AtomData(){}
	~AtomData(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AtomData& ad);
	
	//==== access - global ====
	const AtomType& atomType()const{return atomType_;}
	const int& nAtoms()const{return nAtoms_;}
	
	//==== access - vectors ====
	//basic properties
	std::vector<std::string>& name(){return name_;}
	const std::vector<std::string>& name()const{return name_;}
	std::vector<int>& an(){return an_;}
	const std::vector<int>& an()const{return an_;}
	std::vector<int>& type(){return type_;}
	const std::vector<int>& type()const{return type_;}
	std::vector<int>& index(){return index_;}
	const std::vector<int>& index()const{return index_;}
	//serial properties
	std::vector<double>& mass(){return mass_;}
	const std::vector<double>& mass()const{return mass_;}
	std::vector<double>& charge(){return charge_;}
	const std::vector<double>& charge()const{return charge_;}
	std::vector<double>& spin(){return spin_;}
	const std::vector<double>& spin()const{return spin_;}
	//vector properties
	std::vector<vec3d>& posn(){return posn_;}
	const std::vector<vec3d>& posn()const{return posn_;}
	std::vector<vec3d>& vel(){return vel_;}
	const std::vector<vec3d>& vel()const{return vel_;}
	std::vector<vec3d>& force(){return force_;}
	const std::vector<vec3d>& force()const{return force_;}
	//nnp
	std::vector<vecXd>& symm(){return symm_;}
	const std::vector<vecXd>& symm()const{return symm_;}
	
	//==== access - atoms ====
	//basic properties
	std::string& name(int i){return name_[i];}
	const std::string& name(int i)const{return name_[i];}
	int& an(int i){return an_[i];}
	const int& an(int i)const{return an_[i];}
	int& type(int i){return type_[i];}
	const int& type(int i)const{return type_[i];}
	int& index(int i){return index_[i];}
	const int& index(int i)const{return index_[i];}
	//serial properties
	double& mass(int i){return mass_[i];}
	const double& mass(int i)const{return mass_[i];}
	double& charge(int i){return charge_[i];}
	const double& charge(int i)const{return charge_[i];}
	double& spin(int i){return spin_[i];}
	const double& spin(int i)const{return spin_[i];}
	//vector properties
	vec3d& posn(int i){return posn_[i];}
	const vec3d& posn(int i)const{return posn_[i];}
	vec3d& vel(int i){return vel_[i];}
	const vec3d& vel(int i)const{return vel_[i];}
	vec3d& force(int i){return force_[i];}
	const vec3d& force(int i)const{return force_[i];}
	//nnp
	vecXd& symm(int i){return symm_[i];}
	const vecXd& symm(int i)const{return symm_[i];}
		
	//==== member functions ====
	void clear();
	void resize(int nAtoms, const AtomType& atomT);
	
	//==== static functions ====
	static AtomData& make_super(const Eigen::Vector3i& s, const AtomData& ad1, AtomData& ad2);
};

//**********************************************************************************************
//Structure
//**********************************************************************************************

class Structure: public Cell, public Thermo, public AtomData{
public:
	//==== constructors/destructors ====
	Structure(){}
	~Structure(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Structure& sim);
	
	//==== member functions ====
	void clear();
	
	//==== static functions ====
	static void write_binary(const Structure& struc, const char* file);
	static void read_binary(Structure& struc, const char* file);
	static Structure& make_super(const Eigen::Vector3i& s, const Structure& ad1, Structure& ad2);
};

//**********************************************************************************************
//AtomSpecies
//**********************************************************************************************

class AtomSpecies{
protected:
	//==== atomic data ====
	int nSpecies_;
	std::vector<std::string> species_;//the names of each species
	std::vector<int> nAtoms_;//the number of atoms of each species
	std::vector<int> offsets_;//the offsets for each species
public:
	//==== constructors/destructors ====
	AtomSpecies(){defaults();}
	AtomSpecies(const std::vector<std::string>& names, const std::vector<int>& nAtoms){resize(names,nAtoms);}
	AtomSpecies(const Structure& struc){resize(struc);}
	~AtomSpecies(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AtomSpecies& as);
	
	//==== access - number ====
	int nSpecies()const{return nSpecies_;}
	int nTot()const;
	
	//==== access - species ====
	std::vector<std::string>& species(){return species_;}
	const std::vector<std::string>& species()const{return species_;}
	std::string& species(int i){return species_[i];}
	const std::string& species(int i)const{return species_[i];}
	
	//==== access - numbers ====
	std::vector<int>& nAtoms(){return nAtoms_;}
	const std::vector<int>& nAtoms()const{return nAtoms_;}
	int& nAtoms(int i){return nAtoms_[i];}
	const int& nAtoms(int i)const{return nAtoms_[i];}
	
	//==== access - offsets ====
	std::vector<int>& offsets(){return offsets_;}
	const std::vector<int>& offsets()const{return offsets_;}
	int& offsets(int i){return offsets_[i];}
	const int& offsets(int i)const{return offsets_[i];}
	
	//==== static functions ====
	static int index_species(const std::string& str, const std::vector<std::string>& names);
	static int index_species(const char* str, const std::vector<std::string>& names);
	static std::vector<int>& read_atoms(const AtomSpecies& as, const char* str, std::vector<int>& ids);
	static int read_natoms(const char* str);
	static std::vector<int>& read_indices(const char* str, std::vector<int>& indices);
	static std::vector<std::string>& read_names(const char* str, std::vector<std::string>& names);
	static void set_species(const AtomSpecies& as, Structure& struc);
	
	//==== member functions ====
	void clear(){defaults();}
	void defaults();
	void resize(const std::vector<std::string>& names, const std::vector<int>& nAtoms);
	void resize(const Structure& struc);
	int index_species(const std::string& str)const{return index_species(str,species_);}
	int index_species(const char* str)const{return index_species(str,species_);}
	int index(int si, int ai)const{return offsets_[si]+ai;}
	
};

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const AtomType& obj);
	template <> int nbytes(const Thermo& obj);
	template <> int nbytes(const AtomData& obj);
	template <> int nbytes(const Structure& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const AtomType& obj, char* arr);
	template <> int pack(const Thermo& obj, char* arr);
	template <> int pack(const AtomData& obj, char* arr);
	template <> int pack(const Structure& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(AtomType& obj, const char* arr);
	template <> int unpack(Thermo& obj, const char* arr);
	template <> int unpack(AtomData& obj, const char* arr);
	template <> int unpack(Structure& obj, const char* arr);
	
}

#endif