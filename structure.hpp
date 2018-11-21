#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

//c libraries
#include <cstdlib>
#include <stdexcept>
//c++ libraries
#include <iostream>
#include <string>
#include <type_traits>
//Eigen
#include <Eigen/Dense>
//#include "label.hpp"
#include "cell.hpp"
#include "string.hpp"
#include "ptable.hpp"

#ifndef DEBUG_STRUCTURE
#define DEBUG_STRUCTURE 0
#endif

//**********************************************************************************************
//AtomType
//**********************************************************************************************

struct AtomType{
	//data
	bool name;
	bool an;
	bool specie;
	bool index;
	bool mass;
	bool charge;
	bool posn;
	bool velocity;
	bool force;
	bool dipole;
	bool alpha;
	bool jzero;
	bool symm;
	bool neighlist;
	bool frac;
	//constructors/destructors
	AtomType(){defaults();};
	~AtomType(){};
	//operators
	friend std::ostream& operator<<(std::ostream& out, const AtomType& atomT);
	//member functions
	void defaults();
	void clear(){defaults();};
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
		GAUSSIAN,//Gaussian output file
		DFTB,//DFTB output files
		XYZ,//XYZ file
		CAR,//CAR file
		LAMMPS,//LAMMPS input,data,dump files
		GROMACS,//GROMACS trajectory files
		QE,//quantum espresso output files
		PROPHET,//PROPhet xml file
		XSF,//xcrysden format
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
	Interval(int b,int e,int s):beg(b),end(e),stride(s){};
	Interval():beg(0),end(0),stride(1){};
	~Interval(){};
	friend std::ostream& operator<<(std::ostream& out, const Interval& i);
	static Interval read(const char* str);
};

//**********************************************************************************************
//List Atomic
//**********************************************************************************************

template <class T>
class ListData{
private:
	unsigned int N_;//total number of objects
	T* list_;//the list of atoms
public:
	//macros
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW//just in case we are storing Eigen objects
	
	//constructors/destructors
	ListData(){init();};
	ListData(unsigned int N){init(); resize(N);};
	ListData(const ListData<T>& l);//deep copy
	~ListData();
	
	//operators
	ListData<T>& operator=(const ListData<T>& l);//deep copy
	
	//access
	//primitives
		unsigned int N()const{return N_;};
	//list access - operators
		inline T& operator[](unsigned int i){return list_[i];};
		inline const T& operator[](unsigned int i)const{return list_[i];};
	//list access - stored array
		const T* list()const{return list_;};
	
	//static member functions
	static unsigned int size(const ListData<T>& list){return sizeof(T)*list.N();};
	
	//member functions
	void init();
	void clear();
	bool empty()const{return (N_==0)?true:false;};
	void resize(unsigned int N);
	unsigned int size()const{return N_;};
};

//constructors/destructors

template <class T>
ListData<T>::ListData(const ListData<T>& l){
	if(DEBUG_STRUCTURE>0) std::cout<<"ListData<T>::ListData(const ListData<T>&):\n";
	init();
	resize(l.N());
	for(unsigned int i=0; i<size(); ++i) list_[i]=l[i];
}

template <class T>
ListData<T>::~ListData(){
	if(DEBUG_STRUCTURE>0) std::cout<<"ListData<T>::~ListData()\n";
	clear();
}

//operators

template <class T>
ListData<T>& ListData<T>::operator=(const ListData<T>& l){
	if(DEBUG_STRUCTURE>0) std::cout<<"ListData<T>::operator=(const ListData<T>&):\n";
	if(&l!=this){
		resize(l.N());
		for(unsigned int i=0; i<size(); ++i) list_[i]=l[i];
	}
	return *this;
}

//member functions

template <class T>
void ListData<T>::init(){
	if(DEBUG_STRUCTURE>0) std::cout<<"ListData<T>::init():\n";
	list_=NULL;
	N_=0;
}

template <class T>
void ListData<T>::clear(){
	if(DEBUG_STRUCTURE>0) std::cout<<"ListData<T>::clear():\n";
	if(list_!=NULL){delete[] list_; list_=NULL;}
	N_=0;
}

template <class T>
void ListData<T>::resize(unsigned int N){
	if(DEBUG_STRUCTURE>0) std::cout<<"ListData<T>::resize(unsigned int):\n";
	if(list_!=NULL){delete[] list_; list_=NULL;}
	N_=N;
	if(N_>0) list_=new T[N_];
}

//operators

template <class T>
bool operator==(const ListData<T>& l1, const ListData<T>& l2){
	return (l1.N()==l2.N());
}

template <class T>
bool operator!=(const ListData<T>& l1, const ListData<T>& l2){
	return !(l1.N()==l2.N());
}

//**********************************************************************************************
//Structure Interface
//**********************************************************************************************

class StructureI{
public:
	static const unsigned int D=3;
protected:
	//cell data
	Cell cell_;//unit cell
	//simulation data
	double energy_;//energy
	double temp_;//temperature
	//atomic data
	unsigned int nSpecies_;//number of atomic species
	unsigned int nAtomsT_;//total number of atoms
	std::vector<unsigned int> nAtoms_;//the number of atoms of each species
	std::vector<unsigned int> offsets_;
	std::vector<std::string> atomNames_;//the names of each species
public:
	//constructors/destructors
	StructureI(){defaults();};
	StructureI(const StructureI& arg);
	StructureI(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& atomNames){resize(nAtoms,atomNames);};
	~StructureI(){};
	
	//operators
	StructureI& operator=(const StructureI& arg);
	friend std::ostream& operator<<(std::ostream& out, const StructureI& struc);
	
	//access
	//cells
		Cell& cell(){return cell_;};
		const Cell& cell()const{return cell_;};
	//energy
		double& energy(){return energy_;};
		const double& energy()const{return energy_;};
	//temperature
		double& temp(){return temp_;};
		const double& temp()const{return temp_;};
	//atoms
		unsigned int nSpecies()const{return nSpecies_;};
		unsigned int nAtoms()const{return nAtomsT_;};
		const std::vector<unsigned int>& nAtomsVec()const{return nAtoms_;};
		unsigned int nAtoms(unsigned int i)const{return nAtoms_[i];};
		unsigned int offset(unsigned int i)const{return offsets_[i];};
		const std::string& atomNames(unsigned int i)const{return atomNames_[i];};
		const std::vector<std::string>& atomNames()const{return atomNames_;};
	
	//template functions
	template <class AtomT> unsigned int index(const AtomT& atom){return atom.index()+offsets_[atom.species()];};
	
	//member functions
	void defaults();
	void clear(){defaults();};
	void resize(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& atomNames);
	int speciesIndex(const std::string& str)const{return speciesIndex(str,atomNames_);};
	int speciesIndex(const char* str)const{return speciesIndex(str,atomNames_);};
	
	//static functions
	static int speciesIndex(const std::string& str, const std::vector<std::string>& atomNames);
	static int speciesIndex(const char* str, const std::vector<std::string>& atomNames);
};

//**********************************************************************************************
//Sim Atomic Storage
//**********************************************************************************************

class StructureS{
protected:
	//atom properties
	ListData<std::string>		name_;		//name
	ListData<unsigned int> 		an_;		//atomic_number
	ListData<unsigned int> 		specie_;	//specie
	ListData<unsigned int> 		index_;	//index
	ListData<double> 			mass_;		//mass
	ListData<double> 			charge_;	//charge
	ListData<Eigen::Vector3d>	posn_;		//position
	ListData<Eigen::Vector3d>	velocity_;	//velocity
	ListData<Eigen::Vector3d>	force_;	//force
	ListData<Eigen::Vector3d>	dipole_;	//dipole
	ListData<Eigen::Matrix3d>	alpha_;	//polarizability
	ListData<double> 			jzero_;	//idempotential
	ListData<Eigen::VectorXd>	symm_;		//symmetry_function
	ListData<std::vector<unsigned int> >	neighlist_;		//neighborlist
	
	//resizing functions
	void resizeAtomArrays(unsigned int nAtomsT, const AtomType& atomT);
	
public:
	//constructors/destructors
	StructureS(){};
	StructureS(const StructureS& struc);
	~StructureS(){};
	
	//operators
	StructureS& operator=(const StructureS& struc);
	friend std::ostream& operator<<(std::ostream& out, const StructureS& struc){};
	
	//access - properties
	//name
		const ListData<std::string>& name()const{return name_;};
	//atomic_number
		const ListData<unsigned int>& an()const{return an_;};
	//specie
		const ListData<unsigned int>& specie()const{return specie_;};
	//index
		const ListData<unsigned int>& index()const{return index_;};
	//mass
		const ListData<double>& mass()const{return mass_;};
	//charge
		const ListData<double>& charge()const{return charge_;};
	//position
		const ListData<Eigen::Vector3d>& posn()const{return posn_;};
	//velocity
		const ListData<Eigen::Vector3d>& velocity()const{return velocity_;};
	//force
		const ListData<Eigen::Vector3d>& force()const{return force_;};
	//dipole
		const ListData<Eigen::Vector3d>& dipole()const{return dipole_;};
	//polarizability
		const ListData<Eigen::Matrix3d>& alpha()const{return alpha_;};
	//idempotential
		const ListData<double>& jzero()const{return jzero_;};
	//symmetry_function
		const ListData<Eigen::VectorXd>& symm()const{return symm_;};
	//neighbor list
		const ListData<std::vector<unsigned int> >& neighlist()const{return neighlist_;};
	
	//member functions
	void clear();
	void resize(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& speciesNames, const AtomType& atomT);
};

//**********************************************************************************************
//Structure Atomic
//**********************************************************************************************

class Structure: public StructureI, public StructureS{
public:
	//constructors/destructors
	Structure(){};
	Structure(const Structure& sim):StructureI(sim),StructureS(sim){};
	Structure(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& atomNames, const AtomType& atomT){resize(nAtoms,atomNames,atomT);};
	~Structure(){};
	
	//operators
	Structure& operator=(const Structure& sim);
	friend std::ostream& operator<<(std::ostream& out, const Structure& sim);
	
	//member functions
	void clear();
	void resize(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& atomNames, const AtomType& atomT);
	
	//access - properties
	//name
		std::string& name(unsigned int i){return name_[i];};
		const std::string& name(unsigned int i)const{return name_[i];};
		std::string& name(unsigned int i, unsigned int j){return name_[offsets_[i]+j];};
		const std::string& name(unsigned int i, unsigned int j)const{return name_[offsets_[i]+j];};
	//atomic_number
		unsigned int& an(unsigned int i){return an_[i];};
		const unsigned int& an(unsigned int i)const{return an_[i];};
		unsigned int& an(unsigned int i, unsigned int j){return an_[offsets_[i]+j];};
		const unsigned int& an(unsigned int i, unsigned int j)const{return an_[offsets_[i]+j];};
	//specie
		unsigned int& specie(unsigned int i){return specie_[i];};
		const unsigned int& specie(unsigned int i)const{return specie_[i];};
		unsigned int& specie(unsigned int i, unsigned int j){return specie_[offsets_[i]+j];};
		const unsigned int& specie(unsigned int i, unsigned int j)const{return specie_[offsets_[i]+j];};
	//index
		unsigned int& index(unsigned int i){return index_[i];};
		const unsigned int& index(unsigned int i)const{return index_[i];};
		unsigned int& index(unsigned int i, unsigned int j){return index_[offsets_[i]+j];};
		const unsigned int& index(unsigned int i, unsigned int j)const{return index_[offsets_[i]+j];};
	//mass
		double& mass(unsigned int i){return mass_[i];};
		const double& mass(unsigned int i)const{return mass_[i];};
		double& mass(unsigned int i, unsigned int j){return mass_[offsets_[i]+j];};
		const double& mass(unsigned int i, unsigned int j)const{return mass_[offsets_[i]+j];};
	//charge
		double& charge(unsigned int i){return charge_[i];};
		const double& charge(unsigned int i)const{return charge_[i];};
		double& charge(unsigned int i, unsigned int j){return charge_[offsets_[i]+j];};
		const double& charge(unsigned int i, unsigned int j)const{return charge_[offsets_[i]+j];};
	//position
		Eigen::Vector3d& posn(unsigned int i){return posn_[i];};
		const Eigen::Vector3d& posn(unsigned int i)const{return posn_[i];};
		Eigen::Vector3d& posn(unsigned int i, unsigned int j){return posn_[offsets_[i]+j];};
		const Eigen::Vector3d& posn(unsigned int i, unsigned int j)const{return posn_[offsets_[i]+j];};
	//velocity
		Eigen::Vector3d& velocity(unsigned int i){return velocity_[i];};
		const Eigen::Vector3d& velocity(unsigned int i)const{return velocity_[i];};
		Eigen::Vector3d& velocity(unsigned int i, unsigned int j){return velocity_[offsets_[i]+j];};
		const Eigen::Vector3d& velocity(unsigned int i, unsigned int j)const{return velocity_[offsets_[i]+j];};
	//force
		Eigen::Vector3d& force(unsigned int i){return force_[i];};
		const Eigen::Vector3d& force(unsigned int i)const{return force_[i];};
		Eigen::Vector3d& force(unsigned int i, unsigned int j){return force_[offsets_[i]+j];};
		const Eigen::Vector3d& force(unsigned int i, unsigned int j)const{return force_[offsets_[i]+j];};
	//dipole
		Eigen::Vector3d& dipole(unsigned int i){return dipole_[i];};
		const Eigen::Vector3d& dipole(unsigned int i)const{return dipole_[i];};
		Eigen::Vector3d& dipole(unsigned int i, unsigned int j){return dipole_[offsets_[i]+j];};
		const Eigen::Vector3d& dipole(unsigned int i, unsigned int j)const{return dipole_[offsets_[i]+j];};
	//polarizability
		Eigen::Matrix3d& alpha(unsigned int i){return alpha_[i];};
		const Eigen::Matrix3d& alpha(unsigned int i)const{return alpha_[i];};
		Eigen::Matrix3d& alpha(unsigned int i, unsigned int j){return alpha_[offsets_[i]+j];};
		const Eigen::Matrix3d& alpha(unsigned int i, unsigned int j)const{return alpha_[offsets_[i]+j];};
	//idempotential
		double& jzero(unsigned int i){return jzero_[i];};
		const double& jzero(unsigned int i)const{return jzero_[i];};
		double& jzero(unsigned int i, unsigned int j){return jzero_[offsets_[i]+j];};
		const double& jzero(unsigned int i, unsigned int j)const{return jzero_[offsets_[i]+j];};
	//symmetry_function
		Eigen::VectorXd& symm(unsigned int i){return symm_[i];};
		const Eigen::VectorXd& symm(unsigned int i)const{return symm_[i];};
		Eigen::VectorXd& symm(unsigned int i, unsigned int j){return symm_[offsets_[i]+j];};
		const Eigen::VectorXd& symm(unsigned int i, unsigned int j)const{return symm_[offsets_[i]+j];};
	//neighbor_list
		std::vector<unsigned int>& neighlist(unsigned int i){return neighlist_[i];};
		const std::vector<unsigned int>& neighlist(unsigned int i)const{return neighlist_[i];};
		std::vector<unsigned int>& neighlist(unsigned int i, unsigned int j){return neighlist_[offsets_[i]+j];};
		const std::vector<unsigned int>& neighlist(unsigned int i, unsigned int j)const{return neighlist_[offsets_[i]+j];};
};

//**********************************************
// Simulation
//**********************************************

class Simulation{
private:
	std::string name_;
	unsigned int timestep_;
	unsigned int timesteps_;
	unsigned int beg_,end_,stride_;
	bool cell_fixed_;
	AtomType atomT_;
	std::vector<Structure> frames_;
public:
	//constructors/destructors
	Simulation(){defaults();};
	~Simulation(){};
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Simulation& sim);
	
	//access
	std::string& name(){return name_;};
	const std::string& name()const{return name_;};
	unsigned int& timestep(){return timestep_;};
	const unsigned int& timestep()const{return timestep_;};
	unsigned int& timesteps(){return timesteps_;};
	const unsigned int& timesteps()const{return timesteps_;};
	unsigned int& beg(){return beg_;};
	const unsigned int& beg()const{return beg_;};
	unsigned int& end(){return end_;};
	const unsigned int& end()const{return end_;};
	unsigned int& stride(){return stride_;};
	const unsigned int& stride()const{return stride_;};
	bool& cell_fixed(){return cell_fixed_;};
	const bool& cell_fixed()const{return cell_fixed_;};
	AtomType& atomT(){return atomT_;};
	const AtomType& atomT()const{return atomT_;};
	Structure& frame(unsigned int i){return frames_[i];};
	const Structure& frame(unsigned int i)const{return frames_[i];};
	
	//member functions
	void defaults();
	void clear();
	void resize(unsigned int ts, const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& names, const AtomType& atomT);
};

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const ListData<unsigned int>& obj);
	template <> unsigned int nbytes(const ListData<double>& obj);
	template <> unsigned int nbytes(const ListData<std::string>& obj);
	template <> unsigned int nbytes(const ListData<Eigen::Vector3d>& obj);
	template <> unsigned int nbytes(const ListData<Eigen::VectorXd>& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const ListData<unsigned int>& obj, char* arr);
	template <> void pack(const ListData<double>& obj, char* arr);
	template <> void pack(const ListData<std::string>& obj, char* arr);
	template <> void pack(const ListData<Eigen::Vector3d>& obj, char* arr);
	template <> void pack(const ListData<Eigen::VectorXd>& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(ListData<unsigned int>& obj, const char* arr);
	template <> void unpack(ListData<double>& obj, const char* arr);
	template <> void unpack(ListData<std::string>& obj, const char* arr);
	template <> void unpack(ListData<Eigen::Vector3d>& obj, const char* arr);
	template <> void unpack(ListData<Eigen::VectorXd>& obj, const char* arr);
	
}

#endif