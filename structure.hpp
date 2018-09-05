#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

// c libraries
#include <cstdlib>
// c++ libraries
#include <iostream>
#include <stdexcept>
#include <string>
// eigen libraries
#include <Eigen/Dense>
// string
#include "string.hpp"
// cell
#include "cell.hpp"
// atom properties
#include "property.hpp"
#include "ptable.hpp"

#ifndef DEBUG_STRUCTURE
#define DEBUG_STRUCTURE 0
#endif

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
		XSF//xcrysden format
	};
	static FILE_FORMAT::type load(const std::string& str);
};

std::ostream& operator<<(std::ostream& out, FILE_FORMAT::type& format);

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
		T& operator[](unsigned int i){return list_[i];};
		const T& operator[](unsigned int i)const{return list_[i];};
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
	init(); resize(l.N());
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
	ListData<T> templist(l);
	resize(templist.N());
	for(unsigned int i=0; i<size(); ++i) list_[i]=templist[i];
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
	if(list_!=NULL) delete[] list_;
	list_=NULL;
	N_=0;
}

template <class T>
void ListData<T>::resize(unsigned int N){
	if(DEBUG_STRUCTURE>0) std::cout<<"ListData<T>::resize(unsigned int):\n";
	try{
		if(list_!=NULL) delete[] list_;
		N_=N;
		list_=new T[N_];
	}catch(std::exception& e){
		clear();
		throw e;
	}
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
// Structure Interface
//**********************************************************************************************

class StructureI{
public:
	static const unsigned int D=3;
protected:
	//cell data
	Cell cell_;//unit cell
	bool cellFixed_;//whether the cell is static
	//simulation data
	std::string system_;//the name of the simulation
	bool periodic_;//whether the simulation is periodic or not
	double timestep_;//the length in picoseconds of each timestep
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
	friend std::ostream& operator<<(std::ostream& out, const StructureI& sim);
	
	//access
	//name
		std::string& system(){return system_;};
		const std::string& system()const{return system_;};
	//timestep
		double& timestep(){return timestep_;};
		const double& timestep()const{return timestep_;};
	//periodicity
		bool& periodic(){return periodic_;};
		const bool& periodic()const{return periodic_;};
	//cells
		Cell& cell(){return cell_;};
		const Cell& cell()const{return cell_;};
		bool& cellFixed(){return cellFixed_;};
		const bool& cellFixed()const{return cellFixed_;};
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
// Structure Storage
//**********************************************************************************************

template <class AtomT>
class StructureS{
protected:
	//the atoms of the simulation
	ListData<AtomT> atoms_;
	
	//atom properties
	ListData<std::string> atom_name_;//names
	ListData<unsigned int> atom_an_;//atomic numbers
	ListData<double> atom_mass_;//masses
	ListData<unsigned int> atom_specie_;//species 
	ListData<unsigned int> atom_index_;//indices
	ListData<Eigen::Vector3d> atom_posn_;//positions
	ListData<Eigen::Vector3d> atom_force_;//forces
	ListData<Eigen::Vector3d> atom_velocity_;//velocities
	ListData<double> atom_charge_;//charges
	ListData<Eigen::VectorXd> atom_symm_;//symmetry functions
	
	//resizing functions
	void resizeAtomArrays(unsigned int nAtomsT);
	void resizeAtomName(std::true_type, unsigned int nAtomsT); void resizeAtomName(std::false_type, unsigned int nAtomsT){};//names
	void resizeAtomAN(std::true_type, unsigned int nAtomsT); void resizeAtomAN(std::false_type, unsigned int nAtomsT){};//atomic numbers
	void resizeAtomMass(std::true_type, unsigned int nAtomsT); void resizeAtomMass(std::false_type, unsigned int nAtomsT){};//masses
	void resizeAtomSpecie(std::true_type, unsigned int nAtomsT); void resizeAtomSpecie(std::false_type, unsigned int nAtomsT){};//species
	void resizeAtomIndex(std::true_type, unsigned int nAtomsT); void resizeAtomIndex(std::false_type, unsigned int nAtomsT){};//indices
	void resizeAtomPosn(std::true_type, unsigned int nAtomsT); void resizeAtomPosn(std::false_type, unsigned int nAtomsT){};//positions
	void resizeAtomForce(std::true_type, unsigned int nAtomsT); void resizeAtomForce(std::false_type, unsigned int nAtomsT){};//forces
	void resizeAtomVelocity(std::true_type, unsigned int nAtomsT); void resizeAtomVelocity(std::false_type, unsigned int nAtomsT){};//forces
	void resizeAtomCharge(std::true_type, unsigned int nAtomsT); void resizeAtomCharge(std::false_type, unsigned int nAtomsT){};//charges
	void resizeAtomSymm(std::true_type, unsigned int nAtomsT); void resizeAtomSymm(std::false_type, unsigned int nAtomsT){};//symmetry functions
	
	//assigment functions
	void assignAtomArrays();
	void assignAtomName(std::true_type); void assignAtomName(std::false_type){};//names
	void assignAtomAN(std::true_type); void assignAtomAN(std::false_type){};//atomic numbers
	void assignAtomMass(std::true_type); void assignAtomMass(std::false_type){};//masses
	void assignAtomSpecie(std::true_type); void assignAtomSpecie(std::false_type){};//species
	void assignAtomPosn(std::true_type); void assignAtomPosn(std::false_type){};//indices
	void assignAtomForce(std::true_type); void assignAtomForce(std::false_type){};//indices
	void assignAtomVelocity(std::true_type); void assignAtomVelocity(std::false_type){};//indices
	void assignAtomIndex(std::true_type); void assignAtomIndex(std::false_type){};//positions
	void assignAtomCharge(std::true_type); void assignAtomCharge(std::false_type){};//charges
	void assignAtomSymm(std::true_type); void assignAtomSymm(std::false_type){};//symmetry functions
public:
	//constructors/destructors
	StructureS(){};
	StructureS(const StructureS<AtomT>& sim);
	~StructureS(){};
	
	//operators
	StructureS<AtomT>& operator=(const StructureS<AtomT>& sim);
	template <class AtomT_> friend std::ostream& operator<<(std::ostream& out, const StructureS<AtomT_>& sim);
	
	//access - properties
	//names
		ListData<std::string>& atom_name(){return atom_name_;};
		const ListData<std::string>& atom_name()const{return atom_name_;};
	//atomic numbers
		ListData<unsigned int>& atom_an(){return atom_an_;};
		const ListData<unsigned int>& atom_an()const{return atom_an_;};
	//masses
		ListData<double>& atom_mass(){return atom_mass_;};
		const ListData<double>& atom_mass()const{return atom_mass_;};
	//species
		ListData<unsigned int>& atom_specie(){return atom_specie_;};
		const ListData<unsigned int>& atom_specie()const{return atom_specie_;};
	//species
		ListData<unsigned int>& atom_index(){return atom_index_;};
		const ListData<unsigned int>& atom_index()const{return atom_index_;};
	//positions
		ListData<Eigen::Vector3d>& atom_posn(){return atom_posn_;};
		const ListData<Eigen::Vector3d>& atom_posn()const{return atom_posn_;};
	//forces
		ListData<Eigen::Vector3d>& atom_force(){return atom_force_;};
		const ListData<Eigen::Vector3d>& atom_force()const{return atom_force_;};
	//velocities
		ListData<Eigen::Vector3d>& atom_velocity(){return atom_velocity_;};
		const ListData<Eigen::Vector3d>& atom_velocity()const{return atom_velocity_;};
	//charges
		ListData<double>& atom_charge(){return atom_charge_;};
		const ListData<double>& atom_charge()const{return atom_charge_;};
	//symmetry functions
		ListData<Eigen::VectorXd>& atom_symm(){return atom_symm_;};
		const ListData<Eigen::VectorXd>& atom_symm()const{return atom_symm_;};
	
	//access - atoms
	const ListData<AtomT>& atoms()const{return atoms_;};
	
	//member functions
	void clear();
	void resize(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& speciesNames);
};

//constructors/destructors

template <class AtomT>
StructureS<AtomT>::StructureS(const StructureS<AtomT>& sim){
	//set the property arrays
	atom_name_=sim.atom_name();
	atom_an_=sim.atom_an();
	atom_mass_=sim.atom_mass();
	atom_specie_=sim.atom_specie();
	atom_index_=sim.atom_index();
	atom_posn_=sim.atom_posn();
	atom_force_=sim.atom_force();
	atom_velocity_=sim.atom_velocity();
	atom_charge_=sim.atom_charge();
	atom_symm_=sim.atom_symm();
	//resize the atom array
	atoms_.resize(sim.atoms_.size());
	//assign the atom arrays
	assignAtomArrays();
};

//operators

template <class AtomT>
StructureS<AtomT>& StructureS<AtomT>::operator=(const StructureS<AtomT>& sim){
	StructureI::operator=(sim);
	//set the property arrays
	atom_name_=sim.atom_name();
	atom_an_=sim.atom_an();
	atom_mass_=sim.atom_mass();
	atom_specie_=sim.atom_specie();
	atom_index_=sim.atom_index();
	atom_posn_=sim.atom_posn();
	atom_force_=sim.atom_force();
	atom_velocity_=sim.atom_velocity();
	atom_charge_=sim.atom_charge();
	atom_symm_=sim.atom_symm();
	//resize the atom array
	atoms_.resize(sim.atoms().nAtoms());
	//assign the atom arrays
	assignAtomArrays();
	return *this;
}

template <class AtomT>
std::ostream& operator<<(std::ostream& out, const StructureS<AtomT>& sim){
	return out<<static_cast<const StructureI&>(sim);
}

//member functions

template <class AtomT>
void StructureS<AtomT>::clear(){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::clear():\n";
	StructureI::clear();
	//clear property arrays
	atom_name_.clear();
	atom_an_.clear();
	atom_mass_.clear();
	atom_specie_.clear();
	atom_index_.clear();
	atom_posn_.clear();
	atom_force_.clear();
	atom_velocity_.clear();
	atom_charge_.clear();
	atom_symm_.clear();
	//clear the atoms
	atoms_.clear();
}

template <class AtomT>
void StructureS<AtomT>::resize(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& speciesNames){
	//find the total number of atoms
	unsigned int nAtomsT=0;
	for(unsigned int i=0; i<nAtoms.size(); ++i) nAtomsT+=nAtoms[i];
	//resize the atom/molecule arrays
	atoms_.resize(nAtomsT);
	//resize the property arrays
	resizeAtomArrays(nAtomsT);
	//assign atom arrays
	assignAtomArrays();
}

//resizing atom properties

template <class AtomT>
void StructureS<AtomT>::resizeAtomArrays(unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeArrays():\n";
	resizeAtomName(std::is_base_of<Name,AtomT>(),nAtomsT);
	resizeAtomAN(std::is_base_of<AN,AtomT>(),nAtomsT);
	resizeAtomMass(std::is_base_of<Mass,AtomT>(),nAtomsT);
	resizeAtomSpecie(std::is_base_of<Species,AtomT>(),nAtomsT);
	resizeAtomIndex(std::is_base_of<Index,AtomT>(),nAtomsT);
	resizeAtomPosn(std::is_base_of<Position,AtomT>(),nAtomsT);
	resizeAtomForce(std::is_base_of<Force,AtomT>(),nAtomsT);
	resizeAtomVelocity(std::is_base_of<Velocity,AtomT>(),nAtomsT);
	resizeAtomCharge(std::is_base_of<Charge,AtomT>(),nAtomsT);
	resizeAtomSymm(std::is_base_of<Symm,AtomT>(),nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomName(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomName(std::true_type, unsigned int nAtomsT):\n";
	atom_name_.resize(nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomAN(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomAN(std::true_type, unsigned int nAtomsT):\n";
	atom_an_.resize(nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomMass(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomMass(std::true_type, unsigned int nAtomsT):\n";
	atom_mass_.resize(nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomSpecie(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomSpecie(std::true_type, unsigned int nAtomsT):\n";
	atom_specie_.resize(nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomIndex(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomIndex(std::true_type, unsigned int nAtomsT):\n";
	atom_index_.resize(nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomPosn(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomPosn(std::true_type, unsigned int nAtomsT):\n";
	atom_posn_.resize(nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomForce(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomForce(std::true_type, unsigned int nAtomsT):\n";
	atom_force_.resize(nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomVelocity(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomVelocity(std::true_type, unsigned int nAtomsT):\n";
	atom_velocity_.resize(nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomCharge(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomCharge(std::true_type, unsigned int nAtomsT):\n";
	atom_charge_.resize(nAtomsT);
}

template <class AtomT>
void StructureS<AtomT>::resizeAtomSymm(std::true_type, unsigned int nAtomsT){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::resizeAtomSymm(std::true_type, unsigned int nAtomsT):\n";
	atom_symm_.resize(nAtomsT);
}

//assigning atom properties

template <class AtomT>
void StructureS<AtomT>::assignAtomArrays(){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomArrays():\n";
	assignAtomName(std::is_base_of<Name,AtomT>());
	assignAtomAN(std::is_base_of<AN,AtomT>());
	assignAtomMass(std::is_base_of<Mass,AtomT>());
	assignAtomSpecie(std::is_base_of<Species,AtomT>());
	assignAtomIndex(std::is_base_of<Index,AtomT>());
	assignAtomPosn(std::is_base_of<Position,AtomT>());
	assignAtomForce(std::is_base_of<Force,AtomT>());
	assignAtomVelocity(std::is_base_of<Velocity,AtomT>());
	assignAtomCharge(std::is_base_of<Charge,AtomT>());
	assignAtomSymm(std::is_base_of<Symm,AtomT>());
}

template <class AtomT>
void StructureS<AtomT>::assignAtomName(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomName(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<Name&>(atoms_[i]).set(&atom_name_[i]);
	}
}

template <class AtomT>
void StructureS<AtomT>::assignAtomAN(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomAN(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<AN&>(atoms_[i]).set(&atom_an_[i]);
	}
}

template <class AtomT>
void StructureS<AtomT>::assignAtomMass(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomMass(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<Mass&>(atoms_[i]).set(&atom_mass_[i]);
	}
}

template <class AtomT>
void StructureS<AtomT>::assignAtomSpecie(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomSpecie(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<Species&>(atoms_[i]).set(&atom_specie_[i]);
	}
}

template <class AtomT>
void StructureS<AtomT>::assignAtomIndex(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomIndex(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<Index&>(atoms_[i]).set(&atom_index_[i]);
	}
}

template <class AtomT>
void StructureS<AtomT>::assignAtomPosn(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomPosn(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<Position&>(atoms_[i]).set(&atom_posn_[i]);
	}
}

template <class AtomT>
void StructureS<AtomT>::assignAtomForce(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomForce(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<Force&>(atoms_[i]).set(&atom_force_[i]);
	}
}

template <class AtomT>
void StructureS<AtomT>::assignAtomVelocity(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomVelocity(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<Velocity&>(atoms_[i]).set(&atom_velocity_[i]);
	}
}

template <class AtomT>
void StructureS<AtomT>::assignAtomCharge(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomCharge(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<Charge&>(atoms_[i]).set(&atom_charge_[i]);
	}
}

template <class AtomT>
void StructureS<AtomT>::assignAtomSymm(std::true_type){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureS<AtomT>::assignAtomSymm(std::true_type):\n";
	for(unsigned int i=0; i<atoms_.size(); ++i){
		static_cast<Symm&>(atoms_[i]).set(&atom_symm_[i]);
	}
}

//**********************************************************************************************
// Structure - Atomic
//**********************************************************************************************

template <class AtomT>
class Structure: public StructureI, public StructureS<AtomT>{
public:
	//constructors/destructors
	Structure(){};
	Structure(const Structure<AtomT>& sim):StructureI(sim),StructureS<AtomT>(sim){};
	~Structure(){};
	
	//operators
	Structure<AtomT>& operator=(const Structure<AtomT>& sim);
	template <class AtomT_> friend std::ostream& operator<<(std::ostream& out, const Structure<AtomT_>& sim);
	
	//member functions
	void clear();
	void resize(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& speciesNames);
	
	//access - atoms
	AtomT& atom(unsigned int n){return this->atoms_[n];};
	const AtomT& atom(unsigned int n)const{return this->atoms_[n];};
	AtomT& atom(unsigned int n, unsigned int m);
	const AtomT& atom(unsigned int n, unsigned int m)const;
};

//access - atoms

template <class AtomT>
AtomT& Structure<AtomT>::atom(unsigned int n, unsigned int m){
	return this->atoms_[offsets_[n]+m];
}

template <class AtomT>
const AtomT& Structure<AtomT>::atom(unsigned int n, unsigned int m)const{
	return this->atoms_[offsets_[n]+m];
}

//operators

template <class AtomT>
Structure<AtomT>& Structure<AtomT>::operator=(const Structure<AtomT>& sim){
	if(DEBUG_STRUCTURE>0) std::cout<<"Structure<AtomT>::operator=(const Structure<AtomT>&):\n";
	StructureI::operator=(sim);
	StructureS<AtomT>::operator=(sim);
	return *this;
}

template <class AtomT>
std::ostream& operator<<(std::ostream& out, const Structure<AtomT>& sim){
	out<<static_cast<const StructureI&>(sim);
	return out;
}

//member functions

template <class AtomT>
void Structure<AtomT>::clear(){
	if(DEBUG_STRUCTURE>0) std::cout<<"Structure<AtomT>::clear():\n";
	StructureI::clear();
	StructureS<AtomT>::clear();
}

template <class AtomT>
void Structure<AtomT>::resize(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& speciesNames){
	if(DEBUG_STRUCTURE>0) std::cout<<"Structure<AtomT>::resize(const std::vector<unsigned int>&,const std::vector<std::string>&):\n";
	//resize StructureI
	StructureI::resize(nAtoms,speciesNames);
	//resize StructureS
	StructureS<AtomT>::resize(nAtoms,speciesNames);
	//set the names, species, and indices
	unsigned int count=0;
	for(unsigned int n=0; n<nSpecies_; ++n){
		for(unsigned int m=0; m<nAtoms_[n]; ++m){
			this->atom_name_[count]=this->atomNames_[n];
			this->atom_an_[count]=PTable::atomicNumber(this->atomNames_[n].c_str());
			this->atom_specie_[count]=n;
			this->atom_index_[count]=m;
			++count;
		}
	}
	
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	//ListData
	
	template <> unsigned int nbytes(const ListData<Name>& obj);
	template <> unsigned int nbytes(const ListData<AN>& obj);
	template <> unsigned int nbytes(const ListData<Species>& obj);
	template <> unsigned int nbytes(const ListData<Index>& obj);
	template <> unsigned int nbytes(const ListData<Mass>& obj);
	template <> unsigned int nbytes(const ListData<Charge>& obj);
	template <> unsigned int nbytes(const ListData<Position>& obj);
	template <> unsigned int nbytes(const ListData<Velocity>& obj);
	template <> unsigned int nbytes(const ListData<Force>& obj);
	template <> unsigned int nbytes(const ListData<Symm>& obj);
	
	//StructureI
	
	template <> unsigned int nbytes(const StructureI& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	//ListData
	
	template <> void pack(const ListData<Name>& obj, char* arr);
	template <> void pack(const ListData<AN>& obj, char* arr);
	template <> void pack(const ListData<Species>& obj, char* arr);
	template <> void pack(const ListData<Index>& obj, char* arr);
	template <> void pack(const ListData<Mass>& obj, char* arr);
	template <> void pack(const ListData<Charge>& obj, char* arr);
	template <> void pack(const ListData<Position>& obj, char* arr);
	template <> void pack(const ListData<Velocity>& obj, char* arr);
	template <> void pack(const ListData<Force>& obj, char* arr);
	template <> void pack(const ListData<Symm>& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	//ListData
	
	template <> void unpack(ListData<Name>& obj, const char* arr);
	template <> void unpack(ListData<AN>& obj, const char* arr);
	template <> void unpack(ListData<Species>& obj, const char* arr);
	template <> void unpack(ListData<Index>& obj, const char* arr);
	template <> void unpack(ListData<Mass>& obj, const char* arr);
	template <> void unpack(ListData<Charge>& obj, const char* arr);
	template <> void unpack(ListData<Position>& obj, const char* arr);
	template <> void unpack(ListData<Velocity>& obj, const char* arr);
	template <> void unpack(ListData<Force>& obj, const char* arr);
	template <> void unpack(ListData<Symm>& obj, const char* arr);
	
}

#endif