#pragma once
#ifndef ANN_NN_POT_HPP
#define ANN_NN_POT_HPP

// c++ libraries
#include <iosfwd>
// ann - lower triangular matrix
#include "lmat.hpp"
// ann - structure
#include "structure_fwd.hpp"
// neural networks
#include "nn.hpp"
// map
#include "map.hpp"
// basis functions
#include "basis_radial.hpp"
#include "basis_angular.hpp"
// atom
#include "atom.hpp"

//***********************************************************************
// COMPILER DIRECTIVES
//***********************************************************************

#ifndef NN_POT_PRINT_FUNC
#define NN_POT_PRINT_FUNC 0
#endif

#ifndef NN_POT_PRINT_STATUS
#define NN_POT_PRINT_STATUS 0
#endif

#ifndef NN_POT_PRINT_DATA
#define NN_POT_PRINT_DATA 0
#endif

//************************************************************
// TERMINOLOGY:
// symmetry function - measure of local symmetry around a given atom
// basis - group of symmetry functions associated with a given interaction
// Each species has its own basis for each pair interaction and each
// triple interaction (i.e. H has (H)-H, (H)-O, (H)-O-O, (H)-H-H, (H)-H-O/(H)-O-H)
// Since the atomic neural networks are different, the basis for pair/triple
// interactions on different species of center atoms can be the same, reducing
// the number of necessary parameters.  Note however, that they can be different.
//************************************************************

//************************************************************
// NEURAL NETWORK HAMILTONIAN (NNH)
//************************************************************

/*
PRIVATE:
	int nspecies_ - the total number of species in a neural network potential
	Atom atom_ - the central atom of the NNH
	NeuralNet::ANN nn_ - the neural network which determines the energy of the central atom
	basisR_ - the radial basis functions
		there is a radial basis for each species in the simulation (nspecies)
		each basis for each species contains a set of radial symmetry functions
		each symmetry function then corresponds to a unique input to the neural network
	basisA_ - the angular basis functions
		there is an angular basis for each unique pair of species in the simulation (nspecies x (nspecies-1)/2)
		each basis for each pair contains a set of angular symmetry functions
		each symmetry function then corresponds to a unique input to the neural network
	nInput_ - the total number of inputs to the network
		defined as the total number of radial and angular symmetry functions in each basis
		the inputs are arranged with the radial inputs preceding the angular inputs
	nInputR_ - the total number of radial inputs to the network
		defined as the total number of radial symmetry functions in each basis
	nInputA_ - the total number of angular inputs to the network
		defined as the total number of angular symmetry functions in each basis
	offsetR_ - the offset of each radial input
		all symmetry functions must be serialized into a single vector - the input to the neural network
		each radial symmetry function thus has an offset defined as the total number of symmetry functions
		in all bases preceding the current basis
	offsetA_ - the offset of each angular input
		all symmetry functions must be serialized into a single vector - the input to the neural network
		each angular symmetry function thus has an offset defined as the total number of symmetry functions
		in all basis pairs preceding the current basis pair
		note that the offset is from the beginning of the angular section of the inputs
		thus, ths offset from the beginning of the input vector is nInputR_ + offsetA_(i,j)
NOTES:
	This class is not meant to be used independently, only as a part of the class NNPot.
	This class alone does not have enough data to define a neural network potential.
	Rather, this class accumulates all the data associated with the inputs and nueral network for a given atom.
	Thus, if one has a valid symmetry function defining the the local symmetry around a given atom of the
	correct species, this class can be used to compute the energy.
	However, the class NNPot is required to define all species, define all neural network potentials, and to
	compute the symmetry functions, total energies, and forces for a given atomic configuration.
*/
class NNH{
private:
	//network configuration
	int nInput_;//number of radial + angular symmetry functions
	int nInputR_;//number of radial symmetry functions
	int nInputA_;//number of angular symmetry functions
	
	//hamiltonian
	int nspecies_;//the total number of species
	Atom atom_;//atom - name, mass, energy, charge
	NeuralNet::ANN nn_;//neural network hamiltonian
	NeuralNet::DOutDVal dOutDVal_;//gradient of the output w.r.t. node values
	
	//basis for pair/triple interactions
	std::vector<BasisR> basisR_;//radial basis functions (nspecies_)
	std::vector<int> offsetR_;//offset for the given radial basis (nspecies_)
	LMat<BasisA> basisA_;//angular basis functions (nspecies x (nspecies+1)/2)
	LMat<int> offsetA_;//offset for the given radial basis (nspecies x (nspecies+1)/2)
public:
	//==== constructors/destructors ====
	NNH(){defaults();}
	~NNH(){}
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const NNH& nmh);
	
	//==== access ====
	//hamiltonian
		const int& nspecies()const{return nspecies_;}
		Atom& atom(){return atom_;}
		const Atom& atom()const{return atom_;}
		NeuralNet::ANN& nn(){return nn_;}
		const NeuralNet::ANN& nn()const{return nn_;}
		NeuralNet::DOutDVal& dOutDVal(){return dOutDVal_;}
		const NeuralNet::DOutDVal& dOutDVal()const{return dOutDVal_;}
	//basis for pair/triple interactions
		BasisR& basisR(int i){return basisR_[i];}
		const BasisR& basisR(int i)const{return basisR_[i];}
		BasisA& basisA(int i, int j){return basisA_(i,j);}
		const BasisA& basisA(int i, int j)const{return basisA_(i,j);}
	//network configuration
		const int& nInput()const{return nInput_;}
		const int& nInputR()const{return nInputR_;}
		const int& nInputA()const{return nInputA_;}
		const int& offsetR(int i)const{return offsetR_[i];}
		const int& offsetA(int i, int j)const{return offsetA_(i,j);}
	
	//==== member functions ====
	//misc
		void defaults();//set defaults
		void clear(){defaults();}//clear the potential
	//resizing
		void resize(int nspecies);//resize
		void init_input();//initialize the inputs
	//output
		double energy(const Eigen::VectorXd& symm);//compute energy of atom
};

//************************************************************
// NEURAL NETWORK POTENTIAL (NNP)
//************************************************************

/*
PRIVATE:
	double rc_ - global cutoff
		this cutoff is used for determining which atoms will be included in the calculations of the symmetry functions and forces
		different cutoffs can then be used for different symmetry functions
	int nspecies_ - the total number of atomic species
	Map<int,int> map_ - map assigning atom ids to the index of a given atom in the NNP
		note that the atom id is a unique integer generated from the atom name
		the index is the position of the atom in the list of NNHs (nnh_)
		thus, this maps assigns the correct index in "nnh_" to each atom id, and thus each atom name
*/
class NNPot{
private:
	//global cutoff
	double rc_;
	
	//atomic species - generic "atom" objects
	int nspecies_;//number of types of atoms
	Map<int,int> map_;//map atom ids to nnpot index
	std::vector<NNH> nnh_;//the hamiltonians for each species
public:
	//==== constructors/destructors ====
	NNPot(){defaults();}
	~NNPot(){}
	
	//friend declarations
	friend class NNPotOpt;
	
	//==== access ====
	//species
		int nspecies()const{return nspecies_;}
		int index(const char* name)const{return map_[string::hash(name)];}
		int index(const std::string& name)const{return map_[string::hash(name)];}
		Map<int,int>& map(){return map_;}
		const Map<int,int>& map()const{return map_;}
		NNH& nnh(int i){return nnh_[i];}
		const NNH& nnh(int i)const{return nnh_[i];}
	//global cutoff
		double& rc(){return rc_;}
		const double& rc()const{return rc_;}
		
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NNPot& nnpot);
	friend FILE* operator<<(FILE* out, const NNPot& nnpot);
	
	//==== member functions ====
	//auxilliary
		void defaults();//set defaults
		void clear(){defaults();};//clear the potential
	//resizing
		void resize(const std::vector<Atom>& species);
	//nn-struc
		void init_symm(Structure& struc)const;//assign vector of all species in the simulations
		void calc_symm(Structure& struc);//calculate inputs - symmetry functions
		void forces(Structure& struc);//calculate forces
		double energy(const Structure& struc);//sum over atomic energyies and return total energy
		double compute(Structure& struc);//calculate forces
		
	//==== static functions ====
	static void read_basis(const char* file, NNPot& nnpot, const char* atomName);//read basis for atomName
	static void read_basis(FILE* reader, NNPot& nnpot, const char* atomName);//read basis for atomName
	static void write(const char* file, const NNPot& nnpot);//write NNPot to "file"
	static void read(const char* file, NNPot& nnpot);//read NNPot from "file"
	static void write(FILE* writer, const NNPot& nnpot);//write NNPot to "writer"
	static void read(FILE* reader, NNPot& nnpot);//read NNPot from "reader"
	
};

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNH& obj);
	template <> int nbytes(const NNPot& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNH& obj, char* arr);
	template <> int pack(const NNPot& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNH& obj, const char* arr);
	template <> int unpack(NNPot& obj, const char* arr);
	
}

#endif
