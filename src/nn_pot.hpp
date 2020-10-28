#pragma once
#ifndef ANN_NN_POT_HPP
#define ANN_NN_POT_HPP

// c++ libraries
#include <iosfwd>
// typedefs
#include "typedef.hpp"
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
// NEURAL NETWORK HAMILTONIAN
//************************************************************

class NNH{
private:
	//hamiltonian
	double rc_;//cutoff radius
	Atom atom_;//atom - name, mass, energy, charge
	NN::Network nn_;//neural network hamiltonian
	
	//interacting species
	int nspecies_;//number of species
	std::vector<Atom> species_;//species
	Map<int,int> map_;//map - atom ids to nnh index
	
	//basis for pair/triple interactions
	std::vector<BasisR> basisR_;//radial basis functions (nspecies_)
	LMat<BasisA> basisA_;//angular basis functions (nspecies x (nspecies+1)/2)
	
	//network configuration
	int nInput_;//number of radial + angular symmetry functions
	int nInputR_;//number of radial symmetry functions
	int nInputA_;//number of angular symmetry functions
	std::vector<int> offsetR_;//offset for the given radial basis (nspecies_)
	LMat<int> offsetA_;//offset for the given radial basis (nspecies x (nspecies+1)/2)
	static const int nOutput_=1;//only output = energy
public:
	//==== constructors/destructors ====
	NNH(){defaults();}
	~NNH(){}
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const NNH& nmh);
	
	//==== access ====
	//hamiltonian
		Atom& atom(){return atom_;}
		const Atom& atom()const{return atom_;}
		NN::Network& nn(){return nn_;}
		const NN::Network& nn()const{return nn_;}
		double& rc(){return rc_;}
		const double& rc()const{return rc_;}
	//interacting species
		const int& nspecies()const{return nspecies_;}
		Atom& species(int i){return species_[i];}
		const Atom& species(int i)const{return species_[i];}
		Map<int,int>& map(){return map_;}
		const Map<int,int>& map()const{return map_;}
		int index(const std::string& name)const{return map_[string::hash(name)];}
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
		void clear(){defaults();};//clear the potential
	//resizing
		void resize(const std::vector<Atom>& species);
		void init_input();//initialize the inputs
	//output
		double energy(const Eigen::VectorXd& symm);//compute energy of atom
	//reading/writing - all
		void write(const std::string& filename)const;
		void write(FILE* writer)const;
		void read(const std::string& filename);
		void read(FILE* reader);
	//reading/writing - all
		void write_basis(const std::string& filename)const;
		void write_basis(FILE* reader)const;
		void read_basis(const std::string& filename);
		void read_basis(FILE* reader);
};

//************************************************************
// NEURAL NETWORK POTENTIAL
//************************************************************

class NNPot{
private:
	//global cutoff
	double rc_;
	
	//atomic species - generic "atom" objects
	int nspecies_;//number of types of atoms
	Map<int,int> map_;//map atom ids to nnpot index
	std::vector<NNH> nnh_;//the hamiltonians for each species
	
	//input/output
	std::string head_;
	std::string tail_;
	
	//utility
	std::vector<Eigen::Vector3d> R_;
public:
	//==== constructors/destructors ====
	NNPot(){defaults();}
	~NNPot(){}
	
	//friend declarations
	friend class NNPotOpt;
	
	//==== access ====
	//species
		int nspecies()const{return nspecies_;}
		int index(const std::string& name)const{return map_[string::hash(name)];}
		Map<int,int>& map(){return map_;}
		const Map<int,int>& map()const{return map_;}
		NNH& nnh(int i){return nnh_[i];}
		const NNH& nnh(int i)const{return nnh_[i];};
	//global cutoff
		double& rc(){return rc_;}
		const double& rc()const{return rc_;}
	//input/output
		std::string& head(){return head_;}
		const std::string& head()const{return head_;}
		std::string& tail(){return tail_;}
		const std::string& tail()const{return tail_;}
		
	//operators
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
		void forces(Structure& struc, bool calc_symm=true);//calculate forces
		double energy(Structure& struc, bool calc_symm=true);//sum over atomic energyies and return total energy
	//reading/writing
		void write()const;
		void read();
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
