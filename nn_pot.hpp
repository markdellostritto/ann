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
// NEURAL NETWORK POTENTIAL
//************************************************************

class NNPot{
private:
	//atomic species - generic "atom" objects
	std::vector<Atom> atoms_;//the atoms in the simulation
	Map<unsigned int,unsigned int> atomMap_;//map atom ids to nnpot index
	
	//global cutoff
	double rc_;
	
	//element nn's
	std::vector<NN::Network> nn_;//neural networks for each specie (nspecies)
	
	//basis for pair/triple interactions
	std::vector<std::vector<BasisR> > basisR_;//radial basis functions (nspecies x nspecies)
	std::vector<LMat<BasisA> > basisA_;//angular basis functions (nspecies (nspecies x (nspecies+1)/2))
	
	//network configuration
	std::vector<unsigned int> nInput_;//number of radial + angular symmetry functions (nspecies)
	std::vector<unsigned int> nInputR_;//number of radial symmetry functions (nspecies)
	std::vector<unsigned int> nInputA_;//number of angular symmetry functions (nspecies)
	std::vector<std::vector<unsigned int> > offsetR_;//offset for the given radial basis
	std::vector<LMat<unsigned int> > offsetA_;//offset for the given radial basis
	static const unsigned int nOutput_=1;//only output = energy
	
	//input/output
	std::string head_;
	std::string tail_;
	
	//utility
	Eigen::Vector3d rIJ_,rIK_,rJK_;
	Eigen::Vector3d rIJt_,rIKt_,rJKt_;
	std::vector<Eigen::Vector3d> R_;
public:
	//initialization data - aggregates data necessary for starting training from scratch
	struct Init{
		//network configuration
		std::vector<unsigned int> nh;//number of hidden nodes
		//regularization
		double lambda;
		//transfer function
		NN::TransferN::type tfType;//transfer function type
		//operators
		friend std::ostream& operator<<(std::ostream& out, const Init& init);
		//constructors/destructors
		Init(){defaults();}
		~Init(){}
		//member functions
		void defaults();
	};
public:
	//==== constructors/destructors ====
	NNPot(){defaults();}
	~NNPot(){}
	
	//friend declarations
	friend class NNPotOpt;
	
	//==== access ====
	//species
		const unsigned int nAtoms()const{return atoms_.size();}
		Atom& atom(unsigned int i){return atoms_[i];}
		const Atom& atom(unsigned int i)const{return atoms_[i];}
		const std::vector<Atom>& atoms()const{return atoms_;}
		unsigned int atom_index(const std::string& name)const{return atomMap_[string::hash(name)];}
		Map<unsigned int,unsigned int>& atomMap(){return atomMap_;}
		const Map<unsigned int,unsigned int>& atomMap()const{return atomMap_;}
	//global cutoff
		double& rc(){return rc_;}
		const double& rc()const{return rc_;}
	//basis
		const std::vector<std::vector<BasisR> >& basisR()const{return basisR_;}
		BasisR& basisR(unsigned int i, unsigned int j){return basisR_[i][j];}
		const BasisR& basisR(unsigned int i, unsigned int j)const{return basisR_[i][j];}
		const unsigned int& offsetR(unsigned int i, unsigned int j)const{return offsetR_[i][j];}
		const std::vector<LMat<BasisA> >& basisA()const{return basisA_;}
		BasisA& basisA(unsigned int n, unsigned int i, unsigned int j){return basisA_[n](i,j);}
		const BasisA& basisA(unsigned int n, unsigned int i, unsigned int j)const{return basisA_[n](i,j);}
		const unsigned int& offsetA(unsigned int n, unsigned int i, unsigned int j)const{return offsetA_[n](i,j);}
	//neural networks
		const std::vector<NN::Network>& nn()const{return nn_;}
		NN::Network& nn(unsigned int i){return nn_[i];}
		const NN::Network& nn(unsigned int i)const{return nn_[i];}
	//nodes
		const unsigned int& nInput(unsigned int i)const{return nInput_[i];}
		const unsigned int& nInputR(unsigned int i)const{return nInputR_[i];}
		const unsigned int& nInputA(unsigned int i)const{return nInputR_[i];}
		const std::vector<unsigned int>& nInput()const{return nInput_;}
		const std::vector<unsigned int>& nInputR()const{return nInputR_;}
		const std::vector<unsigned int>& nInputA()const{return nInputA_;}
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
		void init(const Init& init_);//initialize the basis functions and element networks
	//resizing
		void resize(const Structure& struc);//assign vector of all species in the simulations
		void resize(const std::vector<Structure >& simv);//assign vector of all species in the simulations
		void resize(const std::vector<std::string>& speciesNames);//set the number of species and species names to the total number of species in the simulations
		void resize(const std::vector<Atom>& atoms);//set the number of species and species names to the total number of species in the simulations
	//nn-struc
		void init_symm(Structure& struc)const;//assign vector of all species in the simulations
		void inputs_symm(Structure& struc);//calculate inputs - symmetry functions
		void init_inputs();
		void forces(Structure& struc, bool calc_symm=true);//calculate forces
		double energy(Structure& struc, bool calc_symm=true);//sum over atomic energyies and return total energy
		void forces_radial(Structure& struc);
		void forces_angular(Structure& struc);
	//static functions
		void write()const;
		void read();
		void write(unsigned int index, const std::string& filename)const;
		void write(unsigned int index, FILE* writer)const;
		void read(unsigned int index, const std::string& filename);
		void read(unsigned int index, FILE* reader);
};

bool operator==(const NNPot& nnPot1, const NNPot& nnPot2);
inline bool operator!=(const NNPot& nnPot1, const NNPot& nnPot2){return !(nnPot1==nnPot2);}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const NNPot& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const NNPot& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(NNPot& obj, const char* arr);
	
}

#endif
