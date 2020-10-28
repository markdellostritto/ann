#pragma once
#ifndef ANN_NN_POT_HPP
#define ANN_NN_POT_HPP

// c++ libraries
#include <string>
#include <iosfwd>
// ann - lower triangular matrix
#include "lmat.h"
// neural networks
#include "nn.h"
// ann - string
#include "string_ann.h"
// map
#include "map.h"
// basis functions
#include "basis_radial.h"
#include "basis_angular.h"
// atom
#include "atom_ann.h"

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
	double rc_;
	AtomANN atom_;
	NN::Network nn_;
	
	//interacting species
	int nspecies_;
	std::vector<AtomANN> species_;
	Map<int,int> map_;//map atom ids to nnpot index
	
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
		AtomANN& atom(){return atom_;}
		const AtomANN& atom()const{return atom_;}
		NN::Network& nn(){return nn_;}
		const NN::Network& nn()const{return nn_;}
		double& rc(){return rc_;}
		const double& rc()const{return rc_;}
	//interacting species
		const int& nspecies()const{return nspecies_;}
		AtomANN& species(int i){return species_[i];}
		const AtomANN& species(int i)const{return species_[i];}
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
		void resize(const std::vector<AtomANN>& species);
		void init_input();//initialize the inputs
	//output
		double energy(const Eigen::VectorXd& symm);//compute energy of atom
		//double force();//?
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

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNH& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNH& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNH& obj, const char* arr);
	
}

#endif
