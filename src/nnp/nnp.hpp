#pragma once
#ifndef NNP_HPP
#define NNP_HPP

// c++ libraries
#include <iosfwd>
// structure
#include "struc/structure_fwd.hpp"
#include "struc/neighbor.hpp"
#include "struc/verlet.hpp"
// ml
#include "ml/nn.hpp"
// mem
#include "mem/map.hpp"
// string
#include "str/string.hpp"
// basis functions
#include "nnp/basis_radial.hpp"
#include "nnp/basis_angular.hpp"
// type
#include "nnp/type.hpp"
// nnh
#include "nnp/nnh.hpp"

//***********************************************************************
// COMPILER DIRECTIVES
//***********************************************************************

#ifndef NNP_PRINT_FUNC
#define NNP_PRINT_FUNC 0
#endif

#ifndef NNP_PRINT_STATUS
#define NNP_PRINT_STATUS 0
#endif

#ifndef NNP_PRINT_DATA
#define NNP_PRINT_DATA 0
#endif

//************************************************************
// TERMINOLOGY:
// symmetry function - measure of local symmetry around a given atom
// basis - group of symmetry functions associated with a given interaction
// Each types has its own basis for each pair interaction and each
// triple interaction (i.e. H has (H)-H, (H)-O, (H)-O-O, (H)-H-H, (H)-H-O/(H)-O-H)
// Since the atomic neural networks are different, the basis for pair/triple
// interactions on different types of center atoms can be the same, reducing
// the number of necessary parameters.  Note however, that they can be different.
//************************************************************

//************************************************************
// NEURAL NETWORK POTENTIAL (NNP)
//************************************************************

/**
* Class defining a Neural Network Potential
* This class defines a nueral network potential (NNP) as a collection of neural network hamiltonians (NNH),
* one for each atomic type in a given simulation, along with a global cutoff.  Each NNH contains all the
* information necessary to compute the energy and forces for a given atomic type, and so together they can
* be used to compute the energy and forces of all types in a simulation.
* 
* A map is defined to map arbitrary ids for each type to the index of the type as it is stored in the NNP class.
* This allows one to compute the energy of atom types given arbitrary order in a simulation and even the presence
* of types not addressed by the NNP.
*/
class NNP{
private:
	double rc_;//global cutoff
	int ntypes_;//number of types of atoms
	Map<int,int> map_;//map atom ids to nnpot index
	std::vector<NNH> nnh_;//the hamiltonians for each types
public:
	//==== constructors/destructors ====
	NNP(){defaults();}
	NNP(const std::vector<Type>& types){defaults();resize(types);}
	~NNP(){}
	
	//==== access ====
	//types
		int ntypes()const{return ntypes_;}
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
	friend std::ostream& operator<<(std::ostream& out, const NNP& nnpot);
	friend FILE* operator<<(FILE* out, const NNP& nnpot);
	
	//==== member functions ====
	//info
		int size()const;//number of parameters
	//auxilliary
		void defaults();//set defaults
		void clear(){defaults();};//clear the potential
	//resizing
		void resize(const std::vector<Type>& types);
		
	//==== static functions ====
	//read/write basis
		static void read_basis(const char* file, NNP& nnpot, const char* name);//read basis for name
		static void read_basis(FILE* reader, NNP& nnpot, const char* name);//read basis for name
	//read/write nnp
		static void write(const char* file, const NNP& nnpot);//write NNP to "file"
		static void read(const char* file, NNP& nnpot);//read NNP from "file"
		static void write(FILE* writer, const NNP& nnpot);//write NNP to "writer"
		static void read(FILE* reader, NNP& nnpot);//read NNP from "reader"
	//calculation
		static void init(const NNP& nnp, Structure& struc);
		static void symm(NNP& nnp, Structure& struc, const NeighborList& nlist);//compute symmetry functions
		static void symm(NNP& nnp, Structure& struc, const verlet::List& vlist);
		static double energy(NNP& nnp, const Structure& struc);//compute energy
		static void force(NNP& nnp, Structure& struc, const NeighborList& nlist);//compute forces
		static void force(NNP& nnp, Structure& struc, const verlet::List& nlist);
		static void compute(NNP& nnp, Structure& struc, const NeighborList& nlist);//compute forces
		static void compute(NNP& nnp, Structure& struc, const verlet::List& nlist);
};

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNP& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNP& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNP& obj, const char* arr);
	
}

#endif
