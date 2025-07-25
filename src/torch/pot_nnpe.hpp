#pragma once
#ifndef POT_NNPE_HPP
#define POT_NNPE_HPP

//c++
#include <vector>
//eigen
#include <Eigen/Dense>
// mem
#include "mem/map.hpp"
// nnp
#include "nnp/nnh.hpp"
// torch
#include "torch/pot.hpp"

#ifndef POT_NNPE_PRINT_FUNC
#define POT_NNPE_PRINT_FUNC 0
#endif

namespace ptnl{

class PotNNPE: public Pot{
private:
	Map<int,int> map_;//map atom ids to nnpot index
	std::vector<NNH> nnh_;//the hamiltonians for each species
	std::vector<Eigen::VectorXd> symm_;//symm func array for each species
public:
	//==== constructors/destructors ====
	PotNNPE():Pot(Pot::Name::NNPE){}
	~PotNNPE(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const PotNNPE& pot);
	
	//==== access ====
	int index(const char* name)const{return map_[string::hash(name)];}
	int index(const std::string& name)const{return map_[string::hash(name)];}
	Map<int,int>& map(){return map_;}
	const Map<int,int>& map()const{return map_;}
	NNH& nnh(int i){return nnh_[i];}
	const NNH& nnh(int i)const{return nnh_[i];}
	
	//==== member functions ====
	//pot
		void read(Token& token);
		void coeff(Token& token);
		void resize(int ntypes);
	//read/write basis
		void read_basis(const char* file, const char* atomName);//read basis for atomName
		void read_basis(FILE* reader, const char* atomName);//read basis for atomName
	//read/write nnp
		void write(const char* file);//write NNP to "file"
		void read(const char* file);//read NNP from "file"
		void write(FILE* writer);//write NNP to "writer"
		void read(FILE* reader);//read NNP from "reader"
	//calculation
		double energy(const Structure& struc, const NeighborList& nlist);//compute energy
		double energy(const Structure& struc, const NeighborList& nlist, int i);//compute energy
		double compute(Structure& struc, const NeighborList& nlist);//compute forces
	//calculation
		double energy(const Structure& struc, const verlet::List& vlist);//compute energy
		double energy(const Structure& struc, const verlet::List& vlist, int i);//compute energy
		double compute(Structure& struc, const verlet::List& vlist);//compute forces
};

}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotNNPE& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotNNPE& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotNNPE& obj, const char* arr);
	
}

#endif