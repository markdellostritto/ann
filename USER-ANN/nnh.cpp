// c libraries
#include <cstdio>
// c++ libraries
#include <iostream>
// ann - math
#include "math_const_ann.h"
// ann - print
#include "print.h"
// ann - nnh
#include "nnh.h"

//************************************************************
// NEURAL NETWORK HAMILTONIAN
//************************************************************

//==== operators ====

/**
* print neural network hamiltonian
* @param out - output stream
* @param nnh - neural network hamiltonian
*/
std::ostream& operator<<(std::ostream& out, const NNH& nnh){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	//hamiltonian
	out<<"ATOM     = "<<nnh.atom_<<"\n";
	//species
	out<<"NSPECIES = "<<nnh.nspecies_<<"\n";
	//potential parameters
	out<<"N_INPUT  = "; std::cout<<nnh.nInput_<<" "; std::cout<<"\n";
	out<<"N_INPUTR = "; std::cout<<nnh.nInputR_<<" "; std::cout<<"\n";
	out<<"N_INPUTA = "; std::cout<<nnh.nInputA_<<" "; std::cout<<"\n";
	out<<nnh.nn_<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

/**
* set NNH defaults
*/
void NNH::defaults(){
	//hamiltonian
		nspecies_=0;
		atom_.clear();
		nn_.clear();
	//basis for pair/triple interactions
		basisR_.clear();
		basisA_.clear();
	//network configuration
		nInput_=0;
		nInputR_=0;
		nInputA_=0;
		offsetR_.clear();
		offsetA_.clear();
}

/**
* resize the number of species
* @param nspecies - the total number of species
*/
void NNH::resize(int nspecies){
	if(nspecies<=0) throw std::invalid_argument("NNH::resize(int): invalid number of species.");
	nspecies_=nspecies;
	basisR_.resize(nspecies_);
	basisA_.resize(nspecies_);
	offsetR_.resize(nspecies_);
	offsetA_.resize(nspecies_);
}

/**
* Initialize the number of inputs and offsets associated with the basis functions.
* Must be done after the basis has been defined, otherwise the values will make no sense.
* Different from resizing: resizing sets the number of species, this sets the number of inputs
* associated with the basis associated with each species.
*/
void NNH::init_input(){
	//radial inputs
	nInputR_=0;
	for(int i=0; i<nspecies_; ++i){
		nInputR_+=basisR_[i].nfR();
	}
	//radial offsets
	offsetR_[0]=0;
	for(int i=1; i<nspecies_; ++i){
		offsetR_[i]=offsetR_[i-1]+basisR_[i-1].nfR();
	}
	//angular inputs
	nInputA_=0;
	for(int i=0; i<nspecies_; ++i){
		for(int j=i; j<nspecies_; ++j){
			nInputA_+=basisA_(j,i).nfA();
		}
	}
	//angular offsets
	offsetA_[0]=0;
	for(int i=1; i<basisA_.size(); ++i){
		offsetA_[i]=offsetA_[i-1]+basisA_[i-1].nfA();
	}
	//total number of inputs
	nInput_=nInputR_+nInputA_;
}

/**
* compute energy of atom with symmetry function "symm"
* @param symm - the symmetry function
*/
double NNH::energy(const Eigen::VectorXd& symm){
	return nn_.execute(symm)[0]+atom_.energy();
}

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNH& obj){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"nbytes(const NNH&):\n";
	int size=0;
	//hamiltonian
	size+=nbytes(obj.atom());
	size+=nbytes(obj.nn());
	//species
	size+=nbytes(obj.nspecies());//nspecies_
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		size+=nbytes(obj.basisR(j));
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			size+=nbytes(obj.basisA(j,k));
		}
	}
	//return the size
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNH& obj, char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"pack(const NNH&,char*):\n";
	int pos=0;
	//hamiltonian
	pos+=pack(obj.atom(),arr+pos);
	pos+=pack(obj.nn(),arr+pos);
	//species
	pos+=pack(obj.nspecies(),arr+pos);
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		pos+=pack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			pos+=pack(obj.basisA(j,k),arr+pos);
		}
	}
	//return bytes written
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNH& obj, const char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"unpack(NNH&,const char*):\n";
	int pos=0;
	//hamiltonian
	pos+=unpack(obj.atom(),arr+pos);
	pos+=unpack(obj.nn(),arr+pos);
	obj.dOutDVal().resize(obj.nn());
	//species
	int nspecies=0;
	pos+=unpack(nspecies,arr+pos);
	obj.resize(nspecies);
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		pos+=unpack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			pos+=unpack(obj.basisA(j,k),arr+pos);
		}
	}
	//intialize the inputs and offsets
	obj.init_input();
	//return bytes read
	return pos;
}

}
