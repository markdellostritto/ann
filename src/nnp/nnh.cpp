// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// c++ libraries
#include <iostream>
// structure
#include "struc/structure.hpp"
// math
#include "math/const.hpp"
// str
#include "str/print.hpp"
#include "str/token.hpp"
// nnh
#include "nnp/nnh.hpp"

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
	out<<"type     = "<<nnh.type_<<"\n";
	//types
	out<<"n_types  = "<<nnh.ntypes_<<"\n";
	//potential parameters
	out<<"n_input  = "; std::cout<<nnh.nInput_<<" "; std::cout<<"\n";
	out<<"n_inputr = "; std::cout<<nnh.nInputR_<<" "; std::cout<<"\n";
	out<<"n_inputa = "; std::cout<<nnh.nInputA_<<" "; std::cout<<"\n";
	out<<nnh.nn_<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

/**
* set NNH defaults
*/
void NNH::defaults(){
	if(NNH_PRINT_FUNC>0) std::cout<<"NNH::defaults()\n";
	//hamiltonian
		ntypes_=0;
		type_.clear();
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
* resize the number of types
* @param ntypes - the total number of types
*/
void NNH::resize(int ntypes){
	if(NNH_PRINT_FUNC>0) std::cout<<"NNH::resize(int)\n";
	if(ntypes<0) throw std::invalid_argument("NNH::resize(int): invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		basisR_.resize(ntypes_);
		basisA_.resize(ntypes_);
		offsetR_.resize(ntypes_,0);
		offsetA_.resize(ntypes_,0);
	}
}

/**
* Initialize the number of inputs and offsets associated with the basis functions.
* Must be done after the basis has been defined, otherwise the values will make no sense.
* Different from resizing: resizing sets the number of types, this sets the number of inputs
* associated with the basis associated with each types.
*/
void NNH::init_input(){
	if(NNH_PRINT_FUNC>0) std::cout<<"NNH::init_input()\n";
	//**** radial ****
	nInputR_=0;
	for(int i=0; i<basisR_.size(); ++i){
		nInputR_+=basisR_[i].size();
		if(i==0) offsetR_[i]=0;
		else offsetR_[i]=offsetR_[i-1]+basisR_[i-1].size();
	}
	//**** angular ****
	nInputA_=0;
	for(int i=0; i<basisA_.size(); ++i){
		nInputA_+=basisA_[i].size();
		if(i==0) offsetA_[i]=0;
		else offsetA_[i]=offsetA_[i-1]+basisA_[i-1].size();
	}
	//total number of inputs
	nInput_=nInputR_+nInputA_;
}

/**
* compute energy of atom with symmetry function "symm"
* @param symm - the symmetry function
*/
double NNH::energy(const Eigen::VectorXd& symm){
	if(NNH_PRINT_FUNC>0) std::cout<<"NNH::energy(const Eigen::VectorXd&)\n";
	return nn_.fp(symm)[0]+type_.energy().val();
}

//************************************************************
// serialization
//************************************************************

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNH& obj){
	if(NNH_PRINT_FUNC>0) std::cout<<"nbytes(const NNH&):\n";
	int size=0;
	//hamiltonian
	size+=nbytes(obj.type());
	size+=nbytes(obj.nn());
	//types
	size+=nbytes(obj.ntypes());//ntypes_
	//basis for pair/triple interactions
	for(int j=0; j<obj.ntypes(); ++j){
		size+=nbytes(obj.basisR(j));
	}
	for(int j=0; j<obj.ntypes(); ++j){
		for(int k=j; k<obj.ntypes(); ++k){
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
	if(NNH_PRINT_FUNC>0) std::cout<<"pack(const NNH&,char*):\n";
	int pos=0;
	//hamiltonian
	pos+=pack(obj.type(),arr+pos);
	pos+=pack(obj.nn(),arr+pos);
	//types
	pos+=pack(obj.ntypes(),arr+pos);
	//basis for pair/triple interactions
	for(int j=0; j<obj.ntypes(); ++j){
		pos+=pack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.ntypes(); ++j){
		for(int k=j; k<obj.ntypes(); ++k){
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
	if(NNH_PRINT_FUNC>0) std::cout<<"unpack(NNH&,const char*):\n";
	int pos=0;
	//hamiltonian
	pos+=unpack(obj.type(),arr+pos);
	pos+=unpack(obj.nn(),arr+pos);
	obj.dOdZ().resize(obj.nn());
	//types
	int ntypes=0;
	pos+=unpack(ntypes,arr+pos);
	obj.resize(ntypes);
	//basis for pair/triple interactions
	for(int j=0; j<obj.ntypes(); ++j){
		pos+=unpack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.ntypes(); ++j){
		for(int k=j; k<obj.ntypes(); ++k){
			pos+=unpack(obj.basisA(j,k),arr+pos);
		}
	}
	//intialize the inputs and offsets
	obj.init_input();
	//return bytes read
	return pos;
}

}