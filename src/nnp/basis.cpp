// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// str
#include "str/string.hpp"
// nnp
#include "nnp/basis.hpp"
// math
#include "math/special.hpp"

/**
* constructor
* @param rc - Cutoff radius
* @param cutname - name of Cutoff function
* @param size - number of symmetry functions
*/
Basis::Basis(double rc, Cutoff::Name cutname, int size=0){
	if(BASIS_PRINT_FUNC>0) std::cout<<"Basis(double,Cutoff::Name,int):\n";
	cutoff_=Cutoff(cutname,rc);
	resize(size);
}

//==== member functions ====

/**
* clear basis
*/
void Basis::clear(){
	if(BASIS_PRINT_FUNC>0) std::cout<<"Basis::clear():\n";
	size_=0;
}

void Basis::resize(int size){
	if(BASIS_PRINT_FUNC>0) std::cout<<"Basis::resize(int):\n";
	if(size<0) throw std::invalid_argument("Basis::resize(int): invalid number of functions.");
	size_=size;
	if(size_>0){
		symm_.resize(size_);
	}
}
