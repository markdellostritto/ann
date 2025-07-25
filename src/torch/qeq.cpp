// c++
#include <iostream>
// string
#include "str/string.hpp"
#include "str/print.hpp"
#include "str/token.hpp"
// math
#include "math/const.hpp"
// struc
#include "struc/neighbor.hpp"
// torch
#include "torch/pot_factory.hpp"
#include "torch/qeq.hpp"

//************************************************************
//QEQ
//************************************************************

using math::constant::Rad2;
using math::constant::RadPI;

//==== operators ====

std::ostream& operator<<(std::ostream& out, const QEQ& qeq){
	if(qeq.pot()!=nullptr) out<<"qeq "<<qeq.pot()->name();
	else out<<"qeq UNKNOWN";
	return out;
}

//==== member functions ====

void QEQ::qt(Structure& struc, const NeighborList& nlist){
	if(QEQ_PRINT_FUNC>0) std::cout<<"QEQ::qt(Structure&,const Ewald::Coulomb,const NeighborList&):\n";
	//local variables
	const double ke=units::Consts::ke();
	
	//resize
	A_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	b_.resize(struc.nAtoms()+1);
	x_.resize(struc.nAtoms()+1);
	
	//compute coulomb integrals
	Eigen::MatrixXd J;
	pot_->J(struc,nlist,J);
	
	//add coulomb integrals to A
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<struc.nAtoms(); ++j){
			A_(j,i)=J(j,i);
		}
		A_(struc.nAtoms(),i)=1.0;
	}
	for(int i=0; i<struc.nAtoms(); ++i) A_(i,struc.nAtoms())=1.0;
	A_(struc.nAtoms(),struc.nAtoms())=0.0;
	//add idempotential
	for(int i=0; i<struc.nAtoms(); ++i){
		A_(i,i)+=struc.eta(i);
	}
	
	//calculate the solution vector
	for(int i=0; i<struc.nAtoms(); ++i) b_[i]=-1.0*struc.chi(i);
	b_[struc.nAtoms()]=struc.qtot();
	
	//solve the linear equations
	x_.noalias()=A_.partialPivLu().solve(b_);
	
	//set the atomic charges
	for(int i=0; i<struc.nAtoms(); ++i) struc.charge(i)=x_[i];

}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const QEQ& obj){
		if(QEQ_PRINT_FUNC>0) std::cout<<"nbytes(const QEQ&):\n";
		int size=0;
		size+=nbytes(obj.pot());
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const QEQ& obj, char* arr){
		if(QEQ_PRINT_FUNC>0) std::cout<<"pack(const QEQ&,char*):\n";
		int pos=0;
		pos+=pack(obj.pot(),arr+pos);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(QEQ& obj, const char* arr){
		if(QEQ_PRINT_FUNC>0) std::cout<<"unpack(const QEQ&,char*):\n";
		int pos=0;
		pos+=unpack(obj.pot(),arr+pos);
		return pos;
	}
	
}