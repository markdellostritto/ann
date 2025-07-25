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
#include "torch/qtpie.hpp"

//************************************************************
//QTPIE
//************************************************************

using math::constant::Rad2;
using math::constant::RadPI;

//==== constants ====

const double QTPIE::c=0.0634936359342409697857633049346360201550538992822729240811981742;

//==== operators ====

std::ostream& operator<<(std::ostream& out, const QTPIE& qtpie){
	if(qtpie.pot()!=nullptr) out<<"qtpie "<<qtpie.pot()->name();
	else out<<"qtpie UNKNOWN";
	return out;
}

//==== member functions ====

void QTPIE::qt(Structure& struc, const NeighborList& nlist){
	if(QTPIE_PRINT_FUNC>0) std::cout<<"QTPIE::qt(Structure&,const Ewald::Coulomb,const NeighborList&):\n";
	//local variables
	const double ke=units::Consts::ke();
	Eigen::Vector3d dr;
	
	//resize
	S_.resize(struc.nAtoms(),struc.nAtoms());
	A_.resize(struc.nAtoms()+1,struc.nAtoms()+1);
	b_.resize(struc.nAtoms()+1);
	x_.resize(struc.nAtoms()+1);
	
	//compute coulomb integrals
	Eigen::MatrixXd J;
	pot_->J(struc,nlist,J);
	
	//compute the overlap integrals
	S_.setZero();
	for(int i=0; i<struc.nAtoms(); ++i){
		const double ri=ptable::radius_covalent(struc.an(i));
		const double norm=c*0.35355339059327376220042218105242451/(ri*ri*ri);
		S_(i,i)=norm;
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const double rj=ptable::radius_covalent(struc.an(jj));
			const double g=1.0/std::sqrt(ri*ri+rj*rj);
			const double g2=g*g;
			const double dr=nlist.neigh(i,j).dr();
			S_(i,jj)+=c*g*g2*exp(-0.5*dr*dr*g2);
		}
	}
	
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
	for(int i=0; i<struc.nAtoms(); ++i){
		b_[i]=0.0;
		double norm=0.0;
		for(int j=0; j<struc.nAtoms(); ++j){
			b_[i]+=(struc.chi(i)-struc.chi(j))*S_(i,j);
			norm+=S_(i,j);
		}
		b_[i]*=-1.0/norm;
	}
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
	
	template <> int nbytes(const QTPIE& obj){
		if(QTPIE_PRINT_FUNC>0) std::cout<<"nbytes(const QTPIE&):\n";
		int size=0;
		size+=nbytes(obj.pot());
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const QTPIE& obj, char* arr){
		if(QTPIE_PRINT_FUNC>0) std::cout<<"pack(const QTPIE&,char*):\n";
		int pos=0;
		pos+=pack(obj.pot(),arr+pos);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(QTPIE& obj, const char* arr){
		if(QTPIE_PRINT_FUNC>0) std::cout<<"unpack(const QTPIE&,char*):\n";
		int pos=0;
		pos+=unpack(obj.pot(),arr+pos);
		return pos;
	}
	
}