#pragma once
#ifndef QEQ_HPP
#define QEQ_HPP

//c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
//c++ libraries
#include <vector>
#include <iosfwd>
#include <memory>
// struc
#include "struc/structure.hpp"
// math
#include "math/eigen.hpp"
// torch
#include "torch/pot.hpp"
// chem
#include "chem/ptable.hpp"
#include "chem/units.hpp"

#ifndef QEQ_PRINT_FUNC
#define QEQ_PRINT_FUNC 0
#endif

#ifndef QEQ_PRINT_STATUS
#define QEQ_PRINT_STATUS 0
#endif

//************************************************************
//QEQ
//************************************************************

class QEQ{
private:
	//potential
	std::shared_ptr<ptnl::Pot> pot_;
	
	//matrix utilities
	Eigen::MatrixXd A_;//coulomb matrix
	Eigen::VectorXd b_;//constant vector
	Eigen::VectorXd x_;//solution vector
public:
	//==== constructors/destructors ====
	QEQ(){defaults();}
	~QEQ(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const QEQ& qeq);
	
	//==== access ====
	std::shared_ptr<ptnl::Pot>& pot(){return pot_;}
	const std::shared_ptr<ptnl::Pot>& pot()const{return pot_;}
	const Eigen::MatrixXd& A()const{return A_;}
	const Eigen::VectorXd& b()const{return b_;}
	const Eigen::VectorXd& x()const{return x_;}
	
	//==== member functions ====
	void defaults(){};
	void clear(){defaults();}
	void qt(Structure& struc, const NeighborList& nlist);
};

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const QEQ& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const QEQ& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(QEQ& obj, const char* arr);
	
}

#endif