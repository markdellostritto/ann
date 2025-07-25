#pragma once
#ifndef KSPACE_LONDON_HPP
#define KSPACE_LONDON_HPP

// structure
#include "struc/structure.hpp"
// torch
#include "torch/kspace.hpp"

#ifndef KSPACEL_PRINT_FUNC
#define KSPACEL_PRINT_FUNC 0
#endif

#ifndef KSPACEL_PRINT_STATUS
#define KSPACEL_PRINT_STATUS 0
#endif

namespace KSpace{

class London: public Base{
private:
	double a3_;
	Eigen::MatrixXd b_;
	Eigen::VectorXd bs_;
public:
	//==== constructors/destructors ====
	London(){}
	~London(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const London& l);
	
	//==== access ====
	const double& a3()const{return a3_;}
	Eigen::MatrixXd& b(){return b_;}
	const Eigen::MatrixXd& b()const{return b_;}
	
	//==== static functions ====
	double fv(double pre, double a, double rc, double prec);
	double fd(double pre, double a, double rc, double prec);
	
	//==== member functions ====
	void init(const Structure& struc, const Eigen::MatrixXd& b);
	double energy(const Structure& struc)const;
	double compute(Structure& struc)const;
	//Eigen::MatrixXd& J(const Structure& struc, Eigen::MatrixXd& J)const;
};

}

#endif