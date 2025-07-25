#ifndef THOLE_HPP
#define THOLE_HPP

// c++
#include <iostream>
// eigen
#include <Eigen/Dense>
// structure
#include "struc/structure.hpp"

class Thole{
private:
	//size
		int nAtoms_;
	//scale
		double a_;
	//polarizability
		Eigen::Matrix3d atot_;
	//matrix
		Eigen::VectorXd ai_;
		Eigen::MatrixXd A_;
		Eigen::MatrixXd AI_;
		Eigen::MatrixXd T_;
		Eigen::MatrixXd S_;
public:
	//==== constructors/destructors ====
	Thole():a_(1.0){}
	Thole(int nAtoms):a_(1.0){resize(nAtoms);}
	~Thole(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Thole& thole);
	
	//==== access ====
	//parameters
	const int& nAtoms()const{return nAtoms_;}
	double& a(){return a_;}
	const double& a()const{return a_;}
	//matrix
	Eigen::Matrix3d& atot(){return atot_;}
	const Eigen::Matrix3d& atot()const{return atot_;}
	Eigen::VectorXd& ai(){return ai_;}
	const Eigen::VectorXd& ai()const{return ai_;}
	Eigen::MatrixXd& A(){return A_;}
	const Eigen::MatrixXd& A()const{return A_;}
	Eigen::MatrixXd& AI(){return AI_;}
	const Eigen::MatrixXd& AI()const{return AI_;}
	Eigen::MatrixXd& T(){return T_;}
	const Eigen::MatrixXd& T()const{return T_;}
	Eigen::MatrixXd& S(){return S_;}
	const Eigen::MatrixXd& S()const{return S_;}
	
	//==== member functions ===
	void resize(int nAtoms);
	Eigen::Matrix3d& compute(const Structure& struc);
	void gradient(const Structure& struc);
};

#endif