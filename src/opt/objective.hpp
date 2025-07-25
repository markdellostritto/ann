#pragma once
#ifndef OBJECTIVE_HPP
#define OBJECTIVE_HPP

// c++
#include <iosfwd>
// eigen
#include <Eigen/Dense>
//serialization
#include "mem/serialize.hpp"

#ifndef OPT_OBJ_PRINT_FUNC
#define OPT_OBJ_PRINT_FUNC 0
#endif

namespace opt{

//***************************************************
// Objective
//***************************************************

class Objective{
private:
	//status
		double gamma_;//
		double val_,valOld_;//current, old value
		double dv_,dp_;//change in value, p
	//parameters
		int dim_;//dimension of problem
		Eigen::VectorXd p_,pOld_;//current, old parameters
		Eigen::VectorXd g_,gOld_;//current, old gradients
public:
	//==== constructors/destructors ====
	Objective(){defaults();}
	Objective(int dim){defaults();resize(dim);}
	~Objective(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Objective& data);
	
	//==== access ====
	//status
		double& gamma(){return gamma_;}
		const double& gamma()const{return gamma_;}
		double& val(){return val_;}
		const double& val()const{return val_;}
		double& valOld(){return valOld_;}
		const double& valOld()const{return valOld_;}
		double& dv(){return dv_;}
		const double& dv()const{return dv_;}
		double& dp(){return dp_;}
		const double& dp()const{return dp_;}
	//parameters
		int& dim(){return dim_;}
		const int& dim()const{return dim_;}
		Eigen::VectorXd& p(){return p_;}
		const Eigen::VectorXd& p()const{return p_;}
		Eigen::VectorXd& pOld(){return pOld_;}
		const Eigen::VectorXd& pOld()const{return pOld_;}
		Eigen::VectorXd& g(){return g_;}
		const Eigen::VectorXd& g()const{return g_;}
		Eigen::VectorXd& gOld(){return gOld_;}
		const Eigen::VectorXd& gOld()const{return gOld_;}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
	void resize(int dim);
};

}

namespace serialize{

	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const opt::Objective& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const opt::Objective& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(opt::Objective& obj, const char* arr);
	
}

#endif