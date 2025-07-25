#pragma once
#ifndef PCA_HPP
#define PCA_HPP

// eigen
#include <Eigen/Dense>

class PCA{
private:
	int nobs_,nvars_;
	Eigen::MatrixXd X_;//data matrix
	Eigen::MatrixXd S_;//covariance matrix
	Eigen::MatrixXd W_;//eigenvector matrix
	Eigen::VectorXd w_;//eigenvalue matrix
public:
	//==== constructors/destructors ====
	PCA(){}
	PCA(int nobs, int nvars){resize(nobs,nvars);}
	~PCA(){}
	
	//==== access ====
	const int& nobs()const{return nobs_;}
	const int& nvars()const{return nvars_;}
	Eigen::MatrixXd& X(){return X_;}
	const Eigen::MatrixXd& X()const{return X_;}
	const Eigen::MatrixXd& S()const{return S_;}
	const Eigen::MatrixXd& W()const{return W_;}
	const Eigen::VectorXd& w()const{return w_;}
	int ci(int i)const{return nvars_-i-1;}
	
	//==== member functions ====
	void resize(int nobs, int nvars);
	void compute();
};

#endif