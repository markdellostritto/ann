#pragma once
#ifndef DATA_HPP
#define DATA_HPP

// c++ libaries
#include <vector>
// eigen
#include <Eigen/Dense>

typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecXd;

class MLData{
private:
	int size_;
	int dInp_,dOut_;
	std::vector<VecXd> inp_;
	std::vector<VecXd> out_;
public:
	//==== constructors/destructors ====
	MLData():size_(0),dInp_(0),dOut_(0){}
	MLData(int size, int dInp, int dOut){resize(size,dInp,dOut);}
	~MLData(){}
	
	//==== access ====
	const int& size()const{return size_;}
	const int& dInp()const{return dInp_;}
	const int& dOut()const{return dOut_;}
	VecXd& inp(int n){return inp_[n];}
	const VecXd& inp(int n)const{return inp_[n];}
	std::vector<VecXd>& inp(){return inp_;}
	const std::vector<VecXd>& inp()const{return inp_;}
	VecXd& out(int n){return out_[n];}
	const VecXd& out(int n)const{return out_[n];}
	std::vector<VecXd>& out(){return out_;}
	const std::vector<VecXd>& out()const{return out_;}
	
	//==== member functions ====
	void clear();
	void resize(int dInp, int dOut);
	void resize(int size, int dInp, int dOut);
	void push(const Eigen::VectorXd& inp, const Eigen::VectorXd& out);
};

#endif