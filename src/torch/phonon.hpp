#pragma once
#ifndef PHONON_HPP
#define PHONON_HPP

// c++
#include <iostream>
#include <vector>
// eigen
#include <Eigen/Dense>

#ifndef PHONON_PRINT_FUNC
#define PHONON_PRINT_FUNC 1
#endif

#ifndef PHONON_PRINT_STATUS
#define PHONON_PRINT_STATUS 1
#endif

#ifndef PHONON_PRINT_DATA
#define PHONON_PRINT_DATA 1
#endif

//******************************************************************
// KPath
//******************************************************************

class KPath{
private:
	int nkvecs_;//total number of points
	int nkpts_;//total number of k-points
	std::vector<int> npts_;//number of points between each k-point
	std::vector<Eigen::Vector3d> kpts_;//special k-points
	std::vector<Eigen::Vector3d> kvecs_;//all k-points
	std::vector<Eigen::VectorXd> kval_;//eigenvectors at each k-point
public:
	//==== constructors/destructors ====
	KPath():nkvecs_(0),nkpts_(0){}
	~KPath(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const KPath& kpath);
	
	//==== access ====
	const int& nkvecs(){return nkvecs_;}
	const int& nkpts(){return nkpts_;}
	const std::vector<Eigen::Vector3d>& kpts(){return kpts_;}
	Eigen::Vector3d& kpts(int i){return kpts_[i];}
	const Eigen::Vector3d& kpts(int i)const{return kpts_[i];}
	Eigen::VectorXd& kval(int i){return kval_[i];}
	const Eigen::VectorXd& kval(int i)const{return kval_[i];}
	int& npts(int i){return npts_[i];}
	const int& npts(int i)const{return npts_[i];}
	const Eigen::Vector3d& kvec(int i)const{return kvecs_[i];}
	
	//==== member functions ====
	void resize(int n);
	void init();
};

//******************************************************************
// DOS
//******************************************************************

class DOS{
private:
	std::vector<Eigen::Vector2d> dos_;//density of states
	double wlmin_;//freq min 
	double wlmax_;//freq max
	double dw_;//freq spacing
	double sigma_;//freq smearing
public:
	//==== density of states ====
	DOS():wlmin_(-1),wlmax_(-1),dw_(-1),sigma_(-1){}
	~DOS(){}
	
	//==== access ===
	double& wlmin(){return wlmin_;}
	const double& wlmin()const{return wlmin_;}
	double& wlmax(){return wlmax_;}
	const double& wlmax()const{return wlmax_;}
	double& dw(){return dw_;}
	const double& dw()const{return dw_;}
	double& sigma(){return sigma_;}
	const double& sigma()const{return sigma_;}
	std::vector<Eigen::Vector2d>& dos(){return dos_;}
	const std::vector<Eigen::Vector2d>& dos()const{return dos_;}
	Eigen::Vector2d& dos(int i){return dos_[i];}
	const Eigen::Vector2d& dos(int i)const{return dos_[i];}
	
	//==== member functions ====
	int size(){return dos_.size();}
	void resize(int s){dos_.resize(s,Eigen::Vector2d::Zero());}
	double w(int i){return wlmin_+dw_*i;}
};

#endif