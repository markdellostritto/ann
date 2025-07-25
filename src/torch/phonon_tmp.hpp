#ifndef PHONON_HPP
#define PHONON_HPP

// c++
#include <iostream>
#include <vector>
// fftw3
#include <fftw3.h>
// math
#include "mem/tensor.hpp"
// torch
#include "torch/engine.hpp"

#ifndef PHONON_PRINT_FUNC
#define PHONON_PRINT_FUNC 1
#endif

#ifndef PHONON_PRINT_STATUS
#define PHONON_PRINT_STATUS 1
#endif

#ifndef PHONON_PRINT_DATA
#define PHONON_PRINT_DATA 1
#endif

namespace phonon{

//******************************************************************
// KPath
//******************************************************************

class KPath{
private:
	int N_;//total number of points
	int m_;//total number of k-points
	std::vector<int> npts_;//number of points between each k-point
	std::vector<Eigen::Vector3d> kpts_;//special k-points
	std::vector<Eigen::Vector3d> kpath_;//all k-points
public:
	//==== constructors/destructors ====
	KPath():N_(0),m_(0){}
	~KPath(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const KPath& kpath);
	
	//==== access ====
	const int& N(){return N_;}
	const int& m(){return m_;}
	const std::vector<Eigen::Vector3d>& kpts(){return kpts_;}
	Eigen::Vector3d& kpts(int i){return kpts_[i];}
	const Eigen::Vector3d& kpts(int i)const{return kpts_[i];}
	int& npts(int i){return npts_[i];}
	const int& npts(int i)const{return npts_[i];}
	const Eigen::Vector3d& kpath(int i)const{return kpath_[i];}
	
	//==== member functions ====
	void resize(int n);
	void init();
};

//******************************************************************
// DynMat
//******************************************************************

class DynMat{
private:
	//k-space lattice
	int np_;//total number of k-pace lattice points
	double radnpi_;//inverse of sqrt of N
	Eigen::Vector3i nlat_;//k-space lattice dimension
	
	//dynamical matrix
	int nModes_;
	Tensor<Eigen::MatrixXcd> mat_;//dynamical matrix
	Tensor<Eigen::VectorXcd> evals_;//eigenvalues
	Tensor<Eigen::VectorXd> omega_;//frequencies
	double wmin_,wmax_;
	Eigen::VectorXd mvec_;
	int nasr_;
	
	//vibrational modes
	Structure pcell_;//primitive unit cell
	Structure scell_;//super cell
	std::vector<Tensor<int> > map_;
public:
	//==== constructors/destructors ====
	DynMat():nasr_(0),np_(0),radnpi_(0),nlat_(Eigen::Vector3i::Zero()){}
	~DynMat(){}
	
	//==== access ====
	//k-space lattice
		const int& np()const{return np_;}
		const double& radnpi()const{return radnpi_;}
		const Eigen::Vector3i& nlat()const{return nlat_;}
	//dynamical matrix
		const int& nModes()const{return nModes_;}
		Tensor<Eigen::MatrixXcd>& mat(){return mat_;}
		const Tensor<Eigen::MatrixXcd>& mat()const{return mat_;}
		Tensor<Eigen::VectorXcd>& evals(){return evals_;}
		const Tensor<Eigen::VectorXcd>& evals()const{return evals_;}
		Tensor<Eigen::VectorXd>& omega(){return omega_;}
		const Tensor<Eigen::VectorXd>& omega()const{return omega_;}
		double& wmin(){return wmin_;}
		const double& wmin()const{return wmin_;}
		double& wmax(){return wmax_;}
		const double& wmax()const{return wmax_;}
		int& nasr(){return nasr_;}
		const int& nasr()const{return nasr_;}
	//vibrational modes
		Structure& pcell(){return pcell_;}
		const Structure& pcell()const{return pcell_;}
		Structure& scell(){return scell_;}
		const Structure& scell()const{return scell_;}
		const std::vector<Tensor<int> >& map()const{return map_;}
	
	//==== member functions ====
	void resize(const Structure& pcell, const Eigen::Vector3i& nlat);
	void compute(int N, double dr, Engine& engine);
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
	void compute(const DynMat& dynmat);
};

}

#endif