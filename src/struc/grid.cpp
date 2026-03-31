//c++ libraries
#include <iostream>
// DAME - structure
#include "struc/grid.hpp"

//**********************************************************************************************
//Grid
//**********************************************************************************************


//operators

std::ostream& operator<<(std::ostream& out, const Grid& grid){
	out<<grid.n_.transpose()<<"\n";
	out<<grid.voxel_;
	return out;
}

//member functions

void Grid::clear(){
	n_=Eigen::Vector3i::Zero();
	data_.clear();
}

void Grid::resize(const Eigen::Vector3i& n){
	if(n[0]<=0 || n[1]<=0 || n[2]<=0) throw std::runtime_error("Invalid grid size");
	n_=n;
	const int np=n_[0]*n_[1]*n_[2];
	data_.resize(np,0);
}

void Grid::resize(const Eigen::Matrix3d& R, const Eigen::Vector3d l){
	if(R.col(0).norm()<=0 || R.col(1).norm()<=0 || R.col(1).norm()<=0) throw std::runtime_error("Invalid lattice matrix");
	if(l[0]<=0 || l[1]<=0 || l[2]<=0) throw std::runtime_error("Invalid voxel length");
	Eigen::Vector3d Rl; Rl<<R.col(0).norm(),R.col(1).norm(),R.col(2).norm();
	n_[0]=Rl[0]/l[0];
	n_[1]=Rl[1]/l[1];
	n_[2]=Rl[2]/l[2];
	voxel_.col(0)=R.col(0)/n_[0];
	voxel_.col(1)=R.col(1)/n_[1];
	voxel_.col(2)=R.col(2)/n_[2];
	resize(n_);
}