#pragma once
#ifndef GRID_HPP
#define GRID_HPP

//c++ libraries
#include <iosfwd>
#include <vector>
//Eigen
#include <Eigen/Dense>

//**********************************************************************************************
//Grid
//**********************************************************************************************

class Grid{
private:
	Eigen::Vector3i n_;//number of grid points in the x,y,z directions
	Eigen::Vector3d origin_;//grid origin
	Eigen::Matrix3d voxel_;//voxel lattice vectors
	std::vector<double> data_;//data stored on grid points
public:
	//==== constructors/destructors ====
	Grid():n_(Eigen::Vector3i::Zero()){}
	Grid(const Eigen::Vector3i& n){resize(n);}
	~Grid(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Grid& grid);
	double& operator()(int i){return data_[i];}
	const double& operator()(int i)const{return data_[i];}
	double& operator()(int i, int j, int k){return data_[i*n_[1]*n_[2]+j*n_[2]+k];}
	const double& operator()(int i, int j, int k)const{return data_[i*n_[1]*n_[2]+j*n_[2]+k];}
	
	//==== access ====
	//number of points
	int np()const{return n_[0]*n_[1]*n_[2];}
	const Eigen::Vector3i& n()const{return n_;}
	int n(int i)const{return n_[i];}
	//origin
	Eigen::Vector3d& origin(){return origin_;}
	const Eigen::Vector3d& origin()const{return origin_;}
	//voxel
	Eigen::Matrix3d& voxel(){return voxel_;}
	const Eigen::Matrix3d& voxel()const{return voxel_;}
	//data
	std::vector<double> data(){return data_;}
	int index(int i, int j, int k){return i*n_[1]*n_[2]+j*n_[2]+k;}
	double& data(int i){return data_[i];}
	const double& data(int i)const{return data_[i];}
	double& data(int i, int j, int k){return data_[i*n_[1]*n_[2]+j*n_[2]+k];}
	const double& data(int i, int j, int k)const{return data_[i*n_[1]*n_[2]+j*n_[2]+k];}
	
	//==== member functions ====
	void clear();
	void resize(const Eigen::Vector3i& n);
	void resize(const Eigen::Matrix3d& R, const Eigen::Vector3d l);
	
	//==== static functions ====
};

#endif