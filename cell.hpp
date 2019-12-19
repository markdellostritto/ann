#pragma once
#ifndef CELL_HPP
#define CELL_HPP

// c++ libraries
#include <iosfwd>
// eigen libraries
#include <Eigen/Dense>
// ann - serialization
#include "serialize.hpp"

//****************************************************************
//Cell class
//****************************************************************

class Cell{
private:
	//members
	double scale_;//the scale factor by which the lattice vector matrices are multiplied by
	double vol_;//the volume of the simulation cell
	Eigen::Matrix3d R_;//the lattice vector matrix (lattice vectors are columns of the matrix)
	Eigen::Matrix3d RInv_;//the inverse of the lattice vector matrix
	Eigen::Matrix3d K_;//the repiprocal lattice vector matrix (lattice vectors are columns of the matrix
	Eigen::Matrix3d KInv_;//the inverse of the reciprocal lattice vector matrix
public:	
	//constructors/destructors
	Cell(){defaults();}
	Cell(const Eigen::Matrix3d& R){init(R,1);}
	Cell(const Eigen::Matrix3d& R, double scale){init(R,scale);}
	Cell(const Eigen::Vector3d& R1,const Eigen::Vector3d& R2,const Eigen::Vector3d& R3):scale_(1){init(R1,R2,R3);}
	Cell(const Eigen::Vector3d& R1,const Eigen::Vector3d& R2,const Eigen::Vector3d& R3, double scale):scale_(1){init(R1,R2,R3);}
	~Cell(){}
	
	//operators
	Cell& operator=(const Cell& cell);
	friend std::ostream& operator<<(std::ostream& out, const Cell& cell);
	
	//access
	double& scale(){return scale_;}
	const double& scale()const{return scale_;}
	double& vol(){return vol_;}
	const double& vol()const{return vol_;}
	const Eigen::Matrix3d& R()const{return R_;}
	const Eigen::Matrix3d& RInv()const{return RInv_;}
	const Eigen::Matrix3d& K()const{return K_;}
	const Eigen::Matrix3d& KInv()const{return KInv_;}
	
	//static vector functions
	static Eigen::Vector3d& sum(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& sum, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv);
	static Eigen::Vector3d& diff(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& diff, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv);
	static double dist(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& temp, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv);
	static double dist(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv);
	static Eigen::Vector3d& fracToCart(const Eigen::Vector3d& vFrac, Eigen::Vector3d& vCart, const Eigen::Matrix3d& R);
	static Eigen::Vector3d& cartToFrac(const Eigen::Vector3d& vCart, Eigen::Vector3d& vFrac, const Eigen::Matrix3d& RInv);
	static Eigen::Vector3d& returnToCell(const Eigen::Vector3d& v1, Eigen::Vector3d& v2, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv);
	
	//static functions - modification
	static Cell& super(const Eigen::Vector3i& f, Cell& cell);
	
	//member functions
	void defaults();
	void clear(){defaults();};
	void init(const Eigen::Matrix3d& R, double scale=1.0);
	void init(const Eigen::Vector3d& R1,const Eigen::Vector3d& R2,const Eigen::Vector3d& R3, double scale=1.0);
};

bool operator==(const Cell& c1, const Cell& c2);	
bool operator!=(const Cell& c1, const Cell& c2);

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************

	template <> unsigned int nbytes(const Cell& obj);
	
	//**********************************************
	// packing
	//**********************************************

	template <> unsigned int pack(const Cell& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************

	template <> unsigned int unpack(Cell& obj, const char* arr);
	
}

#endif
