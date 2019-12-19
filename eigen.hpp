#pragma once
#ifndef EIGEN_HPP
#define EIGEN_HPP

#define EIGEN_NO_DEBUG

//c libraries
#include <cstdio>
//c++ libraries
#include <iosfwd>
//eigen
#include <Eigen/Dense>
//ann serialize
#include "serialize.hpp"

namespace eigen{

struct LIN_SOLVER{
	enum type{
		LLT=0,//cholesky decomposition
		LDLT=1,//cholesky-variant
		PPLU=2,//LU - partial pivoting
		FPLU=3,//LU - full pivoting
		HQR=4,//Householder QR
		CPHQR=5,//Householder QR - column pivoting
		UNKNOWN=-1
	};
	static LIN_SOLVER::type read(const char* str);
};
std::ostream& operator<<(std::ostream& out, const LIN_SOLVER::type& t);

}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const Eigen::Vector3d& obj);
template <> unsigned int nbytes(const Eigen::VectorXd& obj);
template <> unsigned int nbytes(const Eigen::Matrix3d& obj);
template <> unsigned int nbytes(const Eigen::MatrixXd& obj);

//**********************************************
// packing
//**********************************************

template <> unsigned int pack(const Eigen::Vector3d& obj, char* arr);
template <> unsigned int pack(const Eigen::VectorXd& obj, char* arr);
template <> unsigned int pack(const Eigen::Matrix3d& obj, char* arr);
template <> unsigned int pack(const Eigen::MatrixXd& obj, char* arr);

//**********************************************
// unpacking
//**********************************************

template <> unsigned int unpack(Eigen::Vector3d& obj, const char* arr);
template <> unsigned int unpack(Eigen::VectorXd& obj, const char* arr);
template <> unsigned int unpack(Eigen::Matrix3d& obj, const char* arr);
template <> unsigned int unpack(Eigen::MatrixXd& obj, const char* arr);

}

#endif