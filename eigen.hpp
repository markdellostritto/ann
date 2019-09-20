#pragma once
#ifndef EIGEN_HPP
#define EIGEN_HPP

#define EIGEN_NO_DEBUG

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <Eigen/Dense>
#include "string.hpp"
#include "serialize.hpp"

namespace eigen{

struct LIN_SOLVER{
	enum type{
		LLT,//cholesky decomposition
		LDLT,//cholesky-variant
		PPLU,//LU - partial pivoting
		FPLU,//LU - full pivoting
		HQR,//Householder QR
		CPHQR,//Householder QR - column pivoting
		UNKNOWN
	};
	static LIN_SOLVER::type load(const char* str);
};
std::ostream& operator<<(std::ostream& out, const LIN_SOLVER::type& t);

Eigen::Vector3d& load(const char* str, Eigen::Vector3d& vec);

const char* print(char* str, const Eigen::Vector3d& vec);

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

template <> void pack(const Eigen::Vector3d& obj, char* arr);
template <> void pack(const Eigen::VectorXd& obj, char* arr);
template <> void pack(const Eigen::Matrix3d& obj, char* arr);
template <> void pack(const Eigen::MatrixXd& obj, char* arr);

//**********************************************
// unpacking
//**********************************************

template <> void unpack(Eigen::Vector3d& obj, const char* arr);
template <> void unpack(Eigen::VectorXd& obj, const char* arr);
template <> void unpack(Eigen::Matrix3d& obj, const char* arr);
template <> void unpack(Eigen::MatrixXd& obj, const char* arr);

}

#endif