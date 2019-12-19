#pragma once
#ifndef TYPEDEF_HPP
#define TYPEDEF_HPP

#define EIGEN_NO_DEBUG

//c++ libraries
#include <vector>
//Eigen
#include <Eigen/Dense>

typedef unsigned short ushort;
typedef unsigned int uint;
typedef std::vector<unsigned int> vuint;
typedef Eigen::Vector3d vec3d;
typedef Eigen::VectorXd vecXd;
typedef Eigen::MatrixXd matXd;
typedef std::vector<Eigen::VectorXd> VecList;
typedef std::vector<Eigen::MatrixXd> MatList;

#endif