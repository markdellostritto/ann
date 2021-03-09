#pragma once
#ifndef TYPEDEF_HPP
#define TYPEDEF_HPP

#define EIGEN_NO_DEBUG

// extern - Eigen
#include <Eigen/Dense>

typedef Eigen::Matrix<double,3,1> Vec3d;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecXd;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatXd;

#endif
