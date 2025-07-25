#pragma once
#ifndef MATH_EUCLID_HPP
#define MATH_EUCLID_HPP

// c++
#include <iostream>
// eigen
#include <Eigen/Dense>

namespace math{

namespace euclid{

Eigen::Matrix3d rotate(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);
	
}

}

#endif