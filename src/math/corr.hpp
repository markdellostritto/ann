#ifndef CORR_HPP
#define CORR_HPP

// c++
#include <vector>
// Eigen
#include <Eigen/Dense>

namespace math{

namespace corr{
	
	double kendall(const double* x, const double* y, int n);
	double kendall(const std::vector<double>& x, const std::vector<double>& y);
	double kendall(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
	
}

}

#endif