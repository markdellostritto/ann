// math
#include "math/corr.hpp"
#include "math/special.hpp"

namespace math{

namespace corr{
	
	double kendall(const double* x, const double* y, int n){
		double tau=0;
		for(int i=0; i<n; ++i){
			const double xi=x[i];
			const double yi=y[i];
			for(int j=i+1; j<n; ++j){
				tau+=math::special::sgn(xi-x[j])*math::special::sgn(yi-y[j]);
			}
		}
		return 2.0*tau/(n*(n-1.0));
	}
	
	
	double kendall(const std::vector<double>& x, const std::vector<double>& y){
		return kendall(x.data(),y.data(),x.size());
	}
	
	double kendall(const Eigen::VectorXd& x, const Eigen::VectorXd& y){
		return kendall(x.data(),y.data(),x.size());
	}
}

}