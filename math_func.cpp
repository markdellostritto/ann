#include "math_func.hpp"

namespace function{
	
	//======== polynomial ========
	
	double poly(double x, const std::vector<double>& a){
		int j=a.size()-1;
		double result=a[j];
		while(j>0) result=x*result+a[--j];
		return result;
	}
	
	double poly(double x, const double* a, unsigned int s){
		double result=a[--s];
		while(s>0) result=x*result+a[--s];
		return result;
	}

}