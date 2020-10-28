#ifndef MATH_SPECIAL_ANN_HPP
#define MATH_SPECIAL_ANN_HPP

// c libaries
#include <cmath>
// c++ libraries
#include <iosfwd>
#include <vector>
// ann - math
#include "math_const_ann.h"

#ifndef DEBUG_MATH_SPECIAL
#define DEBUG_MATH_SPECIAL 0
#endif 

namespace math{
	
namespace special{
	
	static const double prec=1E-8;
	
	//**************************************************************
	//trig (fdlibm)
	//**************************************************************
	
	static const double cos_const[6]={
		4.16666666666666019037e-02,
		-1.38888888888741095749e-03,
		2.48015872894767294178e-05,
		-2.75573143513906633035e-07,
		2.08757232129817482790e-09,
		-1.13596475577881948265e-11
	};
	//cosine function
	double cos(double x);
	
	static const double sin_const[6]={
		-1.66666666666666324348e-01,
		8.33333333332248946124e-03,
		-1.98412698298579493134e-04,
		2.75573137070700676789e-06,
		-2.50507602534068634195e-08,
		1.58969099521155010221e-10
	};
	//sine function
	double sin(double x);
	
	//**************************************************************
	//Exponential
	//**************************************************************
	template <unsigned int N> inline double expl(double x){
		x=1.0+x/std::pow(2,N);
		for(unsigned int i=0; i<N; ++i) x*=x;
		return x;
	}
	template <> inline double expl<4>(double x){
		x=1.0+x/16.0;
		x*=x; x*=x; x*=x; x*=x;
		return x;
	};
	template <> inline double expl<6>(double x){
		x=1.0+x/64.0;
		x*=x; x*=x; x*=x; x*=x; x*=x; x*=x;
		return x;
	};
	template <> inline double expl<8>(double x){
		x=1.0+x/256.0;
		x*=x; x*=x; x*=x; x*=x;
		x*=x; x*=x; x*=x; x*=x;
		return x;
	};
	static union{
		double d;
		struct{
			#ifdef LITTLE_ENDIAN
				int j,i;
			#else
				int i,j;
			#endif
		} n;
	} eco;
	static const double EXPA=1048576.0/math::constant::LOG2;
	static const double EXPB=1072693248.0;
	static const double EXPC=60801.0;
	inline double expb(const double x){return (eco.n.i=EXPA*x+(EXPB-EXPC),eco.d);}
	
	//**************************************************************
	//Logarithm
	//**************************************************************
	
	static const double logp1c[5]={1.0,1.0/3.0,1.0/5.0,1.0/7.0,1.0/9.0};
	double logp1(double x);//log(x+1)
	
	//**************************************************************
	//Sigmoid function
	//**************************************************************
	
	inline double sigmoid(double x){return 1.0/(1.0+std::exp(-x));}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	static const double sfpc[5]={1.0,1.0/3.0,1.0/5.0,1.0/7.0,1.0/9.0};
	double softplus(double x);
	
	//**************************************************************
	//Error Function - Approximations
	//**************************************************************
	
	struct erfa_const{
		static const double a1[5];
		static const double a2[5];
		static const double a3[7];
		static const double a4[7];
	};
	double erfa1(double x);//max error: 5e-4
	double erfa2(double x);//max error: 2.5e-5
	double erfa3(double x);//max error: 3e-7
	double erfa4(double x);//max error: 1.5e-7
	
	//**************************************************************
	//Gamma Function
	//**************************************************************
	
	static const double gammac[15]={
		0.99999999999999709182,
		57.156235665862923517,
		-59.597960355475491248,
		14.136097974741747174,
		-0.49191381609762019978,
		0.33994649984811888699e-4,
		0.46523628927048575665e-4,
		-0.98374475304879564677e-4,
		0.15808870322491248884e-3,
		-0.21026444172410488319e-3,
		0.21743961811521264320e-3,
		-0.16431810653676389022e-3,
		0.84418223983852743293e-4,
		-0.26190838401581408670e-4,
		0.36899182659531622704e-5
	};
	double lgamma(double x);
	double tgamma(double x);
	
	//**************************************************************
	//Beta Function
	//**************************************************************
	
	double beta(double z, double w);
	
}

namespace poly{
	
	//**************************************************************
	//Legendre Poylnomials
	//**************************************************************
	double legendre(int n, double x);
	std::vector<double>& legendre_c(int n, std::vector<double>& c);
	
	//**************************************************************
	//Chebyshev Polynomials
	//**************************************************************
	double chebyshev1l(int n, double x);//chebyshev polynomial of the first kind
	double chebyshev2l(int n, double x);//chebyshev polynomial of the second kind
	std::vector<double>& chebyshev1_c(int n, double x, std::vector<double>& r);//polynomial coefficients
	std::vector<double>& chebyshev2_c(int n, double x, std::vector<double>& r);//polynomial coefficients
	std::vector<double>& chebyshev1_r(int n, std::vector<double>& r);//polynomial roots
	std::vector<double>& chebyshev2_r(int n, std::vector<double>& r);//polynomial roots
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	double jacobi(int n, double a, double b, double x);
	std::vector<double>& jacobi(int n, double a, double b, std::vector<double>& c);
	
	//**************************************************************
	//Laguerre Polynomials
	//**************************************************************
	double laguerre(int n, double x);
	std::vector<double>& laguerre_c(int n, std::vector<double>& c);
	
}

}

#endif
