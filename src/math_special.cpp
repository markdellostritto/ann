// c++ libraries
#include <ostream>
#include <stdexcept>
// ann - math - func
#include "math_func.hpp"
// ann - math - special
#include "math_special.hpp"

namespace math{

namespace special{
	
	//**************************************************************
	//trig (fdlibm)
	//**************************************************************
	
	//cosine function
	double cos(double x)noexcept{
		x*=x;
		return 1.0+x*(-0.5+x*(cos_const[0]+x*(cos_const[1]+x*(cos_const[2]+x*(cos_const[3]+x*(cos_const[4]+x*cos_const[5]))))));
	}
	
	//sine function
	double sin(double x)noexcept{
		const double r=x*x;
		return x*(1.0+r*(sin_const[0]+r*(sin_const[1]+r*(sin_const[2]+r*(sin_const[3]+r*(sin_const[4]+r*sin_const[5]))))));
	}
	
	//**************************************************************
	//Hypberbolic Function
	//**************************************************************
	
	double sinh(double x){
		if(x>=0){
			const double expf=std::exp(-x);
			return (1.0-expf*expf)/(2.0*expf);
		} else {
			const double expf=std::exp(x);
			return (expf*expf-1.0)/(2.0*expf);
		}
	}
	
	double cosh(double x){
		if(x>=0){
			const double expf=std::exp(-x);
			return (1.0+expf*expf)/(2.0*expf);
		} else {
			const double expf=std::exp(x);
			return (expf*expf+1.0)/(2.0*expf);
		}
	}
	
	double tanh(double x){
		if(x>=0){
			const double expf=std::exp(-2.0*x);
			return (1.0-expf)/(1.0+expf);
		} else {
			const double expf=std::exp(2.0*x);
			return (expf-1.0)/(expf+1.0);
		}
	}
	
	double csch(double x){
		if(x>=0){
			const double expf=std::exp(-x);
			return 2.0*expf/(1.0-expf*expf);
		} else {
			const double expf=std::exp(x);
			return 2.0*expf/(expf*expf-1.0);
		}
	}
	
	double sech(double x){
		if(x>=0){
			const double expf=std::exp(-x);
			return 2.0*expf/(1.0+expf*expf);
		} else {
			const double expf=std::exp(x);
			return 2.0*expf/(expf*expf+1.0);
		}
	}
	
	double coth(double x){
		if(x>=0){
			const double expf=std::exp(-2.0*x);
			return (1.0+expf)/(1.0-expf);
		} else {
			const double expf=std::exp(2.0*x);
			return (expf+1.0)/(expf-1.0);
		}
	}
	
	//**************************************************************
	//Logarithm
	//**************************************************************
	
	double logp1(double x)noexcept{
		const double y=x/(x+2.0);
		const double y2=y*y;
		return 2.0*y*(1.0+y2*(1.0/3.0+y2*(1.0/5.0+y2*(1.0/7.0+y2*1.0/9.0))));
	}
	
	//**************************************************************
	//Sigmoid
	//**************************************************************
	
	double sigmoid(double x){
		if(x>=0){
			return 1.0/(1.0+std::exp(-x));
		} else {
			const double expf=std::exp(x);
			return expf/(expf+1.0);
		}
	}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	double softplus(double x)noexcept{
		if(x>=1.0) return x+logp1(std::exp(-x));
		else return logp1(std::exp(x));
	}
	
	//**************************************************************
	//Complementary Error Function - Approximations
	//**************************************************************
	
	const double erfa_const::a1[5]={1.0,0.278393,0.230389,0.000972,0.078108};
	const double erfa_const::a2[5]={0.0,0.3480242,-0.0958798,0.7478556,0.47047};
	const double erfa_const::a3[7]={1.0,0.0705230784,0.0422820123,0.0092705272,0.0001520143,0.0002765672,0.0000430638};
	const double erfa_const::a4[7]={0.0,0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429,0.3275911};
	
	double erfa1(double x){const double s=func::sign(x); x*=s; x=func::poly<4>(x,erfa_const::a1); x*=x; x*=x; return s*(1.0-1.0/x);}
	double erfa2(double x){const double s=func::sign(x); x*=s; return s*(1.0-func::poly<3>(1.0/(1.0+erfa_const::a2[4]*x),erfa_const::a2)*std::exp(-x*x));}
	double erfa3(double x){const double s=func::sign(x); x*=s; x=func::poly<6>(x,erfa_const::a3); x*=x; x*=x; x*=x; x*=x; return s*(1.0-1.0/x);}
	double erfa4(double x){const double s=func::sign(x); x*=s; return s*(1.0-func::poly<5>(1.0/(1.0+erfa_const::a4[6]*x),erfa_const::a4)*std::exp(-x*x));}
	
	//**************************************************************
	//Gamma Function
	//**************************************************************
	
	double lgamma(double x){
		const double r2pi=2.5066282746310005;
		const double g=671.0/128.0;
		double s=gammac[0];
		for(int i=1; i<15; ++i) s+=gammac[i]/(x+i);
		return (x+0.5)*std::log(x+g)-(x+g)+std::log(r2pi*s/x);
	}
	double tgamma(double x){return std::exp(lgamma(x));}
	
	//**************************************************************
	//Beta Function
	//**************************************************************
	
	double beta(double z, double w){return std::exp(lgamma(z)+lgamma(w)-lgamma(z+w));}
	
	//**************************************************************
	//Kummer's (confluent hypergeometric) function 
	//**************************************************************
	
	double M(double a, double b, double z, double prec){
		const int nMax=1e8;
		double fac=1,result=1;
		for(int n=1; n<=nMax; ++n){
			fac*=a*z/(n*b);
			result+=fac;
			++a; ++b;
			if(std::fabs(fac/result)*100<prec) break;
		}
		return result;
	}

}

namespace poly{
	
	//**************************************************************
	//Legendre Polynomials
	//**************************************************************
	
	double legendre(int n, double x){
		if(n<0) throw std::runtime_error("legendre(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1.0,rm1=x,r=x;
			for(int i=2; i<=n; ++i){
				r=((2.0*n-1.0)*x*rm1-(n-1.0)*rm2)/i;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	std::vector<double>& legendre_c(int n, std::vector<double>& c){
		if(n==0) c.resize(n+1,1.0);
		else if(n==1){c.resize(n+1,0.0); c[1]=1.0;}
		else {
			c.resize(n+1,0.0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1.0; ct2[1]=1.0;
			for(int m=2; m<=n; ++m){
				for(int l=m; l>0; --l) c[l]=(2.0*m-1.0)/m*ct2[l-1];
				c[0]=0.0;
				for(int l=m; l>=0; --l) c[l]-=(m-1.0)/m*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
	
	//**************************************************************
	//Chebyshev Polynomials
	//**************************************************************
	
	double chebyshev1(int n, double x){
		if(n<0) throw std::runtime_error("chebyshev1(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1.0,rm1=x,r=x;
			for(int i=2; i<=n; ++i){
				r=2*x*rm1-rm2;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	double chebyshev2(int n, double x){
		if(n<0) throw std::runtime_error("chebyshev2(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1,rm1=2.0*x,r=2.0*x;
			for(int i=2; i<=n; ++i){
				r=2.0*x*rm1-rm2;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	std::vector<double>& chebyshev1_c(int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=x;
			for(int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	std::vector<double>& chebyshev2_c(int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=2*x;
			for(int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	std::vector<double>& chebyshev1_r(int n, std::vector<double>& r){
		r.resize(n);
		for(int i=0; i<n; i++) r[i]=std::cos((2.0*i+1.0)/(2.0*n)*constant::PI);
		return r;
	}
	
	std::vector<double>& chebyshev2_r(int n, std::vector<double>& r){
		r.resize(n);
		for(int i=0; i<n; i++) r[i]=std::cos((i+1.0)/(n+1.0)*constant::PI);
		return r;
	}
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	
	double jacobi(int n, double a, double b, double x){
		if(n==0) return 1;
		else if(n==1) return 0.5*(2*(a+1)+(a+b+2)*(x-1));
		else return 
			(2*n+a+b-1)*((2*n+a+b)*(2*n+a+b-2)*x+a*a-b*b)/(2*n*(n+a+b)*(2*n+a+b-2))*jacobi(n-1,a,b,x)
			-(n+a-1)*(n+b-1)*(2*n+a+b)/(n*(n+a+b)*(2*n+a+b-2))*jacobi(n-2,a,b,x);
	}
	
	std::vector<double>& jacobi(int n, double a, double b, std::vector<double>& c){
		if(n==0) c.resize(1,1);
		else if(n==1){
			c.resize(2,0);
			c[0]=0.5*(a-b);
			c[1]=0.5*(a+b+2);
		} else {
			c.resize(n+1,0.0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1.0;
			ct2[0]=0.5*(a-b);
			ct2[1]=0.5*(a+b+2);
			for(int m=2; m<=n; ++m){
				c[0]=0.0;
				for(int l=m; l>0; --l) c[l]=(2*m+a+b-1)*(2*m+a+b)/(2*m*(m+a+b))*ct2[l-1];
				for(int l=m; l>=0; --l) c[l]+=(2*m+a+b-1)*(a*a-b*b)/(2*m*(m+a+b)*(2*m+a+b-2))*ct2[l];
				for(int l=m; l>=0; --l) c[l]-=(m+a-1)*(m+b-1)*(2*m+a+b)/(m*(m+a+b)*(2*m+a+b-2))*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
	
	//**************************************************************
	//Laguerre Polynomials
	//**************************************************************
	
	double laguerre(int n, double x){
		if(n<0) throw std::runtime_error("chebyshev2(int,double): invalid order");
		else if(n==0) return 1;
		else {
			double rm2=1.0,rm1=1.0-x,r=1.0-x;
			for(int i=2; i<=n; ++i){
				r=((2.0*i-1.0-x)*rm1-(i-1.0)*rm2)/i;
				rm2=rm1; rm1=r;
			}
			return r;
		}
	}
	
	std::vector<double>& laguerre(int n, std::vector<double>& c){
		if(n==0) c.resize(1,1);
		else if(n==1){
			c.resize(2,1);
			c[1]=-1;
		} else {
			c.resize(n+1,0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1;
			ct2[0]=1; ct2[1]=-1;
			for(int m=2; m<=n; ++m){
				c[0]=0;
				for(int l=m; l>0; --l) c[l]=-1.0/m*ct2[l-1];
				for(int l=m; l>=0; --l) c[l]+=(2.0*m-1.0)/m*ct2[l];
				for(int l=m; l>=0; --l) c[l]-=(m-1.0)/m*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}

}

namespace pdist{
	
	//******************************************************
	// Distribution - Exp
	//******************************************************
	
	double Exp::p(double x){
		return beta_*std::exp(-beta_*x);
	}
	double Exp::cdf(double x){
		return 1.0-std::exp(-beta_*x);
	}
	double Exp::icdf(double x){
		return -std::log(1.0-x)/beta_;
	}
	
	//******************************************************
	// Distribution - Normal
	//******************************************************
	
	const double Normal::Rad2PI=std::sqrt(2.0*constant::PI);
	double Normal::p(double x){
		return std::exp(-(x-mu_)*(x-mu_)/(2.0*sigma_*sigma_))/(sigma_*Rad2PI);
	}
	double Normal::cdf(double x){
		return 0.5*std::erfc(-1.0/constant::Rad2*(x-mu_)/sigma_);
	}
	double Normal::icdf(double x){
		//return mu_-constant::Rad2*sigma_*1.0/std::erfc(2.0*x);
		return 0;
	}
	
	//******************************************************
	// Distribution - Logistic
	//******************************************************
	
	const double Logistic::a_=constant::PI/constant::Rad3;
	double Logistic::p(double x){
		const double expf=(x>=0)?std::exp(-x):std::exp(+x);
		return a_/sigma_*x/((1.0+x)*(1.0+x));
	}
	double Logistic::cdf(double x){
		return 0.5+1.0/constant::RadPI*std::atan((x-mu_)/sigma_);
	}
	double Logistic::icdf(double x){
		return mu_+sigma_*std::tan(constant::PI*(x-0.5));
	}
	
	//******************************************************
	// Distribution - Cauchy
	//******************************************************
	
	double Cauchy::p(double x){
		return 1.0/(sigma_*constant::PI*(1.0+(x-mu_)*(x-mu_)/sigma_*sigma_));
	}
	double Cauchy::cdf(double x){
		return 0.5+1.0/constant::RadPI*std::atan((x-mu_)/sigma_);
	}
	double Cauchy::icdf(double x){
		return mu_+sigma_*std::tan(constant::PI*(x-0.5));
	}
	
	//******************************************************
	// Distribution - LogNormal
	//******************************************************
	
	const double LogNormal::rad2Pi=constant::Rad2*constant::RadPI;
	double LogNormal::p(double x){
		const double l=std::log(x);
		return 1.0/(rad2Pi*sigma_*x)*std::exp(-(l-mu_)*(l-mu_)/(2.0*sigma_*sigma_));
	}
	double LogNormal::cdf(double x){
		return 0.5*std::erfc(-(std::log(x)-mu_)/(constant::Rad2*sigma_));
	}
	double LogNormal::icdf(double x){
		//return std::exp(mu_-std::Rad2*sigma_*erfci(2.0*x));
		return 0;
	}
	
	//******************************************************
	// Distribution - Gamma
	//******************************************************
	
	double Gamma::p(double x){
		return std::pow(beta_,alpha_)/std::tgamma(alpha_)*std::pow(x,alpha_-1.0)*std::exp(-beta_*x);
	}
	double Gamma::cdf(double x){
		return 0;
	}
	double Gamma::icdf(double x){
		return 0;
	}

}

}