#include "math_special.hpp"

namespace special{
	
	//**************************************************************
	//Kummer's (confluent hypergeometric) function 
	//**************************************************************
	
	double M(double a, double b, double z, double prec){
		if(DEBUG_MATH_SPECIAL>0) std::cout<<"M(unsigned int,unsigned int,double):\n";
		unsigned int nMax=1e8;
		double fac=1,result=1;
		for(unsigned int n=1; n<=nMax; ++n){
			fac*=a*z/(n*b);
			result+=fac;
			++a; ++b;
			if(std::fabs(fac/result)*100<prec) break;
		}
		return result;
	}
	
	//**************************************************************
	//Legendre Polynomials
	//**************************************************************
	
	std::vector<double>& legendre(unsigned int n, std::vector<double>& c){
		if(n==0) c.resize(n+1,1.0);
		else if(n==1){c.resize(n+1,0.0); c[1]=1.0;}
		else {
			c.resize(n+1,0.0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1.0; ct2[1]=1.0;
			for(unsigned int m=2; m<=n; ++m){
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
	
	double chebyshev1r(unsigned int n, double x){
		if(n==0) return 1;
		else if(n==1) return x;
		else return 2*x*chebyshev1r(n-1,x)-chebyshev1r(n-2,x);
	}
	
	double chebyshev1l(unsigned int n, double x){
		if(n==0) return 1;
		double rnm1=x,rnm2=1,rn=x;
		for(unsigned int i=2; i<=n; ++i){
			rn=2*x*rnm1-rnm2;
			rnm2=rnm1; rnm1=rn;
		}
		return rn;
	}
	
	std::vector<double>& chebyshev1l(unsigned int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=x;
			for(unsigned int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	double chebyshev2r(unsigned int n, double x){
		if(n==0) return 1;
		else if(n==1) return 2*x;
		else return 2*x*chebyshev2r(n-1,x)-chebyshev2r(n-2,x);
	}
	
	double chebyshev2l(unsigned int n, double x){
		if(n==0) return 1;
		double rnm1=2*x,rnm2=1,rn=2*x;
		for(unsigned int i=2; i<=n; ++i){
			rn=2*x*rnm1-rnm2;
			rnm2=rnm1; rnm1=rn;
		}
		return rn;
	}
	
	std::vector<double>& chebyshev2l(unsigned int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=2*x;
			for(unsigned int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	std::vector<double>& chebyshev1_root(unsigned int n, std::vector<double>& r){
		r.resize(n);
		for(unsigned int i=0; i<n; i++) r[i]=std::cos((2.0*i+1.0)/(2.0*n)*num_const::PI);
		return r;
	}
	
	std::vector<double>& chebyshev2_root(unsigned int n, std::vector<double>& r){
		r.resize(n);
		for(unsigned int i=0; i<n; i++) r[i]=std::cos((i+1.0)/(n+1.0)*num_const::PI);
		return r;
	}
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	
	double jacobi(unsigned int n, double a, double b, double x){
		if(n==0) return 1;
		else if(n==1) return 0.5*(2*(a+1)+(a+b+2)*(x-1));
		else return 
			(2*n+a+b-1)*((2*n+a+b)*(2*n+a+b-2)*x+a*a-b*b)/(2*n*(n+a+b)*(2*n+a+b-2))*jacobi(n-1,a,b,x)
			-(n+a-1)*(n+b-1)*(2*n+a+b)/(n*(n+a+b)*(2*n+a+b-2))*jacobi(n-2,a,b,x);
	}
	
	std::vector<double>& jacobi(unsigned int n, double a, double b, std::vector<double>& c){
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
			for(unsigned int m=2; m<=n; ++m){
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
	
	std::vector<double>& laguerre(unsigned int n, std::vector<double>& c){
		if(n==0) c.resize(1,1);
		else if(n==1){
			c.resize(2,1);
			c[1]=-1;
		} else {
			c.resize(n+1,0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1;
			ct2[0]=1; ct2[1]=-1;
			for(unsigned int m=2; m<=n; ++m){
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
