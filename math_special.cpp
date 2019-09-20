#include "math_special.hpp"

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
	//Logarithm
	//**************************************************************
	
	double logp1(double x)noexcept{
		return x*(logp1c[0]+x*(logp1c[1]+x*(logp1c[2]+x*(logp1c[3]+x*(logp1c[4]+x*logp1c[5])))));
	}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	double softplus(double x)noexcept{
		if(x>=1){
			return x+logp1(std::exp(-x));
		} else {
			const double exp=std::exp(x);
			const double f=exp/(exp+2.0);
			const double f2=f*f;
			return 2.0*f*(sfpc[0]+f2*(sfpc[1]+f2*(sfpc[2]+f2*(sfpc[3]+f2*sfpc[4]))));
		}
	}
	
	//**************************************************************
	//Complementary Error Function - Approximations
	//**************************************************************
	
	const double erfa_const::a1[5]={1.0,0.278393,0.230389,0.000972,0.078108};
	const double erfa_const::a2[5]={0.0,0.3480242,-0.0958798,0.7478556,0.47047};
	const double erfa_const::a3[7]={1.0,0.0705230784,0.0422820123,0.0092705272,0.0001520143,0.0002765672,0.0000430638};
	const double erfa_const::a4[7]={0.0,0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429,0.3275911};
	
	double erfa1(double x){double s=sign(x); x*=s; x=function::poly<4>(x,erfa_const::a1); x*=x; x*=x; return s*(1.0-1.0/x);}
	double erfa2(double x){double s=sign(x); x*=s; return s*(1.0-function::poly<3>(1.0/(1.0+erfa_const::a2[4]*x),erfa_const::a2)*std::exp(-x*x));}
	double erfa3(double x){double s=sign(x); x*=s; x=function::poly<6>(x,erfa_const::a3); x*=x; x*=x; x*=x; x*=x; return s*(1.0-1.0/x);}
	double erfa4(double x){double s=sign(x); x*=s; return s*(1.0-function::poly<5>(1.0/(1.0+erfa_const::a4[6]*x),erfa_const::a4)*std::exp(-x*x));}
	
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
