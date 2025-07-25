// c++ libraries
#include <iostream>
#include <stdexcept>
// math 
#include "math/special.hpp"
#include "math/poly.hpp"

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
	
	double coscut(double x)noexcept{
		x-=0.5*math::constant::PI;
		const double x2=x*x;
		//return x*(-1.0+x2*(1.0/6.0+x2*(-1.0/120.0+x2*(1.0/5040.0+x2*(-1.0/362880.0+x2*1.0/39916800.0)))));//n11
		//return x*(-1.0+x2*(1.0/6.0+x2*(-1.0/120.0+x2*(1.0/5040.0+x2*(-1.0/362880.0+x2*(1.0/39916800.0-x2*1.0/6227020800.0))))));//n13
		return x*(-1.0+x2*(1.0/6.0+x2*(-1.0/120.0+x2*(1.0/5040.0+x2*(-1.0/362880.0+x2*(1.0/39916800.0+x2*(-1.0/6227020800.0+x2*1.0/1307674368000.0)))))));//n15
	}
	
	//sine function
	double sin(double x)noexcept{
		const double r=x*x;
		return x*(1.0+r*(sin_const[0]+r*(sin_const[1]+r*(sin_const[2]+r*(sin_const[3]+r*(sin_const[4]+r*sin_const[5]))))));
	}
	
	double sincut(double x)noexcept{
		x-=0.5*math::constant::PI;
		const double x2=x*x;
		//return 1.0+x2*(-1.0/2.0+x2*(1.0/24.0+x2*(-1.0/720.0+x2*(1.0/40320.0+x2*(-1.0/3628800.0+x2*1.0/479001600.0)))));//n12
		//return 1.0+x2*(-1.0/2.0+x2*(1.0/24.0+x2*(-1.0/720.0+x2*(1.0/40320.0+x2*(-1.0/3628800.0+x2*(1.0/479001600.0-x2*1.0/87178291200.0))))));//n14
		return 1.0+x2*(-1.0/2.0+x2*(1.0/24.0+x2*(-1.0/720.0+x2*(1.0/40320.0+x2*(-1.0/3628800.0+x2*(1.0/479001600.0+x2*(-1.0/87178291200.0+x2*1.0/20922789888000.0)))))));//n16
	}
	
	//**************************************************************
	//Hypberbolic Function
	//**************************************************************
	
	double sinh(double x){
		if(x>=0){
			const double expf=exp(-x);
			return (1.0-expf*expf)/(2.0*expf);
		} else {
			const double expf=exp(x);
			return (expf*expf-1.0)/(2.0*expf);
		}
	}
	
	double cosh(double x){
		if(x>=0){
			const double expf=exp(-x);
			return (1.0+expf*expf)/(2.0*expf);
		} else {
			const double expf=exp(x);
			return (expf*expf+1.0)/(2.0*expf);
		}
	}
	
	double tanh(double x){
		if(x>=0){
			const double expf=exp(-2.0*x);
			return (1.0-expf)/(1.0+expf);
		} else {
			const double expf=exp(2.0*x);
			return (expf-1.0)/(expf+1.0);
		}
	}
	
	double csch(double x){
		if(x>=0){
			const double expf=exp(-x);
			return 2.0*expf/(1.0-expf*expf);
		} else {
			const double expf=exp(x);
			return 2.0*expf/(expf*expf-1.0);
		}
	}
	
	double sech(double x){
		if(x>=0){
			const double expf=exp(-x);
			return 2.0*expf/(1.0+expf*expf);
		} else {
			const double expf=exp(x);
			return 2.0*expf/(expf*expf+1.0);
		}
	}
	
	double coth(double x){
		if(x>=0){
			const double expf=exp(-2.0*x);
			return (1.0+expf)/(1.0-expf);
		} else {
			const double expf=exp(2.0*x);
			return (expf+1.0)/(expf-1.0);
		}
	}
	
	void tanhsech(double x, double& ftanh, double& fsech){
		if(x>=0){
			const double fexp=exp(-x);
			const double fexp2=fexp*fexp;
			const double den=1.0/(1.0+fexp2);
			ftanh=(1.0-fexp2)*den;
			fsech=2.0*fexp*den;
		} else {
			const double fexp=exp(x);
			const double fexp2=fexp*fexp;
			const double den=1.0/(1.0+fexp2);
			ftanh=(fexp2-1.0)*den;
			fsech=2.0*fexp*den;
		}
	}
	
	double sech2(double x){
		if(x>=0){
			const double expf=exp(-2.0*x);
			const double den=1.0/(1.0+expf);
			return 4.0*expf*den*den;
		} else {
			const double expf=exp(2.0*x);
			const double den=1.0/(expf+1.0);
			return 4.0*expf*den*den;
		}
	}
	
	//**************************************************************
	//Power
	//**************************************************************
	
	double powint(double x, const int n){
		double yy, ww;
		if (n == 0) return 1.0;
		if (x == 0.0) return 0.0;
		int nn = (n > 0) ? n : -n;
		ww = x;
		for (yy = 1.0; nn != 0; nn >>= 1, ww *= ww)
		if (nn & 1) yy *= ww;
		return (n > 0) ? yy : 1.0 / yy;
	}
	
	
	double sqrta(double x){
		return (256.0+x*(1792.0+x*(1120.0+x*(112.0+x))))/(1024.0+x*(1792.0+x*(448.0+x*16)));
	}
	
	//**************************************************************
	//Sigmoid
	//**************************************************************
	
	double sigmoid(double x){
		if(x>=0){
			return 1.0/(1.0+exp(-x));
		} else {
			const double expf=exp(x);
			return expf/(expf+1.0);
		}
	}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	double softplus(double x)noexcept{
		if(x>=0.0) return x+log1p(exp(-x));
		else return log1p(exp(x));
	}
	
	double softplus2(double x)noexcept{
		if(x==0.0) return 1.0;
		else return x/(1.0-exp(-x*math::constant::Rad2));
	}
	
	double logcosh(double x)noexcept{
		if(x<-3.5){
			return -x-math::constant::LOG2;
		} else if(x>3.5){
			return x-math::constant::LOG2;
		} else {
			const double x2=x*x;
			return (
				x2*(1.0/2.0+
				x2*(11892482635958650.0/29240404197921777.0+
				x2*(102713256950199143.0/946832135932705160.0+
				x2*(376460448859668655.0/35790254738256255048.0+
				x2*133718113656599531929.0/488536977177197881405200.0))))
			)/(
				1.0+
				x2*(57316731943141859.0/58480808395843554.0+
				x2*(500864620636190840.0/1491260614094010627.0+
				x2*(700103608640134057.0/14912606140940106270.0+
				x2*(28019278674099557687.0/12213424429429947035130.0+
				x2*71434381678873132901.0/4030430061711882521592900.0))))
			);
		}
		/*
		if(x>0.0) return x+std::log(0.5*(1.0+expn10(-2.0*x)));
		else return -x+std::log(0.5*(1.0+expn10(2.0*x)));
		*/
	}
	
	//**************************************************************
	//Exponential
	//**************************************************************
	
	/* optimizer friendly implementation of exp2(x).
	*
	* strategy:
	*
	* split argument into an integer part and a fraction:
	* ipart = floor(x+0.5);
	* fpart = x - ipart;
	*
	* compute exp2(ipart) from setting the ieee754 exponent
	* compute exp2(fpart) using a pade' approximation for x in [-0.5;0.5[
	*
	* the result becomes: exp2(x) = exp2(ipart) * exp2(fpart)
	*/

	/* IEEE 754 double precision floating point data manipulation */
	typedef union {
		double   f;
		uint64_t u;
		struct {int32_t  i0,i1;} s;
	}  udi_t;

	static const double fm_exp2_q[] = {
	/*  1.00000000000000000000e0, */
		2.33184211722314911771e2,
		4.36821166879210612817e3
	};
	static const double fm_exp2_p[] = {
		2.30933477057345225087e-2,
		2.02020656693165307700e1,
		1.51390680115615096133e3
	};

	/* double precision constants */
	#define FM_DOUBLE_LOG2OFE  1.4426950408889634074
	
	double exp2_x86(double x){
	#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
		double   ipart, fpart, px, qx;
		udi_t    epart;

		ipart = floor(x+0.5);
		fpart = x - ipart;
		epart.s.i0 = 0;
		epart.s.i1 = (((int) ipart) + 1023) << 20;

		x = fpart*fpart;

		px =        fm_exp2_p[0];
		px = px*x + fm_exp2_p[1];
		qx =    x + fm_exp2_q[0];
		px = px*x + fm_exp2_p[2];
		qx = qx*x + fm_exp2_q[1];

		px = px * fpart;

		x = 1.0 + 2.0*(px/(qx-px));
		return epart.f*x;
	#else
		return pow(2.0, x);
	#endif
	}
	
	double fmexp(double x){
	#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
	    if (x < -1022.0/FM_DOUBLE_LOG2OFE) return 0;
	    if (x > 1023.0/FM_DOUBLE_LOG2OFE) return INFINITY;
	    return exp2_x86(FM_DOUBLE_LOG2OFE * x);
	#else
	    return ::exp(x);
	#endif
	}
	
	double expn10(double x) {
		x = 1.0 + x / 1024.0;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x; x *= x; x *= x;
		x *= x; x *= x;
		return x;
	}
	
	double gauss10(double x){
		x*=x;
		return 1.0/(1.0+x*(1.0+x*(1.0/2.0+x*(1.0/6.0+x*(1.0/24.0+x*1.0/120.0)))));
	}
	
	//**************************************************************
	//Error Function
	//**************************************************************
	
	double erfa1(double x){
		const double f=1.0/(1.0+x*(0.278393+x*(0.230389+x*(0.000972+x*0.078108))));
		const double f2=f*f;
		return 1.0-f2*f2;
	}
	
	//**************************************************************
	//Beta Function
	//**************************************************************
	
	double beta(double z, double w){return exp(lgamma(z)+lgamma(w)-lgamma(z+w));}
	double fratio(double num, double den){return exp(lgamma(num+1)-lgamma(den+1));}
	
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
			if(fabs(fac/result)*100<prec) break;
		}
		return result;
	}
	
	//**************************************************************
	//spherical harmonics - real
	//**************************************************************
	
	/*void YLMR::init(int l){
		if(l<0) throw std::invalid_argument("YLMR::init(int): Invalid angular momentum.");
		//clear
		clear();
		//set angular momentum constants
		l_=l; nl_=sqrt((2.0*l_+1.0)/(4.0*PI));
		//resize
		val_.resize(2*l_+1);
		nm_.resize(l_);
		pl_.resize(l_);
		//set constants
		for(int m=0; m<=l_; ++m){
			nm_[m]=sqrt(2.0*tgamma(l_-m+1.0)/tgamma(l_+m+1.0));
		}
	}
	
	void YLMR::clear(){
		val_.clear();
		nm_.clear();
		pl_.clear();
	}
	
	void YLMR::compute(double theta, double phi){
		const double fcos=cos(theta);
		//legendre polynomials
		for(int m=0; m<=l_; ++m) pl_[m]=poly::legendre(l_,fcos);
		//ylm
		for(int m=-l_; m<0; ++m){
			val_[l_+m]=pow(-1.0,m)*nl_*nm_[-m]*sin(-m*phi)*pl_[-m];
		}
		val_[l_+0]=nl_*pl_[0];
		for(int m=0; m<l_; ++m){
			val_[l_+m]=pow(-1.0,m)*nl_*nm_[m]*cos(m*phi)*pl_[m];
		}
	}
	
	double YLMR::val(int m){
		return val_[l_+m];
	}*/
	
}

}
