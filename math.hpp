#ifndef MATH_HPP
#define MATH_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <boost/function.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include <Eigen/Dense>
#include "interpolation.hpp"
#include "num_const.hpp"

#ifndef DEBUG_MATH
#define DEBUG_MATH 0
#endif 

namespace cmp{
	
	template <class T> T max(T x1, T x2){return (x1>x2)?x1:x2;}
	template <class T> T min(T x1, T x2){return (x1>x2)?x2:x1;}
	template <class T> unsigned int delta(const T& x1, const T& x2){return (x1==x2);}
	
};

namespace misc{
	
	template <class T> inline T mod(T n, T z){return (n%z+z)%z;}
	template <class T> inline T mod(T n, T lLim, T uLim){return mod<T>(n-lLim,uLim-lLim)+lLim;}
	
	template<> inline int mod<int>(int n, int z){return (n%z+z)%z;}
	template<> inline int mod<int>(int n, int lLim, int uLim){return mod<int>(n-lLim,uLim-lLim)+lLim;}
	
	template<> inline double mod<double>(double n, double z){return fmod(fmod(n,z)+z,z);}
	template<> inline double mod<double>(double n, double lLim, double uLim){return mod<double>(n-lLim,uLim-lLim)+lLim;}
	
	template <class T> inline int sign(T x){return (x>0)-(x<0);}
	
	template <class T>
	int round(T x){return (std::fabs(x-(int)x)>=0.5) ? (int)x+1*sign(x) : (int)x;}
	
}

namespace function{
	
	double poly(double x, const std::vector<double>& a);
	double poly(double x, const double* a, unsigned int s);
	
	class Function{
	protected:
		std::vector<double> p_;
	public:
		Function(){};
		Function(const std::vector<double>& p):p_(p){};
		Function(double x0):p_(1){p_[0]=x0;};
		Function(double x0, double x1):p_(2){p_[0]=x0;p_[1]=x1;};
		Function(double x0, double x1, double x2):p_(3){p_[0]=x0;p_[1]=x1;p_[2]=x2;};
		virtual ~Function(){};
		std::vector<double>& p(){return p_;};
		const std::vector<double>& p()const{return p_;};
		virtual double operator()(double x)const=0;
	};
	
	class Step: public Function{
	public:
		Step():Function(0){};
		Step(double x0):Function(x0){};
		~Step(){};
		double& x0(){return p_[0];};
		const double& x0()const{return p_[0];};
		double operator()(double x)const{return x>p_[0];};
	};
	
	class Rect: public Function{
	public:
		Rect():Function(0,0){};
		Rect(double x1,double x2):Function(x1,x2){};
		~Rect(){};
		double& x1(){return p_[0];};
		const double& x1()const{return p_[0];};
		double& x2(){return p_[1];};
		const double& x2()const{return p_[1];};
		double operator()(double x)const{return (x>p_[0])*(x<p_[1]);};
	};
	
	class Gaussian: public Function{
	public:
		Gaussian():Function(0,1,1){};
		Gaussian(double c, double s, double a):Function(c,s,a){}
		~Gaussian(){};
		double& c(){return p_[0];};
		const double& c()const{return p_[0];};
		double& s(){return p_[1];};
		const double& s()const{return p_[1];};
		double& a(){return p_[2];};
		const double& a()const{return p_[2];};
		double operator()(double x)const{return p_[2]*std::exp(-(x-p_[0])*(x-p_[0])/(2*p_[1]*p_[1]));};
	};
	
	class StepCos: public Function{
	public:
		StepCos():Function(0,0){};
		StepCos(double x1, double x2):Function(x1,x2){};
		~StepCos(){};
		double& x1(){return p_[0];};
		const double& x1()const{return p_[0];};
		double& x2(){return p_[1];};
		const double& x2()const{return p_[1];};
		double operator()(double x)const;
	};
}

namespace special{
	
	static const double prec=1E-8;
	
	//**************************************************************
	//Sigmoid function
	//**************************************************************
	
	inline double sigmoid(double x){return 1.0/(1.0+std::exp(-x));}
	
	//**************************************************************
	//Kummer's (confluent hypergeometric) function 
	//**************************************************************
	
	double M(double a, double b, double z, double prec=1e-8);
	
	//**************************************************************
	//Legendre Poylnomials
	//**************************************************************
	std::vector<double>& legendre(unsigned int n, std::vector<double>& c);
	
	//**************************************************************
	//Chebyshev Polynomials
	//**************************************************************
	double chebyshev1r(unsigned int n, double x);
	double chebyshev1l(unsigned int n, double x);
	std::vector<double>& chebyshev1l(unsigned int n, double x, std::vector<double>& r);
	double chebyshev2r(unsigned int n, double x);
	double chebyshev2l(unsigned int n, double x);
	std::vector<double>& chebyshev2l(unsigned int n, double x, std::vector<double>& r);
	std::vector<double>& chebyshev1_root(unsigned int n, std::vector<double>& r);
	std::vector<double>& chebyshev2_root(unsigned int n, std::vector<double>& r);
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	double jacobi(unsigned int n, double a, double b, double x);
	std::vector<double>& jacobi(unsigned int n, double a, double b, std::vector<double>& c);
	
	//**************************************************************
	//Laguerre Polynomials
	//**************************************************************
	std::vector<double>& laguerre(unsigned int n, std::vector<double>& c);
}

namespace gradient{
	
	//*********************************************************
	//Vectors - Scalars
	//*********************************************************
	
	/*
		Symmetric first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	template <class T> T dc1o2(const std::vector<T>& v, T step, unsigned int t){
		return 0.5*(v[t+1]-v[t-1])/step;
	}
	template <class T> T dc1o4(const std::vector<T>& v, T step, unsigned int t){
		return (1.0/12.0*v[t-2]-2.0/3.0*v[t-1]+2.0/3.0*v[t+1]-1.0/12.0*v[t+2])/step;
	}
	template <class T> T dc1o6(const std::vector<T>& v, T step, unsigned int t){
		return (-1.0/60.0*v[t-3]+3.0/20.0*v[t-2]-3.0/4.0*v[t-1]+3.0/4.0*v[t+1]-3.0/20.0*v[t+2]+1.0/60.0*v[t+3])/step;
	}
	template <class T> T dc1o8(const std::vector<T>& v, T step, unsigned int t){
		return (1.0/280.0*v[t-4]-4.0/105.0*v[t-3]+1.0/5.0*v[t-2]-4.0/5.0*v[t-1]+4.0/5.0*v[t+1]-1.0/5.0*v[t+2]+4.0/105.0*v[t+3]-1.0/280.0*v[t+4])/step;
	}
	
	/*
		Forward first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	template <class T> T df1o1(const std::vector<T>& v, T step, unsigned int t){
		return (-v[t]+v[t+1])/step;
	}
	template <class T> T df1o2(const std::vector<T>& v, T step, unsigned int t){
		return (-3.0/2.0*v[t]+2*v[t+1]-1.0/2.0*v[t+2])/step;
	}
	template <class T> T df1o3(const std::vector<T>& v, T step, unsigned int t){
		return (-11.0/6.0*v[t]+3*v[t+1]-3.0/2.0*v[t+2]+1.0/3.0*v[t+3])/step;
	}
	template <class T> T df1o4(const std::vector<T>& v, T step, unsigned int t){
		return (-25.0/12.0*v[t]+4*v[t+1]-3*v[t+2]+4.0/3.0*v[t+3]-1.0/4.0*v[t+4])/step;
	}
	template <class T> T df1o5(const std::vector<T>& v, T step, unsigned int t){
		return (-137.0/60.0*v[t]+5*v[t+1]-5*v[t+2]+10.0/3.0*v[t+3]-5.0/4.0*v[t+4]+1.0/5.0*v[t+5])/step;
	}
	template <class T> T df1o6(const std::vector<T>& v, T step, unsigned int t){
		return (-49.0/20.0*v[t]+6*v[t+1]-15.0/2.0*v[t+2]+20.0/3.0*v[t+3]-15.0/4.0*v[t+4]+6.0/5.0*v[t+5]-1.0/6.0*v[t+6])/step;
	}
	
	/*
		Backward first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	template <class T> T db1o1(const std::vector<T>& v, T step, unsigned int t){
		return (v[t-1]+v[t])/step;
	}
	template <class T> T db1o2(const std::vector<T>& v, T step, unsigned int t){
		return (1.0/2.0*v[t-2]-2*v[t-1]+3.0/2.0*v[t])/step;
	}
	template <class T> T db1o3(const std::vector<T>& v, T step, unsigned int t){
		return (-1.0/3.0*v[t-3]+3.0/2.0*v[t-2]-3*v[t-1]+11.0/6.0*v[t])/step;
	}
	template <class T> T db1o4(const std::vector<T>& v, T step, unsigned int t){
		return (1.0/4.0*v[t-4]-4.0/3.0*v[t-3]+3*v[t-2]-4*v[t-1]+25.0/12.0*v[t])/step;
	}
	template <class T> T db1o5(const std::vector<T>& v, T step, unsigned int t){
		return (-1.0/5.0*v[t-5]+5.0/4.0*v[t-4]-10.0/3.0*v[t-3]+5*v[t-2]-5*v[t-1]+137.0/60.0*v[t])/step;
	}
	template <class T> T db1o6(const std::vector<T>& v, T step, unsigned int t){
		return (1.0/6.0*v[t-6]-6.0/5.0*v[t-5]+15.0/4.0*v[t-4]-20.0/3.0*v[t-3]+15.0/2.0*v[t-2]-6*v[t-1]+49.0/20.0*v[t])/step;
	}
	
	//*********************************************************
	//Vectors - Eigen
	//*********************************************************
	
	/*
		Symmetric first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	inline Eigen::Vector3d& dc1o2(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=0.5*(r[t+1]-r[t-1])/step;
	}
	inline Eigen::Vector3d& dc1o4(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/12.0*r[t-2]-2.0/3.0*r[t-1]+2.0/3.0*r[t+1]-1.0/12.0*r[t+2])/step;
	}
	inline Eigen::Vector3d& dc1o6(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-1.0/60.0*r[t-3]+3.0/20.0*r[t-2]-3.0/4.0*r[t-1]+3.0/4.0*r[t+1]-3.0/20.0*r[t+2]+1.0/60.0*r[t+3])/step;
	}
	inline Eigen::Vector3d& dc1o8(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/280.0*r[t-4]-4.0/105.0*r[t-3]+1.0/5.0*r[t-2]-4.0/5.0*r[t-1]+4.0/5.0*r[t+1]-1.0/5.0*r[t+2]+4.0/105.0*r[t+3]-1.0/280.0*r[t+4])/step;
	}
	
	/*
		Forward first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	inline Eigen::Vector3d& df1o1(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-r[t]+r[t+1])/step;
	}
	inline Eigen::Vector3d& df1o2(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-3.0/2.0*r[t]+2*r[t+1]-1.0/2.0*r[t+2])/step;
	}
	inline Eigen::Vector3d& df1o3(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-11.0/6.0*r[t]+3*r[t+1]-3.0/2.0*r[t+2]+1.0/3.0*r[t+3])/step;
	}
	inline Eigen::Vector3d& df1o4(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-25.0/12.0*r[t]+4*r[t+1]-3*r[t+2]+4.0/3.0*r[t+3]-1.0/4.0*r[t+4])/step;
	}
	inline Eigen::Vector3d& df1o5(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-137.0/60.0*r[t]+5*r[t+1]-5*r[t+2]+10.0/3.0*r[t+3]-5.0/4.0*r[t+4]+1.0/5.0*r[t+5])/step;
	}
	inline Eigen::Vector3d& df1o6(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-49.0/20.0*r[t]+6*r[t+1]-15.0/2.0*r[t+2]+20.0/3.0*r[t+3]-15.0/4.0*r[t+4]+6.0/5.0*r[t+5]-1.0/6.0*r[t+6])/step;
	}
	
	/*
		Backward first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	inline Eigen::Vector3d& db1o1(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(r[t-1]+r[t])/step;
	}
	inline Eigen::Vector3d& db1o2(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/2.0*r[t-2]-2*r[t-1]+3.0/2.0*r[t])/step;
	}
	inline Eigen::Vector3d& db1o3(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-1.0/3.0*r[t-3]+3.0/2.0*r[t-2]-3*r[t-1]+11.0/6.0*r[t])/step;
	}
	inline Eigen::Vector3d& db1o4(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/4.0*r[t-4]-4.0/3.0*r[t-3]+3*r[t-2]-4*r[t-1]+25.0/12.0*r[t])/step;
	}
	inline Eigen::Vector3d& db1o5(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-1.0/5.0*r[t-5]+5.0/4.0*r[t-4]-10.0/3.0*r[t-3]+5*r[t-2]-5*r[t-1]+137.0/60.0*r[t])/step;
	}
	inline Eigen::Vector3d& db1o6(const std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/6.0*r[t-6]-6.0/5.0*r[t-5]+15.0/4.0*r[t-4]-20.0/3.0*r[t-3]+15.0/2.0*r[t-2]-6*r[t-1]+49.0/20.0*r[t])/step;
	}
	
	//*********************************************************
	//Arrays - Scalar
	//*********************************************************
	
	/*
		Symmetric first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	template <class T> T dc1o2(const T* v, T step, unsigned int t){
		return 0.5*(v[t+1]-v[t-1])/step;
	}
	template <class T> T dc1o4(const T* v, T step, unsigned int t){
		return (1.0/12.0*v[t-2]-2.0/3.0*v[t-1]+2.0/3.0*v[t+1]-1.0/12.0*v[t+2])/step;
	}
	template <class T> T dc1o6(const T* v, T step, unsigned int t){
		return (-1.0/60.0*v[t-3]+3.0/20.0*v[t-2]-3.0/4.0*v[t-1]+3.0/4.0*v[t+1]-3.0/20.0*v[t+2]+1.0/60.0*v[t+3])/step;
	}
	template <class T> T dc1o8(const T* v, T step, unsigned int t){
		return (1.0/280.0*v[t-4]-4.0/105.0*v[t-3]+1.0/5.0*v[t-2]-4.0/5.0*v[t-1]+4.0/5.0*v[t+1]-1.0/5.0*v[t+2]+4.0/105.0*v[t+3]-1.0/280.0*v[t+4])/step;
	}
	
	/*
		Forward first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	template <class T> T df1o1(const T* v, T step, unsigned int t){
		return (-v[t]+v[t+1])/step;
	}
	template <class T> T df1o2(const T* v, T step, unsigned int t){
		return (-3.0/2.0*v[t]+2*v[t+1]-1.0/2.0*v[t+2])/step;
	}
	template <class T> T df1o3(const T* v, T step, unsigned int t){
		return (-11.0/6.0*v[t]+3*v[t+1]-3.0/2.0*v[t+2]+1.0/3.0*v[t+3])/step;
	}
	template <class T> T df1o4(const T* v, T step, unsigned int t){
		return (-25.0/12.0*v[t]+4*v[t+1]-3*v[t+2]+4.0/3.0*v[t+3]-1.0/4.0*v[t+4])/step;
	}
	template <class T> T df1o5(const T* v, T step, unsigned int t){
		return (-137.0/60.0*v[t]+5*v[t+1]-5*v[t+2]+10.0/3.0*v[t+3]-5.0/4.0*v[t+4]+1.0/5.0*v[t+5])/step;
	}
	template <class T> T df1o6(const T* v, T step, unsigned int t){
		return (-49.0/20.0*v[t]+6*v[t+1]-15.0/2.0*v[t+2]+20.0/3.0*v[t+3]-15.0/4.0*v[t+4]+6.0/5.0*v[t+5]-1.0/6.0*v[t+6])/step;
	}
	
	/*
		Backward first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	template <class T> T db1o1(const T* v, T step, unsigned int t){
		return (v[t-1]+v[t])/step;
	}
	template <class T> T db1o2(const T* v, T step, unsigned int t){
		return (1.0/2.0*v[t-2]-2*v[t-1]+3.0/2.0*v[t])/step;
	}
	template <class T> T db1o3(const T* v, T step, unsigned int t){
		return (-1.0/3.0*v[t-3]+3.0/2.0*v[t-2]-3*v[t-1]+11.0/6.0*v[t])/step;
	}
	template <class T> T db1o4(const T* v, T step, unsigned int t){
		return (1.0/4.0*v[t-4]-4.0/3.0*v[t-3]+3*v[t-2]-4*v[t-1]+25.0/12.0*v[t])/step;
	}
	template <class T> T db1o5(const T* v, T step, unsigned int t){
		return (-1.0/5.0*v[t-5]+5.0/4.0*v[t-4]-10.0/3.0*v[t-3]+5*v[t-2]-5*v[t-1]+137.0/60.0*v[t])/step;
	}
	template <class T> T db1o6(const T* v, T step, unsigned int t){
		return (1.0/6.0*v[t-6]-6.0/5.0*v[t-5]+15.0/4.0*v[t-4]-20.0/3.0*v[t-3]+15.0/2.0*v[t-2]-6*v[t-1]+49.0/20.0*v[t])/step;
	}
	
	//*********************************************************
	//Arrays - Eigen
	//*********************************************************
	
	/*
		Symmetric first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	inline Eigen::Vector3d& dc1o2(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=0.5*(r[t+1]-r[t-1])/step;
	}
	inline Eigen::Vector3d& dc1o4(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/12.0*r[t-2]-2.0/3.0*r[t-1]+2.0/3.0*r[t+1]-1.0/12.0*r[t+2])/step;
	}
	inline Eigen::Vector3d& dc1o6(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-1.0/60.0*r[t-3]+3.0/20.0*r[t-2]-3.0/4.0*r[t-1]+3.0/4.0*r[t+1]-3.0/20.0*r[t+2]+1.0/60.0*r[t+3])/step;
	}
	inline Eigen::Vector3d& dc1o8(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/280.0*r[t-4]-4.0/105.0*r[t-3]+1.0/5.0*r[t-2]-4.0/5.0*r[t-1]+4.0/5.0*r[t+1]-1.0/5.0*r[t+2]+4.0/105.0*r[t+3]-1.0/280.0*r[t+4])/step;
	}
	
	/*
		Forward first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	inline Eigen::Vector3d& df1o1(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-r[t]+r[t+1])/step;
	}
	inline Eigen::Vector3d& df1o2(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-3.0/2.0*r[t]+2*r[t+1]-1.0/2.0*r[t+2])/step;
	}
	inline Eigen::Vector3d& df1o3(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-11.0/6.0*r[t]+3*r[t+1]-3.0/2.0*r[t+2]+1.0/3.0*r[t+3])/step;
	}
	inline Eigen::Vector3d& df1o4(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-25.0/12.0*r[t]+4*r[t+1]-3*r[t+2]+4.0/3.0*r[t+3]-1.0/4.0*r[t+4])/step;
	}
	inline Eigen::Vector3d& df1o5(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-137.0/60.0*r[t]+5*r[t+1]-5*r[t+2]+10.0/3.0*r[t+3]-5.0/4.0*r[t+4]+1.0/5.0*r[t+5])/step;
	}
	inline Eigen::Vector3d& df1o6(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-49.0/20.0*r[t]+6*r[t+1]-15.0/2.0*r[t+2]+20.0/3.0*r[t+3]-15.0/4.0*r[t+4]+6.0/5.0*r[t+5]-1.0/6.0*r[t+6])/step;
	}
	
	/*
		Backward first-order derivatives
		o(n) corresponds to error in h (i.e. error~O(h^n))
	*/
	
	inline Eigen::Vector3d& db1o1(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(r[t-1]+r[t])/step;
	}
	inline Eigen::Vector3d& db1o2(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/2.0*r[t-2]-2*r[t-1]+3.0/2.0*r[t])/step;
	}
	inline Eigen::Vector3d& db1o3(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-1.0/3.0*r[t-3]+3.0/2.0*r[t-2]-3*r[t-1]+11.0/6.0*r[t])/step;
	}
	inline Eigen::Vector3d& db1o4(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/4.0*r[t-4]-4.0/3.0*r[t-3]+3*r[t-2]-4*r[t-1]+25.0/12.0*r[t])/step;
	}
	inline Eigen::Vector3d& db1o5(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(-1.0/5.0*r[t-5]+5.0/4.0*r[t-4]-10.0/3.0*r[t-3]+5*r[t-2]-5*r[t-1]+137.0/60.0*r[t])/step;
	}
	inline Eigen::Vector3d& db1o6(const Eigen::Vector3d*& r, Eigen::Vector3d& v, double step, unsigned int t){
		return v.noalias()=(1.0/6.0*r[t-6]-6.0/5.0*r[t-5]+15.0/4.0*r[t-4]-20.0/3.0*r[t-3]+15.0/2.0*r[t-2]-6*r[t-1]+49.0/20.0*r[t])/step;
	}
}

namespace roots{
	
	typedef boost::function<double(double)> func;
	
	struct Root{
		//global variables
		static double prec;//requested accuracy
		static unsigned int nMax;//max number of iterations
		static unsigned int N;//number of iterations (testing purposes)
		//root finding functions
		static double bisect(const func& f, double xMin, double xMax);
		static double ridder(const func& f, double xMin, double xMax);
		static double NR(const func& f, const func& df, double xMin, double xMax);
	};
	
	struct RootPoly{
		//root finding functions
		static std::vector<double>& eigen(const std::vector<double>& a, std::vector<double>& r);
	};
	
}

namespace integration{
	
	//***************************************************************
	//Quadrature class
	//***************************************************************
	
	class Quadrature{
	protected:
		unsigned int n_;
		unsigned int nMin_;
		unsigned int nMax_;
		double prec_;
		
		//member functions
		double init(boost::function<double(double)>& f, double& a, double& b, double& s);
		double next(boost::function<double(double)>& f, double& a, double& b, double& s);
	public:
		//constructors/destructors
		Quadrature():n_(0),nMin_(4),nMax_(20),prec_(1E-6){};
		~Quadrature(){};
		
		//access
		double& prec(){return prec_;};
		const double& prec()const{return prec_;};
		unsigned int& nMin(){return nMin_;};
		const unsigned int nMin()const{return nMin_;};
		unsigned int n()const{return n_;};
		
		//member functions
		double error(boost::function<double(double)>& f, double a, double b, unsigned int n);
		
		//virtual functions
		virtual double integrate(boost::function<double(double)>& f, double a, double b)=0;
	};
	
	//***************************************************************
	//Trapezoid class
	//***************************************************************
	
	class Trapezoid: public Quadrature{
	public:
		//constructors/destructors
		Trapezoid(){};
		~Trapezoid(){};
		
		//member functions
		double integrate(boost::function<double(double)>& f, double a, double b);
	};
	
	//***************************************************************
	//Simpson class
	//***************************************************************
	
	class Simpson: public Quadrature{
	public:
		//constructors/destructors
		Simpson(){};
		~Simpson(){};
		
		//member functions
		double integrate(boost::function<double(double)>& f, double a, double b);
	};
	
	//***************************************************************
	//Romberg class
	//***************************************************************
	
	class Romberg: public Quadrature{
	public:
		//constructors/destructors
		Romberg(){};
		~Romberg(){};
		
		//member functions
		double integrate(boost::function<double(double)>& f, double a, double b);
	};
	
	//***************************************************************
	//Gauss-Legendre Quadrature class
	//***************************************************************
	
	class QuadGaussLegendre{
	private:
		double prec_;
		unsigned int order_;
		std::vector<double> coeffs_,x_,w_;
	public:
		//constructors/destructors
		QuadGaussLegendre();
		QuadGaussLegendre(unsigned int order);
		QuadGaussLegendre(double prec, unsigned int order);
		~QuadGaussLegendre(){};
		
		//access
		double prec()const{return prec_;};
		unsigned int order()const{return order_;};
		const std::vector<double>& coeffs()const{return coeffs_;};
		const std::vector<double>& x()const{return x_;};
		const std::vector<double>& w()const{return w_;};
		
		//member functions
		static void weights(unsigned int n, std::vector<double>& x, std::vector<double>& w);
		double integrate(boost::function<double(double)>& f, double a, double b);
	};
	
	//***************************************************************
	//Gauss-Laguerre Quadrature class
	//***************************************************************
	
	class QuadGaussLaguerre{
	private:
		double prec_;
		unsigned int order_;
		std::vector<double> coeffs_,x_,w_;
	public:
		//constructors/destructors
		QuadGaussLaguerre();
		QuadGaussLaguerre(unsigned int order);
		QuadGaussLaguerre(double prec, unsigned int order);
		~QuadGaussLaguerre(){};
		
		//access
		double prec()const{return prec_;};
		unsigned int order()const{return order_;};
		const std::vector<double>& coeffs()const{return coeffs_;};
		const std::vector<double>& x()const{return x_;};
		const std::vector<double>& w()const{return w_;};
		
		//member functions
		static void weights(unsigned int n, std::vector<double>& x, std::vector<double>& w);
		double integrate(boost::function<double(double)>& f);
	};
	
	//***************************************************************
	//Gauss-Jacobi Quadrature class
	//***************************************************************
	
	class QuadGaussJacobi{
	private:
		double prec_;
		unsigned int order_;
		double a_,b_;//exponents
		std::vector<double> coeffs_,x_,w_;
	public:
		//constructors/destructors
		QuadGaussJacobi();
		QuadGaussJacobi(unsigned int order, double a, double b);
		QuadGaussJacobi(double prec, unsigned int order, double a, double b);
		~QuadGaussJacobi(){};
		
		//access
		double prec()const{return prec_;};
		unsigned int order()const{return order_;};
		double a()const{return a_;};
		double b()const{return b_;};
		const std::vector<double>& coeffs()const{return coeffs_;};
		const std::vector<double>& x()const{return x_;};
		const std::vector<double>& w()const{return w_;};
		
		//member functions
		static void weights(unsigned int n, double a, double b, std::vector<double>& x, std::vector<double>& w);
		double integrate(boost::function<double(double)>& f, double a, double b);
	};
	
}

namespace geom{

	Eigen::Matrix3d& cosineM(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Matrix3d& mat, bool norm=true);
	Eigen::Matrix3d& RX(double theta, Eigen::Matrix3d& mat);
	Eigen::Matrix3d& RY(double theta, Eigen::Matrix3d& mat);
	Eigen::Matrix3d& RZ(double theta, Eigen::Matrix3d& mat);
}

#endif
