#ifndef ACCUMULATOR_HPP
#define ACCUMULATOR_HPP

//c libaries
#include <cstdlib>
#include <cmath>
#include <cfloat>
//c++ libraries
#include <stdexcept>
#include <complex>
//local - math constants
#include "math_const.hpp"
//local - fft
#ifdef ACC_FFT
	#include "fft.hpp"
#endif

//***********************************************
//Accumulator1D
//***********************************************

template <typename... Args>
class Accumulator1D: public Args...{
public:
	//constructors/destructors
	Accumulator1D():Args()...{};
	Accumulator1D(const Accumulator1D<Args...>& a):Args(a)...{};
	~Accumulator1D(){};
	//operators
	Accumulator1D<Args...>& operator=(const Accumulator1D<Args...>& a);
	//members
	void clear();
	void push(double x);
};
template <typename... Args>
Accumulator1D<Args...>& Accumulator1D<Args...>::operator=(const Accumulator1D<Args...>& a){
	int arr[sizeof...(Args)]={(Args::operator=(a),0)...};
}
template <typename... Args>
void Accumulator1D<Args...>::clear(){
	int arr[sizeof...(Args)]={(Args::clear(),0)...};
}
template <typename... Args>
void Accumulator1D<Args...>::push(double x){
	int arr[sizeof...(Args)]={(Args::push(x),0)...};
}

//***********************************************
//Min
//***********************************************

class Min{
private:
	double min_;
public:
	//constructors/destructors
	Min():min_(FLT_MAX){};
	~Min(){};
	//access
	const double& min(){return min_;};
	//member functions
	void clear(){min_=FLT_MAX;};
	void push(double x){min_=(x<min_)?x:min_;};
};

//***********************************************
//Max
//***********************************************

class Max{
private:
	double max_;
public:
	//constructors/destructors
	Max():max_(FLT_MIN){};
	~Max(){};
	//access
	double max(){return max_;};
	//member functions
	void clear(){max_=FLT_MIN;};
	void push(double x){max_=(x>max_)?x:max_;};
};

//***********************************************
//Average
//***********************************************

class Avg{
private:
	unsigned int N_;
	double avg_;
public:
	//constructors/destructors
	Avg():N_(0),avg_(0){};
	~Avg(){};
	//access
	const unsigned int& N(){return N_;};
	double avg(){return avg_;};
	//member functions
	void clear(){N_=0;avg_=0;};
	void push(double x){avg_+=(x-avg_)/(++N_);}
};

//***********************************************
//Variance
//***********************************************

class Var{
private:
	unsigned int N_;
	double avg_;
	double m2_;
public:
	//constructors/destructors
	Var():N_(0),avg_(0),m2_(0){};
	~Var(){};
	//access
	const unsigned int& N()const{return N_;};
	double var(){return m2_/(N_-1);};
	//member functions
	void clear(){N_=0;avg_=0;m2_=0;};
	void push(double x){
		++N_;
		double delta_=(x-avg_);
		avg_+=delta_/N_;
		m2_+=delta_*(x-avg_);
	}
};

//***********************************************
//Distribution
//***********************************************

class Dist{
private:
	unsigned int N_,M_,bc_,gc_;
	double len_,min_,max_;
	double* hist_;
	double* buf_;
public:
	//constructors/destructors
	Dist():N_(0),M_(1000),bc_(0),gc_(0),len_(0),min_(0),max_(0),hist_(NULL),buf_(NULL){};
	Dist(const Dist& dist);
	~Dist();
	//operators
	Dist& operator=(const Dist& dist);
	//access
	const unsigned int& N()const{return N_;};
	unsigned int& M(){return M_;};
	const unsigned int& M()const{return M_;};
	const unsigned int& bc()const{return bc_;};
	const unsigned int& gc()const{return gc_;};
	const double& dist_len()const{return len_;};
	const double& dist_min()const{return min_;};
	const double& dist_max()const{return min_;};
	double hist(unsigned int i)const{return hist_[i];};
	double buf(unsigned int i)const{return buf_[i];};
	double abscissa(unsigned int i)const{return min_+len_*i;};
	double ordinate(unsigned int i)const{return hist_[i];};
	//member functions
	void clear();
	void init(double min, double max, unsigned int N);
	unsigned int bin(double x);
	void push(double x);
};

//***********************************************
//Fourier
//***********************************************

#ifdef ACC_FFT
class Fourier{
private:
	unsigned int N_,n_;
	fourier::FFT_R2C fft_;
	fftw_complex* f_;
	static const std::complex<double> I;
public:
	//construtors/destructors
	Fourier():N_(0),n_(0),f_(NULL){};
	Fourier(unsigned int N){init(N);};
	~Fourier(){clear();};
	//access
	const unsigned int& N()const{return N_;};
	const unsigned int& n()const{return n_;};
	const fftw_complex& f(unsigned int i)const{return f_[i];};
	const double fr(unsigned int i)const{return f_[i][0];};
	const double fi(unsigned int i)const{return f_[i][1];};
	//member functions
	void clear();
	void init(unsigned int N);
	void push(double x);
};
#endif

//***********************************************
//Accumulator2D
//***********************************************

template <typename... Args>
class Accumulator2D: public Args...{
private:

public:
	//constructors/destructors
	Accumulator2D():Args()...{};
	Accumulator2D(const Accumulator2D<Args...>& a):Args(a)...{};
	~Accumulator2D(){};
	//operators
	Accumulator2D<Args...>& operator=(const Accumulator2D<Args...>& a);
	//members
	void clear();
	void push(double x, double y);
};
template <typename... Args>
Accumulator2D<Args...>& Accumulator2D<Args...>::operator=(const Accumulator2D<Args...>& a){
	int arr[sizeof...(Args)]={(Args::operator=(a),0)...};
}
template <typename... Args>
void Accumulator2D<Args...>::clear(){
	int arr[sizeof...(Args)]={(Args::clear(),0)...};
}
template <typename... Args>
void Accumulator2D<Args...>::push(double x, double y){
	int arr[sizeof...(Args)]={(Args::push(x,y),0)...};
}

//***********************************************
//Pearson Correlation Coefficient
//***********************************************

class PCorr{
private:
	unsigned int N_;
	double avgX_,avgY_;
	double m2X_,m2Y_;
	double covar_;
public:
	//constructors/destructors
	PCorr():N_(0),avgX_(0),avgY_(0),covar_(0){};
	~PCorr(){};
	//access
	const unsigned int& N(){return N_;};
	double pcorr(){return covar_/std::sqrt(m2X_*m2Y_);};
	//member functions
	void clear(){N_=0;avgX_=0;avgX_=0;covar_=0;};
	void push(double x, double y){
		++N_;
		double dX_=(x-avgX_);
		double dY_=(y-avgY_);
		avgX_+=dX_/N_;
		avgY_+=dY_/N_;
		m2X_+=dX_*(x-avgX_);
		m2Y_+=dY_*(y-avgY_);
		covar_+=dX_*(y-avgY_);
	}
};

//***********************************************
//Covariance
//***********************************************

class Covar{
private:
	unsigned int N_;
	double avgX_,avgY_;
	double covar_;
public:
	//constructors/destructors
	Covar():N_(0),avgX_(0),avgY_(0),covar_(0){};
	~Covar(){};
	//access
	const unsigned int& N(){return N_;};
	double covar(){return covar_/(N_-1);};
	//member functions
	void clear(){N_=0;avgX_=0;avgX_=0;covar_=0;};
	void push(double x, double y){
		++N_;
		double dX_=(x-avgX_);
		avgX_+=dX_/N_;
		avgY_+=(y-avgY_)/N_;
		covar_+=dX_*(y-avgY_);
	}
};

//***********************************************
//LinReg
//***********************************************

class LinReg{
private:
	unsigned int N_;
	double avgX_,avgY_;
	double m2X_,m2Y_;
	double covar_;
public:
	//constructors/destructors
	LinReg():N_(0),avgX_(0),avgY_(0),m2X_(0),m2Y_(0),covar_(0){};
	~LinReg(){};
	//access
	const unsigned int& N(){return N_;};
	double m(){return covar_/m2X_;};
	double b(){return avgY_-m()*avgX_;};
	double r2(){return covar_*covar_/(m2X_*m2Y_);};
	//member functions
	void clear(){N_=0;avgX_=0;avgY_=0;m2X_=0;m2Y_=0;covar_=0;};
	void push(double x, double y){
		++N_;
		double dX_=(x-avgX_);
		double dY_=(y-avgY_);
		avgX_+=dX_/N_;
		avgY_+=dY_/N_;
		m2X_+=dX_*(x-avgX_);
		m2Y_+=dY_*(y-avgY_);
		covar_+=dX_*(y-avgY_);
	}
};

#endif