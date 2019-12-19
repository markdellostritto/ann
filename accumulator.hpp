#pragma once
#ifndef ACCUMULATOR_HPP
#define ACCUMULATOR_HPP

// c libaries
#include <cmath>
#include <cfloat>
// c++ libraries
#include <iosfwd>
// ann - fft
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
	Accumulator1D():Args()...{}
	Accumulator1D(const Accumulator1D<Args...>& a):Args(a)...{}
	~Accumulator1D(){}
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
	Min():min_(FLT_MAX){}
	~Min(){}
	//operators
	Min& operator+=(const Min& min){min_=(min.min()<min_)?min.min():min_;return *this;}
	//access
	const double& min()const{return min_;}
	//member functions
	void clear(){min_=FLT_MAX;}
	void push(double x){if(x<min_)min_=x;}
};

//***********************************************
//Max
//***********************************************

class Max{
private:
	double max_;
public:
	//constructors/destructors
	Max():max_(FLT_MIN){}
	~Max(){}
	//operators
	Max& operator+=(const Max& max){max_=(max.max()>max_)?max.max():max_;return *this;}
	//access
	const double& max()const{return max_;}
	//member functions
	void clear(){max_=FLT_MIN;}
	void push(double x){if(x>max_)max_=x;}
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
	Avg():N_(0),avg_(0){}
	~Avg(){}
	//operators
	Avg& operator+=(const Avg& avg){avg_+=avg.avg();N_+=avg.N();return *this;}
	//access
	const unsigned int& N()const{return N_;}
	double avg()const{return avg_;}
	//member functions
	void clear(){N_=0;avg_=0;}
	void push(double x){avg_+=(x-avg_)/(++N_);}
};

//***********************************************
//Variance
//***********************************************

class Var{
private:
	unsigned int N_;//total number of datums
	double avg_;//average of data
	double m2_;//sum of squares of differences from average
public:
	//constructors/destructors
	Var():N_(0),avg_(0),m2_(0){}
	~Var(){}
	//access
	const unsigned int& N()const{return N_;}
	double var()const{return m2_/(N_-1);}
	//member functions
	void clear(){N_=0;avg_=0;m2_=0;};
	void push(double x){
		++N_;
		const double delta_=(x-avg_);
		avg_+=delta_/N_;
		m2_+=delta_*(x-avg_);
	}
};

//***********************************************
//Velocity
//***********************************************

/*class Vel{
private:
	unsigned int t_;//total number of datums
	double ts_;//timestep
	double vel_;//velocity
public:
	//constructors/destructors
	Vel():t_(0),ts_(1.0),vel_(0.0){};
	~Vel(){};
	//access
	const unsigned int& t()const{return t_;};
	//member functions
	void clear(){t_=0;vel_=0;};
	void push();
};*/

//***********************************************
//Distribution
//***********************************************

#ifndef DEBUG_ACC_DIST
#define DEBUG_ACC_DIST 0
#endif 

class Dist{
private:
	unsigned int N_,M_,bc_,gc_;//nbins,buf size,buf count,global count
	double len_,min_,max_;//bin len, min, max
	bool norm_;//whether to normalize histogram
	double* hist_;//histogram
	double* bufx_;//buffer - x values
	double* bufy_;//buffer - y values
public:
	//constructors/destructors
	Dist():N_(0),M_(0),bc_(0),gc_(0),len_(0),min_(0),max_(0),norm_(true),hist_(NULL),bufx_(NULL),bufy_(NULL){}
	Dist(const Dist& dist);
	~Dist();
	//operators
	Dist& operator=(const Dist& dist);
	Dist& operator+=(const Dist& dist);
	//access
	bool& norm(){return norm_;};
	const bool& norm()const{return norm_;}
	const unsigned int& N()const{return N_;}
	const unsigned int& M()const{return M_;}
	const unsigned int& bc()const{return bc_;}
	const unsigned int& gc()const{return gc_;}
	const double& len()const{return len_;}
	const double& min()const{return min_;}
	const double& max()const{return min_;}
	double hist(unsigned int i)const{return hist_[i];}
	double bufx(unsigned int i)const{return bufx_[i];}
	double bufy(unsigned int i)const{return bufy_[i];}
	double abscissa(unsigned int i)const{return min_+len_*i;}
	double ordinate(unsigned int i)const{return hist_[i];}
	//member functions
	void clear();
	void init(double min, double max, unsigned int N, unsigned int M=1);
	unsigned int bin(double x);
	void push(double x);
	void push(double x, double y);
};

//operators

bool operator==(const Dist& dist1, const Dist& dist2);
inline bool operator!=(const Dist& dist1, const Dist& dist2){return !(dist1==dist2);};

//***********************************************
//Fourier
//***********************************************

#ifdef ACC_FFT
class Fourier{
private:
	unsigned int N_,n_;//fourier size, buf count
	fourier::FFT_R2C fft_;//fft object
	fftw_complex* f_;//fourier transform
	static const std::complex<double> I;//sqrt(-1)
public:
	//construtors/destructors
	Fourier():N_(0),n_(0),f_(NULL){}
	Fourier(unsigned int N){init(N);}
	~Fourier(){clear();}
	//operators
	Fourier& operator+=(const Fourier& fourier);
	//access
	const unsigned int& N()const{return N_;}
	const unsigned int& n()const{return n_;}
	const fourier::FFT_R2C& fft()const{return fft_;}
	const fftw_complex& f(unsigned int i)const{return f_[i];}
	const double fr(unsigned int i)const{return f_[i][0];}
	const double fi(unsigned int i)const{return f_[i][1];}
	//member functions
	void clear();
	void init(unsigned int N);
	void push(double x);
};
//operators
bool operator==(const Fourier& f1, const Fourier& f2);
bool operator!=(const Fourier& f1, const Fourier& f2){!(f1==f2);}
#endif

//***********************************************
//Accumulator2D
//***********************************************

template <typename... Args>
class Accumulator2D: public Args...{
private:

public:
	//constructors/destructors
	Accumulator2D():Args()...{}
	Accumulator2D(const Accumulator2D<Args...>& a):Args(a)...{}
	~Accumulator2D(){}
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
	PCorr():N_(0),avgX_(0),avgY_(0),covar_(0){}
	~PCorr(){}
	//access
	const unsigned int& N(){return N_;}
	double pcorr(){return covar_/std::sqrt(m2X_*m2Y_);}
	//member functions
	void clear(){N_=0;avgX_=0;avgX_=0;covar_=0;}
	void push(double x, double y){
		++N_;
		const double dX_=(x-avgX_);
		const double dY_=(y-avgY_);
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
	Covar():N_(0),avgX_(0),avgY_(0),covar_(0){}
	~Covar(){}
	//access
	const unsigned int& N(){return N_;}
	double covar(){return covar_/(N_-1);}
	//member functions
	void clear(){N_=0;avgX_=0;avgX_=0;covar_=0;}
	void push(double x, double y){
		++N_;
		const double dX_=(x-avgX_);
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
	LinReg():N_(0),avgX_(0),avgY_(0),m2X_(0),m2Y_(0),covar_(0){}
	~LinReg(){}
	//access
	const unsigned int& N(){return N_;}
	double m(){return covar_/m2X_;}
	double b(){return avgY_-m()*avgX_;}
	double r2(){return covar_*covar_/(m2X_*m2Y_);}
	//member functions
	void clear(){N_=0;avgX_=0;avgY_=0;m2X_=0;m2Y_=0;covar_=0;}
	void push(double x, double y){
		++N_;
		const double dX_=(x-avgX_);
		const double dY_=(y-avgY_);
		avgX_+=dX_/N_;
		avgY_+=dY_/N_;
		m2X_+=dX_*(x-avgX_);
		m2Y_+=dY_*(y-avgY_);
		covar_+=dX_*(y-avgY_);
	}
};

#endif