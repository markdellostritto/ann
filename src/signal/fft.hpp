#ifndef FFT_HPP
#define FFT_HPP

//c libraries
#include <cstdio>
#include <cmath>
#include <cstring>
//c++ libraries
#include <stdexcept>
#include <iosfwd>
#include <vector>
// eigen
#include <Eigen/Dense>
//fftw3
#include <fftw3.h>

#ifndef FFT_PRINT_FUNC
#define FFT_PRINT_FUNC 0
#endif

namespace fourier{
	
	//*****************************************************************
	// Data Type
	//*****************************************************************
	
	struct DataT{
		enum type{
			REAL=0,//real-valued
			COMPLEX=1,//complex-valued
			UNKNOWN=2
		};
	};
	std::ostream& operator<<(std::ostream& out, const DataT::type& t);
	
	template <int R, int I, int O>
	class FFT{
		
	};
	
	//*****************************************************************
	// FFT - 1D - COMPLEX - COMPLEX
	//*****************************************************************
	
	template <>
	class FFT<1,DataT::COMPLEX,DataT::COMPLEX>{
	private:
		int N_;//the number of data points
		int sign_;//the sign of the transform
		fftw_plan plan_;//the fourier transform plan
		fftw_complex* in_;
		fftw_complex* out_;
	public:
		//==== constructors/destructors ====
		FFT():N_(0),sign_(FFTW_FORWARD),in_(NULL),out_(NULL),plan_(NULL){}
		FFT(int N, int sign):N_(0),sign_(sign),in_(NULL),out_(NULL),plan_(NULL){resize(N);init();}
		FFT(const FFT<1,DataT::COMPLEX,DataT::COMPLEX>& fft);
		~FFT(){clear();}
		
		//==== operators ====
		FFT<1,DataT::COMPLEX,DataT::COMPLEX>& operator=(const FFT<1,DataT::COMPLEX,DataT::COMPLEX>& fft);
		
		//==== access ====
		const int& N()const{return N_;}
		int& sign(){return sign_;}
		const int& sign()const{return sign_;}
		fftw_complex* in(){return in_;}
		const fftw_complex* in()const{return in_;}
		fftw_complex& in(int i){return in_[i];}
		const fftw_complex& in(int i)const{return in_[i];}
		fftw_complex* out(){return out_;}
		const fftw_complex* out()const{return out_;}
		fftw_complex& out(int i){return out_[i];}
		const fftw_complex& out(int i)const{return out_[i];}
			
		//==== member functions ====
		void resize(int N);
		void init();
		void clear();
		void transform();
	};
	
	//*****************************************************************
	// FFT - 1D - REAL - COMPLEX
	//*****************************************************************
	
	/*template <>
	class FFT<1,DataT::REAL,DataT::COMPLEX>{
	private:
		int N_;//the number of data points
		int sign_;//the sign of the transform
		fftw_plan planf_;//the fourier transform plan - forward
		fftw_plan planr_;//the fourier transform plan - backward
		double* in_;
		fftw_complex* out_;
	public:
		//==== constructors/destructors ====
		FFT():N_(0),sign_(FFTW_FORWARD),in_(NULL),out_(NULL),planf_(NULL),planr_(NULL){}
		FFT(int N):N_(0),sign_(FFTW_FORWARD),in_(NULL),out_(NULL),planf_(NULL),planr_(NULL){resize(N);}
		FFT(int N, int sign):N_(0),sign_(sign),in_(NULL),out_(NULL),planf_(NULL),planr_(NULL){resize(N);}
		~FFT(){clear();}
		
		//==== access ====
		const int& N()const{return N_;}
		int& sign(){return sign_;}
		const int& sign()const{return sign_;}
		double* in(){return in_;}
		const double* in()const{return in_;}
		double& in(int i){return in_[i];}
		const double& in(int i)const{return in_[i];}
		fftw_complex* out(){return out_;}
		const fftw_complex* out()const{return out_;}
		fftw_complex& out(int i){return out_[i];}
		const fftw_complex& out(int i)const{return out_[i];}
			
		//==== member functions ====
		void resize(int N);
		void clear();
		void transform();
	};*/
	
	//*****************************************************************
	// FFT - 2D - COMPLEX - COMPLEX
	//*****************************************************************
	
	template <>
	class FFT<2,DataT::COMPLEX,DataT::COMPLEX>{
	private:
		Eigen::Vector2i N_;//the number of data points
		int sign_;//the sign of the transform
		fftw_plan plan_;//the fourier transform plan
		fftw_complex* in_;
		fftw_complex* out_;
	public:
		//==== constructors/destructors ====
		FFT():N_(Eigen::Vector2i::Zero()),sign_(FFTW_FORWARD),in_(NULL),out_(NULL),plan_(NULL){}
		FFT(const Eigen::Vector2i& N):N_(Eigen::Vector2i::Zero()),sign_(FFTW_FORWARD),in_(NULL),out_(NULL),plan_(NULL){resize(N);}
		FFT(const Eigen::Vector2i& N, int sign):N_(Eigen::Vector2i::Zero()),sign_(sign),in_(NULL),out_(NULL),plan_(NULL){resize(N);}
		~FFT(){clear();}
		
		//==== access ====
		const Eigen::Vector2i& N()const{return N_;}
		int& sign(){return sign_;}
		const int& sign()const{return sign_;}
		fftw_complex* in(){return in_;}
		const fftw_complex* in()const{return in_;}
		fftw_complex& in(int i){return in_[i];}
		const fftw_complex& in(int i)const{return in_[i];}
		fftw_complex* out(){return out_;}
		const fftw_complex* out()const{return out_;}
		fftw_complex& out(int i){return out_[i];}
		const fftw_complex& out(int i)const{return out_[i];}
			
		//==== member functions ====
		void resize(const Eigen::Vector2i& N);
		void clear();
		void transform();
	};
	
	//*****************************************************************
	// FFT - 3D - COMPLEX - COMPLEX
	//*****************************************************************
	
	template <>
	class FFT<3,DataT::COMPLEX,DataT::COMPLEX>{
	private:
		Eigen::Vector3i N_;//the number of data points
		int sign_;//the sign of the transform
		fftw_plan plan_;//the fourier transform plan
		fftw_complex* in_;
		fftw_complex* out_;
	public:
		//==== constructors/destructors ====
		FFT():N_(Eigen::Vector3i::Zero()),sign_(FFTW_FORWARD),in_(NULL),out_(NULL),plan_(NULL){}
		FFT(const Eigen::Vector3i& N, int sign):N_(Eigen::Vector3i::Zero()),sign_(sign),in_(NULL),out_(NULL),plan_(NULL){resize(N);}
		~FFT(){clear();}
		
		//==== access ====
		const Eigen::Vector3i& N()const{return N_;}
		int& sign(){return sign_;}
		const int& sign()const{return sign_;}
		fftw_complex* in(){return in_;}
		const fftw_complex* in()const{return in_;}
		fftw_complex& in(int i){return in_[i];}
		const fftw_complex& in(int i)const{return in_[i];}
		fftw_complex* out(){return out_;}
		const fftw_complex* out()const{return out_;}
		fftw_complex& out(int i){return out_[i];}
		const fftw_complex& out(int i)const{return out_[i];}
			
		//==== member functions ====
		void resize(const Eigen::Vector3i& N);
		void clear();
		void transform();
	};
	
}

namespace signala{
	
	double smooth(std::vector<double>& arr, double fwidth);
	
}

#endif