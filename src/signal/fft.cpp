//c++ libraries
#include <iostream>
// ann - math
#include "math/const.hpp"
// ann - signal
#include "signal/fft.hpp"

namespace fourier{
	
	//*****************************************************************
	// Data Type
	//*****************************************************************
	
	std::ostream& operator<<(std::ostream& out, const DataT::type& t){
		switch(t){
			case DataT::REAL: out<<"REAL";
			case DataT::COMPLEX: out<<"COMPLEX";
			default: out<<"UKNOWN\n";
		}
		return out;
	}
	
	//*****************************************************************
	// FFT - 1D - COMPLEX - COMPLEX
	//*****************************************************************
	
	//==== constructors/destructors ====
	
	FFT<1,DataT::COMPLEX,DataT::COMPLEX>::FFT(const FFT<1,DataT::COMPLEX,DataT::COMPLEX>& fft):N_(0),sign_(FFTW_FORWARD),in_(NULL),out_(NULL),plan_(NULL){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<1,DataT::COMPLEX,DataT::COMPLEX>::FFT(const FFT<1,DataT::COMPLEX,DataT::COMPLEX>&)\n";
		resize(fft.N());
		for(int i=0; i<N_; ++i){
			in_[i][0]=fft.in(i)[0]; in_[i][1]=fft.in(i)[1];
			out_[i][0]=fft.out(i)[0]; out_[i][1]=fft.out(i)[1];
		}
		sign_=fft.sign();
	}
	
	//==== operators ====
	
	FFT<1,DataT::COMPLEX,DataT::COMPLEX>& FFT<1,DataT::COMPLEX,DataT::COMPLEX>::operator=(const FFT<1,DataT::COMPLEX,DataT::COMPLEX>& fft){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<1,DataT::COMPLEX,DataT::COMPLEX>::operator=(const FFT<1,DataT::COMPLEX,DataT::COMPLEX>&)\n";
		resize(fft.N());
		for(int i=0; i<N_; ++i){
			in_[i][0]=fft.in(i)[0]; in_[i][1]=fft.in(i)[1];
			out_[i][0]=fft.out(i)[0]; out_[i][1]=fft.out(i)[1];
		}
		sign_=fft.sign();
		return *this;
	}
	
	//==== member functions ====
	
	void FFT<1,DataT::COMPLEX,DataT::COMPLEX>::resize(int N){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<1,DataT::COMPLEX,DataT::COMPLEX>::resize(int)\n";
		//if resizing is necessary, clear old data, allocate new vectors
		if(N!=N_){
			if(N<=0) throw std::invalid_argument("FFT<1,DataT::COMPLEX,DataT::COMPLEX>::resize(int): invalid size.");
			clear();
			N_=N;
			in_=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N_);
			if(in_==NULL) throw std::runtime_error("Could not allocate memory for input buffer");
			out_=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N_);
			if(out_==NULL) throw std::runtime_error("Could not allocate memory for output buffer");
		}
	}
	
	void FFT<1,DataT::COMPLEX,DataT::COMPLEX>::init(){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<1,DataT::COMPLEX,DataT::COMPLEX>::init()\n";
		if(N_>0){
			if(plan_!=NULL) fftw_destroy_plan(plan_);
			plan_=fftw_plan_dft_1d(N_,in_,out_,sign_,FFTW_ESTIMATE);
			if(plan_==NULL) throw std::runtime_error("Could not create fftw plan.");
		} else throw std::invalid_argument("Invalid size for fftw plan.");
	}
	
	void FFT<1,DataT::COMPLEX,DataT::COMPLEX>::clear(){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<1,DataT::COMPLEX,DataT::COMPLEX>::clear()\n";
		if(in_!=NULL){
			fftw_free(in_);
			in_=NULL;
		}
		if(out_!=NULL){
			fftw_free(out_);
			out_=NULL;
		}
		if(plan_!=NULL){
			fftw_destroy_plan(plan_);
			plan_=NULL;
		}
	}
	
	void FFT<1,DataT::COMPLEX,DataT::COMPLEX>::transform(){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<1,DataT::COMPLEX,DataT::COMPLEX>::transform()\n";
		fftw_execute(plan_);
	}
		
	//*****************************************************************
	// FFT - 1D - REAL - COMPLEX
	//*****************************************************************
	
	/*void FFT<1,DataT::REAL,DataT::COMPLEX>::resize(int N){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<1,DataT::REAL,DataT::COMPLEX>::resize(int)\n";
		if(N<=0) throw std::invalid_argument("FFT<1,DataT::REAL,DataT::COMPLEX>::resize(int): invalid size.");
		clear();
		N_=N;
		in_=(double*)fftw_malloc(sizeof(double)*N_);
		out_=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N_);
		if(sign_==FFTW_FORWARD){
			planf_=fftw_plan_dft_r2c_1d(N_,in_,out_,FFTW_ESTIMATE);
		} else if(sign_==FFTW_BACKWARD){
			planr_=fftw_plan_dft_c2r_1d(N_,in_,out_,FFTW_ESTIMATE);
		} else throw std::invalid_argument("FFT<1,DataT::REAL,DataT::COMPLEX>::resize(int): invalid sign.");
	}
	
	void FFT<1,DataT::REAL,DataT::COMPLEX>::clear(){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<1,DataT::REAL,DataT::COMPLEX>::clear()\n";
		if(in_!=NULL) fftw_free(in_);
		if(out_!=NULL) fftw_free(out_);
		if(planf_!=NULL) fftw_destroy_plan(planf_);
		if(planr_!=NULL) fftw_destroy_plan(planr_);
	}
	
	void FFT<1,DataT::REAL,DataT::COMPLEX>::transform(){
		if(FFT_PRINT_FUNC<0) std::cout<<"FFT<1,DataT::REAL,DataT::COMPLEX>::transform()\n";
		if(sign_==FFTW_FORWARD){
			fftw_execute(planf_);
			for(int t=N_/2+1; t<N_; ++t){
				out_[t][0]=out_[N_-t][0];
				out_[t][1]=-out_[N_-t][1];
			}
		} else if(sign_==FFTW_BACKWARD){
			fftw_execute(planr_);
		}
	}*/
	
	//*****************************************************************
	// FFT - 2D - COMPLEX - COMPLEX
	//*****************************************************************
	
	void FFT<2,DataT::COMPLEX,DataT::COMPLEX>::resize(const Eigen::Vector2i& N){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<2,DataT::COMPLEX,DataT::COMPLEX>::resize(const Eigen::Vector2i&)\n";
		if(N[0]<=0 && N[1]<=0) throw std::invalid_argument("FFT<2,DataT::COMPLEX,DataT::COMPLEX>::resize(const Eigen::Vector2i&): invalid size.");
		clear();
		N_=N;
		const int NT=N_.prod();
		in_=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NT);
		out_=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NT);
		plan_=fftw_plan_dft_2d(N_[0],N_[1],in_,out_,sign_,FFTW_ESTIMATE);
	}
	
	void FFT<2,DataT::COMPLEX,DataT::COMPLEX>::clear(){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<2,DataT::COMPLEX,DataT::COMPLEX>::clear()\n";
		if(in_!=NULL) fftw_free(in_);
		if(out_!=NULL) fftw_free(out_);
		if(plan_!=NULL) fftw_destroy_plan(plan_);
	}
	
	void FFT<2,DataT::COMPLEX,DataT::COMPLEX>::transform(){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<2,DataT::COMPLEX,DataT::COMPLEX>::transform()\n";
		fftw_execute(plan_);
	}
	
	//*****************************************************************
	// FFT - 3D - COMPLEX - COMPLEX
	//*****************************************************************
	
	void FFT<3,DataT::COMPLEX,DataT::COMPLEX>::resize(const Eigen::Vector3i& N){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<3,DataT::COMPLEX,DataT::COMPLEX>::resize(const Eigen::Vector3i&)\n";
		if(N[0]<=0 && N[1]<=0 && N[2]<=0) throw std::invalid_argument("FFT<3,DataT::COMPLEX,DataT::COMPLEX>::resize(const Eigen::Vector3i&): invalid size.");
		clear();
		N_=N;
		const int NT=N_.prod();
		in_=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NT);
		out_=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*NT);
		plan_=fftw_plan_dft_3d(N_[0],N_[1],N_[2],in_,out_,sign_,FFTW_ESTIMATE);
	}
	
	void FFT<3,DataT::COMPLEX,DataT::COMPLEX>::clear(){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<3,DataT::COMPLEX,DataT::COMPLEX>::clear()\n";
		if(in_!=NULL) fftw_free(in_);
		if(out_!=NULL) fftw_free(out_);
		if(plan_!=NULL) fftw_destroy_plan(plan_);
	}
	
	void FFT<3,DataT::COMPLEX,DataT::COMPLEX>::transform(){
		if(FFT_PRINT_FUNC>0) std::cout<<"FFT<3,DataT::COMPLEX,DataT::COMPLEX>::transform()\n";
		fftw_execute(plan_);
	}
	
}

namespace signala{
	
	double smooth(std::vector<double>& arr, double fwidth){
		double error=0;
		if(fwidth>math::constant::ZERO){
			const int N=arr.size();
			fourier::FFT<1,fourier::DataT::COMPLEX,fourier::DataT::COMPLEX> fft_r_(2*N,FFTW_FORWARD);
			fourier::FFT<1,fourier::DataT::COMPLEX,fourier::DataT::COMPLEX> fft_f_(2*N,FFTW_FORWARD);
			fourier::FFT<1,fourier::DataT::COMPLEX,fourier::DataT::COMPLEX> fft_(2*N,FFTW_BACKWARD);
			//init ffts
			for(int i=0; i<N; ++i){
				fft_r_.in(i)[0]=arr[i];
				fft_r_.in(i)[1]=0.0;
			}
			for(int i=N; i<2*N; ++i){
				fft_r_.in(i)[0]=0.0;
				fft_r_.in(i)[1]=0.0;
			}
			const double denom=1.0/(2.0*fwidth*fwidth);
			for(int i=0; i<N; ++i){
				fft_f_.in(i)[0]=std::exp(-1.0*i*i*denom);
				fft_f_.in(i)[1]=0;
			}
			for(int i=N; i<2*N; ++i){
				fft_f_.in(i)[0]=std::exp(-1.0*(i-2*N)*(i-2*N)*denom);
				fft_f_.in(i)[1]=0.0;
			}
			//fft - forward
			fft_r_.transform();
			fft_f_.transform();
			//product
			for(int i=0; i<2*N; ++i){
				fft_.in(i)[0]=fft_r_.out(i)[0]*fft_f_.out(i)[0]-fft_r_.out(i)[1]*fft_f_.out(i)[1];
				fft_.in(i)[1]=fft_r_.out(i)[0]*fft_f_.out(i)[1]+fft_r_.out(i)[1]*fft_f_.out(i)[0];
			}
			//fft - reverse
			fft_.transform();
			//compute smoothed data
			const double norm=1.0/((std::sqrt(2.0*math::constant::PI)*fwidth)*2.0*N);
			for(int i=0; i<N; ++i){
				arr[i]=norm*fft_.out(i)[0];
				error+=norm*fft_.out(i)[1];
			}
		}
		return error;
	}
	
}