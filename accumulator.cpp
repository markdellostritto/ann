#include "accumulator.hpp"

//***********************************************
//Distribution
//***********************************************

//constructors/destructors

Dist::Dist(const Dist& dist){
	clear();
	min_=dist.dist_min();
	max_=dist.dist_max();
	N_=dist.N();
	M_=dist.M();
	bc_=dist.bc();
	gc_=dist.gc();
	len_=(max_-min_)/N_;
	hist_=new double[N_];
	buf_=new double[M_];
	for(unsigned int i=0; i<N_; ++i) hist_[i]=dist.hist(i);
	for(unsigned int i=0; i<M_; ++i) buf_[i]=dist.buf(i);
}

Dist::~Dist(){
	if(hist_!=NULL) delete[] hist_;
	if(buf_!=NULL) delete[] buf_;
}

//operators

Dist& Dist::operator=(const Dist& dist){
	clear();
	min_=dist.dist_min();
	max_=dist.dist_max();
	N_=dist.N();
	M_=dist.M();
	bc_=dist.bc();
	gc_=dist.gc();
	len_=(max_-min_)/N_;
	hist_=new double[N_];
	buf_=new double[M_];
	for(unsigned int i=0; i<N_; ++i) hist_[i]=dist.hist(i);
	for(unsigned int i=0; i<M_; ++i) buf_[i]=dist.buf(i);
	return *this;
}

//member functions

void Dist::clear(){
	N_=0; M_=1000;
	bc_=0; gc_=0;
	len_=0; min_=0; max_=0;
	if(hist_!=NULL) delete[] hist_; hist_=NULL;
	if(buf_!=NULL) delete[] buf_; buf_=NULL;
}

void Dist::init(double min, double max, unsigned int N){
	if(min>=max) throw std::runtime_error("Invalid Dist limits.");
	if(N==0) throw std::runtime_error("Invalid Dist N.");
	clear();
	min_=min;
	max_=max;
	N_=N;
	len_=(max-min)/N_;
	hist_=new double[N_];
	buf_=new double[M_];
}

unsigned int Dist::bin(double x){
	unsigned int uLim=N_;
	unsigned int lLim=0;
	unsigned int mid;
	while(uLim-lLim>1){
		mid=lLim+(uLim-lLim)/2;
		if(min_+len_*lLim<=x && x<=min_+len_*mid) uLim=mid;
		else lLim=mid;
	}
	return lLim;
}

void Dist::push(double x){
	buf_[bc_++]=x;
	if(bc_==M_){
		for(unsigned int i=0; i<N_; ++i) hist_[i]*=gc_/(gc_+1.0);
		++gc_;
		for(unsigned int i=0; i<M_; ++i) hist_[bin(buf_[i])]+=1.0/(M_*gc_);
		bc_=0;
	}
}

//***********************************************
//Fourier
//***********************************************

#ifdef ACC_FFT

//constants
const std::complex<double> Fourier::I=std::complex<double>(0.0,1.0);

void Fourier::clear(){
	fft_.clear();
	if(f_!=NULL) delete[] f_;
	f_=NULL;
	n_=0;
	N_=0;
}

void Fourier::init(unsigned int N){
	n_=0;
	N_=N;
	fft_=fourier::FFT_R2C(N_);
	f_=new fftw_complex[N_];
	for(int i=N_-1; i>=0; --i){f_[i][0]=0;f_[i][1]=0;};
}

void Fourier::push(double x){
	if(n_<N_-1) fft_.in(n_++)=x;
	else {
		fft_.in(n_++)=x;
		fft_.transformf();
		for(int k=N_-1; k>=0; --k){
			f_[k][0]+=fft_.out(k)[0];
			f_[k][1]+=fft_.out(k)[1];
		}
		n_=0;
	}
}

#endif