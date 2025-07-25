#include <iostream>
#include <cmath>
#include "test_unit.hpp"

void test_unit_fft_gauss(){
	//gaussian function
	const int N=1000;
	const double sigma=1.0;
	const double mu=0.0;
	const double xmin=mu-5.0*sigma;
	const double xmax=mu+5.0*sigma;
	const double dx=(xmax-xmin)/(N-1.0);
	//compute fourier transform
	typedef fourier::FFT<1,fourier::DataT::COMPLEX,fourier::DataT::COMPLEX> FFT1D;
	FFT1D fft(N,FFTW_FORWARD);
	for(int i=0; i<N; ++i){
		const double x=xmin+i*dx;
		fft.in(i)[0]=std::exp(-x*x/(2.0*sigma*sigma))/sqrt(2.0*3.14159);
		fft.in(i)[1]=0.0;
	}
	fft.transform();
	//compute average, stddev
	double avg_in=0,dev_in=0,avg_out=0,dev_out=0;
	for(int i=0; i<N; ++i) avg_in+=fft.in(i)[0]*(xmin+i*dx);
	for(int i=0; i<N; ++i) avg_out+=fft.out(i)[0];
	avg_in/=N; avg_out/=N;
	for(int i=0; i<N; ++i) dev_in+=(fft.in(i)[0]*(xmin+i*dx)-avg_in)*(fft.in(i)[0]*(xmin+i*dx)-avg_in);
	for(int i=0; i<N; ++i) dev_out+=(fft.out(i)[0]-avg_out)*(fft.out(i)[0]-avg_out);
	dev_in=std::sqrt(dev_in/(N-1.0)/dx);
	dev_out=std::sqrt(dev_out/(N-1.0));
	//integral and imaginary
	double sum=0,re=0,im=0;
	for(int i=0; i<N; ++i) sum+=fft.in(i)[0]*dx;
	for(int i=0; i<N; ++i) re+=fft.out(i)[0]*dx;
	for(int i=0; i<N; ++i) im+=fft.out(i)[1]*dx;
	//print
	std::cout<<"avg_in  = "<<avg_in<<"\n";
	std::cout<<"avg_out = "<<avg_out<<"\n";
	std::cout<<"dev_in  = "<<dev_in<<"\n";
	std::cout<<"dev_out = "<<dev_out<<"\n";
	std::cout<<"sum     = "<<sum<<"\n";
	std::cout<<"re      = "<<re<<"\n";
	std::cout<<"im      = "<<im<<"\n";
}

int main(int argc, char* argv[]){
	
	std::cout<<"FFT - GAUSS\n";
	test_unit_fft_gauss();
	std::cout<<"FFT - GAUSS\n";
}