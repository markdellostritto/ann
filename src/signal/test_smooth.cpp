#include <iostream>
#include <cmath>
#include <ctime>
#include "signal/fft.hpp"

double func(double x){
	return -0.5*(std::tanh(2.0*(x-3.0))+1.0)+0.5*(std::tanh(2.0*(x+3.0))+1.0);
}

int main(int argc, char* argv[]){
	
	FILE* writer=NULL;
	
	const double xmin=-5;
	const double xmax=+5;
	const double dx=0.01;
	const double noise=0.1;
	const int N=(xmax-xmin)/dx+1;
	std::cout<<"x     = "<<xmin<<" "<<xmax<<" "<<dx<<"\n";
	std::cout<<"noise = "<<noise<<"\n";
	std::cout<<"N     = "<<N<<"\n";
	
	std::vector<double> ydata(N);
	std::vector<double> ysmth(N);
	std::srand(std::time(NULL));
	
	std::cout<<"generating data\n";
	for(int i=0; i<N; ++i){
		const double x=xmin+dx*i;
		const double r=2.0*((1.0*std::rand())/RAND_MAX-0.5);
		ydata[i]=func(x)+noise*r;
	}
	
	std::cout<<"smoothing data\n";
	ysmth=ydata;
	signala::smooth(ysmth,1.0);
	
	std::cout<<"writing data\n";
	writer=fopen("test_smooth_ydata.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X Y\n");
		for(int i=0; i<N; ++i){
			const double x=xmin+dx*i;
			fprintf(writer,"%f %f\n",x,ydata[i]);
		}
		fclose(writer);
		writer=NULL;
	}
	writer=fopen("test_smooth_ysmth.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X Y\n");
		for(int i=0; i<N; ++i){
			const double x=xmin+dx*i;
			fprintf(writer,"%f %f\n",x,ysmth[i]);
		}
		fclose(writer);
		writer=NULL;
	}
	
	return 0;
}
