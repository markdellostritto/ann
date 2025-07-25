//c
#include <cmath>
//c++
#include <iostream>
#include <random>
//pca
#include "ml/pca.hpp"

int main(int arg, char* argv){
	//random number generator
		std::default_random_engine generator;
		std::normal_distribution<double> dx(0.0,1.0);
		std::normal_distribution<double> dy(0.0,2.0);
	//data
		const int nobs=5000;
		const int nvars=2;
	//pca
		PCA pca;
	//file i/o
		FILE* writer=NULL;
		const char* file_data="test_pca_gauss.dat";
	//constants
		const double pi=3.141592653589793238462643;
		
	std::cout<<"GAUSSIAN DISTRIBUTION - 2D\n";
	std::cout<<"stddev - x = 1.0\n";
	std::cout<<"stddev - y = 2.0\n";
	std::cout<<"rotation - 45 degrees\n";
	
	//==== generate distribution ====
	std::cout<<"generating distribution\n";
	Eigen::Matrix2d R;
	const double phi=-45.0*pi/180.0;
	R(0,0)=std::cos(phi); R(0,1)=-sin(phi);
	R(1,0)=std::sin(phi); R(1,1)=cos(phi);
	std::vector<Eigen::Vector2d> data(nobs);
	for(int i=0; i<nobs; ++i){
		data[i][0]=dx(generator);
		data[i][1]=dy(generator);
		data[i]=R*data[i];
	}
	
	//==== write distribution ====
	std::cout<<"writing distribution\n";
	writer=fopen(file_data,"w");
	if(writer==NULL) throw std::runtime_error("Unable to open data file.");
	for(int i=0; i<nobs; ++i){
		fprintf(writer,"%f %f\n",data[i][0],data[i][1]);
	}
	fclose(writer); writer=NULL;
	
	//==== set PCA ====
	std::cout<<"setting PCA\n";
	pca.resize(nobs,nvars);
	for(int i=0; i<nobs; ++i){
		pca.X().row(i)=data[i];
	}
	
	//==== compute PCA ====
	std::cout<<"computing PCA\n";
	pca.compute();
	
	//==== principal components ====
	std::cout<<"printing principal components\n";
	for(int i=0; i<nvars; ++i){
		std::cout<<"w["<<i<<"] = "<<pca.w()[pca.ci(i)]<<"\n";
	}
	for(int i=0; i<nvars; ++i){
		std::cout<<"W["<<i<<"] = "<<pca.W().row(pca.ci(i))<<"\n";
	}
}