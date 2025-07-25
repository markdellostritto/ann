// c++
#include <iostream>
#include <chrono>
// math
#include "math/density.hpp"
// rand
#include "rand/rng.hpp"

void test_gauss_1D(){
	//density
	const int D=1;
	Density<D> den;
	std::array<double,D> min={-8.0};
	std::array<double,D> max={8.0};
	std::array<double,D> len={0.1};
	den.init(min,max,len);
	//normal distribution
	const int seed=std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 rng=std::mt19937(seed);
	rng::dist::Normal nrng=rng::dist::Normal(0.0,1.0);
	//bin distribution
	const int N=1e6;
	for(int i=0; i<N; ++i){
		std::array<double,1> tmp={nrng.rand(rng)};
		den.push(tmp);
	}
	//print distribution
	FILE* writer=fopen("test_density_gauss_1D.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#a o\n");
		for(int i=0; i<den.nbins(0); ++i){
			double o;
			std::array<double,D> a;
			const std::array<int,D> index={i};
			den.abscissa(index,a);
			o=den.ordinate(index);
			fprintf(writer,"%f %f\n",a[0],o);
		}
	}
	//count
	std::cout<<"c = "<<den.c()<<"\n";
	std::cout<<"m = "<<den.m()<<"\n";
}

void test_gauss_2D(){
	//density
	const int D=2;
	Density<D> den;
	std::array<double,D> min={-6.0,-6.0};
	std::array<double,D> max={6.0,6.0};
	std::array<double,D> len={0.1,0.1};
	den.init(min,max,len);
	//normal distribution
	const int seed=std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 rng=std::mt19937(seed);
	rng::dist::Normal nrng=rng::dist::Normal(0.0,1.0);
	//bin distribution
	const int N=1e4;
	for(int i=0; i<N; ++i){
		std::array<double,D> tmp={nrng.rand(rng),nrng.rand(rng)};
		den.push(tmp);
	}
	//print distribution
	FILE* writer=fopen("test_density_gauss_2D.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#a1 a2 o\n");
		for(int i=0; i<den.nbins(0); ++i){
			for(int j=0; j<den.nbins(1); ++j){
				double o;
				std::array<double,D> a;
				const std::array<int,D> index={i,j};
				den.abscissa(index,a);
				o=den.ordinate(index);
				fprintf(writer,"%f %f %f\n",a[0],a[1],o);
			}
		}
	}
	//count
	std::cout<<"c = "<<den.c()<<"\n";
	std::cout<<"m = "<<den.m()<<"\n";
}

void test_gauss_3D(){
	//density
	const int D=3;
	Density<D> den;
	std::array<double,D> min={-6.0,-6.0,-6.0};
	std::array<double,D> max={6.0,6.0,6.0};
	std::array<double,D> len={0.1,0.1,0.1};
	den.init(min,max,len);
	//normal distribution
	const int seed=std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 rng=std::mt19937(seed);
	rng::dist::Normal nrng=rng::dist::Normal(0.0,1.0);
	//bin distribution
	const int N=1e3;
	for(int i=0; i<N; ++i){
		std::array<double,D> tmp={nrng.rand(rng),nrng.rand(rng),nrng.rand(rng)};
		den.push(tmp);
	}
	//print distribution
	FILE* writer=fopen("test_density_gauss_3D.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#a1 a2 a3 o\n");
		for(int i=0; i<den.nbins(0); ++i){
			for(int j=0; j<den.nbins(1); ++j){
				for(int k=0; k<den.nbins(2); ++k){
					double o;
					std::array<double,D> a;
					const std::array<int,D> index={i,j,k};
					den.abscissa(index,a);
					o=den.ordinate(index);
					fprintf(writer,"%f %f %f %f\n",a[0],a[1],o);
				}
			}
		}
	}
	//count
	std::cout<<"c = "<<den.c()<<"\n";
	std::cout<<"m = "<<den.m()<<"\n";
}

int main(int argc, char* argv[]){
	
	test_gauss_1D();
	test_gauss_2D();
	test_gauss_3D();
	
	return 0;
}