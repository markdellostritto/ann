// c
#include <cmath>
#include <ctime>
// c++
#include <iostream>
#include <random>
// math
#include "math/reduce.hpp"

void test_normal(){
	const double m=1.57827852;
	const double s=2.45812678;
	std::default_random_engine generator(std::time(NULL));
	std::normal_distribution<double> distribution(m,s);
	Reduce<1> reduce;
	const int N=1000000;
	for(int i=0; i<N; ++i){
		const double x=distribution(generator);
		reduce.push(x);
	}
	std::cout<<"==========================================\n";
	std::cout<<"TEST - REDUCE - NORMAL\n";
	std::cout<<"N   = "<<N<<"\n";
	std::cout<<"min = "<<reduce.min()<<"\n";
	std::cout<<"max = "<<reduce.max()<<"\n";
	std::cout<<"avg = "<<reduce.avg()<<"\n";
	std::cout<<"dev = "<<sqrt(reduce.var())<<"\n";
	std::cout<<"avg - ref = "<<m<<"\n";
	std::cout<<"dev - ref = "<<s<<"\n";
	std::cout<<"err - avg = "<<fabs((reduce.avg()-m)/m)<<"\n";
	std::cout<<"err - dev = "<<fabs((sqrt(reduce.var())-s)/s)<<"\n";
	
	std::vector<int> Nv(6);
	Nv[0]=100;
	std::cout<<"N err\n";
	for(int i=1; i<Nv.size(); ++i) Nv[i]=Nv[i-1]*10;
	for(int j=0; j<Nv.size(); ++j){
		Reduce<1> rtmp;
		for(int i=0; i<Nv[j]; ++i){
			const double x=distribution(generator);
			rtmp.push(x);
		}
		double err=fabs((rtmp.avg()-m)/m)+fabs((sqrt(rtmp.var())-s)/s);
		std::cout<<Nv[j]<<" "<<err<<"\n";
	}
	std::cout<<"==========================================\n";
	
}

void test_add(){
	const double m=0.2652785289;
	const double s=1.763978352;
	std::default_random_engine generator(std::time(NULL));
	std::normal_distribution<double> distribution(m,s);
	Reduce<1> reduce1;
	Reduce<1> reduce2;
	Reduce<1> reduceT;
	Reduce<1> reduceS;
	const int N=100000;
	for(int i=0; i<N; ++i){
		const double x=distribution(generator);
		reduce1.push(x);
		reduceT.push(x);
	}
	for(int i=0; i<N; ++i){
		const double x=distribution(generator);
		reduce2.push(x);
		reduceT.push(x);
	}
	reduceS=reduce1+reduce2;
	std::cout<<"==========================================\n";
	std::cout<<"TEST - REDUCE - ADDITION\n";
	std::cout<<"err - min = "<<fabs(reduceT.min()-reduceS.min())<<"\n";
	std::cout<<"err - max = "<<fabs(reduceT.max()-reduceS.max())<<"\n";
	std::cout<<"err - avg = "<<fabs(reduceT.avg()-reduceS.avg())<<"\n";
	std::cout<<"err - var = "<<fabs(reduceT.var()-reduceS.var())<<"\n";
	std::cout<<"==========================================\n";
}

int main(int argc, char* argv[]){

	test_normal();
	test_add();
}