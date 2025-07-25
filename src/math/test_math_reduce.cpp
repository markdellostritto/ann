// c
#include <cmath>
#include <ctime>
// c++
#include <iostream>
#include <random>
// math
#include "math/reduce.hpp"
//str
#include "str/print.hpp"

void test_normal(int ns){
	//random
	std::default_random_engine generator(std::time(NULL));
	std::uniform_real_distribution<double> udist(0.0,1.0);
	//local variables
	const int nt=100;//number of tests
	const int np=nt/10;
	//normal distribution limits
	const double cmin=-5.0;
	const double cmax=5.0;
	const double smin=1.0;
	const double smax=5.0;
	//error
	double err_ca=0;
	double err_cp=0;
	double err_sa=0;
	double err_sp=0;
	//compute
	for(int t=0; t<nt; ++t){
		//generate distribution
		const double c=(cmax-cmin)*udist(generator)+cmin;
		const double s=(smax-smin)*udist(generator)+smin;
		std::normal_distribution<double> distribution(c,s);
		//compute distribution
		Reduce<1> reduce;
		for(int i=0; i<ns; ++i){
			const double x=distribution(generator);
			reduce.push(x);
		}
		//compute error
		const double cc=reduce.avg();
		const double ss=sqrt(reduce.var());
		err_ca+=fabs(cc-c);
		err_cp+=fabs((cc-c)/c)*100.0;
		err_sa+=fabs(ss-s);
		err_sp+=fabs((ss-s)/s)*100.0;
	}
	//normalize error
	err_ca/=nt;
	err_cp/=nt;
	err_sa/=nt;
	err_sp/=nt;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - REDUCE - NORMAL",str)<<"\n";
	std::cout<<"c = ["<<cmin<<","<<cmax<<"]\n";
	std::cout<<"s = ["<<smin<<","<<smax<<"]\n";
	std::cout<<"n - test   = "<<nt<<"\n";
	std::cout<<"n - sample = "<<ns<<"\n";
	std::cout<<"err - c - abs = "<<err_ca<<"\n";
	std::cout<<"err - c - (%) = "<<err_cp<<"\n";
	std::cout<<"err - s - abs = "<<err_sa<<"\n";
	std::cout<<"err - s - (%) = "<<err_sp<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
}

void test_add(){
	//random
	std::default_random_engine generator(std::time(NULL));
	std::uniform_real_distribution<double> udist(0.0,1.0);
	//normal distribution limits
	const double cmin=-5.0;
	const double cmax=5.0;
	const double smin=1.0;
	const double smax=5.0;
	const double c=(cmax-cmin)*udist(generator)+cmin;
	const double s=(smax-smin)*udist(generator)+smin;
	std::normal_distribution<double> distribution(c,s);
	//compute
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
	//print
	std::cout<<"==========================================\n";
	std::cout<<"TEST - REDUCE - ADDITION\n";
	std::cout<<"err - min = "<<fabs(reduceT.min()-reduceS.min())<<"\n";
	std::cout<<"err - max = "<<fabs(reduceT.max()-reduceS.max())<<"\n";
	std::cout<<"err - avg = "<<fabs(reduceT.avg()-reduceS.avg())<<"\n";
	std::cout<<"err - var = "<<fabs(reduceT.var()-reduceS.var())<<"\n";
	std::cout<<"==========================================\n";
}

int main(int argc, char* argv[]){

	test_normal(1e2);
	test_normal(1e4);
	test_normal(1e6);
	test_normal(1e8);
	test_add();
}