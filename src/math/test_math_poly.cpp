// c
#include <cmath>
#include <cstdlib>
#include <ctime>
// c++
#include <iostream>
#include <vector>
#include <chrono>
//math
#include "math/poly.hpp"
#include "math/special.hpp"
//util
#include "util/time.hpp"

using namespace std::chrono;
using math::constant::PI;

void test_alegendre_error(){
	std::srand(std::time(NULL));
	//constants
	const int N=100;
	std::vector<double> x(N);
	//generate x
	for(int i=0; i<N; ++i){
		x[i]=2.0*((1.0*std::rand())/RAND_MAX-0.5);
	}
	//compute - 00
	double err00=0;
	for(int i=0; i<N; ++i){
		const double fe=1.0;
		const double fa=math::poly::alegendre(0,0,x[i]);
		err00+=fabs(fe-fa);
	}
	err00/=N;
	//compute - 10
	double err10=0;
	for(int i=0; i<N; ++i){
		const double fe=x[i];
		const double fa=math::poly::alegendre(1,0,x[i]);
		err10+=fabs(fe-fa);
	}
	err10/=N;
	//compute - 11
	double err11=0;
	for(int i=0; i<N; ++i){
		const double fe=-sqrt(1.0-x[i]*x[i]);
		const double fa=math::poly::alegendre(1,1,x[i]);
		err11+=fabs(fe-fa);
	}
	err11/=N;
	//compute - 20
	double err20=0;
	for(int i=0; i<N; ++i){
		const double fe=0.5*(3.0*x[i]*x[i]-1.0);
		const double fa=math::poly::alegendre(2,0,x[i]);
		err20+=fabs(fe-fa);
	}
	err20/=N;
	//compute - 21
	double err21=0;
	for(int i=0; i<N; ++i){
		const double fe=-3.0*sqrt(1.0-x[i]*x[i])*x[i];
		const double fa=math::poly::alegendre(2,1,x[i]);
		err21+=fabs(fe-fa);
	}
	err21/=N;
	//compute - 22
	double err22=0;
	for(int i=0; i<N; ++i){
		const double fe=3.0*(1.0-x[i]*x[i]);
		const double fa=math::poly::alegendre(2,2,x[i]);
		err22+=fabs(fe-fa);
	}
	err22/=N;
	//print
	std::cout<<"test - alegendre\n";
	std::cout<<"err00 = "<<err00<<"\n";
	std::cout<<"err10 = "<<err10<<"\n";
	std::cout<<"err11 = "<<err11<<"\n";
	std::cout<<"err20 = "<<err20<<"\n";
	std::cout<<"err21 = "<<err21<<"\n";
	std::cout<<"err22 = "<<err22<<"\n";
}

void test_alegendre_time(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	//constants
	const int N=1e8;
	std::vector<double> x(N);
	//generate x
	for(int i=0; i<N; ++i){
		x[i]=2.0*((1.0*std::rand())/RAND_MAX-0.5);
	}
	//compute - 00
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::alegendre(0,0,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time00 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 10
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::alegendre(1,0,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time10 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 11
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::alegendre(1,1,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time11 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 20
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::alegendre(2,0,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time20 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 21
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::alegendre(2,1,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time21 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 22
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::alegendre(2,2,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time22 = duration_cast<duration<double>>(tend-tbeg);
	//print
	std::cout<<"test - alegendre\n";
	std::cout<<"time00 = "<<time00.count()/N<<"\n";
	std::cout<<"time10 = "<<time10.count()/N<<"\n";
	std::cout<<"time11 = "<<time11.count()/N<<"\n";
	std::cout<<"time20 = "<<time20.count()/N<<"\n";
	std::cout<<"time21 = "<<time21.count()/N<<"\n";
	std::cout<<"time22 = "<<time22.count()/N<<"\n";
}

void test_simple_time(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	//constants
	const int N=1e8;
	std::vector<double> x(N);
	//generate x
	for(int i=0; i<N; ++i){
		x[i]=2.0*((1.0*std::rand())/RAND_MAX-0.5);
	}
	//compute - 00
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fe=1.0;
	}
	tend=high_resolution_clock::now();
	duration<double> time00 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 10
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fe=x[i];
	}
	tend=high_resolution_clock::now();
	duration<double> time10 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 11
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fe=-sqrt(1.0-x[i]*x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time11 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 20
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fe=0.5*(3.0*x[i]*x[i]-1.0);
	}
	tend=high_resolution_clock::now();
	duration<double> time20 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 21
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fe=-3.0*sqrt(1.0-x[i]*x[i])*x[i];
	}
	tend=high_resolution_clock::now();
	duration<double> time21 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 22
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fe=3.0*(1.0-x[i]*x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time22 = duration_cast<duration<double>>(tend-tbeg);
	//print
	std::cout<<"test - alegendre\n";
	std::cout<<"time00 = "<<time00.count()/N<<"\n";
	std::cout<<"time10 = "<<time10.count()/N<<"\n";
	std::cout<<"time11 = "<<time11.count()/N<<"\n";
	std::cout<<"time20 = "<<time20.count()/N<<"\n";
	std::cout<<"time21 = "<<time21.count()/N<<"\n";
	std::cout<<"time22 = "<<time22.count()/N<<"\n";
}

void test_legendre_alegendre_error(){
	std::srand(std::time(NULL));
	//constants
	const int N=100;
	std::vector<double> x(N);
	const int lmax=4;
	std::vector<double> error(lmax);
	//generate x
	for(int i=0; i<N; ++i){
		x[i]=2.0*((1.0*std::rand())/RAND_MAX-0.5);
	}
	for(int l=0; l<lmax; ++l){
		error[l]=0;
		for(int i=0; i<N; ++i){
			const double f1=math::poly::legendre(l,x[i]);
			const double f2=math::poly::alegendre(l,0,x[i]);
			error[l]+=fabs(f1-f2);
		}
		error[l]/=N;
	}
	//print
	std::cout<<"test - legendre - alegendre\n";
	for(int l=0; l<lmax; ++l){
		std::cout<<"error["<<l<<"] = "<<error[l]<<"\n";
	}
}

void test_plegendre_error(){
	std::srand(std::time(NULL));
	//constants
	const int N=100;
	std::vector<double> x(N);
	//generate x
	for(int i=0; i<N; ++i){
		x[i]=2.0*((1.0*std::rand())/RAND_MAX-0.5);
	}
	//compute - 00
	double err00=0;
	for(int i=0; i<N; ++i){
		const int l=0;
		const int m=0;
		const double a=sqrt((2.0*l+1.0)/(4.0*PI)*math::special::fratio(l-m,l+m));
		const double fe=a*1.0;
		const double fa=math::poly::plegendre(l,m,x[i]);
		err00+=fabs(fe-fa);
	}
	err00/=N;
	//compute - 1m1
	double err1m1=0;
	for(int i=0; i<N; ++i){
		const int l=1;
		const int m=-1;
		const double a=sqrt((2.0*l+1.0)/(4.0*PI)*math::special::fratio(l-m,l+m));
		const double fe=-0.5*a*-sqrt(1.0-x[i]*x[i]);
		const double fa=math::poly::plegendre(l,m,x[i]);
		err1m1+=fabs(fe-fa);
	}
	err1m1/=N;
	//compute - 10
	double err10=0;
	for(int i=0; i<N; ++i){
		const int l=1;
		const int m=0;
		const double a=sqrt((2.0*l+1.0)/(4.0*PI)*math::special::fratio(l-m,l+m));
		const double fe=a*x[i];
		const double fa=math::poly::plegendre(l,m,x[i]);
		err10+=fabs(fe-fa);
	}
	err10/=N;
	//compute - 1p1
	double err1p1=0;
	for(int i=0; i<N; ++i){
		const int l=1;
		const int m=1;
		const double a=sqrt((2.0*l+1.0)/(4.0*PI)*math::special::fratio(l-m,l+m));
		const double fe=a*-sqrt(1.0-x[i]*x[i]);
		const double fa=math::poly::plegendre(l,m,x[i]);
		err1p1+=fabs(fe-fa);
	}
	err1p1/=N;
	//compute - 2m2
	double err2m2=0;
	for(int i=0; i<N; ++i){
		const int l=2;
		const int m=-2;
		const double a=sqrt((2.0*l+1.0)/(4.0*PI)*math::special::fratio(l-m,l+m));
		const double fe=1.0/24.0*a*3.0*(1.0-x[i]*x[i]);
		const double fa=math::poly::plegendre(l,m,x[i]);
		err2m2+=fabs(fe-fa);
	}
	err2m2/=N;
	//compute - 2m1
	double err2m1=0;
	for(int i=0; i<N; ++i){
		const int l=2;
		const int m=-1;
		const double a=sqrt((2.0*l+1.0)/(4.0*PI)*math::special::fratio(l-m,l+m));
		const double fe=-1.0/6.0*a*-3.0*sqrt(1.0-x[i]*x[i])*x[i];
		const double fa=math::poly::plegendre(l,m,x[i]);
		err2m1+=fabs(fe-fa);
	}
	err2m1/=N;
	//compute - 20
	double err20=0;
	for(int i=0; i<N; ++i){
		const int l=2;
		const int m=0;
		const double a=sqrt((2.0*l+1.0)/(4.0*PI)*math::special::fratio(l-m,l+m));
		const double fe=a*0.5*(3.0*x[i]*x[i]-1.0);
		const double fa=math::poly::plegendre(l,m,x[i]);
		err20+=fabs(fe-fa);
	}
	err20/=N;
	//compute - 2p1
	double err2p1=0;
	for(int i=0; i<N; ++i){
		const int l=2;
		const int m=1;
		const double a=sqrt((2.0*l+1.0)/(4.0*PI)*math::special::fratio(l-m,l+m));
		const double fe=a*-3.0*sqrt(1.0-x[i]*x[i])*x[i];
		const double fa=math::poly::plegendre(l,m,x[i]);
		err2p1+=fabs(fe-fa);
	}
	err2p1/=N;
	//compute - 2p2
	double err2p2=0;
	for(int i=0; i<N; ++i){
		const int l=2;
		const int m=2;
		const double a=sqrt((2.0*l+1.0)/(4.0*PI)*math::special::fratio(l-m,l+m));
		const double fe=a*3.0*(1.0-x[i]*x[i]);
		const double fa=math::poly::plegendre(l,m,x[i]);
		err2p2+=fabs(fe-fa);
	}
	err2p2/=N;
	//print
	std::cout<<"test - plegendre\n";
	std::cout<<"err00  = "<<err00<<"\n";
	std::cout<<"err1m1 = "<<err1m1<<"\n";
	std::cout<<"err10  = "<<err10<<"\n";
	std::cout<<"err1p1 = "<<err1p1<<"\n";
	std::cout<<"err2m2 = "<<err2m2<<"\n";
	std::cout<<"err2m1 = "<<err2m1<<"\n";
	std::cout<<"err20  = "<<err20<<"\n";
	std::cout<<"err2p1 = "<<err2p1<<"\n";
	std::cout<<"err2p2 = "<<err2p2<<"\n";
}

void test_plegendre_time(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	//constants
	const int N=1e8;
	std::vector<double> x(N);
	//generate x
	for(int i=0; i<N; ++i){
		x[i]=2.0*((1.0*std::rand())/RAND_MAX-0.5);
	}
	//compute - 00
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::plegendre(0,0,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time00 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 10
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::plegendre(1,0,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time10 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 11
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::plegendre(1,1,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time11 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 20
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::plegendre(2,0,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time20 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 21
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::plegendre(2,1,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time21 = duration_cast<duration<double>>(tend-tbeg);
	//compute - 22
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fa=math::poly::plegendre(2,2,x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time22 = duration_cast<duration<double>>(tend-tbeg);
	//print
	std::cout<<"test - plegendre\n";
	std::cout<<"time00 = "<<time00.count()/N<<"\n";
	std::cout<<"time10 = "<<time10.count()/N<<"\n";
	std::cout<<"time11 = "<<time11.count()/N<<"\n";
	std::cout<<"time20 = "<<time20.count()/N<<"\n";
	std::cout<<"time21 = "<<time21.count()/N<<"\n";
	std::cout<<"time22 = "<<time22.count()/N<<"\n";
}

void test_legendre_plegendre_error(){
	std::srand(std::time(NULL));
	//constants
	const int N=100;
	std::vector<double> x(N);
	const int lmax=4;
	std::vector<double> error(lmax);
	//generate x
	for(int i=0; i<N; ++i){
		x[i]=2.0*((1.0*std::rand())/RAND_MAX-0.5);
	}
	for(int l=0; l<lmax; ++l){
		error[l]=0;
		for(int i=0; i<N; ++i){
			const double f1=math::poly::legendre(l,x[i]);
			const double f2=math::poly::plegendre(l,0,x[i]);
			error[l]+=fabs(f1-f2);
		}
		error[l]/=N;
	}
	//print
	std::cout<<"test - legendre - plegendre\n";
	for(int l=0; l<lmax; ++l){
		std::cout<<"error["<<l<<"] = "<<error[l]<<"\n";
	}
}

void test_chebyshev_error(){
	std::srand(std::time(NULL));
	//constants
	const int N=100;
	std::vector<double> x(N);
	//generate x
	for(int i=0; i<N; ++i){
		x[i]=2.0*((1.0*std::rand())/RAND_MAX-0.5);
	}
	//compute error
	const int L=5;
	std::vector<double> error(L,0);
	for(int n=0; n<L; ++n){
		for(int i=0; i<N; ++i){
			double fe=0.0;
			switch(n){
				case 0: fe=1.0; break;
				case 1: fe=x[i]; break;
				case 2: fe=2.0*x[i]*x[i]-1.0; break;
				case 3: fe=4.0*x[i]*x[i]*x[i]-3.0*x[i]; break;
				case 4: fe=8.0*x[i]*x[i]*x[i]*x[i]-8.0*x[i]*x[i]+1.0; break;
			}
			const double fa=math::poly::chebyshev(n,x[i]);
			error[n]+=fabs(fe-fa);
		}
		error[n]/=N;
	}
	//print
	std::cout<<"test - chebyshev\n";
	for(int n=0; n<L; ++n){
		std::cout<<"error["<<n<<"]  = "<<error[n]<<"\n";
	}
}

void test_chebysheva_error(){
	std::srand(std::time(NULL));
	//constants
	const int N=100;
	std::vector<double> x(N);
	//generate x
	for(int i=0; i<N; ++i){
		x[i]=2.0*((1.0*std::rand())/RAND_MAX-0.5);
	}
	//compute error
	const int L=4;
	std::vector<double> fa;
	std::vector<double> error(L,0);
	for(int i=0; i<N; ++i){
		math::poly::chebyshev(L,x[i],fa);
		for(int n=0; n<=L; ++n){
			double fe=0.0;
			switch(n){
				case 0: fe=1.0; break;
				case 1: fe=x[i]; break;
				case 2: fe=2.0*x[i]*x[i]-1.0; break;
				case 3: fe=4.0*x[i]*x[i]*x[i]-3.0*x[i]; break;
				case 4: fe=8.0*x[i]*x[i]*x[i]*x[i]-8.0*x[i]*x[i]+1.0; break;
			}
			error[n]+=fabs(fe-fa[n]);
		}
	}
	//print
	std::cout<<"test - chebysheva\n";
	for(int n=0; n<=L; ++n){
		error[n]/=N;
		std::cout<<"error["<<n<<"]  = "<<error[n]<<"\n";
	}
}

int main(int argc, char* argv[]){
	
	//test_simple_time();
	
	test_alegendre_error();
	//test_alegendre_time();
	//test_legendre_alegendre_error();
	
	test_plegendre_error();
	//test_plegendre_time();
	//test_legendre_plegendre_error();
	
	test_chebyshev_error();
	test_chebysheva_error();
	
	return 0;
}