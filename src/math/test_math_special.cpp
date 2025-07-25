// c
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
// c++
#include <iostream>
#include <vector>
//math
#include "math/special.hpp"
#include "math/reduce.hpp"
//str
#include "str/print.hpp"
//util
#include "util/time.hpp"

using namespace std::chrono;

void test_mod(){
	const int len=10;
	std::cout<<"len = "<<len<<"\n";
	std::cout<<"  0 mod len = "<<math::special::mod(0,len)<<"\n";
	std::cout<<"  5 mod len = "<<math::special::mod(5,len)<<"\n";
	std::cout<<" 10 mod len = "<<math::special::mod(10,len)<<"\n";
	std::cout<<" -1 mod len = "<<math::special::mod(-1,len)<<"\n";
	std::cout<<" -9 mod len = "<<math::special::mod(-9,len)<<"\n";
	std::cout<<"-15 mod len = "<<math::special::mod(-15,len)<<"\n";
	std::cout<<"-19 mod len = "<<math::special::mod(-19,len)<<"\n";
	std::cout<<"-20 mod len = "<<math::special::mod(-20,len)<<"\n";
}

void test_fma(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000;
	std::vector<double> x(N);
	std::vector<double> y(N);
	std::vector<double> z(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX;
		y[i]=(1.0*std::rand())/RAND_MAX;
		z[i]=(1.0*std::rand())/RAND_MAX;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		for(int j=0; j<N; ++j){
			volatile double f=x[j]*y[j]+z[j];
		}
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//fma
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		for(int j=0; j<N; ++j){
			volatile double f=std::fma(x[j],y[j],z[j]);
		}
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0;
	for(int i=0; i<N; ++i){
		const double f1=x[i]*y[i]+z[i];
		const double f2=std::fma(x[i],y[i],z[i]);
		err+=fabs(f2-f1);
		reduce.push(f1,f2);
	}
	err/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - FMA",str)<<"\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err = "<<err<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_powint(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000000;
	double xmin=-5.0;
	double xmax=5.0;
	std::vector<double> x(N);
	std::vector<double> p(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)-xmin;
		p[i]=std::rand()%32-16;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=pow(x[i],p[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::powint(x[i],p[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=pow(x[i],p[i]);
		const double f2=math::special::powint(x[i],p[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - POWINT",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_fmexp(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000000;
	std::vector<double> x(N);
	const double xmin=-20.0;
	const double xmax=20.0;
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=exp(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::fmexp(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=exp(x[i]);
		const double f2=math::special::fmexp(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//write
	/*const int Nw=1000;
	std::vector<double> xw(Nw);
	FILE* writer=fopen("fmexp.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#x exact approx\n");
		for(int i=0; i<Nw; ++i){
			xw[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
			fprintf(writer,"%f %f %f\n",xw[i],
				exp(xw[i]),math::special::fmexp(xw[i])
			);
		}
		fclose(writer);
		writer=NULL;
	}*/
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - FMEXP",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_exp10(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000000;
	std::vector<double> x(N);
	const double xmin=-20.0;
	const double xmax=0.0;
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=exp(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::expn10(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=exp(x[i]);
		const double f2=math::special::expn10(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//write
	/*const int Nw=1000;
	std::vector<double> xw(Nw);
	FILE* writer=fopen("expn10.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#x exact approx\n");
		for(int i=0; i<Nw; ++i){
			xw[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
			fprintf(writer,"%f %f %f\n",xw[i],
				exp(xw[i]),math::special::expn10(xw[i])
			);
		}
		fclose(writer);
		writer=NULL;
	}*/
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - EXP10",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_gauss10(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000000;
	std::vector<double> x(N);
	std::vector<double> f_std(N);
	std::vector<double> f_new(N);
	const double xmin=-20.0;
	const double xmax=20.0;
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=exp(-x[i]*x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::gauss10(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=exp(-x[i]*x[i]);
		const double f2=math::special::gauss10(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//write
	/*const int Nw=1000;
	std::vector<double> xw(Nw);
	FILE* writer=fopen("gauss10.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#x exact approx\n");
		for(int i=0; i<Nw; ++i){
			xw[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
			fprintf(writer,"%f %f %f\n",xw[i],
				exp(xw[i]),math::special::gauss10(xw[i])
			);
		}
		fclose(writer);
		writer=NULL;
	}*/
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - GAUSS10",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_logcosh(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=10000;
	const double xmin=-8.0;
	const double xmax=8.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=std::log(std::cosh(x[i]));
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::logcosh(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=std::log(std::cosh(x[i]));
		const double f2=math::special::logcosh(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//write
	FILE* writer=fopen("test.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#x exact approx\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",x[i],
				std::log(std::cosh(x[i])),
				math::special::logcosh(x[i])
			);
		}
		fclose(writer);
		writer=NULL;
	}
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - LNCOSH",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_softplus(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=10000;
	const double xmin=-8.0;
	const double xmax=8.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::softplus(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::softplus2(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=math::special::softplus(x[i]);
		const double f2=math::special::softplus2(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//write
	FILE* writer=fopen("test_math_special_softplus.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#x exact approx\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",x[i],
				math::special::softplus(x[i]),
				math::special::softplus2(x[i])
			);
		}
		fclose(writer);
		writer=NULL;
	}
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - SOTPLUS",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_coscut(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=100000;
	const double xmin=0.0;
	const double xmax=3.14159;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=std::cos(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::coscut(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=std::cos(x[i]);
		const double f2=math::special::coscut(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//write
	/*const int Nw=1000;
	std::vector<double> xw(Nw);
	FILE* writer=fopen("coscut.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#x exact approx\n");
		for(int i=0; i<Nw; ++i){
			xw[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
			fprintf(writer,"%f %f %f\n",xw[i],
				std::cos(xw[i]),math::special::coscut(xw[i])
			);
		}
		fclose(writer);
		writer=NULL;
	}*/
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - COSCUT",str)<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_sincut(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=100000;
	const double xmin=0.0;
	const double xmax=3.14159;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=std::sin(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::sincut(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=std::sin(x[i]);
		const double f2=math::special::sincut(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//write
	/*const int Nw=1000;
	std::vector<double> xw(Nw);
	FILE* writer=fopen("sincut.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#x exact approx\n");
		for(int i=0; i<Nw; ++i){
			xw[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
			fprintf(writer,"%f %f %f\n",xw[i],
				std::sin(xw[i]),math::special::sincut(xw[i])
			);
		}
		fclose(writer);
		writer=NULL;
	}*/
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - SINCUT",str)<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_tanh(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000000;
	const double xmin=-10.0;
	const double xmax=10.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*xmax-xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=std::tanh(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::tanh(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=std::tanh(x[i]);
		const double f2=math::special::tanh(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - TANH",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_sech(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000000;
	const double xmin=-10.0;
	const double xmax=10.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*xmax-xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=1.0/std::cosh(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::sech(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=1.0/std::cosh(x[i]);
		const double f2=math::special::sech(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - SECH",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_sech2(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000000;
	const double xmin=-10.0;
	const double xmax=10.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*xmax-xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=1.0/(std::cosh(x[i])*std::cosh(x[i]));
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::sech2(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=1.0/(std::cosh(x[i])*std::cosh(x[i]));
		const double f2=math::special::sech2(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - SECH2",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_tanhsech(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000000;
	const double xmin=-10.0;
	const double xmax=10.0;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*xmax-xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double fsech=1.0/std::cosh(x[i]);
		volatile double ftanh=std::tanh(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		double fsech,ftanh;
		math::special::tanhsech(x[i],ftanh,fsech);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double ftanh1=std::tanh(x[i]);
		const double fsech1=1.0/std::cosh(x[i]);
		double fsech2,ftanh2;
		math::special::tanhsech(x[i],ftanh2,fsech2);
		err+=fabs(ftanh2-ftanh1);
		err+=fabs(fsech2-fsech1);
		errp+=fabs((ftanh2-ftanh1)/ftanh1)*100.0;
		errp+=fabs((fsech2-fsech1)/fsech1)*100.0;
		reduce.push(fsech1,fsech2);
		reduce.push(ftanh1,ftanh2);
	}
	err/=N;
	errp/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - TANHSECH",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_ylmr_add(){
	std::srand(std::time(NULL));
	const int lmax=1;
	for(int l=0; l<lmax; ++l){
		/*x[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
		const double theta=(1.0*std::rand())/RAND_MAX*(PI-0.0)+0.0;
		const double phi=(1.0*std::rand())/RAND_MAX*(PI-(-PI))+(-PI);
		math::special::YLMR ylmr(l);
		ylmr.compute(theta,phi);
		double sum1=0;
		for(int m=-l; m<l; m++){
			sum1+=pow(-1.0,m)*ylmr.val(m)*ylmr.val(-m)
		}
		double sum2=(2.0+l)/(4.0*PI)*math::poly::legendre(l,cos(*/
	}
}

void test_erfa1(){
	std::srand(std::time(NULL));
	high_resolution_clock::time_point tbeg,tend;
	Reduce<2> reduce;
	//constants
	const int N=1000000;
	const double xmin=-1.0e1;
	const double xmax=1.0e1;
	std::vector<double> x(N);
	for(int i=0; i<N; ++i){
		x[i]=(1.0*std::rand())/RAND_MAX*xmax-xmin;
	}
	//standard
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=std::erf(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_std = duration_cast<duration<double>>(tend-tbeg);
	//new
	tbeg=high_resolution_clock::now();
	for(int i=0; i<N; ++i){
		volatile double f=math::special::erfa1(x[i]);
	}
	tend=high_resolution_clock::now();
	duration<double> time_new = duration_cast<duration<double>>(tend-tbeg);
	//accuracy
	double err=0,errp=0;
	for(int i=0; i<N; ++i){
		const double f1=std::erf(x[i]);
		const double f2=math::special::erfa1(x[i]);
		err+=fabs(f2-f1);
		errp+=fabs((f2-f1)/f1*100.0);
		reduce.push(f1,f2);
	}
	err/=N;
	errp/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - ERF",str)<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<"]\n";
	std::cout<<"time - std = "<<time_std.count()<<"\n";
	std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

int main(int argc, char* argv[]){
	
	test_fma();
	test_powint();
	test_fmexp();
	test_exp10();
	test_gauss10();
	test_logcosh();
	test_softplus();
	test_coscut();
	test_sincut();
	test_tanh();
	test_sech();
	test_sech2();
	test_tanhsech();
	test_mod();
	test_ylmr_add();
	test_erfa1();
	
	return 1;
}