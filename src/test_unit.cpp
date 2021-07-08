// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
#include <iomanip>
// ann - math
#include "math_const.hpp"
#include "math_special.hpp"
#include "accumulator.hpp"
// ann - eigen
#include "eigen.hpp"
// ann - string
#include "string.hpp"
// ann - optimization
#include "optimize.hpp"
// ann - ewald
#include "ewald3D.hpp"
// ann - cutoff
#include "cutoff.hpp"
// ann - symmetry functions
#include "symm_radial_t1.hpp"
#include "symm_radial_g1.hpp"
#include "symm_radial_g2.hpp"
#include "symm_angular_g3.hpp"
#include "symm_angular_g4.hpp"
// ann - neural network
#include "nn.hpp"
// ann - neural network potential
#include "nn_pot.hpp"
// ann - structure
#include "structure.hpp"
#include "cell_list.hpp"
#include "vasp.hpp"
// ann - units
#include "units.hpp"
// ann - compiler
#include "compiler.hpp"
// ann - print
#include "print.hpp"
// ann - random
#include "random.hpp"
// ann - list
#include "list.hpp"
// ann - test - unit
#include "test_unit.hpp"

//**********************************************
// string
//**********************************************

void test_string(){
	char* buf=new char[print::len_buf];
	
	//==== copying ====
	{
		std::cout<<print::buf(buf,char_buf)<<"\n";
		char* dest=new char[50];
		char* src=new char[50];
		std::strcpy(src,"thisisatest");
		std::cout<<"string::copy(char*,char*)\n";
		string::copy(dest,src);
		std::cout<<"src  = "<<src<<"\n";
		std::cout<<"dest = "<<dest<<"\n";
		std::cout<<print::buf(buf,char_buf)<<"\n";
		std::cout<<"string::copy(char*,char*,int)\n";
		string::copy(dest,src,5);
		std::cout<<"src  = "<<src<<"\n";
		std::cout<<"dest = "<<dest<<"\n";
		std::cout<<print::buf(buf,char_buf)<<"\n";
	}
	
	delete[] buf;
}

//**********************************************
// math_special
//**********************************************

void test_math_special_cos(){
	//local variables
	const int N=10000;
	const double xmin=0;
	const double xmax=math::constant::PI;
	std::vector<double> x(N,0);
	std::vector<double> cos_exact(N,0);
	std::vector<double> cos_approx(N,0);
	double time_exact,time_approx;
	clock_t start,stop;
	
	//generate abscissae
	for(int i=N-1; i>=0; --i) x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
	//generate exact ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) cos_exact[i]=std::cos(x[i]);
	stop=std::clock();
	time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
	//generate approximate ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) cos_approx[i]=math::special::cos(x[i]);
	stop=std::clock();
	time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
	//compute error
	double err_avg=0,err_max=0;
	for(int i=N-1; i>=0; --i){
		double err=std::fabs(cos_approx[i]-cos_exact[i]);
		err_avg+=err;
		err_max=(err>err_max)?err:err_max;
	}
	err_avg/=N;
	
	time_exact*=1e9/N;
	time_approx*=1e9/N;
	
	//print results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - MATH - SPECIAL - COS\n";
	std::cout<<"N             = "<<N<<"\n";
	std::cout<<"interval      = "<<xmin<<" "<<xmax<<"\n";
	std::cout<<"error - avg   = "<<err_avg<<"\n";
	std::cout<<"error - max   = "<<err_max<<"\n";
	std::cout<<"time - exact  = "<<time_exact<<" ns\n";
	std::cout<<"time - approx = "<<time_approx<<" ns\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_math_special_sin(){
	//local variables
	const int N=10000;
	const double xmin=0;
	const double xmax=math::constant::PI;
	std::vector<double> x(N,0);
	std::vector<double> sin_exact(N,0);
	std::vector<double> sin_approx(N,0);
	double time_exact,time_approx;
	clock_t start,stop;
	
	//generate abscissae
	for(int i=N-1; i>=0; --i) x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
	//generate exact ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) sin_exact[i]=std::sin(x[i]);
	stop=std::clock();
	time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
	//generate approximate ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) sin_approx[i]=math::special::sin(x[i]);
	stop=std::clock();
	time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
	//compute error
	double err_avg=0,err_max=0;
	for(int i=N-1; i>=0; --i){
		double err=std::fabs(sin_approx[i]-sin_exact[i]);
		err_avg+=err;
		err_max=(err>err_max)?err:err_max;
	}
	err_avg/=N;
	
	time_exact*=1e9/N;
	time_approx*=1e9/N;
	
	//print results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - MATH - SPECIAL - SIN\n";
	std::cout<<"N             = "<<N<<"\n";
	std::cout<<"interval      = "<<xmin<<" "<<xmax<<"\n";
	std::cout<<"error - avg   = "<<err_avg<<"\n";
	std::cout<<"error - max   = "<<err_max<<"\n";
	std::cout<<"time - exact  = "<<time_exact<<" ns\n";
	std::cout<<"time - approx = "<<time_approx<<" ns\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_math_special_logp1(){
	//local variables
	const int N=10000;
	const double xmin=-0.5;
	const double xmax=1.0;
	std::vector<double> x(N,0);
	std::vector<double> exact(N,0);
	std::vector<double> approx(N,0);
	double time_exact,time_approx;
	clock_t start,stop;
	
	//generate abscissae
	for(int i=N-1; i>=0; --i) x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
	//generate exact ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) exact[i]=std::log(x[i]+1.0);
	stop=std::clock();
	time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
	//generate approximate ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) approx[i]=math::special::logp1(x[i]);
	stop=std::clock();
	time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
	//compute error
	double err_avg=0,err_max=0;
	for(int i=N-1; i>=0; --i){
		double err=std::fabs(approx[i]-exact[i]);
		err_avg+=err;
		err_max=(err>err_max)?err:err_max;
	}
	err_avg/=N;
	
	time_exact*=1e9/N;
	time_approx*=1e9/N;
	
	//print results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - MATH - SPECIAL - LOGP1\n";
	std::cout<<"N             = "<<N<<"\n";
	std::cout<<"interval      = "<<xmin<<" "<<xmax<<"\n";
	std::cout<<"error - avg   = "<<err_avg<<"\n";
	std::cout<<"error - max   = "<<err_max<<"\n";
	std::cout<<"time - exact  = "<<time_exact<<" ns\n";
	std::cout<<"time - approx = "<<time_approx<<" ns\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// list
//**********************************************

void test_list_shuffle(){
	const int N=20;
	int* arr=new int[N];
	for(int i=0; i<N; ++i) arr[i]=i+1;
	rng::gen::CG2 cg2=rng::gen::CG2(std::time(NULL));
	
	const int M=10;
	char* str=new char[print::len_buf];
	std::cout<<"TEST - LIST - SHUFFLE\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	for(int j=0; j<M; ++j){
		std::cout<<"arr = "; for(int i=0; i<N; ++i) std::cout<<arr[i]<<" "; std::cout<<"\n";
		list::shuffle(arr,N,cg2);
	}
	std::cout<<print::buf(str,char_buf)<<"\n";
	
	delete[] str;
	delete[] arr;
}

//**********************************************
// accumulator
//**********************************************

void test_acc_1D(){
	//local data
	const int N=100;
	std::vector<double> data(N);
	Accumulator1D<Min,Max,Avg,Var> acc;
	
	//initialize random number generator
	std::srand(std::time(NULL));
	
	//generate the data
	for(int i=0; i<N; ++i){
		data[i]=((double)std::rand())/RAND_MAX;
		acc.push(data[i]);
	}
	
	//compute min
	double min=data[0];
	for(int i=1; i<N; ++i){
		if(data[i]<min) min=data[i];
	}
	std::cout<<"err - min = "<<std::fabs(min-acc.min())<<"\n";
	
	//compute max
	double max=data[0];
	for(int i=1; i<N; ++i){
		if(data[i]>max) max=data[i];
	}
	std::cout<<"err - max = "<<std::fabs(max-acc.max())<<"\n";
	
	//compute avg
	double avg=data[0];
	for(int i=1; i<N; ++i){
		avg+=data[i];
	}
	avg/=N;
	std::cout<<"err - avg = "<<std::fabs(avg-acc.avg())<<"\n";
	
	//compute var
	double var=0;
	for(int i=0; i<N; ++i){
		var+=(data[i]-avg)*(data[i]-avg);
	}
	var/=(N-1);
	std::cout<<"err - var = "<<std::fabs(var-acc.var())<<"\n";
}

void test_acc_2D(){
	//local data
	const int N=1000;
	const double w=2.0*math::constant::PI*5.0;
	std::vector<double> cos(N);
	std::vector<double> cos_shift(N);
	std::vector<double> rand1(N);
	std::vector<double> rand2(N);
	Accumulator2D<PCorr> acc;
	
	//initialize random number generator
	std::srand(std::time(NULL));
	
	//generate data
	for(int i=0; i<N; ++i){
		rand1[i]=((double)std::rand())/RAND_MAX;
		rand2[i]=((double)std::rand())/RAND_MAX;
		cos[i]=std::cos(w*i/((double)N));
		cos_shift[i]=std::cos(w*i/((double)N)+0.5*math::constant::PI);
	}
	
	//test - pcorr
	acc.clear();
	for(int i=0; i<N; ++i) acc.push(rand1[i],rand2[i]);
	std::cout<<"pcorr - rand  = "<<acc.pcorr()<<"\n";
	acc.clear();
	for(int i=0; i<N; ++i) acc.push(cos[i],cos[i]);
	std::cout<<"pcorr - cos   = "<<acc.pcorr()<<"\n";
	acc.clear();
	for(int i=0; i<N-1; ++i) acc.push(cos[i],cos[i+1]);
	std::cout<<"pcorr - cos+1 = "<<acc.pcorr()<<"\n";
	acc.clear();
	for(int i=0; i<N; ++i) acc.push(cos[i],cos_shift[i]);
	std::cout<<"pcorr - shift = "<<acc.pcorr()<<"\n";
}

//**********************************************
// cutoff
//**********************************************

void test_cutoff_cos(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//compute error
	double integral=0.5*(cutf.val(0.0)+cutf.val(rc));
	for(int i=N-2; i>=1; --i){
		integral+=cutf.val(i/(N-1.0)*rc);
	}
	integral*=dr;
	double errg=0;
	for(int i=N-2; i>=1; --i){
		const double g=0.5*(cutf.val((i+1.0)/(N-1.0)*rc)-cutf.val((i-1.0)/(N-1.0)*rc))/dr;
		errg+=std::fabs(g-cutf.grad(i/(N-1.0)*rc));
	}
	errg/=(N-1.0);
	//compute time
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--) volatile double val=cutf.val(std::rand()/RAND_MAX*rc);
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--) volatile double val=cutf.grad(std::rand()/RAND_MAX*rc);
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - CUTOFF - COS\n";
	std::cout<<"err  - integral = "<<std::fabs(0.5*rc-integral)<<"\n";
	std::cout<<"err  - gradient = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// symm
//**********************************************

void test_symm_t1(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//compute error
	double errg=0,r;
	PhiR_T1 t1(1.4268,0.56278905);
	for(int i=N-2; i>=1; --i){
		r=(i+1.0)/(N-1.0)*rc;
		const double f2=t1.val(r,cutf.val(r));
		r=(i-1.0)/(N-1.0)*rc;
		const double f1=t1.val(r,cutf.val(r));
		const double g=0.5*(f2-f1)/dr;
		r=i/(N-1.0)*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		errg+=std::fabs(g-t1.grad(r,cut,gcut));
	}
	errg/=(N-1.0);
	//compute time
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=t1.val(r,cutf.val(r));
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		volatile double val=t1.grad(r,cut,gcut);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiR_T1 t1c(1.5427856487,0.816578674);
	const int size=serialize::nbytes(t1);
	char* arr=new char[size];
	const int size_pack=serialize::pack(t1,arr);
	const int size_unpack=serialize::unpack(t1c,arr);
	const double errs=(
		std::abs(t1.rs-t1c.rs)
		+std::abs(t1.eta-t1c.eta)
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - T1\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_symm_g1(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//compute error
	double errg=0,r;
	PhiR_G1 g1;
	for(int i=N-2; i>=1; --i){
		r=(i+1.0)/(N-1.0)*rc;
		const double f2=g1.val(r,cutf.val(r));
		r=(i-1.0)/(N-1.0)*rc;
		const double f1=g1.val(r,cutf.val(r));
		const double g=0.5*(f2-f1)/dr;
		r=i/(N-1.0)*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		errg+=std::fabs(g-g1.grad(r,cut,gcut));
	}
	errg/=(N-1.0);
	//compute time
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=g1.val(r,cutf.val(r));
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		volatile double val=g1.grad(r,cut,gcut);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiR_G1 g1c;
	const int size=serialize::nbytes(g1);
	char* arr=new char[size];
	const int size_pack=serialize::pack(g1,arr);
	const int size_unpack=serialize::unpack(g1c,arr);
	const double errs=(
		std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - G1\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_symm_g2(){
	const double rc=6.0;
	const int N=100;
	cutoff::Cos cutf=cutoff::Cos(rc);
	//compute error
	const double dr=rc/(N-1.0);
	double errg=0,r;
	PhiR_G2 g2(1.4268,0.56278905);
	for(int i=N-2; i>=1; --i){
		r=(i+1.0)/(N-1.0)*rc;
		const double f2=g2.val(r,cutf.val(r));
		r=(i-1.0)/(N-1.0)*rc;
		const double f1=g2.val(r,cutf.val(r));
		const double g=0.5*(f2-f1)/dr;
		r=i/(N-1.0)*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		errg+=std::fabs(g-g2.grad(r,cut,gcut));
	}
	errg/=(N-1.0);
	//compute time
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=g2.val(r,cutf.val(r));
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r=std::rand()/RAND_MAX*rc;
		const double cut=cutf.val(r);
		const double gcut=cutf.grad(r);
		volatile double val=g2.grad(r,cut,gcut);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiR_G2 g2c;
	const int size=serialize::nbytes(g2);
	char* arr=new char[size];
	const int size_pack=serialize::pack(g2,arr);
	const int size_unpack=serialize::unpack(g2c,arr);
	const double errs=(
		std::abs(g2.rs-g2c.rs)
		+std::abs(g2.eta-g2c.eta)
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - G2\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_symm_g3(){
	//local variables
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double errgd[3]={0,0,0};
	double errga=0;
	double r[3],c[3];
	PhiA_G3 g3(1.4268,2.5,1);
	const double cos=1.0/std::sqrt(2.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//grad - dist - 0
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[0]=(i+1.0)/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double f2=g3.val(cos,r,c);
		//second point
		r[0]=(i-1.0)/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double f1=g3.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[0]=i/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double gcut=cutf.grad(r[0]);
		//error
		errgd[0]+=std::fabs(g-g3.grad_dist_0(r,c,gcut)*g3.angle(cos));
	}
	errgd[0]/=(N-1.0);
	//grad - dist - 1
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[1]=(i+1.0)/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double f2=g3.val(cos,r,c);
		//second point
		r[1]=(i-1.0)/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double f1=g3.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[1]=i/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double gcut=cutf.grad(r[1]);
		//error
		errgd[1]+=std::fabs(g-g3.grad_dist_1(r,c,gcut)*g3.angle(cos));
	}
	errgd[1]/=(N-1.0);
	//grad - dist - 2
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[2]=(i+1.0)/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double f2=g3.val(cos,r,c);
		//second point
		r[2]=(i-1.0)/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double f1=g3.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[2]=i/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double gcut=cutf.grad(r[2]);
		//error
		errgd[2]+=std::fabs(g-g3.grad_dist_2(r,c,gcut)*g3.angle(cos));
	}
	errgd[2]/=(N-1.0);
	//grad - angle
	for(int i=N-2; i>=1; --i){
		double cosv;
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		cosv=(i+1.0)/(N-1.0)*math::constant::PI;
		const double f2=g3.val(cosv,r,c);
		//second point
		cosv=(i-1.0)/(N-1.0)*math::constant::PI;
		const double f1=g3.val(cosv,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		cosv=i/(N-1.0)*math::constant::PI;
		const double gcut=cutf.grad(r[2]);
		//error
		errga+=std::fabs(g-g3.grad_angle(cosv)*g3.dist(r,c));
	}
	errga/=(N-1.0);
	//time - value
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r[0]=std::rand()/RAND_MAX*rc;
		r[1]=std::rand()/RAND_MAX*rc;
		r[2]=std::rand()/RAND_MAX*rc;
		c[0]=cutf.val(r[0]);
		c[1]=cutf.val(r[1]);
		c[2]=cutf.val(r[2]);
		volatile double val=g3.val(cos,r,c);
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r[0]=std::rand()/RAND_MAX*rc;
		r[1]=std::rand()/RAND_MAX*rc;
		r[2]=std::rand()/RAND_MAX*rc;
		c[0]=cutf.val(r[0]);
		c[1]=cutf.val(r[1]);
		c[2]=cutf.val(r[2]);
		const double gcut=cutf.grad(r[0]);
		volatile double val=g3.grad_dist_2(r,c,gcut)*g3.angle(cos);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiA_G3 g3c;
	const int size=serialize::nbytes(g3);
	char* arr=new char[size];
	const int size_pack=serialize::pack(g3,arr);
	const int size_unpack=serialize::unpack(g3c,arr);
	const double errs=(
		std::abs(g3.zeta-g3c.zeta)
		+std::abs(g3.eta-g3c.eta)
		+std::abs(g3.lambda-g3c.lambda)
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - G3\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad - dist[0] = "<<errgd[0]<<"\n";
	std::cout<<"err - grad - dist[1] = "<<errgd[1]<<"\n";
	std::cout<<"err - grad - dist[2] = "<<errgd[2]<<"\n";
	std::cout<<"err - grad - angle   = "<<errga<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_symm_g4(){
	//local variables
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double errgd[3]={0,0,0};
	double errga=0;
	double r[3],c[3];
	PhiA_G4 g4(1.4268,2.5,1);
	const double cos=1.0/std::sqrt(2.0);
	cutoff::Cos cutf=cutoff::Cos(rc);
	//grad - dist - 0
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[0]=(i+1.0)/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double f2=g4.val(cos,r,c);
		//second point
		r[0]=(i-1.0)/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double f1=g4.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[0]=i/(N-1.0)*rc; c[0]=cutf.val(r[0]);
		const double gcut=cutf.grad(r[0]);
		//error
		errgd[0]+=std::fabs(g-g4.grad_dist_0(r,c,gcut)*g4.angle(cos));
	}
	errgd[0]/=(N-1.0);
	//grad - dist - 1
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[1]=(i+1.0)/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double f2=g4.val(cos,r,c);
		//second point
		r[1]=(i-1.0)/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double f1=g4.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[1]=i/(N-1.0)*rc; c[1]=cutf.val(r[1]);
		const double gcut=cutf.grad(r[1]);
		//error
		errgd[1]+=std::fabs(g-g4.grad_dist_1(r,c,gcut)*g4.angle(cos));
	}
	errgd[1]/=(N-1.0);
	//grad - dist - 2
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		r[2]=(i+1.0)/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double f2=g4.val(cos,r,c);
		//second point
		r[2]=(i-1.0)/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double f1=g4.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[2]=i/(N-1.0)*rc; c[2]=cutf.val(r[2]);
		const double gcut=cutf.grad(r[2]);
		errgd[2]+=std::fabs(g-g4.grad_dist_2(r,c,gcut)*g4.angle(cos));
	}
	errgd[2]/=(N-1.0);
	//grad - angle
	for(int i=N-2; i>=1; --i){
		double cosv;
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=cutf.val(r[0]); c[1]=cutf.val(r[1]); c[2]=cutf.val(r[2]);
		//first point
		cosv=(i+1.0)/(N-1.0)*math::constant::PI;
		const double f2=g4.val(cosv,r,c);
		//second point
		cosv=(i-1.0)/(N-1.0)*math::constant::PI;
		const double f1=g4.val(cosv,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		cosv=i/(N-1.0)*math::constant::PI;
		errga+=std::fabs(g-g4.grad_angle(cosv)*g4.dist(r,c));
	}
	errga/=(N-1.0);
	//time - value
	double timef,timed;
	clock_t start,stop;
	const int Nt=1e5;
	std::srand(std::time(NULL));
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r[0]=std::rand()/RAND_MAX*rc;
		r[1]=std::rand()/RAND_MAX*rc;
		r[2]=std::rand()/RAND_MAX*rc;
		c[0]=cutf.val(r[0]);
		c[1]=cutf.val(r[1]);
		c[2]=cutf.val(r[2]);
		volatile double val=g4.val(cos,r,c);
	}
	stop=std::clock();
	timef=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	start=std::clock();
	for(int i=Nt-1; i>=0; i--){
		r[0]=std::rand()/RAND_MAX*rc;
		r[1]=std::rand()/RAND_MAX*rc;
		r[2]=std::rand()/RAND_MAX*rc;
		c[0]=cutf.val(r[0]);
		c[1]=cutf.val(r[1]);
		c[2]=cutf.val(r[2]);
		const double gcut=cutf.grad(r[0]);
		volatile double val=g4.grad_dist_2(r,c,gcut)*g4.angle(cos);
	}
	stop=std::clock();
	timed=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	//serialization
	PhiA_G4 g4c;
	const int size=serialize::nbytes(g4);
	char* arr=new char[size];
	const int size_pack=serialize::pack(g4,arr);
	const int size_unpack=serialize::unpack(g4c,arr);
	const double errs=(
		std::abs(g4.zeta-g4c.zeta)
		+std::abs(g4.eta-g4c.eta)
		+std::abs(g4.lambda-g4c.lambda)
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM - G4\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - gradient - dist[0] = "<<errgd[0]<<"\n";
	std::cout<<"err - gradient - dist[1] = "<<errgd[1]<<"\n";
	std::cout<<"err - gradient - dist[2] = "<<errgd[2]<<"\n";
	std::cout<<"err - gradient - angle   = "<<errga<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<"time - function = "<<timef<<" ns\n";
	std::cout<<"time - gradient = "<<timed<<" ns\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// eigen
//**********************************************

void test_unit_eigen_vec3d(){
	int memsize=0;
	char* memarr=NULL;
	Eigen::Vector3d vec1=Eigen::Vector3d::Random();
	Eigen::Vector3d vec2=Eigen::Vector3d::Zero();
	memsize=serialize::nbytes(vec1);
	memarr=new char[memsize];
	serialize::pack(vec1,memarr);
	serialize::unpack(vec2,memarr);
	delete[] memarr;
	double err=(vec1-vec2).norm();
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - EIGEN - VECTOR3D\n";
	std::cout<<"err - serialization = "<<err<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_eigen_vecxd(){
	int memsize=0;
	char* memarr=NULL;
	Eigen::VectorXd vec1=Eigen::VectorXd::Random(7);
	Eigen::VectorXd vec2;
	memsize=serialize::nbytes(vec1);
	memarr=new char[memsize];
	serialize::pack(vec1,memarr);
	serialize::unpack(vec2,memarr);
	delete[] memarr;
	double err=(vec1-vec2).norm();
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - EIGEN - VECTORXD\n";
	std::cout<<"err - serialization = "<<err<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_eigen_mat3d(){
	int memsize=0;
	char* memarr=NULL;
	Eigen::Matrix3d vec1=Eigen::Matrix3d::Random();
	Eigen::Matrix3d vec2=Eigen::Matrix3d::Zero();
	memsize=serialize::nbytes(vec1);
	memarr=new char[memsize];
	serialize::pack(vec1,memarr);
	serialize::unpack(vec2,memarr);
	delete[] memarr;
	double err=(vec1-vec2).norm();
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - EIGEN - MATRIX3D\n";
	std::cout<<"err - serialization = "<<err<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_eigen_matxd(){
	int memsize=0;
	char* memarr=NULL;
	Eigen::MatrixXd vec1=Eigen::MatrixXd::Random(7,7);
	Eigen::MatrixXd vec2;
	memsize=serialize::nbytes(vec1);
	memarr=new char[memsize];
	serialize::pack(vec1,memarr);
	serialize::unpack(vec2,memarr);
	delete[] memarr;
	double err=(vec1-vec2).norm();
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - EIGEN - MATRIXXD\n";
	std::cout<<"err - serialization = "<<err<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// optimize
//**********************************************

void test_unit_opt_sgd(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::SGD;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::SGD sgd(data.dim());
	sgd.decay()=Opt::DECAY::CONST;
	sgd.alpha()=1;
	sgd.gamma()=1e-3;
	//optimize
	for(int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		sgd.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		switch(data.optVal()){
			case Opt::VAL::FTOL_REL: if(data.dv()<data.tol()) break;
			case Opt::VAL::XTOL_REL: if(data.dp()<data.tol()) break;
			case Opt::VAL::FTOL_ABS: if(data.val()<data.tol()) break;
		}
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	//serialization
	Opt::SGD sgdc;
	const int size=serialize::nbytes(sgd);
	char* arr=new char[size];
	const int size_pack=serialize::pack(sgd,arr);
	const int size_unpack=serialize::unpack(sgdc,arr);
	const double errs=(
		std::abs(sgd.dim()-sgdc.dim())
		+std::fabs(sgd.alpha()-sgdc.alpha())
		+std::fabs(sgd.lambda()-sgdc.lambda())
		+std::fabs(sgd.gamma()-sgdc.gamma())
		+std::fabs(sgd.power()-sgdc.power())
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	delete[] arr;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - SGD\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_opt_sdm(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::SDM;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::SDM sdm(data.dim());
	sdm.decay()=Opt::DECAY::CONST;
	sdm.alpha()=1;
	sdm.gamma()=1e-3;
	sdm.eta()=0.9;
	//optimize
	for(int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		sdm.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		switch(data.optVal()){
			case Opt::VAL::FTOL_REL: if(data.dv()<data.tol()) break;
			case Opt::VAL::XTOL_REL: if(data.dp()<data.tol()) break;
			case Opt::VAL::FTOL_ABS: if(data.val()<data.tol()) break;
		}
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	//serialization
	Opt::SDM sdmc;
	const int size=serialize::nbytes(sdm);
	char* arr=new char[size];
	const int size_pack=serialize::pack(sdm,arr);
	const int size_unpack=serialize::unpack(sdmc,arr);
	const double errs=(
		std::abs(sdm.dim()-sdmc.dim())
		+std::fabs(sdm.alpha()-sdmc.alpha())
		+std::fabs(sdm.lambda()-sdmc.lambda())
		+std::fabs(sdm.gamma()-sdmc.gamma())
		+std::fabs(sdm.power()-sdmc.power())
		+std::fabs(sdm.eta()-sdmc.eta())
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - SDM\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_opt_nag(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::NAG;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::NAG nag(data.dim());
	nag.decay()=Opt::DECAY::CONST;
	nag.alpha()=1;
	nag.gamma()=1e-3;
	nag.eta()=0.9;
	//optimize
	for(int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		nag.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		switch(data.optVal()){
			case Opt::VAL::FTOL_REL: if(data.dv()<data.tol()) break;
			case Opt::VAL::XTOL_REL: if(data.dp()<data.tol()) break;
			case Opt::VAL::FTOL_ABS: if(data.val()<data.tol()) break;
		}
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	//serialization
	Opt::NAG nagc;
	const int size=serialize::nbytes(nag);
	char* arr=new char[size];
	const int size_pack=serialize::pack(nag,arr);
	const int size_unpack=serialize::unpack(nagc,arr);
	const double errs=(
		std::abs(nag.dim()-nagc.dim())
		+std::fabs(nag.alpha()-nagc.alpha())
		+std::fabs(nag.lambda()-nagc.lambda())
		+std::fabs(nag.gamma()-nagc.gamma())
		+std::fabs(nag.power()-nagc.power())
		+std::fabs(nag.power()-nagc.power())
		+std::fabs(nag.eta()-nagc.eta())
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - NAG\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_opt_adam(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::ADAM;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::ADAM adam(data.dim());
	adam.decay()=Opt::DECAY::CONST;
	adam.alpha()=1;
	adam.gamma()=1e-3;
	//optimize
	for(int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		adam.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		switch(data.optVal()){
			case Opt::VAL::FTOL_REL: if(data.dv()<data.tol()) break;
			case Opt::VAL::XTOL_REL: if(data.dp()<data.tol()) break;
			case Opt::VAL::FTOL_ABS: if(data.val()<data.tol()) break;
		}
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	//serialization
	Opt::ADAM adamc;
	const int size=serialize::nbytes(adam);
	char* arr=new char[size];
	const int size_pack=serialize::pack(adam,arr);
	const int size_unpack=serialize::unpack(adamc,arr);
	const double errs=(
		std::abs(adam.dim()-adamc.dim())
		+std::fabs(adam.alpha()-adamc.alpha())
		+std::fabs(adam.lambda()-adamc.lambda())
		+std::fabs(adam.gamma()-adamc.gamma())
		+std::fabs(adam.power()-adamc.power())
		+std::fabs(adam.beta1i()-adamc.beta1i())
		+std::fabs(adam.beta2i()-adamc.beta2i())
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - ADAM\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_opt_nadam(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::NADAM;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::NADAM nadam(data.dim());
	nadam.decay()=Opt::DECAY::CONST;
	nadam.alpha()=1;
	nadam.gamma()=1e-3;
	//optimize
	for(int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		nadam.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		switch(data.optVal()){
			case Opt::VAL::FTOL_REL: if(data.dv()<data.tol()) break;
			case Opt::VAL::XTOL_REL: if(data.dp()<data.tol()) break;
			case Opt::VAL::FTOL_ABS: if(data.val()<data.tol()) break;
		}
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	//serialization
	Opt::NADAM nadamc;
	const int size=serialize::nbytes(nadam);
	char* arr=new char[size];
	const int size_pack=serialize::pack(nadam,arr);
	const int size_unpack=serialize::unpack(nadamc,arr);
	const double errs=(
		std::abs(nadam.dim()-nadamc.dim())
		+std::fabs(nadam.alpha()-nadamc.alpha())
		+std::fabs(nadam.lambda()-nadamc.lambda())
		+std::fabs(nadam.gamma()-nadamc.gamma())
		+std::fabs(nadam.power()-nadamc.power())
		+std::fabs(nadam.beta1i()-nadamc.beta1i())
		+std::fabs(nadam.beta2i()-nadamc.beta2i())
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - NADAM\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_opt_amsgrad(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::AMSGRAD;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::AMSGRAD amsgrad(data.dim());
	amsgrad.decay()=Opt::DECAY::CONST;
	amsgrad.alpha()=1;
	amsgrad.gamma()=1e-3;
	//optimize
	for(int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		amsgrad.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		switch(data.optVal()){
			case Opt::VAL::FTOL_REL: if(data.dv()<data.tol()) break;
			case Opt::VAL::XTOL_REL: if(data.dp()<data.tol()) break;
			case Opt::VAL::FTOL_ABS: if(data.val()<data.tol()) break;
		}
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	//serialization
	Opt::AMSGRAD amsgradc;
	const int size=serialize::nbytes(amsgrad);
	char* arr=new char[size];
	const int size_pack=serialize::pack(amsgrad,arr);
	const int size_unpack=serialize::unpack(amsgradc,arr);
	const double errs=(
		std::abs(amsgrad.dim()-amsgradc.dim())
		+std::fabs(amsgrad.alpha()-amsgradc.alpha())
		+std::fabs(amsgrad.lambda()-amsgradc.lambda())
		+std::fabs(amsgrad.gamma()-amsgradc.gamma())
		+std::fabs(amsgrad.power()-amsgradc.power())
		+std::fabs(amsgrad.beta1i()-amsgradc.beta1i())
		+std::fabs(amsgrad.beta2i()-amsgradc.beta2i())
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - AMSGRAD\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_opt_cg(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::CG;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::CG cg(data.dim());
	cg.decay()=Opt::DECAY::CONST;
	cg.alpha()=1;
	cg.gamma()=1e-3;
	//optimize
	for(int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		cg.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		switch(data.optVal()){
			case Opt::VAL::FTOL_REL: if(data.dv()<data.tol()) break;
			case Opt::VAL::XTOL_REL: if(data.dp()<data.tol()) break;
			case Opt::VAL::FTOL_ABS: if(data.val()<data.tol()) break;
		}
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	//serialization
	Opt::CG cgc;
	const int size=serialize::nbytes(cg);
	char* arr=new char[size];
	const int size_pack=serialize::pack(cg,arr);
	const int size_unpack=serialize::unpack(cgc,arr);
	const double errs=(
		std::abs(cg.dim()-cgc.dim())
		+std::fabs(cg.alpha()-cgc.alpha())
		+std::fabs(cg.lambda()-cgc.lambda())
		+std::fabs(cg.gamma()-cgc.gamma())
		+std::fabs(cg.power()-cgc.power())
		+std::abs(size_pack-size)
		+std::abs(size_unpack-size)
	);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - CG\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"err - s = "<<errs<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// ewald
//**********************************************

void test_unit_ewald_madelung(){
	//generate NaCl crystal
	units::System::type unitsys=units::System::AU;
	units::consts::init(unitsys);
	double a0=5.6199998856;
	if(unitsys==units::System::AU) a0*=units::BOHRpANG;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=true; atomT.posn=true;
	Structure strucg;
	const int natoms=8;
	strucg.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	strucg.init(lv);
	strucg.posn(0)<<0.000000000,0.000000000,0.000000000;
	strucg.posn(1)<<0.000000000,0.500000000,0.500000000;
	strucg.posn(2)<<0.500000000,0.000000000,0.500000000;
	strucg.posn(3)<<0.500000000,0.500000000,0.000000000;
	strucg.posn(4)<<0.500000000,0.500000000,0.500000000;
	strucg.posn(5)<<0.500000000,0.000000000,0.000000000;
	strucg.posn(6)<<0.000000000,0.500000000,0.000000000;
	strucg.posn(7)<<0.000000000,0.000000000,0.500000000;
	for(int i=0; i<8; ++i) strucg.posn(i)*=a0;
	for(int i=0; i<4; ++i) strucg.charge(i)=1;
	for(int i=4; i<8; ++i) strucg.charge(i)=-1;
	for(int i=0; i<4; ++i) strucg.name(i)="Na";
	for(int i=4; i<8; ++i) strucg.name(i)="Cl";
	//ewald
	Ewald3D::Coulomb ewald;
	static const double mc=1.74756;
	//initialize the ewald object
	const double prec=1e-8;
	ewald.init(strucg,prec);
	//compute the total energy
	const double r0=0.5*a0;//min dist
	const double mce=-2*ewald.energy(strucg)/strucg.nAtoms()*r0/units::consts::ke();
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - EWALD - MADELUNG\n";
	std::cout<<"prec              = "<<prec<<"\n";
	std::cout<<"madelung constant = "<<mc<<"\n";
	std::cout<<"madelung (ewald)  = "<<mce<<"\n";
	std::cout<<"error (percent)   = "<<std::fabs((mce-mc)/mc*100.0)<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_ewald_potential(){
	//generate NaCl crystal
	units::System::type unitsys=units::System::AU;
	units::consts::init(unitsys);
	double a0=5.6199998856;
	if(unitsys==units::System::AU) a0*=units::BOHRpANG;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=true; atomT.posn=true;
	Structure strucg;
	const int natoms=8;
	strucg.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	strucg.init(lv);
	strucg.posn(0)<<0.000000000,0.000000000,0.000000000;
	strucg.posn(1)<<0.000000000,0.500000000,0.500000000;
	strucg.posn(2)<<0.500000000,0.000000000,0.500000000;
	strucg.posn(3)<<0.500000000,0.500000000,0.000000000;
	strucg.posn(4)<<0.500000000,0.500000000,0.500000000;
	strucg.posn(5)<<0.500000000,0.000000000,0.000000000;
	strucg.posn(6)<<0.000000000,0.500000000,0.000000000;
	strucg.posn(7)<<0.000000000,0.000000000,0.500000000;
	for(int i=0; i<8; ++i) strucg.posn(i)*=a0;
	for(int i=0; i<4; ++i) strucg.charge(i)=1;
	for(int i=4; i<8; ++i) strucg.charge(i)=-1;
	for(int i=0; i<4; ++i) strucg.name(i)="Na";
	for(int i=4; i<8; ++i) strucg.name(i)="Cl";
	//ewald
	Ewald3D::Coulomb ewald;
	//initialize the ewald object
	ewald.init(strucg,1e-8);
	//compute the potentials
	double energy=0;
	for(int i=0; i<strucg.nAtoms(); ++i){
		const double pot=ewald.phi(strucg,i);
		energy+=pot*strucg.charge(i);
		std::cout<<"pot "<<strucg.name(i)<<" "<<pot<<" "<<strucg.charge(i)<<"\n";
	}
	energy*=0.5;
	//compute the total energy
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - EWALD - POTENTIAL\n";
	std::cout<<"energy/atom (ewald) = "<<ewald.energy(strucg)/strucg.nAtoms()<<"\n";
	std::cout<<"energy/atom (pot)   = "<<energy/strucg.nAtoms()<<"\n";
	std::cout<<"error (percent)     = "<<std::fabs((energy-ewald.energy(strucg))/ewald.energy(strucg)*100.0)<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// string
//**********************************************

void test_unit_string_hash(){
	const char* str1="lnt$*(nGnlj#o*t^nuoVho$&lks";
	const char* str2="lnt$*(nGnlj#o*t^nuoVho$&lks";
	const char* str3="lnt$*(nGnlj#o*t^nCoVho$&lks";
	const int hash1=string::hash(str1);
	const int hash2=string::hash(str2);
	const int hash3=string::hash(str3);
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - UNIT - STRING - HASH\n";
	std::cout<<"err -     equal = "<<std::fabs(1.0*hash1-1.0*hash2)<<"\n";
	std::cout<<"err - non-equal = "<<1.0/std::fabs(1.0*hash1-1.0*hash3)<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// random
//**********************************************

void test_random_seed(){
	
	rng::gen::CG1 cg1v1;
	rng::gen::CG1 cg1v2;
	rng::gen::CG1 cg1v3;
	
	cg1v1.init(std::time(NULL));
	cg1v2.init(std::time(NULL));
	cg1v3.init(std::time(NULL));
	
	for(int i=0; i<5; ++i){
		std::cout<<"cg1v1 "<<cg1v1.randf()<<"\n";
	}
	for(int i=0; i<5; ++i){
		std::cout<<"cg1v2 "<<cg1v2.randf()<<"\n";
	}
	for(int i=0; i<5; ++i){
		std::cout<<"cg1v3 "<<cg1v3.randf()<<"\n";
	}
}

void test_random_time(){
	//rng's
	rng::gen::LCG lcg;
	rng::gen::XOR xorc;
	rng::gen::MWC mwc;
	rng::gen::CG1 cg1;
	rng::gen::CG2 cg2;
	
	//init
	lcg.init(1);
	xorc.init(1);
	mwc.init(1);
	cg1.init(1);
	cg2.init(1);
	
	//time
	const int N=1000000;
	clock_t start,stop;
	
	//lcg
	start=std::clock();
	for(int i=0; i<N; ++i) volatile double f=lcg.randf();
	stop=std::clock();
	const double time_lcg=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	
	//xor
	start=std::clock();
	for(int i=0; i<N; ++i) volatile double f=xorc.randf();
	stop=std::clock();
	const double time_xor=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	
	//mwc
	start=std::clock();
	for(int i=0; i<N; ++i) volatile double f=mwc.randf();
	stop=std::clock();
	const double time_mwc=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	
	//cg1
	start=std::clock();
	for(int i=0; i<N; ++i) volatile double f=cg1.randf();
	stop=std::clock();
	const double time_cg1=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	
	//cg2
	start=std::clock();
	for(int i=0; i<N; ++i) volatile double f=cg2.randf();
	stop=std::clock();
	const double time_cg2=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	
	std::cout<<"time_lcg = "<<time_lcg<<"\n";
	std::cout<<"time_xor = "<<time_xor<<"\n";
	std::cout<<"time_mwc = "<<time_mwc<<"\n";
	std::cout<<"time_cg1 = "<<time_cg1<<"\n";
	std::cout<<"time_cg2 = "<<time_cg2<<"\n";
}

void test_random_dist(){
	//==== global variables ===
	//random number generator
	rng::gen::CG2 cg2=rng::gen::CG2(); cg2.init(1);
	//histogram
	const int N=1e6;
	const int nbins=100;
	//print
	std::cout<<"N     = "<<N<<"\n";
	std::cout<<"nbins = "<<nbins<<"\n";
	
	//=== exp ===
	{
		//parameters
		const double beta=1.26426748297829;
		//dist
		rng::dist::Exp dist=rng::dist::Exp(beta);
		//histogram
		const double xmin=0;
		const double xmax=5.0*beta;
		const double binw=(xmax-xmin)/nbins;
		//accumulator
		Accumulator1D<Dist> acc;
		acc.init(xmin,xmax,nbins,1);
		//compute dist
		for(int i=0; i<N; ++i){
			double f=0;
			do{f=dist(cg2);}while(f<xmin || xmax<f);
			acc.push(f);
		}
		//compute error
		double error=0;
		math::pdist::Exp exp(beta);
		for(int i=0; i<nbins; ++i){
			const double exact=exp.p(acc.abscissa(i));
			const double approx=acc.ordinate(i)/binw;
			error+=(exact-approx)*(exact-approx);
		}
		error=std::sqrt(error)/nbins;
		std::cout<<"error - exp    = "<<error<<"\n";
	}
	
	//=== normal ===
	{
		//parameters
		const double mu=0.0;
		const double sigma=1.0;
		//dist
		rng::dist::Normal dist=rng::dist::Normal(mu,sigma);
		//histogram
		const double xmin=mu-5.0*sigma;
		const double xmax=mu+5.0*sigma;
		const double binw=(xmax-xmin)/nbins;
		//accumulator
		Accumulator1D<Dist> acc;
		acc.init(xmin,xmax,nbins,1);
		//compute dist
		for(int i=0; i<N; ++i){
			double f=0;
			do{f=dist(cg2);}while(f<xmin || xmax<f);
			acc.push(f);
		}
		//compute error
		double error=0;
		math::pdist::Normal normal(mu,sigma);
		for(int i=0; i<nbins; ++i){
			const double exact=normal.p(acc.abscissa(i));
			const double approx=acc.ordinate(i)/binw;
			error+=(exact-approx)*(exact-approx);
		}
		error=std::sqrt(error)/nbins;
		std::cout<<"error - normal = "<<error<<"\n";
	}
	
}

//**********************************************
// nn
//**********************************************

void test_unit_nn(){
	//local function variables
	NeuralNet::ANN nn,nn_copy;
	NeuralNet::ANNInit init;
	//resize the nn
	init.sigma()=1.0;
	init.initType()=NeuralNet::InitN::HE;
	nn.tfType()=NeuralNet::TransferN::TANH;
	std::vector<int> nh(2);
	nh[0]=7; nh[1]=5;
	nn.resize(init,2,nh,3);
	//pack
	const int size=serialize::nbytes(nn);
	char* memarr=new char[size];
	serialize::pack(nn,memarr);
	serialize::unpack(nn_copy,memarr);
	std::cout<<"nn = \n"<<nn<<"\n";
	std::cout<<"nn_copy = \n"<<nn_copy<<"\n";
}

void test_unit_nn_tfunc(){
	FILE* writer=NULL;
	const int N=1000;
	const double xmin=-5;
	const double xmax=5;
	const double dx=(xmax-xmin)/(N-1.0);
	Eigen::VectorXd ff,fd;
	ff=Eigen::VectorXd::Zero(N);
	fd=Eigen::VectorXd::Zero(N);
	
	//linear
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_lin(ff,fd);
	writer=fopen("tfunc_linear.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//sigmoid
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_sigmoid(ff,fd);
	writer=fopen("tfunc_sigmoid.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//tanh
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_tanh(ff,fd);
	writer=fopen("tfunc_tanh.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//isru
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_isru(ff,fd);
	writer=fopen("tfunc_isru.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//arctan
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_arctan(ff,fd);
	writer=fopen("tfunc_arctan.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//softsign
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_softsign(ff,fd);
	writer=fopen("tfunc_softsign.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//softsign
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_softsign(ff,fd);
	writer=fopen("tfunc_softsign.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//relu
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_relu(ff,fd);
	writer=fopen("tfunc_relu.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//softplus
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_softplus(ff,fd);
	writer=fopen("tfunc_softplus.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//softplus
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_softplus2(ff,fd);
	writer=fopen("tfunc_softplus2.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//elu
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_elu(ff,fd);
	writer=fopen("tfunc_elu.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
	
	//gelu
	for(int i=0; i<N; ++i) ff[i]=xmin+dx*i;
	NeuralNet::TransferFFDV::f_gelu(ff,fd);
	writer=fopen("tfunc_gelu.dat","w");
	if(writer!=NULL){
		fprintf(writer,"#X F D\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f\n",xmin+dx*i,ff[i],fd[i]);
		}
	}
}

void test_unit_nn_out(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	const int nIn=2;
	const int nOut=3;
	//init rand
	std::srand(std::time(NULL));
	//resize the nn
	NeuralNet::ANN nn;
	NeuralNet::ANNInit init;
	init.sigma()=1.0;
	init.initType()=NeuralNet::InitN::HE;
	nn.tfType()=NeuralNet::TransferN::TANH;
	std::vector<int> nh(2);
	nh[0]=7; nh[1]=5;
	//loop over all samples
	for(int n=0; n<N; ++n){
		nn.resize(init,nIn,nh,nOut);
		//set input/output scaling
		nn.inw()=Eigen::VectorXd::Random(nIn);
		nn.inb()=Eigen::VectorXd::Random(nIn);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int i=0; i<nn.nIn(); ++i) nn.in()[i]=std::rand()/RAND_MAX-0.5;
		//execute the network
		nn.execute();
		//compute error
		Eigen::VectorXd vec0,vec1,vec2,vec3,vec4;
		Eigen::VectorXd grad1,grad2,grad3;
		Eigen::VectorXd gradd1,gradd2,gradd3;
		vec0=nn.inw().cwiseProduct(nn.in()+nn.inb());
		vec1=nn.edge(0)*vec0+nn.bias(0); grad1=vec1; gradd1=vec1;
		#ifndef NN_COMPUTE_D2
		nn.tffdv(0)(vec1,grad1);
		#else
		nn.tffdv(0)(vec1,grad1,gradd1);
		#endif
		vec2=nn.edge(1)*vec1+nn.bias(1); grad2=vec2; gradd2=vec2;
		#ifndef NN_COMPUTE_D2
		nn.tffdv(1)(vec2,grad2);
		#else
		nn.tffdv(1)(vec2,grad2,gradd2);
		#endif
		vec3=nn.edge(2)*vec2+nn.bias(2); grad3=vec3; gradd3=vec3;
		#ifndef NN_COMPUTE_D2
		nn.tffdv(2)(vec3,grad3);
		#else
		nn.tffdv(2)(vec3,grad3,gradd3);
		#endif
		vec4=nn.outb()+nn.outw().cwiseProduct(vec3);
		erra+=(vec4-nn.out()).norm();
		errp+=(vec4-nn.out()).norm()/vec4.norm()*100;
	}
	//compute the error
	erra/=N;
	errp/=N;
	char* str=new char[print::len_buf];
	//print the results
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - ANN - OUT\n";
	std::cout<<"transfer = "<<nn.tfType()<<"\n";
	std::cout<<"config   = "<<nn.nIn()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra     = "<<erra<<"\n";
	std::cout<<"errp     = "<<errp<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_dOutdVal(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	double time=0;
	NeuralNet::DOutDVal dOutDVal;
	NeuralNet::ANN nn;
	NeuralNet::ANNInit init;
	std::vector<int> nh(2);
	const int nIn=4;
	const int nOut=3;
	nh[0]=7; nh[1]=5;
	init.sigma()=1.0;
	init.initType()=NeuralNet::InitN::HE;
	nn.tfType()=NeuralNet::TransferN::TANH;
	//init rand
	std::srand(std::time(NULL));
	//loop over all samples
	for(int m=0; m<N; ++m){
		//resize the nn
		Eigen::MatrixXd dOutExact,dOutApprox;
		nn.resize(init,nIn,nh,nOut);
		dOutDVal.resize(nn);
		//set input/output scaling
		nn.inw()=Eigen::VectorXd::Random(nIn);
		nn.inb()=Eigen::VectorXd::Random(nIn);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int n=0; n<nn.nIn(); ++n) nn.in()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		Eigen::VectorXd in=nn.in();
		//execute the network, compute analytic gradient
		nn.execute();
		clock_t start=std::clock();
		dOutDVal.grad(nn);
		clock_t stop=std::clock();
		time+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		dOutApprox=dOutDVal.dodi();
		//compute brute force gradient
		dOutExact=Eigen::MatrixXd::Zero(nn.nOut(),nn.nIn());
		Eigen::VectorXd delta=Eigen::VectorXd::Random(nn.nIn())/100.0;
		for(int i=0; i<nn.nIn(); ++i){
			//point 1
			for(int n=0; n<nn.nIn(); ++n) nn.in()[n]=in[n];//reset input
			nn.in()[i]+=delta[i];//add small change
			nn.execute();//execute
			Eigen::VectorXd outNew1=nn.out();//store output
			//point 2
			for(int n=0; n<nn.nIn(); ++n) nn.in()[n]=in[n];//reset input
			nn.in()[i]-=delta[i];//add small change
			nn.execute();//execute
			Eigen::VectorXd outNew2=nn.out();//store output
			//gradient
			for(int j=0; j<nn.out().size(); ++j){
				dOutExact(j,i)=0.5*(outNew1[j]-outNew2[j])/delta[i];
			}
		}
		erra+=(dOutExact-dOutApprox).norm();
		errp+=(dOutExact-dOutApprox).norm()/dOutExact.norm()*100;
	}
	//compute the error
	erra/=N;
	errp/=N;
	time/=N;
	//print the results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - ANN - DOutDVal\n";
	std::cout<<"transfer = "<<nn.tfType()<<"\n";
	std::cout<<"config   = "<<nn.nIn()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra     = "<<erra<<"\n";
	std::cout<<"errp     = "<<errp<<"\n";
	std::cout<<"time     = "<<time<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_dOutDP(){
	//local function variables
	const int N=100;
	double erra=0,errp=0;
	clock_t start,stop;
	double time=0;
	NeuralNet::DOutDP dOutDP;
	NeuralNet::ANN nn,nn_copy;
	NeuralNet::ANNInit init;
	std::vector<int> nh(3);
	const int nIn=4;
	const int nOut=3;
	nh[0]=7; nh[1]=5; nh[2]=4;
	init.sigma()=1.0;
	init.initType()=NeuralNet::InitN::HE;
	nn.tfType()=NeuralNet::TransferN::TANH;
	//init rand
	std::srand(std::time(NULL));
	//loop over all samples
	for(int iter=0; iter<N; ++iter){
		//resize the nn
		nn.resize(init,nIn,nh,nOut);
		dOutDP.resize(nn);
		//set input/output scaling
		nn.inw()=Eigen::VectorXd::Random(nIn);
		nn.inb()=Eigen::VectorXd::Random(nIn);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		//initialize the input nodes
		for(int n=0; n<nn.nIn(); ++n) nn.in()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		//execute the network, compute analytic gradient
		nn.execute();
		start=std::clock();
		dOutDP.grad(nn);
		stop=std::clock();
		time+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		std::vector<std::vector<Eigen::VectorXd> > gradBApprox=dOutDP.dodb();
		std::vector<std::vector<Eigen::MatrixXd> > gradWApprox=dOutDP.dodw();
		//compute brute force gradient - bias
		std::vector<std::vector<Eigen::VectorXd> > gradBExact;
		gradBExact.resize(nn.nOut());
		for(int n=0; n<nn.nOut(); ++n){
			gradBExact[n].resize(nn.nlayer());
			for(int l=0; l<nn.nlayer(); ++l){
				gradBExact[n][l]=Eigen::VectorXd::Zero(nn.bias(l).size());
			}
		}
		for(int l=0; l<nn.nlayer(); ++l){
			for(int i=0; i<nn.bias(l).size(); ++i){
				const double delta=nn.bias(l)[i]/100.0;
				//point 1
				nn_copy=nn;
				nn_copy.bias(l)[i]-=delta;
				nn_copy.execute();
				const Eigen::VectorXd pt1=nn_copy.out();
				//point 2
				nn_copy=nn;
				nn_copy.bias(l)[i]+=delta;
				nn_copy.execute();
				const Eigen::VectorXd pt2=nn_copy.out();
				//gradient
				const Eigen::VectorXd grad=0.5*(pt2-pt1)/delta;
				for(int n=0; n<nn.nOut(); ++n){
					gradBExact[n][l](i)=grad(n);
				}
			}
		}
		std::vector<std::vector<Eigen::MatrixXd> > gradWExact;
		gradWExact.resize(nn.nOut());
		for(int n=0; n<nn.nOut(); ++n){
			gradWExact[n].resize(nn.nlayer());
			for(int l=0; l<nn.nlayer(); ++l){
				gradWExact[n][l]=Eigen::MatrixXd::Zero(nn.edge(l).rows(),nn.edge(l).cols());
			}
		}
		for(int l=0; l<nn.nlayer(); ++l){
			for(int j=0; j<nn.edge(l).rows(); ++j){
				for(int k=0; k<nn.edge(l).cols(); ++k){
					const double delta=nn.edge(l)(j,k)/100.0;
					//point 1
					nn_copy=nn;
					nn_copy.edge(l)(j,k)-=delta;
					nn_copy.execute();
					const Eigen::VectorXd pt1=nn_copy.out();
					//point 2
					nn_copy=nn;
					nn_copy.edge(l)(j,k)+=delta;
					nn_copy.execute();
					const Eigen::VectorXd pt2=nn_copy.out();
					//grad
					const Eigen::VectorXd grad=0.5*(pt2-pt1)/delta;
					for(int n=0; n<nn.nOut(); ++n){
						gradWExact[n][l](j,k)=grad[n];
					}
				}
			}
		}
		//compute the error
		for(int n=0; n<nn.nOut(); ++n){
			for(int l=0; l<nn.nlayer(); ++l){
				//std::cout<<"gradBApprox["<<n<<"]["<<l<<"] = \n"<<gradBApprox[n][l]<<"\n";
				//std::cout<<"gradBExact["<<n<<"]["<<l<<"] = \n"<<gradBExact[n][l]<<"\n";
				erra+=(gradBExact[n][l]-gradBApprox[n][l]).norm();
				errp+=(gradBExact[n][l]-gradBApprox[n][l]).norm()/gradBExact[n][l].norm()*100;
			}
		}
		for(int n=0; n<nn.nOut(); ++n){
			for(int l=0; l<nn.nlayer(); ++l){
				//std::cout<<"gradWApprox["<<n<<"]["<<l<<"] = \n"<<gradWApprox[n][l]<<"\n";
				//std::cout<<"gradWExact["<<n<<"]["<<l<<"] = \n"<<gradWExact[n][l]<<"\n";
				erra+=(gradWExact[n][l]-gradWApprox[n][l]).norm();
				errp+=(gradWExact[n][l]-gradWApprox[n][l]).norm()/gradWExact[n][l].norm()*100;
			}
		}
	}
	//compute the error
	erra/=N;
	errp/=N;
	time/=N;
	//print the results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - ANN - DOutDP\n";
	std::cout<<"transfer = "<<nn.tfType()<<"\n";
	std::cout<<"config   = "<<nn.nIn()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra     = "<<erra<<"\n";
	std::cout<<"errp     = "<<errp<<"\n";
	std::cout<<"time     = "<<time<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_time(){
	//local function variables
	const int N=1000000;
	const int M=4;
	std::vector<double> time_exec(M,0);
	std::vector<double> time_grad(M,0);
	clock_t start,stop;
	std::vector<std::vector<int> > nh(M);
	nh[1].resize(1,5);
	nh[2].resize(2,5);
	nh[3].resize(3,5);
	std::srand(std::time(NULL));
	for(int m=0; m<M; ++m){
		NeuralNet::ANN nn;
		NeuralNet::ANNInit init;
		init.sigma()=1.0;
		init.initType()=NeuralNet::InitN::HE;
		nn.tfType()=NeuralNet::TransferN::TANH;
		if(nh[m].size()>=0) nn.resize(init,3,3);
		else nn.resize(init,3,nh[m],3);
		NeuralNet::Cost cost(nn);
		start=std::clock();
		for(int n=0; n<N; ++n){
			//initialize the input nodes
			for(int i=0; i<nn.nIn(); ++i) nn.in()[i]=std::rand()/RAND_MAX-0.5;
			//execute the network
			nn.execute();
		}
		stop=std::clock();
		time_exec[m]=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		start=std::clock();
		for(int n=0; n<N; ++n){
			//initialize the input nodes
			for(int i=0; i<nn.nIn(); ++i) nn.in()[i]=std::rand()/RAND_MAX-0.5;
			Eigen::VectorXd grad=Eigen::VectorXd::Random(nn.size());
			Eigen::VectorXd dcdo=Eigen::VectorXd::Random(3);
			//compute gradient
			cost.grad(nn,dcdo);
		}
		stop=std::clock();
		time_grad[m]=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	}
	std::vector<std::vector<int> > config(M);
	config[0].resize(2,5); config[0].front()=3; config[0].back()=3;
	config[1].resize(3,5); config[1].front()=3; config[1].back()=3;
	config[2].resize(4,5); config[2].front()=3; config[2].back()=3;
	config[3].resize(5,5); config[3].front()=3; config[3].back()=3;
	double time_avg=0;
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - ANN - TIME - EXECUTION\n";
	std::cout<<"transfer = TANH\n";
	std::cout<<"time - "<<time_exec[0]<<" ns - "; for(int i=0; i<config[0].size(); ++i) std::cout<<config[0][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_exec[1]<<" ns - "; for(int i=0; i<config[1].size(); ++i) std::cout<<config[1][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_exec[2]<<" ns - "; for(int i=0; i<config[2].size(); ++i) std::cout<<config[2][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_exec[3]<<" ns - "; for(int i=0; i<config[3].size(); ++i) std::cout<<config[3][i]<<" "; std::cout<<"\n";
	time_avg=0;
	for(int i=0; i<time_exec.size(); ++i) time_avg+=time_exec[i];
	std::cout<<"time - avg = "<<time_avg/time_exec.size()<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	std::cout<<"TEST - ANN - TIME - GRADIENT\n";
	std::cout<<"transfer = TANH\n";
	std::cout<<"time - "<<time_grad[0]<<" ns - "; for(int i=0; i<config[0].size(); ++i) std::cout<<config[0][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_grad[1]<<" ns - "; for(int i=0; i<config[1].size(); ++i) std::cout<<config[1][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_grad[2]<<" ns - "; for(int i=0; i<config[2].size(); ++i) std::cout<<config[2][i]<<" "; std::cout<<"\n";
	std::cout<<"time - "<<time_grad[3]<<" ns - "; for(int i=0; i<config[3].size(); ++i) std::cout<<config[3][i]<<" "; std::cout<<"\n";
	time_avg=0;
	for(int i=0; i<time_grad.size(); ++i) time_avg+=time_grad[i];
	std::cout<<"time - avg = "<<time_avg/time_grad.size()<<"\n";
	std::cout<<print::buf(str,char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// nn-pot
//**********************************************

void test_unit_nnh(){
	NNH nnh,nnh_copy;
	NeuralNet::ANNInit init;
	//initialize neural network
	std::cout<<"initializing neural network\n";
	std::vector<int> nh(4);
	nh[0]=12; nh[1]=8; nh[2]=4; nh[3]=2;
	init.sigma()=1.0;
	init.initType()=NeuralNet::InitN::HE;
	init.seed()=-1;
	nnh.nn().tfType()=NeuralNet::TransferN::ARCTAN;
	nnh.nn().resize(init,nh);
	//initialize basis
	std::cout<<"initializing basis\n";
	BasisR basisR; basisR.init_G2(6,NormN::UNIT,cutoff::Name::COS,10.0);
	BasisA basisA; basisA.init_G4(4,NormN::UNIT,cutoff::Name::COS,10.0);
	std::vector<Atom> species(2);
	species[0].name()="Ar";
	species[0].id()=string::hash("Ar");
	species[0].mass()=5.0;
	species[0].energy()=-7.0;
	species[1].name()="Ne";
	species[1].id()=string::hash("Ne");
	species[1].mass()=12.0;
	species[1].energy()=-9.0;
	nnh.atom()=species[0];
	nnh.resize(species.size());
	nnh.basisR(0)=basisR;
	nnh.basisR(1)=basisR;
	nnh.basisA(0,0)=basisA;
	nnh.basisA(0,1)=basisA;
	nnh.basisA(1,1)=basisA;
	nnh.init_input();
	//print
	std::cout<<nnh<<"\n";
	//pack
	std::cout<<"serialization\n";
	const int memsize=serialize::nbytes(nnh);
	char* memarr=new char[memsize];
	serialize::pack(nnh,memarr);
	serialize::unpack(nnh_copy,memarr);
	delete[] memarr;
	//print
	std::cout<<nnh_copy<<"\n";
}

void test_unit_nnp(){
	NNH nnh;
	NNPot nnp,nnp_copy;
	NeuralNet::ANNInit init;
	//initialize neural network
	std::cout<<"initializing neural network\n";
	std::vector<int> nh(4);
	nh[0]=12; nh[1]=8; nh[2]=4; nh[3]=2;
	init.sigma()=1.0;
	init.initType()=NeuralNet::InitN::HE;
	init.seed()=-1;
	nnh.nn().tfType()=NeuralNet::TransferN::ARCTAN;
	nnh.nn().resize(init,nh);
	//initialize basis
	std::cout<<"initializing basis\n";
	BasisR basisR; basisR.init_G2(6,NormN::UNIT,cutoff::Name::COS,10.0);
	BasisA basisA; basisA.init_G4(4,NormN::UNIT,cutoff::Name::COS,10.0);
	std::vector<Atom> species(2);
	species[0].name()="Ar";
	species[0].id()=string::hash("Ar");
	species[0].mass()=5.0;
	species[0].energy()=-7.0;
	species[1].name()="Ne";
	species[1].id()=string::hash("Ne");
	species[1].mass()=12.0;
	species[1].energy()=-9.0;
	nnh.atom()=species[0];
	nnh.resize(species.size());
	nnh.basisR(0)=basisR;
	nnh.basisR(1)=basisR;
	nnh.basisA(0,0)=basisA;
	nnh.basisA(0,1)=basisA;
	nnh.basisA(1,1)=basisA;
	nnh.init_input();
	//resize potential
	std::cout<<"resizing potential\n";
	nnp.resize(species);
	nnp.nnh(0)=nnh;
	nnp.nnh(1)=nnh;
	//print
	std::cout<<nnp<<"\n";
	//pack
	std::cout<<"serialization\n";
	const int memsize=serialize::nbytes(nnp);
	char* memarr=new char[memsize];
	serialize::pack(nnp,memarr);
	serialize::unpack(nnp_copy,memarr);
	delete[] memarr;
	//print
	std::cout<<nnp_copy<<"\n";
}

void test_unit_nnp_csymm(){
	//local function variables
	std::vector<Atom> atoms;
	NNPot nnpot;
	Structure struc_small;
	Structure struc_large;
	AtomType atomT;
	atomT.name=true; atomT.an=false; atomT.type=true; atomT.index=false;
	atomT.posn=true; atomT.force=false; atomT.symm=true; atomT.charge=false;
	atomT.neigh=true;
	//set the atoms
	std::cout<<"setting atoms\n";
	atoms.resize(1);
	atoms[0].name()="Ar";
	atoms[0].mass()=22.90;
	atoms[0].energy()=0.0;
	atoms[0].charge()=0.0;
	//resize the nnpot
	std::cout<<"setting nnpot\n";
	nnpot.rc()=9.0;
	nnpot.resize(atoms);
	NNPot::read_basis("basis_Ar",nnpot,"Ar");
	//print the nnpot
	std::cout<<nnpot<<"\n";
	//read the structures
	std::cout<<"reading structures\n";
	VASP::POSCAR::read("Ar.vasp",atomT,struc_small);
	VASP::POSCAR::read("Ar_l4.vasp",atomT,struc_large);
	//set the type
	for(int i=0; i<struc_small.nAtoms(); ++i) struc_small.type(i)=nnpot.index(struc_small.name(i));
	for(int i=0; i<struc_large.nAtoms(); ++i) struc_large.type(i)=nnpot.index(struc_large.name(i));
	//compute neighbor lists
	Structure::neigh_list(struc_small,nnpot.rc());
	Structure::neigh_list(struc_large,nnpot.rc());
	//init the symmetry functions
	std::cout<<"initializing symmetry functions\n";
	nnpot.init_symm(struc_small);
	nnpot.init_symm(struc_large);
	//compute the symmetry functions
	std::cout<<"computing symmetry functions\n";
	nnpot.calc_symm(struc_small);
	nnpot.calc_symm(struc_large);
	//find average symmetry function
	Eigen::VectorXd symm_small=Eigen::VectorXd::Zero(struc_small.symm(0).size());
	Eigen::VectorXd symm_large=Eigen::VectorXd::Zero(struc_large.symm(0).size());
	std::cout<<"symm_small = "<<struc_small.symm(0).transpose()<<"\n";
	std::cout<<"symm_large = "<<struc_large.symm(0).transpose()<<"\n";
	/*for(int i=0; i<struc_small.nAtoms(); ++i){
		std::cout<<"symm - small - "<<i<<" = "<<struc_small.symm(i).transpose()<<"\n";
	}
	for(int i=0; i<struc_large.nAtoms(); ++i){
		std::cout<<"symm - large - "<<i<<" = "<<struc_large.symm(i).transpose()<<"\n";
	}*/
}

//**********************************************
// structure
//**********************************************

void test_unit_struc(){
	char* buf=new char[print::len_buf];
	std::cout<<print::buf(buf,char_buf)<<"\n";
	//generate Ar crystal
	units::System::type unitsys=units::System::METAL;
	units::consts::init(unitsys);
	const double a0=10.512;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=true;
	Structure struc;
	const int natoms=32;
	struc.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	struc.init(lv);
	struc.energy()=-8.527689257;
	struc.posn(0)<<0,0,0;
	struc.posn(1)<<0,0,5.25600004195;
	struc.posn(2)<<0,5.25600004195,0;
	struc.posn(3)<<0,5.25600004195,5.25600004195;
	struc.posn(4)<<5.25600004195,0,0;
	struc.posn(5)<<5.25600004195,0,5.25600004195;
	struc.posn(6)<<5.25600004195,5.25600004195,0;
	struc.posn(7)<<5.25600004195,5.25600004195,5.25600004195;
	struc.posn(8)<<0,2.628000020975,2.628000020975;
	struc.posn(9)<<0,2.628000020975,7.883999821149;
	struc.posn(0)<<0,7.883999821149,2.628000020975;
	struc.posn(11)<<0,7.883999821149,7.883999821149;
	struc.posn(12)<<5.25600004195,2.628000020975,2.628000020975;
	struc.posn(13)<<5.25600004195,2.628000020975,7.883999821149;
	struc.posn(14)<<5.25600004195,7.883999821149,2.628000020975;
	struc.posn(15)<<5.25600004195,7.883999821149,7.883999821149;
	struc.posn(16)<<2.628000020975,0,2.628000020975;
	struc.posn(17)<<2.628000020975,0,7.883999821149;
	struc.posn(18)<<2.628000020975,5.25600004195,2.628000020975;
	struc.posn(19)<<2.628000020975,5.25600004195,7.883999821149;
	struc.posn(20)<<7.883999821149,0,2.628000020975;
	struc.posn(21)<<7.883999821149,0,7.883999821149;
	struc.posn(22)<<7.883999821149,5.25600004195,2.628000020975;
	struc.posn(23)<<7.883999821149,5.25600004195,7.883999821149;
	struc.posn(24)<<2.628000020975,2.628000020975,0;
	struc.posn(25)<<2.628000020975,2.628000020975,5.25600004195;
	struc.posn(26)<<2.628000020975,7.883999821149,0;
	struc.posn(27)<<2.628000020975,7.883999821149,5.25600004195;
	struc.posn(28)<<7.883999821149,2.628000020975,0;
	struc.posn(29)<<7.883999821149,2.628000020975,5.25600004195;
	struc.posn(30)<<7.883999821149,7.883999821149,0;
	struc.posn(31)<<7.883999821149,7.883999821149,5.25600004195;
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.symm(i)=Eigen::VectorXd::Random(12);
	}
	std::cout<<"struc = "<<struc<<"\n";
	/*for(int i=0; i<struc.nAtoms(); ++i){
		std::cout<<struc.name(i)<<struc.index(i)+1<<" "<<struc.posn(i).transpose()<<"\n";
	}
	for(int i=0; i<struc.nAtoms(); ++i){
		std::cout<<struc.name(i)<<struc.index(i)+1<<" "<<struc.symm(i).transpose()<<"\n";
	}*/
	//pack
	int size=serialize::nbytes(struc);
	std::cout<<"size = "<<size<<"\n";
	char* memarr=new char[size];
	std::cout<<"packing structure\n";
	serialize::pack(struc,memarr);
	//unpack
	Structure struc_new;
	std::cout<<"unpacking structure\n";
	serialize::unpack(struc_new,memarr);
	std::cout<<"struc_new = "<<struc_new<<"\n";
	/*for(int i=0; i<struc_new.nAtoms(); ++i){
		std::cout<<struc_new.name(i)<<struc_new.index(i)+1<<" "<<struc_new.posn(i).transpose()<<"\n";
	}
	for(int i=0; i<struc_new.nAtoms(); ++i){
		std::cout<<struc_new.name(i)<<struc_new.index(i)+1<<" "<<struc_new.symm(i).transpose()<<"\n";
	}*/
	//compute error
	double err_posn=0;
	for(int i=0; i<32; ++i){
		err_posn+=(struc.posn(i)-struc_new.posn(i)).norm();
	}
	double err_symm=0;
	for(int i=0; i<32; ++i){
		err_symm+=(struc.symm(i)-struc_new.symm(i)).norm();
	}
	std::cout<<"err_lv   = "<<(struc.R()-struc_new.R()).norm()<<"\n";
	std::cout<<"err_posn = "<<err_posn<<"\n";
	std::cout<<"err_symm = "<<err_symm<<"\n";
	std::cout<<print::buf(buf,char_buf)<<"\n";
	delete[] buf;
}

void test_cell_list_square(){
	char* buf=new char[print::len_buf];
	std::cout<<print::buf(buf,char_buf)<<"\n";
	clock_t start,stop;
	//generate Ar crystal
	std::cout<<"generating crystal\n";
	units::System::type unitsys=units::System::METAL;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=false;
	const double a0=60.0;
	const double r0=3.0;
	const int nside=a0/r0;
	const int N=nside*nside*nside;
	const int nspecies=1;
	std::vector<int> natoms(nspecies,N);
	std::vector<std::string> names(nspecies);
	Structure struc;
	struc.resize(N,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	struc.init(lv);
	//generate positions
	std::cout<<"generating positions\n";
	start=std::clock();
	std::srand(std::time(NULL));
	Eigen::Vector3d rand;
	int count=0;
	for(int i=0; i<nside; ++i){
		for(int j=0; j<nside; ++j){
			for(int k=0; k<nside; ++k){
				struc.posn(count)<<i*r0,j*r0,k*r0;
				rand=Eigen::Vector3d::Random();
				struc.posn(count).noalias()+=rand*0.1;
				Cell::returnToCell(struc.posn(count),struc.posn(count),struc.R(),struc.RInv());
				++count;
			}
		}
	}
	std::cout<<struc<<"\n";
	std::cout<<"N_ATOMS = "<<N<<"\n";
	//set interaction potential
	Eigen::Vector3d r;
	const double rc=10.0;
	LJ lj(0.010831987910334,3.345);
	//compute energy - pair
	std::cout<<"computing energy - pair\n";
	double energy_pair=0;
	for(int i=0; i<N; ++i){
		for(int j=i+1; j<N; ++j){
			const double dr=struc.dist(struc.posn(i),struc.posn(j),r);
			if(dr<rc) energy_pair-=lj(dr);
		}
	}
	energy_pair/=N;
	stop=std::clock();
	const double time_pair=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	//compute energy - list
	std::cout<<"computing energy - list\n";
	start=std::clock();
	double energy_list=0;
	CellList cellList(rc,struc);
	std::vector<Eigen::Vector3i> nnc;
	for(int i=-1; i<=1; ++i){
		for(int j=-1; j<=1; ++j){
			for(int k=-1; k<=1; ++k){
				Eigen::Vector3i vec;
				vec<<i,j,k;
				nnc.push_back(vec);
			}
		}
	}
	for(int n=0; n<N; ++n){
		const Eigen::Vector3i cell=cellList.cell(n);
		for(int i=0; i<nnc.size(); ++i){
			const Eigen::Vector3i neigh=cell+nnc[i];
			for(int m=0; m<cellList.atoms(neigh).size(); ++m){
				const double dr=struc.dist(struc.posn(n),struc.posn(cellList.atoms(neigh)[m]),r);
				if(1e-8<dr && dr<rc) energy_list-=lj(dr);
			}
		}
	}
	energy_list*=0.5;
	energy_list/=N;
	stop=std::clock();
	const double time_list=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
	std::cout<<"dim  = "<<cellList.dim(0)<<" "<<cellList.dim(1)<<" "<<cellList.dim(2)<<"\n";
	std::cout<<"flen = "<<cellList.flen(0)<<" "<<cellList.flen(1)<<" "<<cellList.flen(2)<<"\n";
	//print
	std::cout<<"energy_pair = "<<energy_pair<<"\n";
	std::cout<<"energy_list = "<<energy_list<<"\n";
	std::cout<<"error       = "<<std::fabs((energy_pair-energy_list)/energy_pair*100.0)<<"\n";
	std::cout<<"time_pair (ns/atom) = "<<time_pair<<"\n";
	std::cout<<"time_list (ns/atom) = "<<time_list<<"\n";
	std::cout<<print::buf(buf,char_buf)<<"\n";
	delete[] buf;
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("SIZE - BYTES (BITS)",str)<<"\n";
	std::cout<<"char     = "<<sizeof(char)<<" "<<8*sizeof(char)<<"\n";
	std::cout<<"uchar    = "<<sizeof(unsigned char)<<" "<<8*sizeof(unsigned char)<<"\n";
	std::cout<<"short    = "<<sizeof(short)<<" "<<8*sizeof(short)<<"\n";
	std::cout<<"ushort   = "<<sizeof(unsigned short)<<" "<<8*sizeof(unsigned short)<<"\n";
	std::cout<<"int      = "<<sizeof(int)<<" "<<8*sizeof(int)<<"\n";
	std::cout<<"uint     = "<<sizeof(unsigned int)<<" "<<8*sizeof(unsigned int)<<"\n";
	std::cout<<"l int    = "<<sizeof(long int)<<" "<<8*sizeof(long int)<<"\n";
	std::cout<<"l uint   = "<<sizeof(unsigned long int)<<" "<<8*sizeof(unsigned long int)<<"\n";
	std::cout<<"ll int   = "<<sizeof(long long int)<<" "<<8*sizeof(long long int)<<"\n";
	std::cout<<"ll uint  = "<<sizeof(unsigned long long int)<<" "<<8*sizeof(unsigned long long int)<<"\n";
	std::cout<<"float    = "<<sizeof(float)<<" "<<8*sizeof(float)<<"\n";
	std::cout<<"double   = "<<sizeof(double)<<" "<<8*sizeof(double)<<"\n";
	std::cout<<"l double = "<<sizeof(long double)<<" "<<8*sizeof(long double)<<"\n";
	std::cout<<"size_t   = "<<sizeof(std::size_t)<<" "<<8*sizeof(std::size_t)<<"\n";
	std::cout<<print::title("SIZE - BYTES (BITS)",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("COMPILER",str)<<"\n";
	std::cout<<"date     = "<<compiler::date()<<"\n";
	std::cout<<"time     = "<<compiler::time()<<"\n";
	std::cout<<"compiler = "<<compiler::name()<<"\n";
	std::cout<<"version  = "<<compiler::version()<<"\n";
	std::cout<<"standard = "<<compiler::standard()<<"\n";
	std::cout<<"arch     = "<<compiler::arch()<<"\n";
	std::cout<<"instr    = "<<compiler::instr()<<"\n";
	std::cout<<"os       = "<<compiler::os()<<"\n";
	std::cout<<"omp      = "<<compiler::omp()<<"\n";
	std::cout<<print::title("COMPILER",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("MATH - CONST",str)<<"\n";
	std::cout<<std::setprecision(14);
	std::cout<<"ZERO  = "<<math::constant::ZERO<<"\n";
	std::cout<<"PI    = "<<math::constant::PI<<"\n";
	std::cout<<"RadPI = "<<math::constant::RadPI<<"\n";
	std::cout<<"Rad2  = "<<math::constant::Rad2<<"\n";
	std::cout<<"Rad3  = "<<math::constant::Rad3<<"\n";
	std::cout<<"E     = "<<math::constant::E<<"\n";
	std::cout<<"PHI   = "<<math::constant::PHI<<"\n";
	std::cout<<"LOG2  = "<<math::constant::LOG2<<"\n";
	std::cout<<std::setprecision(6);
	std::cout<<print::title("MATH - CONST",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// string
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("STRING",str)<<"\n";
	test_string();
	std::cout<<print::title("STRING",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// math_special
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("MATH - SPECIAL",str)<<"\n";
	test_math_special_cos();
	test_math_special_sin();
	test_math_special_logp1();
	std::cout<<print::title("MATH - SPECIAL",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// list
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("LIST",str)<<"\n";
	test_list_shuffle();
	std::cout<<print::title("LIST",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// accumulator - 1D
	//**********************************************

	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("ACCUMULATOR - 1D",str)<<"\n";
	test_acc_1D();
	std::cout<<print::title("ACCUMULATOR - 1D",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// accumulator - 2D
	//**********************************************

	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("ACCUMULATOR - 2D",str)<<"\n";
	test_acc_2D();
	std::cout<<print::title("ACCUMULATOR - 2D",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// cutoff
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("CUTOFF",str)<<"\n";
	test_cutoff_cos();
	std::cout<<print::title("CUTOFF",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// symm
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("SYMM",str)<<"\n";
	test_symm_t1();
	test_symm_g1();
	test_symm_g2();
	test_symm_g3();
	test_symm_g4();
	std::cout<<print::title("SYMM",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// eigen
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("EIGEN",str)<<"\n";
	test_unit_eigen_vec3d();
	test_unit_eigen_vecxd();
	test_unit_eigen_mat3d();
	test_unit_eigen_matxd();
	std::cout<<print::title("EIGEN",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// optimize
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("OPTIMIZE",str)<<"\n";
	std::cout<<"ROSENBERG\n";
	std::cout<<"OPT - V =  0.0\n";
	std::cout<<"OPT - X = (1,1)\n";
	//test_unit_opt_sgd();
	test_unit_opt_sdm();
	test_unit_opt_nag();
	test_unit_opt_adam();
	test_unit_opt_nadam();
	test_unit_opt_amsgrad();
	test_unit_opt_cg();
	std::cout<<print::title("OPTIMIZE",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// ewald
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("EWALD",str)<<"\n";
	test_unit_ewald_madelung();
	test_unit_ewald_potential();
	std::cout<<print::title("EWALD",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// string
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("STRING",str)<<"\n";
	test_unit_string_hash();
	std::cout<<print::title("STRING",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// random
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("RANDOM",str)<<"\n";
	test_random_seed();
	test_random_time();
	test_random_dist();
	std::cout<<print::title("RANDOM",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// nn
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("ANN",str)<<"\n";
	test_unit_nn();
	test_unit_nn_tfunc();
	test_unit_nn_out();
	test_unit_nn_dOutdVal();
	test_unit_nn_dOutDP();
	//test_unit_nn_grad2();
	test_unit_nn_time();
	std::cout<<print::title("ANN",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// nn
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("NNH",str)<<"\n";
	std::cout<<"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n";
	test_unit_nnh();
	std::cout<<"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n";
	test_unit_nnp();
	std::cout<<"&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n";
	std::cout<<"**************************************************************************\n";
	test_unit_nnp_csymm();
	std::cout<<"**************************************************************************\n";
	std::cout<<print::title("NNH",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//**********************************************
	// structure
	//**********************************************
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("STRUCTURE",str)<<"\n";
	test_unit_struc();
	test_cell_list_square();
	std::cout<<print::title("STRUCTURE",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
}