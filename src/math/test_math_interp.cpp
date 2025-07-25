// c libaries
#include <ctime>
#include <cmath>
// c++ libaries
#include <iostream>
#include <chrono>
// math
#include "math/interp.hpp"
// string
#include "str/print.hpp"

using namespace std::chrono;

void test_error_gaussian(math::interp::Name name){
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - ERROR - GAUSSIAN",str)<<"\n";
	//init rand
	std::srand(std::time(NULL));
	//set function data
	std::cout<<"setting function data\n";
	const int Ndata=100;
	const double xmin=-5.0;
	const double xmax=5.0;
	const double dx=(xmax-xmin)/(Ndata+0);
	const double s=1.0;
	const double c=0.0;
	const double pi=3.141592653589793;
	//make the interpolation object
	std::cout<<"making interpolation object\n";
	math::interp::Base* obj;
	switch(name){
		case math::interp::Name::CONST:
			obj=new math::interp::Const();
		break;
		case math::interp::Name::LINEAR:
			obj=new math::interp::Linear();
		break;
		case math::interp::Name::AKIMA:
			obj=new math::interp::Akima();
		break;
		case math::interp::Name::RBFI:
			obj=new math::interp::RBFI();
			static_cast<math::interp::RBFI&>(*obj).rbf()=RBF(RBF::Name::IMQUADRIC,dx);
		break;
		default:
			throw std::invalid_argument("Invalid interpolation name.");
		break;
	}
	//resize interpolation object, set data
	obj->resize(Ndata);
	for(int i=0; i<Ndata; ++i){
		const double xx=xmin+dx*i;
		obj->x(i)=xx;
		obj->y(i)=1.0/sqrt(2.0*pi)*exp(-0.5*(xx-c)*(xx-c)/(s*s));
		obj->d(i)=1.0/sqrt(2.0*pi)*exp(-0.5*(xx-c)*(xx-c)/(s*s))*-(xx-c)/(s*s);
	}
	obj->init();
	//compute error
	const int Ntest=1000;
	double erra=0;
	double errp=0;
	for(int i=0; i<Ntest; ++i){
		const double xx=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
		const double yye=1.0/sqrt(2.0*pi)*exp(-0.5*(xx-c)*(xx-c)/(s*s));
		const double yya=obj->eval(xx);
		erra+=fabs(yye-yya);
		errp+=fabs((yye-yya)/yye*100.0);
	}
	erra/=Ndata;
	errp/=Ndata;
	//print error
	std::cout<<"center    = "<<c<<"\n";
	std::cout<<"sigma     = "<<s<<"\n";
	std::cout<<"name      = "<<name<<"\n";
	std::cout<<"x         = ["<<xmin<<":"<<xmax<<":"<<dx<<"]\n";
	std::cout<<"err - abs = "<<erra<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_write_gaussian(math::interp::Name name){
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - WRITE - GAUSSIAN",str)<<"\n";
	//init rand
	std::srand(std::time(NULL));
	//set function data
	std::cout<<"setting function data\n";
	const int Ndata=100;
	const double xmin=-5.0;
	const double xmax=5.0;
	const double dx=(xmax-xmin)/(Ndata+0);
	const double s=1.0;
	const double c=0.0;
	const double pi=3.141592653589793;
	//make the interpolation object
	std::cout<<"making interpolation object\n";
	math::interp::Base* obj;
	switch(name){
		case math::interp::Name::CONST:
			obj=new math::interp::Const();
		break;
		case math::interp::Name::LINEAR:
			obj=new math::interp::Linear();
		break;
		case math::interp::Name::AKIMA:
			obj=new math::interp::Akima();
		break;
		case math::interp::Name::RBFI:
			obj=new math::interp::RBFI();
			static_cast<math::interp::RBFI&>(*obj).rbf()=RBF(RBF::Name::IMQUADRIC,dx);
		break;
		default:
			throw std::invalid_argument("Invalid interpolation name.");
		break;
	}
	//resize interpolation object, set data
	obj->resize(Ndata);
	for(int i=0; i<Ndata; ++i){
		const double xx=xmin+dx*i;
		obj->x(i)=xx;
		obj->y(i)=1.0/sqrt(2.0*pi)*exp(-0.5*(xx-c)*(xx-c)/(s*s));
		obj->d(i)=1.0/sqrt(2.0*pi)*exp(-0.5*(xx-c)*(xx-c)/(s*s))*-(xx-c)/(s*s);
	}
	obj->init();
	//write
	const std::string file="test_math_interp_"+std::string(math::interp::Name::name(name))+".dat";
	FILE* writer=fopen(file.c_str(),"w");
	if(writer!=NULL){
		const int Ntest=5000;
		fprintf(writer,"#X YE YA\n");
		for(int i=0; i<Ntest; ++i){
			const double xx=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
			const double yye=1.0/sqrt(2.0*pi)*exp(-0.5*(xx-c)*(xx-c)/(s*s));
			const double yya=obj->eval(xx);
			fprintf(writer,"%f %f %f\n",xx,yye,yya);
		}
		fclose(writer);
		writer=NULL;
	} else std::cout<<"Could not open file: \""<<file<<"\"\n";
	//print
	std::cout<<"center  = "<<c<<"\n";
	std::cout<<"sigma   = "<<s<<"\n";
	std::cout<<"name    = "<<name<<"\n";
	std::cout<<"file    = "<<file<<"\n";
	std::cout<<"x       = ["<<xmin<<":"<<xmax<<":"<<dx<<"]\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_time_gaussian(math::interp::Name name){
	typedef high_resolution_clock Clock;
	high_resolution_clock::time_point tbeg,tend;
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - TIME - GAUSSIAN",str)<<"\n";
	//init rand
	std::srand(std::time(NULL));
	//set function data
	std::cout<<"setting function data\n";
	const int Ndata=100;
	const double xmin=-5.0;
	const double xmax=5.0;
	const double dx=(xmax-xmin)/(Ndata+0);
	const double s=1.0;
	const double c=0.0;
	const double pi=3.141592653589793;
	//make the interpolation object
	std::cout<<"making interpolation object\n";
	math::interp::Base* obj;
	switch(name){
		case math::interp::Name::CONST:
			obj=new math::interp::Const();
		break;
		case math::interp::Name::LINEAR:
			obj=new math::interp::Linear();
		break;
		case math::interp::Name::AKIMA:
			obj=new math::interp::Akima();
		break;
		case math::interp::Name::RBFI:
			obj=new math::interp::RBFI();
			static_cast<math::interp::RBFI&>(*obj).rbf()=RBF(RBF::Name::IMQUADRIC,dx);
		break;
		default:
			throw std::invalid_argument("Invalid interpolation name.");
		break;
	}
	//resize interpolation object, set data
	obj->resize(Ndata);
	for(int i=0; i<Ndata; ++i){
		const double xx=xmin+dx*i;
		obj->x(i)=xx;
		obj->y(i)=1.0/sqrt(2.0*pi)*exp(-0.5*(xx-c)*(xx-c)/(s*s));
		obj->d(i)=1.0/sqrt(2.0*pi)*exp(-0.5*(xx-c)*(xx-c)/(s*s))*-(xx-c)/(s*s);
	}
	obj->init();
	const int Ntest=1000000;
	//set x
	std::vector<double> xx(Ntest);
	for(int i=0; i<Ntest; ++i){
		xx[i]=(1.0*std::rand())/RAND_MAX*(xmax-xmin)+xmin;
	}
	//set time
	tbeg=Clock::now();
	for(int i=0; i<Ntest; ++i){
		volatile double yy=1.0/sqrt(2.0*pi)*exp(-0.5*(xx[i]-c)*(xx[i]-c)/(s*s));
	}
	tend=Clock::now();
	duration<double> time_exact = duration_cast<duration<double>>(tend-tbeg);
	//compute error
	tbeg=Clock::now();
	for(int i=0; i<Ntest; ++i){
		volatile double yy=obj->eval(xx[i]);
	}
	tend=Clock::now();
	duration<double> time_approx = duration_cast<duration<double>>(tend-tbeg);
	//print error
	std::cout<<"center   = "<<c<<"\n";
	std::cout<<"sigma    = "<<s<<"\n";
	std::cout<<"name     = "<<name<<"\n";
	std::cout<<"x        = ["<<xmin<<":"<<xmax<<":"<<dx<<"]\n";
	std::cout<<"time - e = "<<time_exact.count()<<"\n";
	std::cout<<"time - a = "<<time_approx.count()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

int main(int argc, char* argv[]){
	
	test_error_gaussian(math::interp::Name::CONST);
	test_error_gaussian(math::interp::Name::LINEAR);
	test_error_gaussian(math::interp::Name::AKIMA);
	
	test_write_gaussian(math::interp::Name::CONST);
	test_write_gaussian(math::interp::Name::LINEAR);
	test_write_gaussian(math::interp::Name::AKIMA);
	
	test_time_gaussian(math::interp::Name::CONST);
	test_time_gaussian(math::interp::Name::LINEAR);
	test_time_gaussian(math::interp::Name::AKIMA);
	
}