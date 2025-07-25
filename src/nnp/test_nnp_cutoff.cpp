// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
#include <chrono>
// str
#include "str/print.hpp"
#include "str/string.hpp"
// cutoff
#include "nnp/cutoff.hpp"

using namespace std::chrono;

//**********************************************
// cutoff
//**********************************************

void test_cutoff(Cutoff::Name name){
	//constants
	const double rc=6.0;
	const int N=500;
	const double dr=rc/(N-1.0);
	Cutoff cutoff(name,rc);
	
	//compute error
	double integral=0.5*(cutoff.cutf(0.0)+cutoff.cutf(rc));
	for(int i=N-2; i>=1; --i){
		integral+=cutoff.cutf(i/(N-1.0)*rc);
	}
	integral*=dr;
	double errg=0;
	for(int i=N-2; i>=1; --i){
		const double g=0.5*(cutoff.cutf((i+1.0)/(N-1.0)*rc)-cutoff.cutf((i-1.0)/(N-1.0)*rc))/dr;
		errg+=std::fabs(g-cutoff.cutg(i/(N-1.0)*rc));
	}
	errg/=(N-1.0);
	
	//compute time
	high_resolution_clock::time_point tbeg,tend;
	const int Nt=1e7;
	std::srand(std::time(NULL));
	tbeg=high_resolution_clock::now();
	for(int i=Nt-1; i>=0; i--) volatile double val=cutoff.cutf(std::rand()/RAND_MAX*rc);
	tend=high_resolution_clock::now();
	const duration<double> timef=duration_cast<duration<double>>(tend-tbeg);
	tbeg=high_resolution_clock::now();
	for(int i=Nt-1; i>=0; i--) volatile double val=cutoff.cutg(std::rand()/RAND_MAX*rc);
	tend=high_resolution_clock::now();
	const duration<double> timed=duration_cast<duration<double>>(tend-tbeg);
	
	//integral
	double iexact=0;
	switch(name){
		case Cutoff::Name::STEP: iexact=rc; break;
		case Cutoff::Name::COS: iexact=0.5*rc; break;
		case Cutoff::Name::TANH: iexact=0.0*rc; break;
		case Cutoff::Name::POLY3: iexact=0.5*rc; break;
	}
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - CUTOFF\n";
	std::cout<<"cutoff          = "<<name<<"\n";
	std::cout<<"r - interval    = ["<<0.0<<","<<rc<<"]\n";
	std::cout<<"n points        = "<<N<<"\n";
	std::cout<<"dr              = "<<dr<<"\n";
	std::cout<<"integral exact  = "<<iexact<<"\n";
	std::cout<<"integral approx = "<<integral<<"\n";
	std::cout<<"err  - integral = "<<std::fabs(iexact-integral)<<"\n";
	std::cout<<"err  - gradient = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef.count()<<" ns\n";
	std::cout<<"time - gradient = "<<timed.count()<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - BASIS - CUTOFF",str)<<"\n";
	test_cutoff(Cutoff::Name::STEP);
	test_cutoff(Cutoff::Name::COS);
	test_cutoff(Cutoff::Name::TANH);
	test_cutoff(Cutoff::Name::POLY3);
	std::cout<<print::title("TEST - BASIS - CUTOFF",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}
