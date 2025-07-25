// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
#include <chrono>
// str
#include "str/print.hpp"
#include "str/string.hpp"
// basis - radial
#include "basis_radial.hpp"

using namespace std::chrono;

//**********************************************
// symm
//**********************************************

void test_symm_grad(BasisR::Name phirn){
	high_resolution_clock::time_point tbeg,tend;
	//constants
	const double rc=6.0;
	const int N=500;
	const double dr=rc/(N-1.0);
	BasisR basisR(rc,Cutoff::Name::COS,1,phirn);
	const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc;
	const double rs=(1.0*std::rand())/RAND_MAX*0.5*rc;
	double r=0;
	//compute error
	double errg=0;
	for(int i=1; i<N-1; ++i){
		//finite difference
		const double r2=(i+1.0)/(N-1.0)*rc;
		const double f2=basisR.symmf(r2,eta,rs);
		const double r1=(i-1.0)/(N-1.0)*rc;
		const double f1=basisR.symmf(r1,eta,rs);
		const double g=(f2-f1)/(r2-r1);
		//exact
		r=i/(N-1.0)*rc;
		errg+=std::fabs(g-basisR.symmd(r,eta,rs));
	}
	errg/=(N-1.0);
	//compute time
	const int Nt=1e7;
	tbeg=high_resolution_clock::now();
	for(int i=0; i<Nt; ++i){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=basisR.symmf(r,eta,rs);
	}
	tend=high_resolution_clock::now();
	duration<double> timef = duration_cast<duration<double>>(tend-tbeg);
	tbeg=high_resolution_clock::now();
	for(int i=0; i<Nt; ++i){
		r=std::rand()/RAND_MAX*rc;
		volatile double val=basisR.symmd(r,eta,rs);
	}
	tend=high_resolution_clock::now();
	duration<double> timed = duration_cast<duration<double>>(tend-tbeg);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - SYMM\n";
	std::cout<<"NAME = "<<phirn<<"\n";
	std::cout<<"r - interval    = ["<<0.0<<","<<rc<<"]\n";
	std::cout<<"n points        = "<<N<<"\n";
	std::cout<<"dr              = "<<dr<<"\n";
	std::cout<<"eta             = "<<eta<<"\n";
	std::cout<<"rs              = "<<rs<<"\n";
	std::cout<<"err  - grad     = "<<errg<<"\n";
	std::cout<<"time - function = "<<timef.count()<<" ns\n";
	std::cout<<"time - gradient = "<<timed.count()<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void write_symm(BasisR::Name phirn, const char* fname){
	//constants
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	const double nf=6;
	BasisR basisR(rc,Cutoff::Name::COS,nf,phirn);
	std::vector<double> rs(nf,1.0);
	std::vector<double> eta(nf,0.0);
	eta[0]=0.10;
	eta[1]=0.25;
	eta[2]=2.50;
	eta[3]=4.00;
	eta[4]=8.00;
	eta[5]=15.00;
	//open file
	FILE* writer=fopen(fname,"w");
	if(writer!=NULL){
		for(int i=0; i<N; ++i){
			const double r=i/(N-1.0)*rc;
			fprintf(writer,"%f ",r);
			for(int j=0; j<nf; ++j){
				const double f=basisR.symmf(r,eta[j],rs[j]);
				const double d=basisR.symmd(r,eta[j],rs[j]);
				fprintf(writer,"%f %f ",f,d);
			}
			fprintf(writer,"\n",r);
		}
		fclose(writer);
		writer=NULL;
	} else std::cout<<"ERROR: Could not open output file.\n";
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - BASIS - RADIAL - GRADIENT",str)<<"\n";
	test_symm_grad(BasisR::Name::GAUSSIAN);
	test_symm_grad(BasisR::Name::SECH);
	test_symm_grad(BasisR::Name::LOGISTIC);
	test_symm_grad(BasisR::Name::TANH);
	test_symm_grad(BasisR::Name::LOGCOSH);
	test_symm_grad(BasisR::Name::LOGCOSH2);
	std::cout<<print::title("TEST - BASIS - RADIAL - GRADIENT",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - BASIS - RADIAL - WRITE",str)<<"\n";
	write_symm(BasisR::Name::GAUSSIAN,"symm_gauss.dat");
	write_symm(BasisR::Name::SECH,"symm_sech.dat");
	write_symm(BasisR::Name::LOGISTIC,"symm_sech.dat");
	write_symm(BasisR::Name::TANH,"symm_tanh.dat");
	write_symm(BasisR::Name::LOGCOSH,"symm_logcosh.dat");
	write_symm(BasisR::Name::LOGCOSH2,"symm_logcosh2.dat");
	std::cout<<print::title("TEST - BASIS - RADIAL - WRITE",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}
