// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
#include <chrono>
// str
#include "str/print.hpp"
#include "str/string.hpp"
// basis - angular
#include "basis_angular.hpp"
// math
#include "math/reduce.hpp"

using namespace std::chrono;

//**********************************************
// symm
//**********************************************

void test_symm_grad_phi(BasisA::Name phian){
	//rand
	std::srand(std::time(NULL));
	//error
	double erra=0;
	double errp=0;
	Reduce<2> reduce;
	const int N=1000;
	const int M=100;
	for(int iter=0; iter<M; ++iter){
		//constants
		const double rc=6.0;
		const double da=math::constant::PI/(N-1.0);
		const double dc=std::cos(da);
		BasisA basisA(rc,Cutoff::Name::COS,1,phian);
		const double zeta=(1.0*std::rand())/RAND_MAX*(8.0-1.0)+1.0;
		const int lambda=((1.0*std::rand())/RAND_MAX<0.5)?1:-1;
		const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double rs=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double dr[2]={
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0
		};
		double cos=0;
		//compute error
		//std::cout<<"eta = "<<eta<<"\n";
		//std::cout<<"zeta = "<<zeta<<"\n";
		//std::cout<<"lambda = "<<lambda<<"\n";
		//std::cout<<"dr = "<<dr[0]<<" "<<dr[1]<<" "<<dr[2]<<"\n";
		for(int i=1; i<N-1; ++i){
			double fphi;
			double feta[3];
			//finite difference
			const double cos2=std::cos((i+1.0)/(N-1.0)*da);
			const double f2=basisA.symmf(cos2,dr,eta,zeta,lambda);
			const double cos1=std::cos((i-1.0)/(N-1.0)*da);
			const double f1=basisA.symmf(cos1,dr,eta,zeta,lambda);
			const double g=(f2-f1)/(cos2-cos1);
			//exact
			cos=std::cos((i*1.0)/(N-1.0)*da);
			basisA.symmd(fphi,feta,cos,dr,eta,zeta,lambda);
			erra+=std::fabs(g-fphi);
			errp+=std::fabs(g-fphi)/std::fabs(g)*100.0;
			reduce.push(g,fphi);
			//std::cout<<"g "<<g<<" "<<fphi<<"\n";
		}
	}
	const int Mt=1;
	const int Nt=1e5;
	high_resolution_clock::time_point tbeg,tend;
	duration<double> timef,timed;
	for(int iter=0; iter<Mt; ++iter){
		//constants
		const double rc=6.0;
		const double da=math::constant::PI/(N-1.0);
		const double dc=std::cos(da);
		BasisA basisA(rc,Cutoff::Name::COS,1,phian);
		const double zeta=(1.0*std::rand())/RAND_MAX*(8.0-1.0)+1.0;
		const int lambda=((1.0*std::rand())/RAND_MAX<0.5)?1:-1;
		const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double rs=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		const double dr[2]={
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0
		};
		double cos=0;
		//compute time
		high_resolution_clock::time_point tbeg,tend;
		tbeg=high_resolution_clock::now();
		for(int i=0; i<Nt; ++i){
			cos=std::cos((i+1.0)/(N-1.0)*math::constant::PI);
			volatile double val=basisA.symmf(cos,dr,eta,zeta,lambda);
		}
		tend=high_resolution_clock::now();
		timef+=duration_cast<duration<double>>(tend-tbeg);
		tbeg=high_resolution_clock::now();
		for(int i=0; i<Nt; ++i){
			cos=std::cos((i+1.0)/(N-1.0)*math::constant::PI);
			double fphi;
			double feta[2];
			basisA.symmd(fphi,feta,cos,dr,eta,zeta,lambda);
		}
		tend=high_resolution_clock::now();
		timed+=duration_cast<duration<double>>(tend-tbeg);
	}
	erra/=(N-1.0)*M;
	errp/=(N-1.0)*M;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - SYMM - GRAD - PHI\n";
	std::cout<<"NAME             = "<<phian<<"\n";
	std::cout<<"err - grad - abs = "<<erra<<"\n";
	std::cout<<"err - grad - per = "<<errp<<"\n";
	std::cout<<"correlation - m  = "<<reduce.m()<<"\n";
	std::cout<<"correlation - r2 = "<<reduce.r2()<<"\n";
	std::cout<<"time - count     = "<<Nt<<"\n";
	std::cout<<"time - function  = "<<timef.count()<<" ns\n";
	std::cout<<"time - gradient  = "<<timed.count()<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_symm_grad_dr(BasisA::Name phian, int index){
	//rand
	std::srand(std::time(NULL));
	//error
	double erra=0;
	double errp=0;
	double timef=0;
	double timed=0;
	const int M=100;
	const int N=10000;
	Reduce<2> reduce;
	for(int iter=0; iter<M; ++iter){
		//constants
		const double rc=6.0;
		BasisA basisA(rc,Cutoff::Name::COS,1,phian);
		const double dr=rc/(N-1.0);
		const double cos=std::cos((1.0*std::rand())/RAND_MAX*math::constant::PI);
		const double zeta=((1.0*std::rand())/RAND_MAX*(8.0-1.0)+1.0)/5.0;
		const int lambda=((1.0*std::rand())/RAND_MAX<0.5)?1:-1;
		const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		double rr[2]={
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0
		};
		//compute error
		//std::cout<<"eta = "<<eta<<"\n";
		//std::cout<<"zeta = "<<zeta<<"\n";
		//std::cout<<"lambda = "<<lambda<<"\n";
		//std::cout<<"dr = "<<rr[0]<<" "<<rr[1]<<" "<<rr[2]<<"\n";
		for(int i=1; i<N-1; ++i){
			double fphi;
			double feta[2];
			//finite difference
			const double r2=(i+1.0)/(N-1.0)*rc;
			rr[index]=r2;
			const double f2=basisA.symmf(cos,rr,eta,zeta,lambda);
			const double r1=(i-1.0)/(N-1.0)*rc;
			rr[index]=r1;
			const double f1=basisA.symmf(cos,rr,eta,zeta,lambda);
			const double g=(f2-f1)/(r2-r1);
			//exact
			const double r0=(i*1.0)/(N-1.0)*rc;
			rr[index]=r0;
			basisA.symmd(fphi,feta,cos,rr,eta,zeta,lambda);
			erra+=std::fabs(g-feta[index]);
			if(g!=0) errp+=std::fabs(g-feta[index])/std::fabs(g)*100.0;
			reduce.push(g,feta[index]);
			//std::cout<<"g "<<g<<" "<<feta[index]<<"\n";
		}
	}
	const int Mt=50;
	for(int iter=0; iter<Mt; ++iter){
		//constants
		const double rc=6.0;
		BasisA basisA(rc,Cutoff::Name::COS,1,phian);
		const double dr=rc/(N-1.0);
		const double cos=std::cos((1.0*std::rand())/RAND_MAX*math::constant::PI);
		const double zeta=((1.0*std::rand())/RAND_MAX*(8.0-1.0)+1.0)/5.0;
		const int lambda=((1.0*std::rand())/RAND_MAX<0.5)?1:-1;
		const double eta=(1.0*std::rand())/RAND_MAX*0.5*rc*5;
		double rr[2]={
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0,
			(1.0*std::rand())/RAND_MAX*(0.2*rc-1.0)+0.0
		};
		//compute time
		clock_t start,stop;
		const int Nt=1e5;
		start=std::clock();
		for(int i=0; i<Nt; ++i){
			volatile double val=basisA.symmf(cos,rr,eta,zeta,lambda);
		}
		stop=std::clock();
		timef+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
		start=std::clock();
		for(int i=0; i<Nt; ++i){
			double fphi;
			double feta[2];
			basisA.symmd(fphi,feta,cos,rr,eta,zeta,lambda);
		}
		stop=std::clock();
		timed+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/Nt;
	}
	timef/=Mt;
	timed/=Mt;
	erra/=(N-1.0)*(M);
	errp/=(N-1.0)*(M);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - SYMM - GRAD - DR["<<index<<"]\n";
	std::cout<<"NAME             = "<<phian<<"\n";
	std::cout<<"err - grad - abs = "<<erra<<"\n";
	std::cout<<"err - grad - per = "<<errp<<"\n";
	std::cout<<"correlation - m  = "<<reduce.m()<<"\n";
	std::cout<<"correlation - r2 = "<<reduce.r2()<<"\n";
	std::cout<<"time - function  = "<<timef<<" ns\n";
	std::cout<<"time - gradient  = "<<timed<<" ns\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void write(BasisA::Name phian){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/N;
	const double cos=0.0;
	const double eta=2.5;
	const double zeta=1.0;
	const int lambda=1;
	
	BasisA basisA(rc,Cutoff::Name::COS,1,phian);
	
	double rr[3];
	FILE* writer=fopen("test.dat","w");
	for(int i=0; i<N; ++i){
		for(int j=0; j<N; ++j){
			rr[0]=dr*i;
			rr[1]=dr*j;
			rr[2]=0;
			const double f=basisA.symmf(cos,rr,eta,zeta,lambda);
			fprintf(writer,"%f %f %f\n",rr[0],rr[1],f);
		}
	}
	fclose(writer);
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - BASIS - ANGULAR - GRADIENT",str)<<"\n";
	test_symm_grad_phi(BasisA::Name::GAUSS);
	test_symm_grad_phi(BasisA::Name::GAUSS2);
	test_symm_grad_phi(BasisA::Name::SECH);
	test_symm_grad_phi(BasisA::Name::STUDENT3);
	test_symm_grad_phi(BasisA::Name::STUDENT4);
	test_symm_grad_phi(BasisA::Name::STUDENT5);
	std::cout<<print::buf(str)<<"\n";
	
	test_symm_grad_dr(BasisA::Name::GAUSS,0);
	test_symm_grad_dr(BasisA::Name::GAUSS,1);
	test_symm_grad_dr(BasisA::Name::GAUSS2,0);
	test_symm_grad_dr(BasisA::Name::GAUSS2,1);
	test_symm_grad_dr(BasisA::Name::SECH,0);
	test_symm_grad_dr(BasisA::Name::SECH,1);
	test_symm_grad_dr(BasisA::Name::STUDENT3,0);
	test_symm_grad_dr(BasisA::Name::STUDENT3,1);
	test_symm_grad_dr(BasisA::Name::STUDENT4,0);
	test_symm_grad_dr(BasisA::Name::STUDENT4,1);
	test_symm_grad_dr(BasisA::Name::STUDENT5,0);
	test_symm_grad_dr(BasisA::Name::STUDENT5,1);
	
	//write(BasisA::Name::SECH);
	
	delete[] str;
	
	return 0;
}
