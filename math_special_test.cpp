// c libraries
#include <cstdlib>
#include <cstdio>
#include <ctime>
// c++ libraries
#include <iostream>
#include <vector>
// math
#include "math_special.hpp"
#include "math_const.hpp"
#include "math_cmp.hpp"
// accumulators
#include "accumulator.hpp"

int main(int argc, char* argv[]){
	
	bool test_cos   = false;
	bool test_sin   = false;
	bool test_expl  = false;
	bool test_expb  = false;
	bool test_erf   = false;
	bool test_logp1 = true;
	bool test_sfp   = true;
	bool test_cmp   = false;
	bool test_sign  = false;
	
	if(test_cos){
	std::cout<<"******************************************\n";
	std::cout<<"*************** TEST - COS ***************\n";
	try{
		//local variables
		const unsigned int N=500;
		const unsigned int NN=10000;
		const double xmin=0;
		const double xmax=num_const::PI;
		std::vector<double> x(N,0);
		std::vector<double> cos_exact(N,0);
		std::vector<double> cos_approx(N,0);
		Accumulator1D<Max,Avg,Var> acc;
		double time_exact,time_approx;
		clock_t start,stop;
		
		std::cout<<"N           = "<<N<<"\n";
		std::cout<<"interval    = "<<xmin<<" "<<xmax<<"\n";
		
		//generate the values
		for(unsigned int i=0; i<N; ++i){
			x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
			cos_exact[i]=std::cos(x[i]);
			cos_approx[i]=special::cos(x[i]);
			acc.push(std::fabs(cos_exact[i]-cos_approx[i]));
		}
		
		//print the error
		std::cout<<"err-avg     = "<<acc.avg()<<"\n";
		std::cout<<"err-stddev  = "<<std::sqrt(acc.var())<<"\n";
		std::cout<<"err-max     = "<<acc.max()<<"\n";
		
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=std::cos(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::cos(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-exact  = "<<time_exact<<"\n";
		std::cout<<"time-approx = "<<time_approx<<"\n";
		
		
		//write the results
		FILE* writer=fopen("math_special_test_cos.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#X COS_EXACT COS_APPROX ERROR\n");
			for(unsigned int i=0; i<N; ++i){
				fprintf(writer,"%f %f %f %f\n",x[i],cos_exact[i],cos_approx[i],std::fabs(cos_exact[i]-cos_approx[i]));
			}
			fclose(writer);
			writer=NULL;
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - COS\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"*************** TEST - COS ***************\n";
	std::cout<<"******************************************\n";
	}
	
	if(test_sin){
	std::cout<<"******************************************\n";
	std::cout<<"*************** TEST - SIN ***************\n";
	try{
		//local variables
		const unsigned int N=500;
		const unsigned int NN=10000;
		const double xmin=0;
		const double xmax=num_const::PI;
		std::vector<double> x(N,0);
		std::vector<double> sin_exact(N,0);
		std::vector<double> sin_approx(N,0);
		Accumulator1D<Max,Avg,Var> acc;
		double time_exact,time_approx;
		clock_t start,stop;
		
		std::cout<<"N           = "<<N<<"\n";
		std::cout<<"interval    = "<<xmin<<" "<<xmax<<"\n";
		
		//generate the values
		for(unsigned int i=0; i<N; ++i){
			x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
			sin_exact[i]=std::sin(x[i]);
			sin_approx[i]=special::sin(x[i]);
			acc.push(std::fabs(sin_exact[i]-sin_approx[i]));
		}
		
		//print the error
		std::cout<<"err-avg     = "<<acc.avg()<<"\n";
		std::cout<<"err-stddev  = "<<std::sqrt(acc.var())<<"\n";
		std::cout<<"err-max     = "<<acc.max()<<"\n";
		
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=std::sin(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::sin(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-exact  = "<<time_exact<<"\n";
		std::cout<<"time-approx = "<<time_approx<<"\n";
		
		
		//write the results
		FILE* writer=fopen("math_special_test_sin.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#X SIN_EXACT SIN_APPROX ERROR\n");
			for(unsigned int i=0; i<N; ++i){
				fprintf(writer,"%f %f %f %f\n",x[i],sin_exact[i],sin_approx[i],std::fabs(sin_exact[i]-sin_approx[i]));
			}
			fclose(writer);
			writer=NULL;
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SIN\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"*************** TEST - SIN ***************\n";
	std::cout<<"******************************************\n";
	}
	
	if(test_logp1){
	std::cout<<"******************************************\n";
	std::cout<<"************** TEST - LOGP1 **************\n";
	try{
		//local variables
		const unsigned int N=500;
		const unsigned int NN=10000;
		const double xmin=0.0;
		const double xmax=1.0;
		std::vector<double> x(N,0);
		std::vector<double> exact(N,0);
		std::vector<double> approx(N,0);
		Accumulator1D<Max,Avg,Var> acc;
		double time_exact,time_approx;
		clock_t start,stop;
		
		std::cout<<"N           = "<<N<<"\n";
		std::cout<<"interval    = "<<xmin<<" "<<xmax<<"\n";
		
		//generate the values
		for(unsigned int i=0; i<N; ++i){
			x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
			exact[i]=std::log(x[i]+1.0);
			approx[i]=special::logp1(x[i]);
			acc.push(std::fabs(exact[i]-approx[i]));
		}
		
		//print the error
		std::cout<<"err-avg     = "<<acc.avg()<<"\n";
		std::cout<<"err-stddev  = "<<std::sqrt(acc.var())<<"\n";
		std::cout<<"err-max     = "<<acc.max()<<"\n";
		
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=std::log(1.0+std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::logp1(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-exact  = "<<time_exact<<"\n";
		std::cout<<"time-approx = "<<time_approx<<"\n";
		
		
		//write the results
		FILE* writer=fopen("math_special_test_logp1.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#X EXACT APPROX ERROR\n");
			for(unsigned int i=0; i<N; ++i){
				fprintf(writer,"%f %f %f %f\n",x[i],exact[i],approx[i],std::fabs(exact[i]-approx[i]));
			}
			fclose(writer);
			writer=NULL;
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - LOGP1\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************** TEST - LOGP1 **************\n";
	std::cout<<"******************************************\n";
	}
	
	if(test_sfp){
	std::cout<<"******************************************\n";
	std::cout<<"************ TEST - SOFTPLUS ************\n";
	try{
		//local variables
		const unsigned int N=500;
		const unsigned int NN=10000;
		const double xmin=-10.0;
		const double xmax=10.0;
		std::vector<double> x(N,0);
		std::vector<double> exact(N,0);
		std::vector<double> approx(N,0);
		Accumulator1D<Max,Avg,Var> acc;
		double time_exact,time_approx;
		clock_t start,stop;
		
		std::cout<<"N           = "<<N<<"\n";
		std::cout<<"interval    = "<<xmin<<" "<<xmax<<"\n";
		
		//generate the values
		for(unsigned int i=0; i<N; ++i){
			x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
			exact[i]=std::log(std::exp(x[i])+1.0);
			approx[i]=special::softplus(x[i]);
			acc.push(std::fabs(exact[i]-approx[i]));
		}
		
		//print the error
		std::cout<<"err-avg    = "<<acc.avg()<<"\n";
		std::cout<<"err-stddev = "<<std::sqrt(acc.var())<<"\n";
		std::cout<<"err-max    = "<<acc.max()<<"\n";
		
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=std::log(1.0+std::exp(std::rand()*(xmax-xmin)/RAND_MAX+xmin));
		}
		stop=std::clock();
		time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::softplus(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-exact  = "<<time_exact<<"\n";
		std::cout<<"time-approx = "<<time_approx<<"\n";
		
		
		//write the results
		FILE* writer=fopen("math_special_test_softplus.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#X EXACT APPROX ERROR\n");
			for(unsigned int i=0; i<N; ++i){
				fprintf(writer,"%f %f %f %f\n",x[i],exact[i],approx[i],std::fabs(exact[i]-approx[i]));
			}
			fclose(writer);
			writer=NULL;
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SOFTPLUS\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************ TEST - SOFTPLUS ************\n";
	std::cout<<"******************************************\n";
	}
	
	if(test_expl){
	std::cout<<"******************************************\n";
	std::cout<<"************** TEST - EXPL **************\n";
	try{
		//local variables
		const unsigned int O=8;
		const unsigned int N=500;
		const unsigned int NN=10000;
		const double xmin=-10.0;
		const double xmax=0.0;
		std::vector<double> x(N,0);
		std::vector<double> exp_exact(N,0);
		std::vector<double> exp_approx(N,0);
		Accumulator1D<Max,Avg,Var> acc;
		double time_exact,time_approx;
		clock_t start,stop;
		
		std::cout<<"N           = "<<N<<"\n";
		std::cout<<"interval    = "<<xmin<<" "<<xmax<<"\n";
		
		//generate the values
		for(unsigned int i=0; i<N; ++i){
			x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
			exp_exact[i]=std::exp(x[i]);
			exp_approx[i]=special::expl<O>(x[i]);
			acc.push(std::fabs(exp_exact[i]-exp_approx[i]));
		}
		
		//print the error
		std::cout<<"err-avg     = "<<acc.avg()<<"\n";
		std::cout<<"err-stddev  = "<<std::sqrt(acc.var())<<"\n";
		std::cout<<"err-max     = "<<acc.max()<<"\n";
		
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=std::exp(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::expl<O>(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-exact  = "<<time_exact<<"\n";
		std::cout<<"time-approx = "<<time_approx<<"\n";
		
		//write the results
		FILE* writer=fopen("math_special_test_expl.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#X EXP_EXACT EXP_APPROX ERROR\n");
			for(unsigned int i=0; i<N; ++i){
				fprintf(writer,"%f %f %f %f\n",x[i],exp_exact[i],exp_approx[i],std::fabs(exp_exact[i]-exp_approx[i]));
			}
			fclose(writer);
			writer=NULL;
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - EXPL\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************** TEST - EXPL **************\n";
	std::cout<<"******************************************\n";
	}
	
	if(test_expb){
	std::cout<<"******************************************\n";
	std::cout<<"************** TEST - EXPB **************\n";
	try{
		//local variables
		const unsigned int N=500;
		const unsigned int NN=10000;
		const double xmin=-10.0;
		const double xmax=0.0;
		std::vector<double> x(N,0);
		std::vector<double> exp_exact(N,0);
		std::vector<double> exp_approx(N,0);
		Accumulator1D<Max,Avg,Var> acc;
		double time_exact,time_approx;
		clock_t start,stop;
		
		std::cout<<"N           = "<<N<<"\n";
		std::cout<<"interval    = "<<xmin<<" "<<xmax<<"\n";
		
		//generate the values
		for(unsigned int i=0; i<N; ++i){
			x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
			exp_exact[i]=std::exp(x[i]);
			exp_approx[i]=special::expb(x[i]);
			acc.push(std::fabs(exp_exact[i]-exp_approx[i]));
		}
		
		//print the error
		std::cout<<"err-avg     = "<<acc.avg()<<"\n";
		std::cout<<"err-stddev  = "<<std::sqrt(acc.var())<<"\n";
		std::cout<<"err-max     = "<<acc.max()<<"\n";
		
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=std::exp(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::expb(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-exact  = "<<time_exact<<"\n";
		std::cout<<"time-approx = "<<time_approx<<"\n";
		
		//write the results
		FILE* writer=fopen("math_special_test_expb.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#X EXP_EXACT EXP_APPROX ERROR\n");
			for(unsigned int i=0; i<N; ++i){
				fprintf(writer,"%f %f %f %f\n",x[i],exp_exact[i],exp_approx[i],std::fabs(exp_exact[i]-exp_approx[i]));
			}
			fclose(writer);
			writer=NULL;
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - EXPB\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************** TEST - EXPB **************\n";
	std::cout<<"******************************************\n";
	}
	
	if(test_erf){
	std::cout<<"******************************************\n";
	std::cout<<"************** TEST - ERFC **************\n";
	try{
		//local variables
		const unsigned int N=500;
		const unsigned int NN=10000;
		const double xmin=-5.0;
		const double xmax=5.0;
		std::vector<double> x(N,0);
		std::vector<double> erf_exact(N,0);
		std::vector<double> erf_approx1(N,0);
		std::vector<double> erf_approx2(N,0);
		std::vector<double> erf_approx3(N,0);
		std::vector<double> erf_approx4(N,0);
		Accumulator1D<Max,Avg,Var> acc1;
		Accumulator1D<Max,Avg,Var> acc2;
		Accumulator1D<Max,Avg,Var> acc3;
		Accumulator1D<Max,Avg,Var> acc4;
		double time_exact,time_approx;
		clock_t start,stop;
		
		std::cout<<"N           = "<<N<<"\n";
		std::cout<<"interval    = "<<xmin<<" "<<xmax<<"\n";
		
		//generate the values
		for(unsigned int i=0; i<N; ++i){
			x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
			erf_exact[i]=std::erf(x[i]);
			erf_approx1[i]=special::erfa1(x[i]);
			erf_approx2[i]=special::erfa2(x[i]);
			erf_approx3[i]=special::erfa3(x[i]);
			erf_approx4[i]=special::erfa4(x[i]);
			acc1.push(std::fabs(erf_exact[i]-erf_approx1[i]));
			acc2.push(std::fabs(erf_exact[i]-erf_approx2[i]));
			acc3.push(std::fabs(erf_exact[i]-erf_approx3[i]));
			acc4.push(std::fabs(erf_exact[i]-erf_approx4[i]));
		}
		
		//print the error
		std::cout<<"approx 1\n";
		std::cout<<"err-avg     = "<<acc1.avg()<<"\n";
		std::cout<<"err-stddev  = "<<std::sqrt(acc1.var())<<"\n";
		std::cout<<"err-max     = "<<acc1.max()<<"\n";
		std::cout<<"approx 2\n";
		std::cout<<"err-avg     = "<<acc2.avg()<<"\n";
		std::cout<<"err-stddev  = "<<std::sqrt(acc2.var())<<"\n";
		std::cout<<"err-max     = "<<acc2.max()<<"\n";
		std::cout<<"approx 3\n";
		std::cout<<"err-avg     = "<<acc3.avg()<<"\n";
		std::cout<<"err-stddev  = "<<std::sqrt(acc3.var())<<"\n";
		std::cout<<"err-max     = "<<acc3.max()<<"\n";
		std::cout<<"approx 4\n";
		std::cout<<"err-avg     = "<<acc4.avg()<<"\n";
		std::cout<<"err-stddev  = "<<std::sqrt(acc4.var())<<"\n";
		std::cout<<"err-max     = "<<acc4.max()<<"\n";
		
		std::srand(std::time(NULL));
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=std::erf(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-exact  = "<<time_exact<<"\n";
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::erfa1(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-approx = "<<time_approx<<"\n";
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::erfa2(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-approx = "<<time_approx<<"\n";
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::erfa3(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-approx = "<<time_approx<<"\n";
		start=std::clock();
		for(unsigned int i=0; i<NN; ++i){
			volatile double val=special::erfa4(std::rand()*(xmax-xmin)/RAND_MAX+xmin);
		}
		stop=std::clock();
		time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"time-approx = "<<time_approx<<"\n";
		
		//write the results
		/*FILE* writer=fopen("math_special_test_expb.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#X EXP_EXACT EXP_APPROX ERROR\n");
			for(unsigned int i=0; i<N; ++i){
				fprintf(writer,"%f %f %f %f\n",x[i],exp_exact[i],exp_approx[i],std::fabs(exp_exact[i]-exp_approx[i]));
			}
			fclose(writer);
			writer=NULL;
		}*/
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - ERFC\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************** TEST - ERFC **************\n";
	std::cout<<"******************************************\n";
	}
	
	if(test_cmp){
		std::cout<<"min(1.1242,2.7824) = "<<cmp::min(1.1242,2.7824)<<"\n";
		std::cout<<"min(2.7824,1.1242) = "<<cmp::min(2.7824,1.1242)<<"\n";
		std::cout<<"min(0,0) = "<<cmp::min(0,0)<<"\n";
	}
	
	if(test_sign){
		std::cout<<"sign(0) = "<<special::sign(0)<<"\n";
		std::cout<<"sign(1) = "<<special::sign(1)<<"\n";
		std::cout<<"sign(-1) = "<<special::sign(-1)<<"\n";
		std::cout<<"sign(0.574278) = "<<special::sign(0.574278)<<"\n";
		std::cout<<"sign(-8.31897) = "<<special::sign(-8.31897)<<"\n";
	}
	
}
