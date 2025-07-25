// c++
#include <iostream>
#include <random>
#include <chrono>
// math
#include "math/reduce.hpp"
#include "math/const.hpp"
// rand
#include "rand/mc.hpp"
//str
#include "str/print.hpp"

using math::constant::RadPI;
using math::constant::Rad2;
using math::constant::PI;

class Uniform1D{
private:
	double a_,b_;
public:
	//==== constructor/destructor ====
	Uniform1D():a_(0.0),b_(0.0){}
	Uniform1D(double a, double b):a_(a),b_(b){}
	~Uniform1D(){}
	
	//==== access ====
	double& a(){return a_;}
	const double& a()const{return a_;}
	double& b(){return b_;}
	const double& b()const{return b_;}
	
	//==== operators ====
	double operator()(const Eigen::VectorXd& v){
		return (a_<v[0] && v[0]<b_)?1.0:0.0;
	}
};

class Normal1D{
private:
	double mu_,sigma_;
public:
	//==== constructor/destructor ====
	Normal1D():mu_(0.0),sigma_(0.0){}
	Normal1D(double mu, double sigma):mu_(mu),sigma_(sigma){}
	
	//==== access ====
	double& mu(){return mu_;}
	const double& mu()const{return mu_;}
	double& sigma(){return sigma_;}
	const double& sigma()const{return sigma_;}
	
	//==== operators ====
	double operator()(const Eigen::VectorXd& v){
		const double arg=(v[0]-mu_)/sigma_;
		return 1.0/(Rad2*RadPI*sigma_)*exp(-0.5*arg*arg);
	}
};

void test_mc_uniform1d(){
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - RAND - MC - UNIFORM",str)<<"\n";
	
	//set random number generator
	std::default_random_engine gen(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<double> dist(1.0,3.0);
	
	//set probability distribution
	Uniform1D rho=Uniform1D(-1.0*dist(gen),1.0*dist(gen));
	std::printf("[a,b] = [%f,%f]\n",rho.a(),rho.b());
	std::printf("avg   = %f\n",0.5*(rho.a()+rho.b()));
	std::printf("var   = %f\n",1.0/12.0*(rho.b()-rho.a())*(rho.b()-rho.a()));
	
	//set ensemble
	const int size=10000;
	const int dim=1;
	mc::Ensemble ensemble=mc::Ensemble(size,dim);
	
	//set metropolis
	mc::Metropolis metr;
	metr.rho()=rho;
	metr.init();
	metr.delta()=0.1;
	std::cout<<"metr  = "<<metr<<"\n";
	
	//set reduction
	Reduce<1> reduce;
	
	//integrate
	const int nstep=100000;
	const int nprint=nstep/10;
	std::printf("ensemble = %i\n",size);
	std::printf("nstep    = %i\n",nstep);
	std::printf("step avg variance\n");
	for(int i=0; i<nstep+1; ++i){
		if(i%nprint==0){
			std::printf("%i %f %f\n",i,reduce.avg(),reduce.var());
		}
		metr.step(ensemble);
		for(int j=0; j<ensemble.size(); ++j){
			reduce.push(ensemble.atom(j).x()[0]);
		}
	}
	
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_mc_normal1d(){
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - RAND - MC - NORMAL",str)<<"\n";
	
	//set random number generator
	std::default_random_engine gen(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<double> mdist(-3.0,3.0);
	std::uniform_real_distribution<double> sdist(0.1,3.0);
	
	//set probability distribution
	Normal1D rho=Normal1D(mdist(gen),sdist(gen));
	std::printf("mu    = %f\n",rho.mu());
	std::printf("sigma = %f\n",rho.sigma());
	std::printf("avg   = %f\n",rho.mu());
	std::printf("var   = %f\n",rho.sigma()*rho.sigma());
	
	//set ensemble
	const int size=10000;
	const int dim=1;
	mc::Ensemble ensemble=mc::Ensemble(size,dim);
	
	//set metropolis
	mc::Metropolis metr;
	metr.rho()=rho;
	metr.init();
	metr.delta()=1.0;
	std::cout<<"metr  = "<<metr<<"\n";
	
	//set reduction
	Reduce<1> reduce;
	
	//integrate
	const int nstep=100000;
	const int nprint=nstep/10;
	std::printf("ensemble = %i\n",size);
	std::printf("nstep    = %i\n",nstep);
	std::printf("step avg variance\n");
	for(int i=0; i<nstep+1; ++i){
		if(i%nprint==0){
			std::printf("%i %f %f\n",i,reduce.avg(),reduce.var());
		}
		metr.step(ensemble);
		for(int j=0; j<ensemble.size(); ++j){
			reduce.push(ensemble.atom(j).x()[0]);
		}
	}
	
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

int main(int argc, char* argv[]){
	
	//test_mc_uniform1d();
	test_mc_normal1d();
	
	return 0;
}