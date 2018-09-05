#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include "optimize.hpp"

double quad(const Eigen::VectorXd x, Eigen::VectorXd& grad){
	return 2.0*x[0]*x[0]-3.0*x[0]+5.0;
}

double a=1.0;
double b=100.0;
double rosen(const Eigen::VectorXd x, Eigen::VectorXd& grad){
	grad[0]=-2*(a-x[0])-4*b*x[0]*(x[1]-x[0]*x[0]);
	grad[1]=2*b*(x[1]-x[0]*x[0]);
	return (a-x[0])*(a-x[0])+b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
}

int main(int argc, char* argv[]){
	
	Opt opt;
	bool test_ln=false;
	bool test_sd=true;
	bool test_sdg=true;
	bool test_bfgs=false;
	bool test_bfgsg=false;
	bool test_sdm=true;
	bool test_nag=true;
	bool test_adagrad=true;
	bool test_adadelta=true;
	bool test_rmsprop=true;
	bool test_adam=true;
	bool test_rprop=false;
	
	double gammaG=0.0;
	double etaG=-1;
	Eigen::VectorXd soln(2);
	soln<<1,1;
	
	if(argc>1){
		//load the arguments
		std::vector<std::string> strlist(argc-1);
		for(unsigned int i=1; i<argc; ++i) strlist[i-1]=std::string(argv[i]);
		//read in the formats
		for(unsigned int i=0; i<strlist.size(); ++i){
			if(strlist[i]=="-gamma"){
				if(i==strlist.size()-1) throw std::invalid_argument("No gamma value provided.");
				else gammaG=std::atof(strlist[i+1].c_str());
			} else if(strlist[i]=="-eta"){
				if(i==strlist.size()-1) throw std::invalid_argument("No eta value provided.");
				else etaG=std::atof(strlist[i+1].c_str());
			}
		}
	}
	
	std::cout<<"gammaG = "<<gammaG<<"\n";
	std::cout<<"etaG = "<<etaG<<"\n";
	
	if(test_ln){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - LN **********************\n";
	try{
		opt.tol()=1e-6;
		opt.maxIter()=100;
		opt.dim()=2;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.01;
		
		Eigen::VectorXd x0(1),x1(1),grad(1);
		x0<<-5; x1<<3;
		grad<<0;
		
		double val=opt.opt_ln(quad,x0,x1);
		std::cout<<"x   = ["<<x0[0]<<","<<x1[0]<<"]\n";
		std::cout<<"v   = ["<<quad(x0,grad)<<","<<quad(x1,grad)<<"]\n";
		std::cout<<"val = "<<val<<"\n";
	} catch(std::exception& e){
		std::cout<<"ERROR in TEST - LN\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - LN **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_sd){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - SD **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.dim()=2;
		opt.gamma()=1.0;
		opt.algo()=OPT_METHOD::SD;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SD\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - SD **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_sdg){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - SDG **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.dim()=2;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.01;
		opt.algo()=OPT_METHOD::SDG;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SDG\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - SDG **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_bfgs){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - BFGS **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=5000;
		opt.nPrint()=100;
		opt.dim()=2;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.01;
		opt.algo()=OPT_METHOD::BFGS;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - BFGS\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - BFGS **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_bfgs){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - BFGSG **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=5000000;
		opt.nPrint()=100000;
		opt.dim()=2;
		opt.gamma()=1.0;
		opt.algo()=OPT_METHOD::BFGSG;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - BFGSG\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - BFGSG **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_sdm){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - SDM **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.dim()=2;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.01;
		if(etaG>=0) opt.eta()=etaG; else opt.eta()=0.01;
		opt.algo()=OPT_METHOD::SDM;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SDM\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - SDM **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_nag){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - NAG **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.dim()=2;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.01;
		if(etaG>0) opt.eta()=etaG; else opt.eta()=0.01;
		opt.algo()=OPT_METHOD::NAG;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - NAG\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - NAG **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_adagrad){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - ADAGRAD **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=100000000;
		opt.nPrint()=10000000;
		opt.dim()=2;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.01;
		opt.algo()=OPT_METHOD::ADAGRAD;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - ADAGRAD\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - ADAGRAD **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_adagrad){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - ADADELTA **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=50000;
		opt.nPrint()=1000;
		opt.dim()=2;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.01;
		opt.algo()=OPT_METHOD::ADADELTA;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - ADADELTA\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - ADADELTA **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_rmsprop){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - RMSPROP **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.dim()=2;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.01;
		opt.algo()=OPT_METHOD::RMSPROP;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - RMSPROP\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - RMSPROP **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_adam){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - ADAM **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=50000;
		opt.nPrint()=1000;
		opt.dim()=2;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.01;
		opt.algo()=OPT_METHOD::ADAM;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - ADAM\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - ADAM **********************\n";
	std::cout<<"*******************************************************\n";
	}
	
	if(test_rprop){
	std::cout<<"*******************************************************\n";
	std::cout<<"********************** TEST - RPROP **********************\n";
	try{
		opt.tol()=1e-8;
		opt.maxIter()=10000;
		opt.nPrint()=1000;
		opt.dim()=2;
		opt.gamma()=1.0;
		opt.algo()=OPT_METHOD::RPROP;
		unsigned int seed=1578942;
		std::srand(seed);
		
		std::cout<<"Setting initial point...\n";
		Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
		x[0]=-5+((double)std::rand())/RAND_MAX*10;
		x[1]=-5+((double)std::rand())/RAND_MAX*10;
		
		std::cout<<"Optimizing...\n";
		opt.opt(rosen,x);
		
		std::cout<<"Results:\n";
		std::cout<<"x-min = ["<<x[0]<<","<<x[1]<<"]\n";
		std::cout<<"val   = "<<opt.val()<<"\n";
		std::cout<<"error = "<<(x-soln).norm()<<"\n";
		std::cout<<"nStep = "<<opt.nStep()<<"\n";
		std::cout<<"nEval = "<<opt.nEval()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - BFGS\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - RPROP **********************\n";
	std::cout<<"*******************************************************\n";
	}
}