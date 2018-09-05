// c libraries
#include <cstdlib>
// c++ libraries
#include <iostream>
#include <chrono>
// local libraries - math
#include "math_cmp.hpp"
#include "math_function.hpp"
// local libraries - nn
#include "nn.hpp"
#include "nn_train.hpp"

class Gaussian{
private:
	double c_,s_,a_;
public:
	Gaussian():c_(0),s_(1),a_(1){};
	Gaussian(double c, double s, double a):c_(c),s_(s),a_(a){};
	~Gaussian(){};
	double operator()(double x)const{return a_*std::exp(-(x-c_)*(x-c_)/(2.0*s_*s_));};
};

int main(int argc, char* argv[]){
	
	bool test_unit_lin=false;
	bool test_unit_nonlin=false;
	bool test_timing=true;
	bool test_io=false;
	bool test_fit_func_gaussian2d=false;
	bool test_unit_lin_grad=false;
	bool test_unit_lin_grad_scale=false;
	bool test_unit_nonlin_grad=false;
	bool test_unit_nonlin_grad_scale=false;
	
	bool test_sdg=false;
	bool test_bfgsg=false;
	bool test_rmsprop=false;
	bool test_adadelta=false;
	bool test_rprop=false;
	
	double gammaG=0.0;
	double etaG=-1;
	unsigned int NBatchG=0;
	bool preCondG=false;
	bool postCondG=false;
	
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
			} else if(strlist[i]=="-nbatch"){
				if(i==strlist.size()-1) throw std::invalid_argument("No eta value provided.");
				else NBatchG=std::atoi(strlist[i+1].c_str());
			}
		}
	}
	
	std::cout<<"gammaG = "<<gammaG<<"\n";
	std::cout<<"etaG = "<<etaG<<"\n";
	std::cout<<"NBatchG = "<<NBatchG<<"\n";
	
	if(test_sdg){
	std::cout<<"**********************************************************\n";
	std::cout<<"*********************** TEST - SDG ***********************\n";
	try{
		/* local function variables */
		//linear function
			double m=1.624786758;
			double b=2.1875389;
		//quartic function
			std::vector<double> p(5,0);
			p[1]=0.048252;
			p[2]=1.48925;
			p[3]=0.72805290;
			p[4]=0.032052;
		//Gaussian function
			double c=0.92787849;
			double s=1.724687;
			double a=2.81378;
			Gaussian gauss(c,s,a);
		//sampling
			unsigned int NTrain=20;//number of training points
			unsigned int NBatch=NTrain;//batch size
			if(NBatchG>0) NBatch=NBatchG;
			unsigned int NVal=50;//number of test points
			unsigned int NTest=50;//number of test points
			NN::VecList xTrain(NTrain),yTrain(NTrain);//training data
			NN::VecList xVal(NVal),yVal(NVal);//validation data
			NN::VecList xTest(NTest),yTest(NTest);//test data
			double xMin=-5,xMax=5;
		//neural network
			std::vector<unsigned int> nh(1,15);
			NN::Network nn;
		//optimization
			NN::NNOpt nnopt;
			Opt opt;
			double error=0;
			double errorP=0;
		//rand
			unsigned int seed=72867429;
		//error
			double errorLV=0,errorLT=0;
			double errorQV=0,errorQT=0;
			double errorGV=0,errorGT=0;
		
		//initialize the random number generator
		std::cout<<"Random number generator ("<<seed<<").\n";
		std::srand(seed);
		
		//set nn parameters
		std::cout<<"Setting nn parameters...\n";
		nn.tfType()=NN::TransferN::TANH;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.lambda()=0.0;
		//print the parameters
		std::cout<<"N-batch = "<<NBatch<<"\n";
		std::cout<<"N-train = "<<NTrain<<"\n";
		std::cout<<"N-test  = "<<NTest<<"\n";
		
		//set optimization parameters
		std::cout<<"Setting optimization parameters...\n";
		opt.tol()=1e-14;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.period()=0;
		opt.decay()=0;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.1;
		if(etaG>=0) opt.eta()=etaG; else opt.eta()=0.1;
		opt.algo()=OPT_METHOD::SDG;
		std::cout<<"Opt     = \n"<<opt<<"\n";
		
		/* TEST LINEAR */
		std::cout<<"**** TEST - LINEAR ****\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=xTrain[i][0]*m+b;
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		
		//resizing the network
		std::cout<<"Resizing the network...\n";
		nn.resize(1,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Linear  = ("<<m<<","<<b<<")\n";
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorLV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorLT=error/NTest;
		
		/* TEST QUARTIC */
		std::cout<<"**** TEST - QUARTIC ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Poly = "; for(unsigned int i=0; i<p.size(); ++i) std::cout<<p[i]<<" "; std::cout<<"\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=function::poly(xTrain[i][0],p);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorQV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorQT=error/NTest;
		
		/* TEST GAUSSIAN */
		std::cout<<"**** TEST - GAUSSIAN ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Gaussian = ("<<c<<","<<s<<","<<a<<")\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=gauss(xTrain[i][0]);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorGV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorGT=error/NTest;
		
		std::cout<<"-----------------------------------------------------------\n";
		std::cout<<"------------------------- Summary -------------------------\n";
		std::cout<<"error - lin      = "<<errorLV<<" "<<errorLT<<"\n";
		std::cout<<"error - quartic  = "<<errorQV<<" "<<errorQT<<"\n";
		std::cout<<"error - gaussian = "<<errorGV<<" "<<errorGT<<"\n";
		std::cout<<"-----------------------------------------------------------\n";
	} catch(std::exception& e){
		std::cout<<"ERROR in TEST - SDG\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"*********************** TEST - SDG ***********************\n";
	std::cout<<"**********************************************************\n";
	}
	
	if(test_bfgsg){
	std::cout<<"**********************************************************\n";
	std::cout<<"********************** TEST - BFGSG **********************\n";
	try{
		/* local function variables */
		//linear function
			double m=1.624786758;
			double b=2.1875389;
		//quartic function
			std::vector<double> p(5,0);
			p[1]=0.048252;
			p[2]=1.48925;
			p[3]=0.72805290;
			p[4]=0.032052;
		//Gaussian function
			double c=0.92787849;
			double s=1.724687;
			double a=2.81378;
			Gaussian gauss(c,s,a);
		//sampling
			unsigned int NTrain=20;//number of training points
			unsigned int NBatch=NTrain;//batch size
			if(NBatchG>0) NBatch=NBatchG;
			unsigned int NVal=50;//number of test points
			unsigned int NTest=50;//number of test points
			NN::VecList xTrain(NTrain),yTrain(NTrain);//training data
			NN::VecList xVal(NVal),yVal(NVal);//validation data
			NN::VecList xTest(NTest),yTest(NTest);//test data
			double xMin=-5,xMax=5;
		//neural network
			std::vector<unsigned int> nh(1,15);
			NN::Network nn;
		//optimization
			NN::NNOpt nnopt;
			Opt opt;
			double error=0;
			double errorP=0;
		//rand
			unsigned int seed=72867429;
		//error
			double errorLV=0,errorLT=0;
			double errorQV=0,errorQT=0;
			double errorGV=0,errorGT=0;
		
		//initialize the random number generator
		std::cout<<"Random number generator ("<<seed<<").\n";
		std::srand(seed);
		
		//set nn parameters
		std::cout<<"Setting nn parameters...\n";
		nn.tfType()=NN::TransferN::TANH;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.lambda()=0.0;
		//print the parameters
		std::cout<<"N-batch = "<<NBatch<<"\n";
		std::cout<<"N-train = "<<NTrain<<"\n";
		std::cout<<"N-test  = "<<NTest<<"\n";
		
		//set optimization parameters
		std::cout<<"Setting optimization parameters...\n";
		opt.tol()=1e-14;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.period()=0;
		opt.decay()=0;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.1;
		if(etaG>=0) opt.eta()=etaG; else opt.eta()=0.1;
		opt.algo()=OPT_METHOD::BFGSG;
		std::cout<<"Opt     = \n"<<opt<<"\n";
		
		//**** TEST LINEAR ****
		std::cout<<"**** TEST - LINEAR ****\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=xTrain[i][0]*m+b;
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		
		
		//resizing the network
		std::cout<<"Resizing the network...\n";
		nn.resize(1,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Linear  = ("<<m<<","<<b<<")\n";
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorLV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorLT=error/NTest;
		
		//**** TEST QUARTIC ****
		std::cout<<"**** TEST - QUARTIC ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Poly = "; for(unsigned int i=0; i<p.size(); ++i) std::cout<<p[i]<<" "; std::cout<<"\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=function::poly(xTrain[i][0],p);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorQV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorQT=error/NTest;
		
		//**** TEST GAUSSIAN ****
		std::cout<<"**** TEST - GAUSSIAN ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Gaussian = ("<<c<<","<<s<<","<<a<<")\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=gauss(xTrain[i][0]);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		//nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		nnopt.train(nn,xTrain,yTrain,opt);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorGV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorGT=error/NTest;
		
		std::cout<<"-----------------------------------------------------------\n";
		std::cout<<"------------------------- Summary -------------------------\n";
		std::cout<<"error - lin      = "<<errorLV<<" "<<errorLT<<"\n";
		std::cout<<"error - quartic  = "<<errorQV<<" "<<errorQT<<"\n";
		std::cout<<"error - gaussian = "<<errorGV<<" "<<errorGT<<"\n";
		std::cout<<"-----------------------------------------------------------\n";
	} catch(std::exception& e){
		std::cout<<"ERROR in TEST - BFGSG\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - BFGSG **********************\n";
	std::cout<<"**********************************************************\n";
	}
	
	if(test_rmsprop){
	std::cout<<"**********************************************************\n";
	std::cout<<"********************* TEST - RMSPROP *********************\n";
	try{
		/* local function variables */
		//linear function
			double m=1.624786758;
			double b=2.1875389;
		//quartic function
			std::vector<double> p(5,0);
			p[1]=0.048252;
			p[2]=1.48925;
			p[3]=0.72805290;
			p[4]=0.032052;
		//Gaussian function
			double c=0.92787849;
			double s=1.724687;
			double a=2.81378;
			Gaussian gauss(c,s,a);
		//sampling
			unsigned int NTrain=20;//number of training points
			unsigned int NBatch=NTrain;//batch size
			if(NBatchG>0) NBatch=NBatchG;
			unsigned int NVal=50;//number of test points
			unsigned int NTest=50;//number of test points
			NN::VecList xTrain(NTrain),yTrain(NTrain);//training data
			NN::VecList xVal(NVal),yVal(NVal);//validation data
			NN::VecList xTest(NTest),yTest(NTest);//test data
			double xMin=-5,xMax=5;
		//neural network
			std::vector<unsigned int> nh(1,15);
			NN::Network nn;
		//optimization
			NN::NNOpt nnopt;
			Opt opt;
			double error=0;
			double errorP=0;
		//rand
			unsigned int seed=72867429;
		//error
			double errorLV=0,errorLT=0;
			double errorQV=0,errorQT=0;
			double errorGV=0,errorGT=0;
		
		//initialize the random number generator
		std::cout<<"Random number generator ("<<seed<<").\n";
		std::srand(seed);
		
		//set nn parameters
		std::cout<<"Setting nn parameters...\n";
		nn.tfType()=NN::TransferN::TANH;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.lambda()=0.0;
		//print the parameters
		std::cout<<"N-batch = "<<NBatch<<"\n";
		std::cout<<"N-train = "<<NTrain<<"\n";
		std::cout<<"N-test  = "<<NTest<<"\n";
		
		//set optimization parameters
		std::cout<<"Setting optimization parameters...\n";
		opt.tol()=1e-14;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.period()=0;
		opt.decay()=0;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.1;
		if(etaG>=0) opt.eta()=etaG; else opt.eta()=0.1;
		opt.algo()=OPT_METHOD::RMSPROP;
		std::cout<<"Opt     = \n"<<opt<<"\n";
		
		/* TEST LINEAR */
		std::cout<<"**** TEST - LINEAR ****\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=xTrain[i][0]*m+b;
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		
		//resizing the network
		std::cout<<"Resizing the network...\n";
		nn.resize(1,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Linear  = ("<<m<<","<<b<<")\n";
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorLV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorLT=error/NTest;
		
		/* TEST QUARTIC */
		std::cout<<"**** TEST - QUARTIC ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Poly = "; for(unsigned int i=0; i<p.size(); ++i) std::cout<<p[i]<<" "; std::cout<<"\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=function::poly(xTrain[i][0],p);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorQV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorQT=error/NTest;
		
		/* TEST GAUSSIAN */
		std::cout<<"**** TEST - GAUSSIAN ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Gaussian = ("<<c<<","<<s<<","<<a<<")\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=gauss(xTrain[i][0]);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorGV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorGT=error/NTest;
		
		std::cout<<"-----------------------------------------------------------\n";
		std::cout<<"------------------------- Summary -------------------------\n";
		std::cout<<"error - lin      = "<<errorLV<<" "<<errorLT<<"\n";
		std::cout<<"error - quartic  = "<<errorQV<<" "<<errorQT<<"\n";
		std::cout<<"error - gaussian = "<<errorGV<<" "<<errorGT<<"\n";
		std::cout<<"-----------------------------------------------------------\n";
	} catch(std::exception& e){
		std::cout<<"ERROR in TEST - RMSPROP\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************* TEST - RMSPROP *********************\n";
	std::cout<<"**********************************************************\n";
	}
	
	if(test_adadelta){
	std::cout<<"**********************************************************\n";
	std::cout<<"******************** TEST - ADADELTA  ********************\n";
	try{
		/* local function variables */
		//linear function
			double m=1.624786758;
			double b=2.1875389;
		//quartic function
			std::vector<double> p(5,0);
			p[1]=0.048252;
			p[2]=1.48925;
			p[3]=0.72805290;
			p[4]=0.032052;
		//Gaussian function
			double c=0.92787849;
			double s=1.724687;
			double a=2.81378;
			Gaussian gauss(c,s,a);
		//sampling
			unsigned int NTrain=20;//number of training points
			unsigned int NBatch=NTrain;//batch size
			if(NBatchG>0) NBatch=NBatchG;
			unsigned int NVal=50;//number of test points
			unsigned int NTest=50;//number of test points
			NN::VecList xTrain(NTrain),yTrain(NTrain);//training data
			NN::VecList xVal(NVal),yVal(NVal);//validation data
			NN::VecList xTest(NTest),yTest(NTest);//test data
			double xMin=-5,xMax=5;
		//neural network
			std::vector<unsigned int> nh(1,15);
			NN::Network nn;
		//optimization
			NN::NNOpt nnopt;
			Opt opt;
			double error=0;
			double errorP=0;
		//rand
			unsigned int seed=72867429;
		//error
			double errorLV=0,errorLT=0;
			double errorQV=0,errorQT=0;
			double errorGV=0,errorGT=0;
		
		//initialize the random number generator
		std::cout<<"Random number generator ("<<seed<<").\n";
		std::srand(seed);
		
		//set nn parameters
		std::cout<<"Setting nn parameters...\n";
		nn.tfType()=NN::TransferN::TANH;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.lambda()=0.0;
		//print the parameters
		std::cout<<"N-batch = "<<NBatch<<"\n";
		std::cout<<"N-train = "<<NTrain<<"\n";
		std::cout<<"N-test  = "<<NTest<<"\n";
		
		//set optimization parameters
		std::cout<<"Setting optimization parameters...\n";
		opt.tol()=1e-14;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.period()=0;
		opt.decay()=0;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.1;
		if(etaG>=0) opt.eta()=etaG; else opt.eta()=0.1;
		opt.algo()=OPT_METHOD::ADADELTA;
		std::cout<<"Opt     = \n"<<opt<<"\n";
		
		/* TEST LINEAR */
		std::cout<<"**** TEST - LINEAR ****\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=xTrain[i][0]*m+b;
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		
		//resizing the network
		std::cout<<"Resizing the network...\n";
		nn.resize(1,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Linear  = ("<<m<<","<<b<<")\n";
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorLV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorLT=error/NTest;
		
		/* TEST QUARTIC */
		std::cout<<"**** TEST - QUARTIC ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Poly = "; for(unsigned int i=0; i<p.size(); ++i) std::cout<<p[i]<<" "; std::cout<<"\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=function::poly(xTrain[i][0],p);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorQV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorQT=error/NTest;
		
		/* TEST GAUSSIAN */
		std::cout<<"**** TEST - GAUSSIAN ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Gaussian = ("<<c<<","<<s<<","<<a<<")\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=gauss(xTrain[i][0]);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorGV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorGT=error/NTest;
		
		std::cout<<"-----------------------------------------------------------\n";
		std::cout<<"------------------------- Summary -------------------------\n";
		std::cout<<"error - lin      = "<<errorLV<<" "<<errorLT<<"\n";
		std::cout<<"error - quartic  = "<<errorQV<<" "<<errorQT<<"\n";
		std::cout<<"error - gaussian = "<<errorGV<<" "<<errorGT<<"\n";
		std::cout<<"-----------------------------------------------------------\n";
	} catch(std::exception& e){
		std::cout<<"ERROR in TEST - ADADELTA\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"******************** TEST - ADADELTA  ********************\n";
	std::cout<<"**********************************************************\n";
	}
	
	if(test_rprop){
	std::cout<<"**********************************************************\n";
	std::cout<<"********************** TEST - RPROP **********************\n";
	try{
		/* local function variables */
		//linear function
			double m=1.624786758;
			double b=2.1875389;
		//quartic function
			std::vector<double> p(5,0);
			p[1]=0.048252;
			p[2]=1.48925;
			p[3]=0.72805290;
			p[4]=0.032052;
		//Gaussian function
			double c=0.92787849;
			double s=1.724687;
			double a=2.81378;
			Gaussian gauss(c,s,a);
		//sampling
			unsigned int NTrain=20;//number of training points
			unsigned int NBatch=NTrain;//batch size
			if(NBatchG>0) NBatch=NBatchG;
			unsigned int NVal=50;//number of test points
			unsigned int NTest=50;//number of test points
			NN::VecList xTrain(NTrain),yTrain(NTrain);//training data
			NN::VecList xVal(NVal),yVal(NVal);//validation data
			NN::VecList xTest(NTest),yTest(NTest);//test data
			double xMin=-5,xMax=5;
		//neural network
			std::vector<unsigned int> nh(1,15);
			NN::Network nn;
		//optimization
			NN::NNOpt nnopt;
			Opt opt;
			double error=0;
			double errorP=0;
		//rand
			unsigned int seed=72867429;
		//error
			double errorLV=0,errorLT=0;
			double errorQV=0,errorQT=0;
			double errorGV=0,errorGT=0;
		
		//initialize the random number generator
		std::cout<<"Random number generator ("<<seed<<").\n";
		std::srand(seed);
		
		//set nn parameters
		std::cout<<"Setting nn parameters...\n";
		nn.tfType()=NN::TransferN::TANH;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.lambda()=0.0;
		//print the parameters
		std::cout<<"N-batch = "<<NBatch<<"\n";
		std::cout<<"N-train = "<<NTrain<<"\n";
		std::cout<<"N-test  = "<<NTest<<"\n";
		
		//set optimization parameters
		std::cout<<"Setting optimization parameters...\n";
		opt.tol()=1e-14;
		opt.maxIter()=100000;
		opt.nPrint()=10000;
		opt.period()=0;
		opt.decay()=0;
		if(gammaG>0) opt.gamma()=gammaG; else opt.gamma()=0.1;
		if(etaG>=0) opt.eta()=etaG; else opt.eta()=0.1;
		opt.algo()=OPT_METHOD::RPROP;
		std::cout<<"Opt     = \n"<<opt<<"\n";
		
		//**** TEST LINEAR ****
		std::cout<<"**** TEST - LINEAR ****\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=xTrain[i][0]*m+b;
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=xTest[i][0]*m+b;
		}
		
		
		//resizing the network
		std::cout<<"Resizing the network...\n";
		nn.resize(1,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Linear  = ("<<m<<","<<b<<")\n";
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorLV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorLT=error/NTest;
		
		//**** TEST QUARTIC ****
		std::cout<<"**** TEST - QUARTIC ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Poly = "; for(unsigned int i=0; i<p.size(); ++i) std::cout<<p[i]<<" "; std::cout<<"\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=function::poly(xTrain[i][0],p);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=function::poly(xTest[i][0],p);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorQV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorQT=error/NTest;
		
		//**** TEST GAUSSIAN ****
		std::cout<<"**** TEST - GAUSSIAN ****\n";
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.resize(1,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the function
		std::cout<<"Gaussian = ("<<c<<","<<s<<","<<a<<")\n";
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=gauss(xTrain[i][0]);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt,NBatch);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		errorGV=error/NTest;
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict  = "<<error/NTest<<"\n";
		std::cout<<"error% - predict  = "<<errorP/NTest<<"\n";
		errorGT=error/NTest;
		
		std::cout<<"-----------------------------------------------------------\n";
		std::cout<<"------------------------- Summary -------------------------\n";
		std::cout<<"error - lin      = "<<errorLV<<" "<<errorLT<<"\n";
		std::cout<<"error - quartic  = "<<errorQV<<" "<<errorQT<<"\n";
		std::cout<<"error - gaussian = "<<errorGV<<" "<<errorGT<<"\n";
		std::cout<<"-----------------------------------------------------------\n";
	} catch(std::exception& e){
		std::cout<<"ERROR in TEST - RPROP\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************** TEST - RPROP **********************\n";
	std::cout<<"**********************************************************\n";
	}
	
	if(test_unit_lin){
	std::cout<<"******************************************************\n";
	std::cout<<"**************** TEST - UNIT - LIN ****************\n";
	try{
		//local function variables
		NN::Network nn;
		
		//resize the nn
		std::cout<<"Resizing the network...\n";
		nn.preCond()=false;
		nn.postCond()=false;
		nn.tfType()=NN::TransferN::TANH;
		nn.resize(1,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//initialize the input nodes
		std::cout<<"Initializing the input nodes...\n";
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=0.1;
		std::cout<<"input = "<<nn.input().transpose()<<"\n";
		std::cout<<"bias = "<<nn.bias(0).transpose()<<"\n";
		std::cout<<"edge = "<<nn.edge(0)<<"\n";
		
		//execute the network
		nn.execute();
		
		//print the output 
		std::cout<<nn.tf(0)(0.5)<<"\n";
		std::cout<<"output - exact = "<<nn.tf(0)(nn.input(0)*nn.edge(0,0,0)+nn.bias(0,0))<<"\n";
		std::cout<<"output - nn = "<<nn.output().transpose()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - UNIT - LIN\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"**************** TEST - UNIT - LIN ****************\n";
	std::cout<<"******************************************************\n";
	}
	
	if(test_unit_nonlin){
	std::cout<<"******************************************************\n";
	std::cout<<"**************** TEST - UNIT - NONLIN ****************\n";
	try{
		//local function variables
		NN::Network nn;
		
		//resize the nn
		std::cout<<"Resizing the network...\n";
		nn.preCond()=false;
		nn.postCond()=false;
		nn.tfType()=NN::TransferN::TANH;
		std::vector<unsigned int> nNodes(1);
		nNodes[0]=5;
		nn.resize(1,nNodes,3);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//initialize the input nodes
		std::cout<<"Initializing the input nodes...\n";
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=0.1;
		std::cout<<"input = "<<nn.input().transpose()<<"\n";
		std::cout<<"bias[0] = "<<nn.bias(0).transpose()<<"\n";
		std::cout<<"edge[0] = "<<nn.edge(0)<<"\n";
		std::cout<<"bias[1] = "<<nn.bias(1).transpose()<<"\n";
		std::cout<<"edge[1] = "<<nn.edge(1)<<"\n";
		
		//execute the network
		nn.execute();
		
		//print the output 
		std::cout<<"output - nn = "<<nn.output().transpose()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - UNIT - NONLIN\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"**************** TEST - UNIT - NONLIN****************\n";
	std::cout<<"******************************************************\n";
	}
	
	if(test_unit_lin_grad){
	std::cout<<"******************************************************\n";
	std::cout<<"**************** TEST - UNIT - LIN - GRAD ****************\n";
	try{
		//local function variables
		NN::Network nn;
		
		//resize the nn
		std::cout<<"Resizing the network...\n";
		nn.preCond()=false;
		nn.postCond()=false;
		nn.tfType()=NN::TransferN::TANH;
		nn.resize(3,2);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//initialize the input nodes
		std::cout<<"Initializing the input nodes...\n";
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=0.1;
		
		//execute the network
		std::cout<<"Executing the network...\n";
		nn.execute();
		
		//print the data
		std::cout<<"input = "<<nn.input().transpose()<<"\n";
		std::cout<<"output = "<<nn.output().transpose()<<"\n";
		std::cout<<"weight = "<<nn.edge(0)<<"\n";
		std::cout<<"bias = "<<nn.bias(0).transpose()<<"\n";
		std::cout<<"output = "<<nn.output().transpose()<<"\n";
		
		//calculate the gradient
		nn.grad_out();
		std::cout<<"dOut = "<<nn.dOut(0)<<"\n";
		
		//print the output 
		std::cout<<"output - exact = "<<nn.tf(0)(nn.input(0)*nn.edge(0,0,0)+nn.bias(0,0))<<"\n";
		std::cout<<"output - nn = "<<nn.output().transpose()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - UNIT - LIN - GRAD\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"**************** TEST - UNIT - LIN - GRAD ****************\n";
	std::cout<<"******************************************************\n";
	}
	
	if(test_unit_lin_grad_scale){
	std::cout<<"******************************************************\n";
	std::cout<<"**************** TEST - UNIT - LIN - GRAD - SCALE ****************\n";
	try{
		//local function variables
		NN::Network nn;
		
		//resize the nn
		std::cout<<"Resizing the network...\n";
		nn.preCond()=false;
		nn.postCond()=false;
		nn.tfType()=NN::TransferN::TANH;
		nn.resize(1,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//initialize the input nodes
		std::cout<<"Initializing the input nodes...\n";
		std::srand(std::time(NULL));
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preScale(i)=1.0*std::rand()/RAND_MAX;
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preBias(i)=1.0*std::rand()/RAND_MAX;
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=1;
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postScale(i)=1.0*std::rand()/RAND_MAX;
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postBias(i)=1.0*std::rand()/RAND_MAX;
		
		//execute the network
		std::cout<<"Executing the network...\n";
		nn.execute();
		
		//print the data
		std::cout<<"prescale = "<<nn.preScale().transpose()<<"\n";
		std::cout<<"input = "<<nn.input().transpose()<<"\n";
		std::cout<<"output = "<<nn.output().transpose()<<"\n";
		std::cout<<"weight = "<<nn.edge(0)<<"\n";
		std::cout<<"bias = "<<nn.bias(0).transpose()<<"\n";
		std::cout<<"output = "<<nn.output().transpose()<<"\n";
		std::cout<<"postscale = "<<nn.postScale().transpose()<<"\n";
		
		//calculate the gradient
		nn.grad_out();
		std::cout<<"dOut[0] = "<<nn.dOut(0)<<"\n";
		std::cout<<"dOut[1] = "<<nn.dOut(1)<<"\n";
		std::cout<<"dOut[2] = "<<nn.dOut(2)<<"\n";
		
		//print the output 
		std::cout<<"output - exact = "<<nn.tf(0)(nn.input(0)*nn.edge(0,0,0)+nn.bias(0,0))<<"\n";
		std::cout<<"output - nn = "<<nn.output().transpose()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - UNIT - LIN - GRAD - SCALE \n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"**************** TEST - UNIT - LIN - GRAD - SCALE ****************\n";
	std::cout<<"******************************************************\n";
	}
	
	if(test_unit_nonlin_grad){
	std::cout<<"******************************************************\n";
	std::cout<<"**************** TEST - UNIT - NONLIN - GRAD ****************\n";
	try{
		//local function variables
		NN::Network nn;
		
		//resize the nn
		std::cout<<"Resizing the network...\n";
		Eigen::MatrixXd dOutExact;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.tfType()=NN::TransferN::LINEAR;
		std::vector<unsigned int> nNodes(1);
		nNodes[0]=7;
		nn.resize(3,nNodes,5);
		nn.bInit()=0;
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//initialize the input nodes
		std::cout<<"Initializing the input nodes...\n";
		Eigen::VectorXd input=Eigen::VectorXd::Constant(nn.nInput(),1);
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=input[i];
		//execute the network
		std::cout<<"Executing the network...\n";
		nn.execute();
		nn.grad_out();
		
		//print the data
		std::cout<<"input = "<<nn.input().transpose()<<"\n";
		std::cout<<"bias[0] = "<<nn.bias(0).transpose()<<"\n";
		std::cout<<"edge[0] = \n"<<nn.edge(0)<<"\n";
		std::cout<<"bias[1] = "<<nn.bias(1).transpose()<<"\n";
		std::cout<<"edge[1] = \n"<<nn.edge(1)<<"\n";
		std::cout<<"output = "<<nn.output().transpose()<<"\n";
		
		std::cout<<"grad[0] = \n"<<nn.dOut(0)<<"\n";
		std::cout<<"grad[1] = \n"<<nn.dOut(1)<<"\n";
		
		dOutExact=nn.dOut(0);
		
		std::cout<<"Testing the gradient...\n";
		Eigen::VectorXd delta=Eigen::VectorXd::Random(nn.nInput());
		Eigen::VectorXd outOld=nn.output();
		Eigen::VectorXd outNew=nn.output();
		std::cout<<"delta = "<<delta.transpose()<<"\n";
		for(unsigned int n=0; n<nn.nInput(); ++n){
			for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=input[i];
			nn.input(n)+=delta[n];
			nn.execute();
			outNew=nn.output();
			Eigen::MatrixXd mat=Eigen::MatrixXd::Zero(nn.output().size(),nn.input().size());
			for(unsigned int i=0; i<nn.output().size(); ++i){
				for(unsigned int j=0; j<nn.input().size(); ++j){
					mat(i,j)=(outNew[i]-outOld[i])/delta[j];
				}
			}
			dOutExact.col(n)=mat.col(n);
		}
		
		std::cout<<"dOut - Exact = \n"<<dOutExact<<"\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - UNIT - NONLIN - GRAD \n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"**************** TEST - UNIT - NONLIN - GRAD ****************\n";
	std::cout<<"******************************************************\n";
	}
	
	if(test_unit_nonlin_grad_scale){
	std::cout<<"******************************************************\n";
	std::cout<<"**************** TEST - UNIT - NONLIN - GRAD ****************\n";
	try{
		//local function variables
		NN::Network nn;
		
		//resize the nn
		std::cout<<"Resizing the network...\n";
		Eigen::MatrixXd dOutExact;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.tfType()=NN::TransferN::LINEAR;
		std::vector<unsigned int> nNodes(1);
		nNodes[0]=7;
		nn.resize(3,nNodes,5);
		nn.bInit()=0;
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//set the scaling
		std::cout<<"Setting the pre- and post-conditioning vectors...\n";
		std::srand(std::time(NULL));
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preScale(i)=1.0*std::rand()/RAND_MAX;
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preBias(i)=1.0*std::rand()/RAND_MAX;
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postScale(i)=1.0*std::rand()/RAND_MAX;
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postBias(i)=1.0*std::rand()/RAND_MAX;
		
		//initialize the input nodes
		std::cout<<"Initializing the input nodes...\n";
		Eigen::VectorXd input=Eigen::VectorXd::Constant(nn.nInput(),1);
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=input[i];
		//execute the network
		std::cout<<"Executing the network...\n";
		nn.execute();
		nn.grad_out();
		
		//print the data
		std::cout<<"pre-scale = "<<nn.preScale().transpose()<<"\n";
		std::cout<<"pre-bias = "<<nn.preScale().transpose()<<"\n";
		std::cout<<"input = "<<nn.input().transpose()<<"\n";
		std::cout<<"bias[0] = "<<nn.bias(0).transpose()<<"\n";
		std::cout<<"edge[0] = \n"<<nn.edge(0)<<"\n";
		std::cout<<"bias[1] = "<<nn.bias(1).transpose()<<"\n";
		std::cout<<"edge[1] = \n"<<nn.edge(1)<<"\n";
		std::cout<<"output = "<<nn.output().transpose()<<"\n";
		std::cout<<"post-scale = "<<nn.postScale().transpose()<<"\n";
		std::cout<<"post-bias = "<<nn.postScale().transpose()<<"\n";
		
		std::cout<<"grad[0] = \n"<<nn.dOut(0)<<"\n";
		std::cout<<"grad[1] = \n"<<nn.dOut(1)<<"\n";
		std::cout<<"grad[2] = \n"<<nn.dOut(2)<<"\n";
		std::cout<<"grad[3] = \n"<<nn.dOut(3)<<"\n";
		
		dOutExact=nn.dOut(0);
		
		std::cout<<"Testing the gradient...\n";
		Eigen::VectorXd delta=Eigen::VectorXd::Random(nn.nInput());
		Eigen::VectorXd outOld=nn.output();
		Eigen::VectorXd outNew=nn.output();
		std::cout<<"delta = "<<delta.transpose()<<"\n";
		for(unsigned int n=0; n<nn.nInput(); ++n){
			for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=input[i];
			nn.input(n)+=delta[n];
			nn.execute();
			outNew=nn.output();
			Eigen::MatrixXd mat=Eigen::MatrixXd::Zero(nn.output().size(),nn.input().size());
			for(unsigned int i=0; i<nn.output().size(); ++i){
				for(unsigned int j=0; j<nn.input().size(); ++j){
					mat(i,j)=(outNew[i]-outOld[i])/delta[j];
				}
			}
			dOutExact.col(n)=mat.col(n);
		}
		
		std::cout<<"dOut - Exact = \n"<<dOutExact<<"\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - UNIT - NONLIN - GRAD \n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"**************** TEST - UNIT - NONLIN - GRAD ****************\n";
	std::cout<<"******************************************************\n";
	}
	
	if(test_timing){
	std::cout<<"******************************************************\n";
	std::cout<<"**************** TEST - TIMING ****************\n";
	try{
		//local function variables
		//nn
			std::vector<unsigned int> nh(2,15);
			NN::Network nn;
		//timing
			unsigned int N=500000;
			std::chrono::high_resolution_clock::time_point start;
			std::chrono::high_resolution_clock::time_point stop;
			std::chrono::duration<double> time;
		
		//resize the nn
		std::cout<<"Resizing the network...\n";
		nn.preCond()=false;
		nn.postCond()=false;
		nn.tfType()=NN::TransferN::TANH;
		nn.resize(3,nh,4);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//initialize the input nodes
		std::cout<<"Initializing the input nodes...\n";
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=0.1;
		
		//execute the network
		start=std::chrono::high_resolution_clock::now();
		for(unsigned int i=0; i<N; ++i) nn.execute();
		stop=std::chrono::high_resolution_clock::now();
		time=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		std::cout<<"N "<<N<<" time - execution = "<<time.count()<<"\n";
		
		//calculate gradient
		Eigen::VectorXd output=Eigen::VectorXd::Constant(4,0.8);
		Eigen::VectorXd grad=Eigen::VectorXd::Constant(nn.size(),0.0);
		start=std::chrono::high_resolution_clock::now();
		for(unsigned int i=0; i<N; ++i) nn.error(output,grad);
		stop=std::chrono::high_resolution_clock::now();
		time=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		std::cout<<"N "<<N<<" time - grad      = "<<time.count()<<"\n";
		
		//calculate grad_out
		start=std::chrono::high_resolution_clock::now();
		for(unsigned int i=0; i<N; ++i) nn.grad_out();
		stop=std::chrono::high_resolution_clock::now();
		time=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		std::cout<<"N "<<N<<" time - grad-out  = "<<time.count()<<"\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - TIMING\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"**************** TEST - TIMING ****************\n";
	std::cout<<"******************************************************\n";
	}
	
	if(test_fit_func_gaussian2d){
	std::cout<<"******************************************************\n";
	std::cout<<"**************** TEST - FIT - FUNC - GAUSSIAN ****************\n";
	try{
		/* local function variables */
		//Gaussian function
			double cx=0.92787849,cy=0.8762889;
			double sx=1.724687,sy=1.95869217;
			double ax=2.81378,ay=1.7457821;
			Gaussian gaussx(cx,sx,ax);
			Gaussian gaussy(cy,sy,ay);
		//sampling
			unsigned int NBatch=10;//batch size
			unsigned int NLine=10;//number of points on each side of the square
			unsigned int NTrain=NLine*NLine;//number of training points
			unsigned int NVal=50;//number of test points
			unsigned int NTest=50;//number of test points
			NN::VecList xTrain(NTrain),yTrain(NTrain);//training data
			NN::VecList xVal(NVal),yVal(NVal);//validation data
			NN::VecList xTest(NTest),yTest(NTest);//test data
			double xMin=-5,xMax=5;
			unsigned int count;
		//neural network
			std::vector<unsigned int> nh(1,15);
			NN::Network nn;
		//optimization
			NN::NNOpt nnopt;
			Opt opt;
			double error=0;
			double errorP=0;
		//rand
			unsigned int seed=72867429;
		
		//initialize the random number generator
		std::cout<<"Random number generator ("<<seed<<").\n";
		std::srand(seed);
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		count=0;
		for(unsigned int i=0; i<NLine; ++i){
			for(unsigned int j=0; j<NLine; ++j){
				xTrain[count].resize(2); yTrain[count].resize(1);
				xTrain[count][0]=xMin+(xMax-xMin)*((double)i)/NLine;
				xTrain[count][1]=xMin+(xMax-xMin)*((double)j)/NLine;
				yTrain[count][0]=gaussx(xTrain[count][0])*gaussy(xTrain[count][1]);
				++count;
			}
		}
		
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(2); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			xTest[i][1]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gaussx(xTest[i][0])*gaussy(xTest[i][1]);
		}
		
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(2); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			xTest[i][1]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gaussx(xTest[i][0])*gaussy(xTest[i][1]);
		}
		
		/* Optimization - BFGSG */
		
		//set the optimization object
		std::cout<<"Setting optimization object...\n";
		opt.algo()=OPT_METHOD::BFGSG;
		opt.tol()=1e-14;
		opt.maxIter()=1000000;
		opt.nPrint()=100000;
		opt.gamma()=0.01;
		opt.period()=0;
		opt.decay()=0;
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.tfType()=NN::TransferN::TANH;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.lambda()=0.0;
		nn.resize(2,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the parameters
		std::cout<<"N-batch = "<<NBatch<<"\n";
		std::cout<<"N-train = "<<NTrain<<"\n";
		std::cout<<"N-test= "<<NTest<<"\n";
		std::cout<<"x-lim = ("<<xMin<<","<<xMax<<")\n";
		std::cout<<"GaussianX = ("<<cx<<","<<sx<<","<<ax<<")\n";
		std::cout<<"GaussianY = ("<<cy<<","<<sy<<","<<ay<<")\n";
		std::cout<<"Opt = \n"<<opt<<"\n";
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.input(1)=xTrain[i][1];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.input(1)=xTest[i][1];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict = "<<error/NTest<<"\n";
		std::cout<<"error% - predict = "<<errorP/NTest<<"\n";
		double error_BFGS=error/NTest;
		
		/* Optimization - SDG */
		
		//set the optimization object
		std::cout<<"Setting optimization object...\n";
		opt.algo()=OPT_METHOD::SDG;
		opt.tol()=1e-14;
		opt.maxIter()=1000000;
		opt.nPrint()=100000;
		opt.period()=0;
		opt.decay()=0;
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.tfType()=NN::TransferN::TANH;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.lambda()=0.0;
		nn.resize(2,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the parameters
		std::cout<<"N-batch = "<<NBatch<<"\n";
		std::cout<<"N-train = "<<NTrain<<"\n";
		std::cout<<"N-test= "<<NTest<<"\n";
		std::cout<<"x-lim = ("<<xMin<<","<<xMax<<")\n";
		std::cout<<"GaussianX = ("<<cx<<","<<sx<<","<<ax<<")\n";
		std::cout<<"GaussianY = ("<<cy<<","<<sy<<","<<ay<<")\n";
		std::cout<<"Opt = \n"<<opt<<"\n";
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.input(1)=xTrain[i][1];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.input(1)=xTest[i][1];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict = "<<error/NTest<<"\n";
		std::cout<<"error% - predict = "<<errorP/NTest<<"\n";
		double error_SDG=error/NTest;
		
		/* Optimization - RPROP */
		
		//set the optimization object
		std::cout<<"Setting optimization object...\n";
		opt.algo()=OPT_METHOD::RPROP;
		opt.tol()=1e-14;
		opt.maxIter()=1000000;
		opt.nPrint()=100000;
		opt.period()=0;
		opt.decay()=0;
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.tfType()=NN::TransferN::TANH;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.lambda()=0.0;
		nn.resize(2,nh,1);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the parameters
		std::cout<<"N-batch = "<<NBatch<<"\n";
		std::cout<<"N-train = "<<NTrain<<"\n";
		std::cout<<"N-test= "<<NTest<<"\n";
		std::cout<<"x-lim = ("<<xMin<<","<<xMax<<")\n";
		std::cout<<"GaussianX = ("<<cx<<","<<sx<<","<<ax<<")\n";
		std::cout<<"GaussianY = ("<<cy<<","<<sy<<","<<ay<<")\n";
		std::cout<<"Opt = \n"<<opt<<"\n";
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		nnopt.train(nn,xTrain,yTrain,opt);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.input(1)=xTrain[i][1];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.input(1)=xTest[i][1];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict = "<<error/NTest<<"\n";
		std::cout<<"error% - predict = "<<errorP/NTest<<"\n";
		double error_RPROP=error/NTest;
		
		std::cout<<"--- SUMMARY ---\n";
		std::cout<<"error - BFGS = "<<error_BFGS<<"\n";
		std::cout<<"error - SDG = "<<error_SDG<<"\n";
		std::cout<<"error - RPROP = "<<error_RPROP<<"\n";
		
	} catch(std::exception& e){
		std::cout<<"ERROR in TEST - FIT - FUNC - GAUSSIAN2D\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"**************** TEST - FIT - FUNC - GAUSSIAN2D ****************\n";
	std::cout<<"******************************************************\n";
	}
	
	if(test_io){
	std::cout<<"******************************************************\n";
	std::cout<<"**************** TEST - IO ****************\n";
	try{
		/* local function variables */
		//Gaussian function
			double c=0.92787849;
			double s=1.724687;
			double a=2.81378;
			Gaussian gauss(c,s,a);
		//sampling
			unsigned int NBatch=10;//batch size
			unsigned int NTrain=20;//number of training points
			unsigned int NVal=50;//number of test points
			unsigned int NTest=50;//number of test points
			NN::VecList xTrain(NTrain),yTrain(NTrain);//training data
			NN::VecList xVal(NVal),yVal(NVal);//validation data
			NN::VecList xTest(NTest),yTest(NTest);//test data
			double xMin=-5,xMax=5;
		//neural network
			std::vector<unsigned int> nh(1,15);
			NN::Network nn;
		//optimization
			NN::NNOpt nnopt;
			Opt opt;
			double error=0;
			double errorP=0;
		//i/o
			const char* file="nn_test.dat";
			const char* filecopy="nn_test_copy.dat";
		//rand
			unsigned int seed=72867429;
		
		//initialize the random number generator
		std::cout<<"Random number generator ("<<seed<<").\n";
		std::srand(seed);
		
		//generate the x and y reference data - uniform
		std::cout<<"Generating uniform data...\n";
		for(unsigned int i=0; i<NTrain; ++i){
			xTrain[i].resize(1); yTrain[i].resize(1);
			xTrain[i][0]=xMin+(xMax-xMin)*((double)i)/NTrain;
			yTrain[i][0]=gauss(xTrain[i][0]);
		}
		//generate the x and y validation data
		std::cout<<"Generating validation data...\n";
		for(unsigned int i=0; i<NVal; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		//generate the x and y test data
		std::cout<<"Generating test data...\n";
		for(unsigned int i=0; i<NTest; ++i){
			xTest[i].resize(1); yTest[i].resize(1);
			xTest[i][0]=xMin+(xMax-xMin)*((double)std::rand())/RAND_MAX;
			yTest[i][0]=gauss(xTest[i][0]);
		}
		
		//set the optimization object
		std::cout<<"Setting optimization object...\n";
		opt.algo()=OPT_METHOD::RPROP;
		opt.tol()=1e-14;
		opt.maxIter()=1;
		opt.nPrint()=1;
		opt.period()=0;
		opt.decay()=0;
		
		//resize the nn
		std::cout<<"Resizing the nn...\n";
		nn.tfType()=NN::TransferN::TANH;
		nn.preCond()=false;
		nn.postCond()=false;
		nn.lambda()=0.0;
		nn.resize(3,nh,5);
		std::cout<<"nn = \n"<<nn<<"\n";
		
		//print the parameters
		std::cout<<"N-batch = "<<NBatch<<"\n";
		std::cout<<"N-train = "<<NTrain<<"\n";
		std::cout<<"N-test= "<<NTest<<"\n";
		std::cout<<"x-lim = ("<<xMin<<","<<xMax<<")\n";
		std::cout<<"Gaussian = ("<<c<<","<<s<<","<<a<<")\n";
		std::cout<<"Opt = \n"<<opt<<"\n";
		
		//fit the training data
		std::cout<<"Fitting the training data...\n";
		//nnopt.train(nn,xTrain,yTrain,opt);
		
		//vaslidate the network
		std::cout<<"Validating the network:\n";
		error=0; errorP=0;
		for(unsigned int i=0; i<NTrain; ++i){
			nn.input(0)=xTrain[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTrain[i][0]);
			errorP+=std::fabs((y-yTrain[i][0])/yTrain[i][0]);
		}
		std::cout<<"error  - validate = "<<error/NTrain<<"\n";
		std::cout<<"error% - validate = "<<errorP/NTrain<<"\n";
		
		//test the network
		error=0; errorP=0;
		for(unsigned int i=0; i<NTest; ++i){
			nn.input(0)=xTest[i][0];
			nn.execute();
			double y=nn.output(0);
			error+=std::fabs(y-yTest[i][0]);
			errorP+=std::fabs((y-yTest[i][0])/yTest[i][0]);
		}
		std::cout<<"error  - predict = "<<error/NTest<<"\n";
		std::cout<<"error% - predict = "<<errorP/NTest<<"\n";
		double error_RPROP=error/NTest;
		
		//write the network
		std::cout<<"Printing the network...\n";
		NN::Network::write(file,nn);
		
		//read the network
		std::cout<<"Loading the network...\n";
		nn.clear();
		NN::Network::read(file,nn);
		
		//write the network again
		std::cout<<"Printing the network again...\n";
		NN::Network::write(filecopy,nn);
		
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - IO\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"**************** TEST - IO ****************\n";
	std::cout<<"******************************************************\n";
	}
}