//c libraries
#include <cstdlib>
#include <ctime>
//c++ libraries
#include <iostream>
//eigen 
#include <Eigen/Dense>
//local libraries
#include "optimize.hpp"
#include "input.hpp"

struct Rosen{
	double a,b;
	Rosen():a(1.0),b(100.0){};
	Rosen(double aa, double bb):a(aa),b(bb){};
	double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& g){
		g[0]=-2.0*(a-x[0])-4.0*b*x[0]*(x[1]-x[0]*x[0]);
		g[1]=2.0*b*(x[1]-x[0]*x[0]);
		return (a-x[0])*(a-x[0])+b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
	};
};

int main(int argc, char* argv[]){
	//test flags
	bool test_opt_rosen=true;
	bool test_serialize=false;
	
	//global variables
	Input input;
	std::string str;
	int seed=-1;
	
	//set input flags
	input.flags().push_back("-algo");//optimization algorithm
	input.flags().push_back("-opt");//optimization value
	input.flags().push_back("-nm");//max number of steps
	input.flags().push_back("-np");//print np steps
	input.flags().push_back("-tol");//tolerance
	input.flags().push_back("-seed");//tolerance
	input.flags().push_back("-gamma");//gamma
	input.flags().push_back("-lambda");//lambda
	
	//parse input
	input.parse(argc,argv);
	
	//set random number generator
	if(input.get("-seed",str).size()>0) seed=std::atoi(str.c_str());
	if(seed<0) std::srand(std::time(NULL));
	else std::srand(seed);
	
	if(test_opt_rosen){
	std::cout<<"***********************************************\n";
	std::cout<<"================ TEST - ROSEN ================\n";
	try{
		//local variables
		Opt::ALGO::type algo=Opt::ALGO::SGD;
		Opt::VAL::type val=Opt::VAL::FTOL_ABS;
		const unsigned dim=2;
		Opt::Model* model=NULL;
		Opt::Data data(dim);
		data.max()=1000;
		data.nPrint()=100;
		data.nWrite()=100;
		data.tol()=1e-6;
		double gamma=1e-4;
		double eta=0.9;
		double lambda=1e-4;
		Rosen rosen;
		
		//read inputs
		std::cout<<"reading inputs\n";
		if(input.get("-algo",str).size()>0) data.algo()=Opt::ALGO::read(str.c_str());
		if(input.get("-opt",str).size()>0) data.optVal()=Opt::VAL::read(str.c_str());
		if(input.get("-nm",str).size()>0) data.max()=std::atoi(str.c_str());
		if(input.get("-np",str).size()>0) data.nPrint()=std::atoi(str.c_str());
		if(input.get("-tol",str).size()>0) data.tol()=std::atof(str.c_str());
		if(input.get("-gamma",str).size()>0) gamma=std::atof(str.c_str());
		if(input.get("-lambda",str).size()>0) lambda=std::atof(str.c_str());
		if(input.get("-eta",str).size()>0) eta=std::atof(str.c_str());
		data.nWrite()=data.nPrint();
		
		//check inputs
		std::cout<<"checking inputs\n";
		if(data.algo()==Opt::ALGO::UNKNOWN) throw std::invalid_argument("Invalid algorithm.");
		if(data.optVal()==Opt::VAL::UNKNOWN) throw std::invalid_argument("Invalid opt val.");
		if(data.max()<=0) throw std::invalid_argument("Invalid nm");
		if(data.nPrint()<=0) throw std::invalid_argument("Invalid nm");
		if(data.tol()<0) throw std::invalid_argument("Invalid tol");
		
		data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
		data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
		data.pOld()=data.p();
		
		//print parameters
		std::cout<<data<<"\n";
		std::cout<<"================\n";
		std::cout<<"X0     = "<<data.p().transpose()<<"\n";
		std::cout<<"GAMMA  = "<<gamma<<"\n";
		std::cout<<"LAMBDA = "<<lambda<<"\n";
		std::cout<<"ETA    = "<<eta<<"\n";
		std::cout<<"================\n";
		
		//make opt object
		std::cout<<"initializing opt\n";
		if(data.algo()==Opt::ALGO::SGD){
			model=new Opt::SGD(data.dim());
			static_cast<Opt::SGD*>(model)->gamma()=gamma;
		} else if(data.algo()==Opt::ALGO::SDM){
			model=new Opt::SDM(data.dim());
			static_cast<Opt::SDM*>(model)->gamma()=gamma;
			static_cast<Opt::SDM*>(model)->eta()=eta;
		} else if(data.algo()==Opt::ALGO::NAG){
			model=new Opt::NAG(data.dim());
			static_cast<Opt::NAG*>(model)->gamma()=gamma;
			static_cast<Opt::NAG*>(model)->eta()=eta;
		} else if(data.algo()==Opt::ALGO::ADAGRAD){
			model=new Opt::ADAGRAD(data.dim());
			static_cast<Opt::ADAGRAD*>(model)->gamma()=gamma;
		} else if(data.algo()==Opt::ALGO::ADADELTA){
			model=new Opt::ADADELTA(data.dim());
			static_cast<Opt::ADADELTA*>(model)->gamma()=gamma;
			static_cast<Opt::ADADELTA*>(model)->eta()=0.9;
		} else if(data.algo()==Opt::ALGO::ADAM){
			model=new Opt::ADAM(data.dim());
			static_cast<Opt::ADAM*>(model)->gamma()=gamma;
		} else if(data.algo()==Opt::ALGO::RMSPROP){
			model=new Opt::RMSPROP(data.dim());
			static_cast<Opt::RMSPROP*>(model)->gamma()=gamma;
		} else if(data.algo()==Opt::ALGO::BFGS){
			model=new Opt::BFGS(data.dim());
			static_cast<Opt::BFGS*>(model)->gamma()=gamma;
		} else if(data.algo()==Opt::ALGO::RPROP){
			model=new Opt::RPROP(data.dim());
		} else if(data.algo()==Opt::ALGO::NADAM){
			model=new Opt::NADAM(data.dim());
			static_cast<Opt::NADAM*>(model)->gamma()=gamma;
		} 
		
		switch(data.algo()){
			case Opt::ALGO::SGD: std::cout<<static_cast<Opt::SGD&>(*model)<<"\n"; break;
			case Opt::ALGO::SDM: std::cout<<static_cast<Opt::SDM&>(*model)<<"\n"; break;
			case Opt::ALGO::NAG: std::cout<<static_cast<Opt::NAG&>(*model)<<"\n"; break;
			case Opt::ALGO::ADAGRAD: std::cout<<static_cast<Opt::ADAGRAD&>(*model)<<"\n"; break;
			case Opt::ALGO::ADADELTA: std::cout<<static_cast<Opt::ADADELTA&>(*model)<<"\n"; break;
			case Opt::ALGO::RMSPROP: std::cout<<static_cast<Opt::RMSPROP&>(*model)<<"\n"; break;
			case Opt::ALGO::ADAM: std::cout<<static_cast<Opt::ADAM&>(*model)<<"\n"; break;
			case Opt::ALGO::BFGS: std::cout<<static_cast<Opt::BFGS&>(*model)<<"\n"; break;
			case Opt::ALGO::RPROP: std::cout<<static_cast<Opt::RPROP&>(*model)<<"\n"; break;
			case Opt::ALGO::NADAM: std::cout<<static_cast<Opt::NADAM&>(*model)<<"\n"; break;
			default: std::cout<<"ERROR: Unrecognized optimization method.\n"; break;
		}
		
		//optimize
		std::cout<<"executing optimization\n";
		for(unsigned int i=0; i<data.max(); ++i){
			//compute the value and gradient
			data.val()=rosen(data.p(),data.g());
			//compute the new position
			model->step(data);
			//calculate the difference
			data.dv()=std::fabs(data.val()-data.valOld());
			data.dp()=(data.p()-data.pOld()).norm();
			//check the break condition
			if(data.optVal()==Opt::VAL::FTOL_REL && data.dv()<data.tol()) break;
			else if(data.optVal()==Opt::VAL::XTOL_REL && data.dp()<data.tol()) break;
			else if(data.optVal()==Opt::VAL::FTOL_ABS && data.val()<data.tol()) break;
			//print the status
			if(i%data.nPrint()==0) std::cout<<"opt step "<<data.step()<<" val "<<data.val()<<" dv "<<data.dv()<<" dp "<<data.dp()<<"\n";
			data.pOld()=data.p();//set "old" p value
			data.gOld()=data.g();//set "old" g value
			data.valOld()=data.val();//set "old" value
			//update step
			++data.step();
		}
		
		//print the results
		std::cout<<"opt - n_steps = "<<data.step()<<"\n";
		std::cout<<"opt - val     = "<<data.val()<<"\n";
		std::cout<<"opt - x       = "<<data.p().transpose()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - ROSEN:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"================ TEST - ROSEN ================\n";
	std::cout<<"***********************************************\n";
	}
	
	if(test_serialize){
	std::cout<<"***********************************************\n";
	std::cout<<"============== TEST - SERIALIZE ==============\n";
	try{
		char* arr;
		unsigned int size;
		const unsigned int dim=7;
		//SGD
			Opt::SGD sgd1,sdg2;
			sgd1.gamma()=1.0*std::rand()/RAND_MAX;
			sgd1.init(dim);
			size=serialize::nbytes(sgd1);
			arr=new char[size];
			serialize::pack(sgd1,arr);
			serialize::unpack(sdg2,arr);
			std::cout<<"SGD      = "<<(sgd1==sdg2)<<"\n";
			delete[] arr;
		//SDM
			Opt::SDM sdm1,sdm2;
			sdm1.gamma()=1.0*std::rand()/RAND_MAX;
			sdm1.eta()=1.0*std::rand()/RAND_MAX;
			sdm1.init(dim);
			size=serialize::nbytes(sdm1);
			arr=new char[size];
			serialize::pack(sdm1,arr);
			serialize::unpack(sdm2,arr);
			std::cout<<"SDM      = "<<(sdm1==sdm2)<<"\n";
			delete[] arr;
		//NAG
			Opt::NAG nag1,nag2;
			nag1.gamma()=1.0*std::rand()/RAND_MAX;
			nag1.eta()=1.0*std::rand()/RAND_MAX;
			nag1.init(dim);
			size=serialize::nbytes(nag1);
			arr=new char[size];
			serialize::pack(nag1,arr);
			serialize::unpack(nag2,arr);
			std::cout<<"NAG      = "<<(nag1==nag2)<<"\n";
			delete[] arr;
		//ADAGRAD
			Opt::ADAGRAD adagrad1,adagrad2;
			adagrad1.gamma()=1.0*std::rand()/RAND_MAX;
			adagrad1.init(dim);
			size=serialize::nbytes(adagrad1);
			arr=new char[size];
			serialize::pack(adagrad1,arr);
			serialize::unpack(adagrad2,arr);
			std::cout<<"ADAGRAD  = "<<(adagrad1==adagrad2)<<"\n";
			delete[] arr;
		//ADADELTA
			Opt::ADADELTA adadelta1,adadelta2;
			adadelta1.gamma()=1.0*std::rand()/RAND_MAX;
			adadelta1.eta()=1.0*std::rand()/RAND_MAX;
			adadelta1.init(dim);
			size=serialize::nbytes(adadelta1);
			arr=new char[size];
			serialize::pack(adadelta1,arr);
			serialize::unpack(adadelta2,arr);
			std::cout<<"ADADELTA = "<<(adadelta1==adadelta2)<<"\n";
			delete[] arr;
		//RMSPROP
			Opt::RMSPROP rmsprop1,rmsprop2;
			rmsprop1.gamma()=1.0*std::rand()/RAND_MAX;
			rmsprop1.init(dim);
			size=serialize::nbytes(rmsprop1);
			arr=new char[size];
			serialize::pack(rmsprop1,arr);
			serialize::unpack(rmsprop2,arr);
			std::cout<<"RMSPROP  = "<<(rmsprop1==rmsprop2)<<"\n";
			delete[] arr;
		//ADAM
			Opt::ADAM adam1,adam2;
			adam1.gamma()=1.0*std::rand()/RAND_MAX;
			adam1.init(dim);
			size=serialize::nbytes(adam1);
			arr=new char[size];
			serialize::pack(adam1,arr);
			serialize::unpack(adam2,arr);
			std::cout<<"ADAM     = "<<(adam1==adam2)<<"\n";
			delete[] arr;
		//RPROP
			Opt::RPROP rprop1,rprop2;
			rprop1.init(dim);
			size=serialize::nbytes(rprop1);
			arr=new char[size];
			serialize::pack(rprop1,arr);
			serialize::unpack(rprop2,arr);
			std::cout<<"RPROP    = "<<(rprop1==rprop2)<<"\n";
			delete[] arr;
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SERIALIZE:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"============== TEST - SERIALIZE ==============\n";
	std::cout<<"***********************************************\n";
	}
}
