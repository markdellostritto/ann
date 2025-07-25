// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// eigen libraries
#include <Eigen/Dense>
// optimization
#include "opt/objective.hpp"
#include "opt/iter.hpp"
#include "opt/algo.hpp"
#include "opt/stop.hpp"
// string
#include "str/print.hpp"

const double g_max=1000000;
const double g_nprint=100;
const opt::Stop g_stop=opt::Stop::FABS;
const double g_tol=1e-12;

//**********************************************************************
// Rosenberg function
//**********************************************************************

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

void opt_rosen(opt::algo::Name name){
	//rosenberg function
	Rosen rosen;
	//objective
	opt::Objective obj;
	obj.resize(2);
	obj.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	obj.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	Eigen::VectorXd beg=obj.p();
	obj.pOld()=obj.p();
	obj.gamma()=1e-3;
	//Iterator
	opt::Iterator iter;
	iter.max()=g_max;
	iter.nPrint()=g_nprint;
	iter.stop()=g_stop;
	iter.tol()=g_tol;
	//algorithm
	std::shared_ptr<opt::algo::Base> algo;
	opt::algo::make(algo,name);
	algo->resize(2);
	std::cout<<algo<<"\n";
	char* memarr=new char[serialize::nbytes(algo)];
	serialize::pack(algo,memarr);
	std::shared_ptr<opt::algo::Base> algonew;
	serialize::unpack(algonew,memarr);
	//std::cout<<algonew<<"\n";
	delete[] memarr;
	//optimize
	bool fbreak=false;
	for(int i=0; i<iter.max(); ++i){
		//compute the value and gradient
		obj.val()=rosen(obj.p(),obj.g());
		//compute the new position
		algo->step(obj);
		//calculate the difference
		obj.dv()=std::fabs(obj.val()-obj.valOld());
		obj.dp()=(obj.p()-obj.pOld()).norm();
		//check the break condition
		switch(iter.stop()){
			case opt::Stop::FREL: fbreak=obj.dv()<iter.tol(); break;
			case opt::Stop::XREL: fbreak=obj.dp()<iter.tol(); break;
			case opt::Stop::FABS: fbreak=obj.val()<iter.tol(); break;
		}
		if(fbreak) break;
		//print the status
		obj.pOld().noalias()=obj.p();//set "old" p value
		obj.gOld().noalias()=obj.g();//set "old" g value
		obj.valOld()=obj.val();//set "old" value
		//update step
		++iter.step();
	}
	//print
	std::cout<<"n_steps = "<<iter.step()<<"\n";
	std::cout<<"val     = "<<obj.val()<<"\n";
	std::cout<<"x - beg = "<<beg.transpose()<<"\n";
	std::cout<<"x - end = "<<obj.p().transpose()<<"\n";
}

//**********************************************************************
// main
//**********************************************************************

int main(int argc, char* argv[]){
	
	//==== random ====
	std::srand(std::time(NULL));
	
	//==== optimizers ====
	const int DIM=2;
	char* str=new char[print::len_buf];
	
	//==== global parameters ====
	std::cout<<"GLOBAL PARAMETERS\n";
	std::cout<<"NMAX = "<<g_max<<"\n";
	std::cout<<"STOP = "<<g_stop<<"\n";
	std::cout<<"TOL  = "<<g_tol<<"\n";
	std::cout<<"ROSENBROCK FUNCTION\n";
	std::cout<<"VAL - OPT = 0.0\n";
	std::cout<<"POS - OPT = 1.0 1.0\n";
	
	//==== sgd ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - SGD\n";
	opt_rosen(opt::algo::Name::SGD);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== sdm ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - SDM\n";
	opt_rosen(opt::algo::Name::SDM);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== nag ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - NAG\n";
	opt_rosen(opt::algo::Name::NAG);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== adam ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - ADAM\n";
	opt_rosen(opt::algo::Name::ADAM);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== adamw ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - ADAMW\n";
	opt_rosen(opt::algo::Name::ADAMW);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== adab ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - ADAB\n";
	opt_rosen(opt::algo::Name::ADAB);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== nadam ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - NADAM\n";
	opt_rosen(opt::algo::Name::NADAM);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== amsgrad ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - AMSGRAD\n";
	opt_rosen(opt::algo::Name::AMSGRAD);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== yogi ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - YOGI\n";
	opt_rosen(opt::algo::Name::YOGI);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== yoni ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - YONI\n";
	opt_rosen(opt::algo::Name::YONI);
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== free memory ====
	delete[] str;
	
	return 0;
}
