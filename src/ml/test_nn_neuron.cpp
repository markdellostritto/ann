// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
#include <chrono>
// str
#include "str/print.hpp"
// ml
#include "ml/nn.hpp"

using namespace std::chrono;

static const double large=1.0e100;
//static const double large=std::numeric_limits<double>::max();
//static const double large=0.1*std::numeric_limits<double>::max();

void test_tfunc_deriv(NN::Neuron type, const double lim){
	//local function variables
	const int N=1e7;
	const double xmin=-lim;
	const double xmax=lim;
	const double cmin=0.1;
	const double cmax=10.0;
	const double dx=(xmax-xmin)/(N-0*1.0);
	Eigen::VectorXd z,a,d,d2;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	d=Eigen::VectorXd::Zero(N);
	d2=Eigen::VectorXd::Zero(N);
	double errd1=0,errd2=0;
	
	//set the function pointer
	typedef void (*TFP)(double,const VecXd&,VecXd&,VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:   func=NN::AFFPBP2::af_lin; break;
		case NN::Neuron::SIGMOID:  func=NN::AFFPBP2::af_sigmoid; break;
		case NN::Neuron::TANH:     func=NN::AFFPBP2::af_tanh; break;
		case NN::Neuron::ISRU:     func=NN::AFFPBP2::af_isru; break;
		case NN::Neuron::ARCTAN:   func=NN::AFFPBP2::af_arctan; break;
		case NN::Neuron::RELU:     func=NN::AFFPBP2::af_relu; break;
		case NN::Neuron::ELU:      func=NN::AFFPBP2::af_elu; break;
		case NN::Neuron::TANHRE:   func=NN::AFFPBP2::af_tanhre; break;
		case NN::Neuron::SQRE:     func=NN::AFFPBP2::af_sqre; break;
		case NN::Neuron::SWISH:    func=NN::AFFPBP2::af_swish; break;
		case NN::Neuron::GELU:     func=NN::AFFPBP2::af_gelu; break;
		case NN::Neuron::MISH:     func=NN::AFFPBP2::af_mish; break;
		case NN::Neuron::PFLU:     func=NN::AFFPBP2::af_pflu; break;
		case NN::Neuron::SERF:     func=NN::AFFPBP2::af_serf; break;
		case NN::Neuron::LOGISH:   func=NN::AFFPBP2::af_logish; break;
		case NN::Neuron::SOFTPLUS: func=NN::AFFPBP2::af_softplus; break;
		case NN::Neuron::SQPLUS:   func=NN::AFFPBP2::af_sqplus; break;
		case NN::Neuron::ATISH:    func=NN::AFFPBP2::af_atish; break;
		case NN::Neuron::TEST:     func=NN::AFFPBP2::af_test; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the derivative
	const double c=1.0*std::rand()/RAND_MAX*(cmax-cmin)+cmin;
	for(int i=0; i<N; ++i) z[i]=xmin+dx*i;
	(*func)(c,z,a,d,d2);
	for(int i=1; i<N-1; ++i){
		const double d11=d[i];
		const double d12=0.5*(a[i+1]-a[i-1])/dx;
		errd1+=std::fabs(d12-d11);
		const double d21=d2[i];
		const double d22=0.5*(d[i+1]-d[i-1])/dx;
		errd2+=std::fabs(d22-d21);
	}
	errd1/=N;
	errd2/=N;
	
	//print the results
	std::cout<<"x=["<<xmin<<":"<<xmax<<":"<<dx<<"] c="<<c<<" transfer "<<type<<" errd1 "<<errd1<<" errd2 "<<errd2<<"\n";
}

void test_tfunc_limit_a(NN::Neuron type){
	//local function variables
	const int N=3;
	const double cmin=0.1;
	const double cmax=10.0;
	Eigen::VectorXd z,a,d,d2,alim;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	d=Eigen::VectorXd::Zero(N);
	d2=Eigen::VectorXd::Zero(N);
	alim=Eigen::VectorXd::Zero(N);
	
	z[0]=-large;
	z[1]=0.0;
	z[2]=large;
	
	//set the function pointer
	typedef void (*TFP)(double,const VecXd&,VecXd&,VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:   func=NN::AFFPBP2::af_lin; break;
		case NN::Neuron::SIGMOID:  func=NN::AFFPBP2::af_sigmoid; break;
		case NN::Neuron::TANH:     func=NN::AFFPBP2::af_tanh; break;
		case NN::Neuron::ISRU:     func=NN::AFFPBP2::af_isru; break;
		case NN::Neuron::ARCTAN:   func=NN::AFFPBP2::af_arctan; break;
		case NN::Neuron::RELU:     func=NN::AFFPBP2::af_relu; break;
		case NN::Neuron::ELU:      func=NN::AFFPBP2::af_elu; break;
		case NN::Neuron::TANHRE:   func=NN::AFFPBP2::af_sqre; break;
		case NN::Neuron::SQRE:     func=NN::AFFPBP2::af_sqre; break;
		case NN::Neuron::SWISH:    func=NN::AFFPBP2::af_swish; break;
		case NN::Neuron::GELU:     func=NN::AFFPBP2::af_gelu; break;
		case NN::Neuron::MISH:     func=NN::AFFPBP2::af_mish; break;
		case NN::Neuron::PFLU:     func=NN::AFFPBP2::af_pflu; break;
		case NN::Neuron::SERF:     func=NN::AFFPBP2::af_serf; break;
		case NN::Neuron::LOGISH:   func=NN::AFFPBP2::af_logish; break;
		case NN::Neuron::SOFTPLUS: func=NN::AFFPBP2::af_softplus; break;
		case NN::Neuron::SQPLUS:   func=NN::AFFPBP2::af_sqplus; break;
		case NN::Neuron::ATISH:    func=NN::AFFPBP2::af_atish; break;
		case NN::Neuron::TEST:     func=NN::AFFPBP2::af_test; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//set the function limits
	const double c=1.0*std::rand()/RAND_MAX*(cmax-cmin)+cmin;
	switch(type){
		case NN::Neuron::LINEAR:{
			alim[0]=-large;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::SIGMOID:{
			alim[0]=0.0;
			alim[1]=0.5;
			alim[2]=1.0;
		} break;
		case NN::Neuron::TANH:{
			alim[0]=-1.0;
			alim[1]=0.0;
			alim[2]=1.0;
		} break;
		case NN::Neuron::ISRU:{
			alim[0]=-1.0;
			alim[1]=0.0;
			alim[2]=1.0;
		} break;
		case NN::Neuron::ARCTAN:{
			alim[0]=-1.0;
			alim[1]=0.0;
			alim[2]=1.0;
		} break;
		case NN::Neuron::RELU:{
			alim[0]=0.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::ELU:{
			alim[0]=-1.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::TANHRE:{
			alim[0]=-1.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::SQRE:{
			alim[0]=-1.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::SWISH:{
			alim[0]=0.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::GELU:{
			alim[0]=0.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::MISH:{
			alim[0]=0.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::PFLU:{
			alim[0]=0.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::SERF:{
			alim[0]=0.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::LOGISH:{
			alim[0]=0.0;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::SOFTPLUS:{
			alim[0]=-0.69314718055994531/c;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::SQPLUS:{
			alim[0]=-0.5/c;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::ATISH:{
			alim[0]=-0.5/c;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		case NN::Neuron::TEST:{
			alim[0]=-0.5;
			alim[1]=0.0;
			alim[2]=large;
		} break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the limit
	(*func)(c,z,a,d,d2);
	const double erra=(a-alim).norm()/N;
	
	//print the results
	std::cout<<"large = "<<large<<" c="<<c<<" transfer "<<type<<" erra "<<erra<<"\n";
	if(erra>1.0e-15 || erra!=erra){
		std::cout<<"alim = "<<alim.transpose()<<"\n";
		std::cout<<"a = "<<a.transpose()<<"\n";
	}
}

void test_tfunc_limit_d(NN::Neuron type){
	//local function variables
	const int N=3;
	const double cmin=0.1;
	const double cmax=10.0;
	Eigen::VectorXd z,a,d,d2,dlim;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	d=Eigen::VectorXd::Zero(N);
	d2=Eigen::VectorXd::Zero(N);
	dlim=Eigen::VectorXd::Zero(N);
	
	z[0]=-large;
	z[1]=0.0;
	z[2]=large;
	
	//set the function pointer
	typedef void (*TFP)(double,const VecXd&,VecXd&,VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:   func=NN::AFFPBP2::af_lin; break;
		case NN::Neuron::SIGMOID:  func=NN::AFFPBP2::af_sigmoid; break;
		case NN::Neuron::TANH:     func=NN::AFFPBP2::af_tanh; break;
		case NN::Neuron::ISRU:     func=NN::AFFPBP2::af_isru; break;
		case NN::Neuron::ARCTAN:   func=NN::AFFPBP2::af_arctan; break;
		case NN::Neuron::RELU:     func=NN::AFFPBP2::af_relu; break;
		case NN::Neuron::ELU:      func=NN::AFFPBP2::af_elu; break;
		case NN::Neuron::TANHRE:   func=NN::AFFPBP2::af_tanhre; break;
		case NN::Neuron::SQRE:   func=NN::AFFPBP2::af_sqre; break;
		case NN::Neuron::SWISH:    func=NN::AFFPBP2::af_swish; break;
		case NN::Neuron::GELU:     func=NN::AFFPBP2::af_gelu; break;
		case NN::Neuron::MISH:     func=NN::AFFPBP2::af_mish; break;
		case NN::Neuron::PFLU:     func=NN::AFFPBP2::af_pflu; break;
		case NN::Neuron::SERF:     func=NN::AFFPBP2::af_serf; break;
		case NN::Neuron::LOGISH:   func=NN::AFFPBP2::af_logish; break;
		case NN::Neuron::SOFTPLUS: func=NN::AFFPBP2::af_softplus; break;
		case NN::Neuron::SQPLUS:   func=NN::AFFPBP2::af_sqplus; break;
		case NN::Neuron::ATISH:    func=NN::AFFPBP2::af_atish; break;
		case NN::Neuron::TEST:     func=NN::AFFPBP2::af_test; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//set the function limits
	const double c=1.0*std::rand()/RAND_MAX*(cmax-cmin)+cmin;
	switch(type){
		case NN::Neuron::LINEAR:{
			dlim[0]=1.0;
			dlim[1]=1.0;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::SIGMOID:{
			dlim[0]=0.0;
			dlim[1]=0.25*c;
			dlim[2]=0.0;
		} break;
		case NN::Neuron::TANH:{
			dlim[0]=0.0;
			dlim[1]=1.0*c;
			dlim[2]=0.0;
		} break;
		case NN::Neuron::ISRU:{
			dlim[0]=0.0;
			dlim[1]=1.0*c;
			dlim[2]=0.0;
		} break;
		case NN::Neuron::ARCTAN:{
			dlim[0]=0.0;
			dlim[1]=1.0*c;
			dlim[2]=0.0;
		} break;
		case NN::Neuron::RELU:{
			dlim[0]=0.0;
			dlim[1]=0.0;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::ELU:{
			dlim[0]=0.0;
			dlim[1]=1.0;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::TANHRE:{
			dlim[0]=0.0;
			dlim[1]=0.0;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::SQRE:{
			dlim[0]=0.0;
			dlim[1]=1.0;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::SWISH:{
			dlim[0]=0.0;
			dlim[1]=0.5;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::GELU:{
			dlim[0]=0.0;
			dlim[1]=0.5;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::MISH:{
			dlim[0]=0.0;
			dlim[1]=3.0/5.0;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::PFLU:{
			dlim[0]=0.0;
			dlim[1]=0.5;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::SERF:{
			dlim[0]=0.0;
			dlim[1]=0.5;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::LOGISH:{
			dlim[0]=0.0;
			dlim[1]=0.5;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::SOFTPLUS:{
			dlim[0]=0.0;
			dlim[1]=0.5;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::SQPLUS:{
			dlim[0]=0.0;
			dlim[1]=0.5;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::ATISH:{
			dlim[0]=0.0;
			dlim[1]=0.5;
			dlim[2]=1.0;
		} break;
		case NN::Neuron::TEST:{
			dlim[0]=0.0;
			dlim[1]=0.5;
			dlim[2]=1.0;
		} break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the limit
	(*func)(c,z,a,d,d2);
	const double erra=(d-dlim).norm()/N;
	
	//print the results
	std::cout<<"large = "<<large<<" c="<<c<<" transfer "<<type<<" erra "<<erra<<"\n";
	if(erra>1.0e-15 || erra!=erra){
		std::cout<<"dlim = "<<dlim.transpose()<<"\n";
		std::cout<<"d = "<<d.transpose()<<"\n";
	}
}

void test_tfunc_time_affp(NN::Neuron type){
	//local function variables
	const int N=1e7;
	const double xmin=-5;
	const double xmax=5;
	const double cmin=0.1;
	const double cmax=3.0;
	const double dx=(xmax-xmin)/(N-1.0);
	Eigen::VectorXd z,a;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	double err=0;
	
	//set the function pointer
	typedef void (*TFP)(double,const VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:   func=NN::AFFP::af_lin; break;
		case NN::Neuron::SIGMOID:  func=NN::AFFP::af_sigmoid; break;
		case NN::Neuron::TANH:     func=NN::AFFP::af_tanh; break;
		case NN::Neuron::ISRU:     func=NN::AFFP::af_isru; break;
		case NN::Neuron::ARCTAN:   func=NN::AFFP::af_arctan; break;
		case NN::Neuron::RELU:     func=NN::AFFP::af_relu; break;
		case NN::Neuron::ELU:      func=NN::AFFP::af_elu; break;
		case NN::Neuron::TANHRE:   func=NN::AFFP::af_sqre; break;
		case NN::Neuron::SQRE:     func=NN::AFFP::af_sqre; break;
		case NN::Neuron::SWISH:    func=NN::AFFP::af_swish; break;
		case NN::Neuron::GELU:     func=NN::AFFP::af_gelu; break;
		case NN::Neuron::MISH:     func=NN::AFFP::af_mish; break;
		case NN::Neuron::PFLU:     func=NN::AFFP::af_pflu; break;
		case NN::Neuron::SERF:     func=NN::AFFP::af_serf; break;
		case NN::Neuron::LOGISH:   func=NN::AFFP::af_logish; break;
		case NN::Neuron::SOFTPLUS: func=NN::AFFP::af_softplus; break;
		case NN::Neuron::SQPLUS:   func=NN::AFFP::af_sqplus; break;
		case NN::Neuron::ATISH:    func=NN::AFFP::af_atish; break;
		case NN::Neuron::TEST:     func=NN::AFFP::af_test; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the derivative
	const double c=1.0*std::rand()/RAND_MAX*(cmax-cmin)+cmin;
	high_resolution_clock::time_point tbeg,tend;
	for(int i=0; i<N; ++i) z[i]=xmin+dx*i;
	tbeg=high_resolution_clock::now();
	(*func)(c,z,a);
	tend=high_resolution_clock::now();
	duration<double> time = duration_cast<duration<double>>(tend-tbeg);
	
	//print the results
	std::cout<<"transfer "<<type<<" time "<<time.count()<<"\n";
}

void test_tfunc_time_affpbp(NN::Neuron type){
	//local function variables
	const int N=1e7;
	const double xmin=-5;
	const double xmax=5;
	const double cmin=0.1;
	const double cmax=3.0;
	const double dx=(xmax-xmin)/(N-1.0);
	Eigen::VectorXd z,a,d;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	d=Eigen::VectorXd::Zero(N);
	double err=0;
	
	//set the function pointer
	typedef void (*TFP)(double,const VecXd&,VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:   func=NN::AFFPBP::af_lin; break;
		case NN::Neuron::SIGMOID:  func=NN::AFFPBP::af_sigmoid; break;
		case NN::Neuron::TANH:     func=NN::AFFPBP::af_tanh; break;
		case NN::Neuron::ISRU:     func=NN::AFFPBP::af_isru; break;
		case NN::Neuron::ARCTAN:   func=NN::AFFPBP::af_arctan; break;
		case NN::Neuron::RELU:     func=NN::AFFPBP::af_relu; break;
		case NN::Neuron::ELU:      func=NN::AFFPBP::af_elu; break;
		case NN::Neuron::TANHRE:   func=NN::AFFPBP::af_tanhre; break;
		case NN::Neuron::SQRE:     func=NN::AFFPBP::af_sqre; break;
		case NN::Neuron::SWISH:    func=NN::AFFPBP::af_swish; break;
		case NN::Neuron::GELU:     func=NN::AFFPBP::af_gelu; break;
		case NN::Neuron::MISH:     func=NN::AFFPBP::af_mish; break;
		case NN::Neuron::PFLU:     func=NN::AFFPBP::af_pflu; break;
		case NN::Neuron::SERF:     func=NN::AFFPBP::af_serf; break;
		case NN::Neuron::LOGISH:   func=NN::AFFPBP::af_logish; break;
		case NN::Neuron::SOFTPLUS: func=NN::AFFPBP::af_softplus; break;
		case NN::Neuron::SQPLUS:   func=NN::AFFPBP::af_sqplus; break;
		case NN::Neuron::ATISH:    func=NN::AFFPBP::af_atish; break;
		case NN::Neuron::TEST:     func=NN::AFFPBP::af_test; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the derivative
	const double c=1.0*std::rand()/RAND_MAX*(cmax-cmin)+cmin;
	high_resolution_clock::time_point tbeg,tend;
	for(int i=0; i<N; ++i) z[i]=xmin+dx*i;
	tbeg=high_resolution_clock::now();
	(*func)(c,z,a,d);
	tend=high_resolution_clock::now();
	duration<double> time = duration_cast<duration<double>>(tend-tbeg);
	
	//print the results
	std::cout<<"transfer "<<type<<" time "<<time.count()<<"\n";
}

void test_tfunc_time_affpbp2(NN::Neuron type){
	//local function variables
	const int N=1e7;
	const double xmin=-5;
	const double xmax=5;
	const double cmin=0.1;
	const double cmax=3.0;
	const double dx=(xmax-xmin)/(N-1.0);
	Eigen::VectorXd z,a,d,d2;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	d=Eigen::VectorXd::Zero(N);
	d2=Eigen::VectorXd::Zero(N);
	double err=0;
	
	//set the function pointer
	typedef void (*TFP)(double,const VecXd&,VecXd&,VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:   func=NN::AFFPBP2::af_lin; break;
		case NN::Neuron::SIGMOID:  func=NN::AFFPBP2::af_sigmoid; break;
		case NN::Neuron::TANH:     func=NN::AFFPBP2::af_tanh; break;
		case NN::Neuron::ISRU:     func=NN::AFFPBP2::af_isru; break;
		case NN::Neuron::ARCTAN:   func=NN::AFFPBP2::af_arctan; break;
		case NN::Neuron::RELU:     func=NN::AFFPBP2::af_relu; break;
		case NN::Neuron::ELU:      func=NN::AFFPBP2::af_elu; break;
		case NN::Neuron::TANHRE:   func=NN::AFFPBP2::af_tanhre; break;
		case NN::Neuron::SQRE:   func=NN::AFFPBP2::af_sqre; break;
		case NN::Neuron::SWISH:    func=NN::AFFPBP2::af_swish; break;
		case NN::Neuron::GELU:     func=NN::AFFPBP2::af_gelu; break;
		case NN::Neuron::MISH:     func=NN::AFFPBP2::af_mish; break;
		case NN::Neuron::PFLU:     func=NN::AFFPBP2::af_pflu; break;
		case NN::Neuron::SERF:     func=NN::AFFPBP2::af_serf; break;
		case NN::Neuron::LOGISH:   func=NN::AFFPBP2::af_logish; break;
		case NN::Neuron::SOFTPLUS: func=NN::AFFPBP2::af_softplus; break;
		case NN::Neuron::SQPLUS:   func=NN::AFFPBP2::af_sqplus; break;
		case NN::Neuron::ATISH:    func=NN::AFFPBP2::af_atish; break;
		case NN::Neuron::TEST:     func=NN::AFFPBP2::af_test; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the derivative
	const double c=1.0*std::rand()/RAND_MAX*(cmax-cmin)+cmin;
	high_resolution_clock::time_point tbeg,tend;
	for(int i=0; i<N; ++i) z[i]=xmin+dx*i;
	tbeg=high_resolution_clock::now();
	(*func)(c,z,a,d,d2);
	tend=high_resolution_clock::now();
	duration<double> time = duration_cast<duration<double>>(tend-tbeg);
	
	//print the results
	std::cout<<"transfer "<<type<<" time "<<time.count()<<"\n";
}void test_tfunc_write(NN::Neuron type, const char* file){
	//local function variables
	const int N=1e4;
	const double xmin=-50;
	const double xmax=50;
	const double cmin=0.1;
	const double cmax=3.0;
	const double dx=(xmax-xmin)/(N-1.0);
	Eigen::VectorXd z,a,d,d2;
	z=Eigen::VectorXd::Zero(N);
	a=Eigen::VectorXd::Zero(N);
	d=Eigen::VectorXd::Zero(N);
	d2=Eigen::VectorXd::Zero(N);
	double err=0;
	
	//set the function pointer
	typedef void (*TFP)(double,const VecXd&,VecXd&,VecXd&,VecXd&);
	TFP func;
	switch(type){
		case NN::Neuron::LINEAR:   func=NN::AFFPBP2::af_lin; break;
		case NN::Neuron::SIGMOID:  func=NN::AFFPBP2::af_sigmoid; break;
		case NN::Neuron::TANH:     func=NN::AFFPBP2::af_tanh; break;
		case NN::Neuron::ISRU:     func=NN::AFFPBP2::af_isru; break;
		case NN::Neuron::ARCTAN:   func=NN::AFFPBP2::af_arctan; break;
		case NN::Neuron::RELU:     func=NN::AFFPBP2::af_relu; break;
		case NN::Neuron::ELU:      func=NN::AFFPBP2::af_elu; break;
		case NN::Neuron::TANHRE:   func=NN::AFFPBP2::af_tanhre; break;
		case NN::Neuron::SQRE:   func=NN::AFFPBP2::af_sqre; break;
		case NN::Neuron::SWISH:    func=NN::AFFPBP2::af_swish; break;
		case NN::Neuron::GELU:     func=NN::AFFPBP2::af_gelu; break;
		case NN::Neuron::MISH:     func=NN::AFFPBP2::af_mish; break;
		case NN::Neuron::PFLU:     func=NN::AFFPBP2::af_pflu; break;
		case NN::Neuron::SERF:     func=NN::AFFPBP2::af_serf; break;
		case NN::Neuron::LOGISH:   func=NN::AFFPBP2::af_logish; break;
		case NN::Neuron::SOFTPLUS: func=NN::AFFPBP2::af_softplus; break;
		case NN::Neuron::SQPLUS:   func=NN::AFFPBP2::af_sqplus; break;
		case NN::Neuron::ATISH:    func=NN::AFFPBP2::af_atish; break;
		case NN::Neuron::TEST:     func=NN::AFFPBP2::af_test; break;
		default: throw std::invalid_argument("Invalid transfer function.");
	}
	
	//compute the derivative
	//const double c=1.0*std::rand()/RAND_MAX*(cmax-cmin)+cmin;
	const double c=1.0;
	for(int i=0; i<N; ++i) z[i]=xmin+dx*i;
	(*func)(c,z,a,d,d2);
	
	//write the results
	FILE* writer=fopen(file,"w");
	if(writer!=NULL){
		fprintf(writer,"z a d d2\n");
		for(int i=0; i<N; ++i){
			fprintf(writer,"%f %f %f %f\n",z[i],a[i],d[i],d2[i]);
		}
		fclose(writer);
		writer=NULL;
	}
}

int main(int argc, char* argv[]){
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NEURON - GRADIENT",str)<<"\n";
	test_tfunc_deriv(NN::Neuron::LINEAR,1e2);
	test_tfunc_deriv(NN::Neuron::SIGMOID,1e2);
	test_tfunc_deriv(NN::Neuron::TANH,1e2);
	test_tfunc_deriv(NN::Neuron::ARCTAN,1e2);
	test_tfunc_deriv(NN::Neuron::ISRU,1e2);
	test_tfunc_deriv(NN::Neuron::RELU,1e2);
	test_tfunc_deriv(NN::Neuron::ELU,1e2);
	test_tfunc_deriv(NN::Neuron::SWISH,1e2);
	test_tfunc_deriv(NN::Neuron::GELU,1e2);
	test_tfunc_deriv(NN::Neuron::MISH,1e2);
	test_tfunc_deriv(NN::Neuron::PFLU,1e2);
	test_tfunc_deriv(NN::Neuron::SERF,1e2);
	test_tfunc_deriv(NN::Neuron::LOGISH,1e2);
	test_tfunc_deriv(NN::Neuron::SOFTPLUS,1e2);
	test_tfunc_deriv(NN::Neuron::SQPLUS,1e2);
	test_tfunc_deriv(NN::Neuron::TANHRE,1e2);
	test_tfunc_deriv(NN::Neuron::SQRE,1e2);
	test_tfunc_deriv(NN::Neuron::ELU,1e2);
	test_tfunc_deriv(NN::Neuron::ATISH,1e2);
	test_tfunc_deriv(NN::Neuron::TEST,1e2);
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NEURON - GRADIENT",str)<<"\n";
	test_tfunc_deriv(NN::Neuron::LINEAR,1e100);
	test_tfunc_deriv(NN::Neuron::SIGMOID,1e100);
	test_tfunc_deriv(NN::Neuron::TANH,1e100);
	test_tfunc_deriv(NN::Neuron::ARCTAN,1e100);
	test_tfunc_deriv(NN::Neuron::ISRU,1e100);
	test_tfunc_deriv(NN::Neuron::RELU,1e100);
	test_tfunc_deriv(NN::Neuron::ELU,1e100);
	test_tfunc_deriv(NN::Neuron::SWISH,1e100);
	test_tfunc_deriv(NN::Neuron::GELU,1e100);
	test_tfunc_deriv(NN::Neuron::MISH,1e100);
	test_tfunc_deriv(NN::Neuron::PFLU,1e100);
	test_tfunc_deriv(NN::Neuron::SERF,1e100);
	test_tfunc_deriv(NN::Neuron::LOGISH,1e100);
	test_tfunc_deriv(NN::Neuron::SOFTPLUS,1e100);
	test_tfunc_deriv(NN::Neuron::SQPLUS,1e100);
	test_tfunc_deriv(NN::Neuron::TANHRE,1e100);
	test_tfunc_deriv(NN::Neuron::SQRE,1e100);
	test_tfunc_deriv(NN::Neuron::ELU,1e100);
	test_tfunc_deriv(NN::Neuron::ATISH,1e100);
	test_tfunc_deriv(NN::Neuron::TEST,1e100);
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NEURON - LIMIT - A",str)<<"\n";
	test_tfunc_limit_a(NN::Neuron::LINEAR);
	test_tfunc_limit_a(NN::Neuron::SIGMOID);
	test_tfunc_limit_a(NN::Neuron::TANH);
	test_tfunc_limit_a(NN::Neuron::ARCTAN);
	test_tfunc_limit_a(NN::Neuron::ISRU);
	test_tfunc_limit_a(NN::Neuron::RELU);
	test_tfunc_limit_a(NN::Neuron::ELU);
	test_tfunc_limit_a(NN::Neuron::SWISH);
	test_tfunc_limit_a(NN::Neuron::GELU);
	test_tfunc_limit_a(NN::Neuron::MISH);
	test_tfunc_limit_a(NN::Neuron::PFLU);
	test_tfunc_limit_a(NN::Neuron::SERF);
	test_tfunc_limit_a(NN::Neuron::LOGISH);
	test_tfunc_limit_a(NN::Neuron::SOFTPLUS);
	test_tfunc_limit_a(NN::Neuron::SQPLUS);
	test_tfunc_limit_a(NN::Neuron::TANHRE);
	test_tfunc_limit_a(NN::Neuron::SQRE);
	test_tfunc_limit_a(NN::Neuron::ELU);
	test_tfunc_limit_a(NN::Neuron::ATISH);
	test_tfunc_limit_a(NN::Neuron::TEST);
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NEURON - LIMIT - D",str)<<"\n";
	test_tfunc_limit_d(NN::Neuron::LINEAR);
	test_tfunc_limit_d(NN::Neuron::SIGMOID);
	test_tfunc_limit_d(NN::Neuron::TANH);
	test_tfunc_limit_d(NN::Neuron::ARCTAN);
	test_tfunc_limit_d(NN::Neuron::ISRU);
	test_tfunc_limit_d(NN::Neuron::RELU);
	test_tfunc_limit_d(NN::Neuron::ELU);
	test_tfunc_limit_d(NN::Neuron::SWISH);
	test_tfunc_limit_d(NN::Neuron::GELU);
	test_tfunc_limit_d(NN::Neuron::MISH);
	test_tfunc_limit_d(NN::Neuron::PFLU);
	test_tfunc_limit_d(NN::Neuron::SERF);
	test_tfunc_limit_d(NN::Neuron::LOGISH);
	test_tfunc_limit_d(NN::Neuron::SOFTPLUS);
	test_tfunc_limit_d(NN::Neuron::SQPLUS);
	test_tfunc_limit_d(NN::Neuron::TANHRE);
	test_tfunc_limit_d(NN::Neuron::SQRE);
	test_tfunc_limit_d(NN::Neuron::ELU);
	test_tfunc_limit_d(NN::Neuron::ATISH);
	test_tfunc_limit_d(NN::Neuron::TEST);
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NEURON - TIME - FP",str)<<"\n";
	test_tfunc_time_affp(NN::Neuron::LINEAR);
	test_tfunc_time_affp(NN::Neuron::SIGMOID);
	test_tfunc_time_affp(NN::Neuron::TANH);
	test_tfunc_time_affp(NN::Neuron::ARCTAN);
	test_tfunc_time_affp(NN::Neuron::ISRU);
	test_tfunc_time_affp(NN::Neuron::RELU);
	test_tfunc_time_affp(NN::Neuron::ELU);
	test_tfunc_time_affp(NN::Neuron::SWISH);
	test_tfunc_time_affp(NN::Neuron::GELU);
	test_tfunc_time_affp(NN::Neuron::MISH);
	test_tfunc_time_affp(NN::Neuron::PFLU);
	test_tfunc_time_affp(NN::Neuron::SERF);
	test_tfunc_time_affp(NN::Neuron::LOGISH);
	test_tfunc_time_affp(NN::Neuron::SOFTPLUS);
	test_tfunc_time_affp(NN::Neuron::SQPLUS);
	test_tfunc_time_affp(NN::Neuron::TANHRE);
	test_tfunc_time_affp(NN::Neuron::SQRE);
	test_tfunc_time_affp(NN::Neuron::ELU);
	test_tfunc_time_affp(NN::Neuron::ATISH);
	test_tfunc_time_affp(NN::Neuron::TEST);
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NEURON - TIME - FPBP",str)<<"\n";
	test_tfunc_time_affpbp(NN::Neuron::LINEAR);
	test_tfunc_time_affpbp(NN::Neuron::SIGMOID);
	test_tfunc_time_affpbp(NN::Neuron::TANH);
	test_tfunc_time_affpbp(NN::Neuron::ARCTAN);
	test_tfunc_time_affpbp(NN::Neuron::ISRU);
	test_tfunc_time_affpbp(NN::Neuron::RELU);
	test_tfunc_time_affpbp(NN::Neuron::ELU);
	test_tfunc_time_affpbp(NN::Neuron::SWISH);
	test_tfunc_time_affpbp(NN::Neuron::GELU);
	test_tfunc_time_affpbp(NN::Neuron::MISH);
	test_tfunc_time_affpbp(NN::Neuron::PFLU);
	test_tfunc_time_affpbp(NN::Neuron::SERF);
	test_tfunc_time_affpbp(NN::Neuron::LOGISH);
	test_tfunc_time_affpbp(NN::Neuron::SOFTPLUS);
	test_tfunc_time_affpbp(NN::Neuron::SQPLUS);
	test_tfunc_time_affpbp(NN::Neuron::TANHRE);
	test_tfunc_time_affpbp(NN::Neuron::SQRE);
	test_tfunc_time_affpbp(NN::Neuron::ELU);
	test_tfunc_time_affpbp(NN::Neuron::ATISH);
	test_tfunc_time_affpbp(NN::Neuron::TEST);
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NEURON - TIME - FPBP2",str)<<"\n";
	test_tfunc_time_affpbp2(NN::Neuron::LINEAR);
	test_tfunc_time_affpbp2(NN::Neuron::SIGMOID);
	test_tfunc_time_affpbp2(NN::Neuron::TANH);
	test_tfunc_time_affpbp2(NN::Neuron::ARCTAN);
	test_tfunc_time_affpbp2(NN::Neuron::ISRU);
	test_tfunc_time_affpbp2(NN::Neuron::RELU);
	test_tfunc_time_affpbp2(NN::Neuron::ELU);
	test_tfunc_time_affpbp2(NN::Neuron::SWISH);
	test_tfunc_time_affpbp2(NN::Neuron::GELU);
	test_tfunc_time_affpbp2(NN::Neuron::MISH);
	test_tfunc_time_affpbp2(NN::Neuron::PFLU);
	test_tfunc_time_affpbp2(NN::Neuron::SERF);
	test_tfunc_time_affpbp2(NN::Neuron::LOGISH);
	test_tfunc_time_affpbp2(NN::Neuron::SOFTPLUS);
	test_tfunc_time_affpbp2(NN::Neuron::SQPLUS);
	test_tfunc_time_affpbp2(NN::Neuron::TANHRE);
	test_tfunc_time_affpbp2(NN::Neuron::SQRE);
	test_tfunc_time_affpbp2(NN::Neuron::ELU);
	test_tfunc_time_affpbp2(NN::Neuron::ATISH);
	test_tfunc_time_affpbp2(NN::Neuron::TEST);
	std::cout<<print::buf(str)<<"\n";
	
	//test_tfunc_write(NN::Neuron::MISH,"mish.dat");
	//test_tfunc_write(NN::Neuron::SWISH,"swish.dat");
	//test_tfunc_write(NN::Neuron::ELU,"elu.dat");
	//test_tfunc_write(NN::Neuron::PFLU,"pflu.dat");
	//test_tfunc_write(NN::Neuron::ISRU,"isru.dat");
	//test_tfunc_write(NN::Neuron::TEST,"test.dat");
	//test_tfunc_write(NN::Neuron::GELU,"gelu.dat");
	//test_tfunc_write(NN::Neuron::SQPLUS,"sqplus.dat");
	
	delete[] str;
	
	return 0;
}