// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
#include <vector>
// eigen libraries
#include <Eigen/Dense>
// optimization
#include "opt/loss.hpp"
// string
#include "str/print.hpp"
// math
#include "math/reduce.hpp"

void test_grad(opt::Loss loss){
	//set arrays
	const int N=6000;
	const double xmin=-6.0;
	const double xmax=6.0;
	const double dx=(xmax-xmin)/(N-1.0);
	std::vector<double> x(N,0);
	std::vector<double> f(N,0);
	std::vector<double> g(N,0);
	Eigen::VectorXd value=Eigen::VectorXd::Constant(1,0.0);
	Eigen::VectorXd target=Eigen::VectorXd::Constant(1,0.0);
	Eigen::VectorXd grad=Eigen::VectorXd::Constant(1,0.0);
	const double delta=(1.0*std::rand())/RAND_MAX*(1.0-0.1)+0.1;
	//const double delta=1.0;
	//compute value
	for(int i=0; i<N; ++i){
		x[i]=xmin+i*dx; value[0]=x[i];
		f[i]=opt::Loss::error2(loss,delta,value,target);
	}
	//compute gradient
	g[0]=(f[1]-f[0])/dx;
	for(int i=1; i<N-1; ++i){
		g[i]=0.5*(f[i+1]-f[i-1])/dx;
	}
	g[N-1]=(f[N-1]-f[N-2])/dx;
	//compute error
	double err=0,errp=0;
	Reduce<2> reduce;
	for(int i=0; i<N; ++i){
		value[0]=x[i];
		f[i]=opt::Loss::error2(loss,delta,value,target,grad);
		err+=fabs(grad[0]-g[i]);
		errp+=fabs((grad[0]-g[i])/g[i]*100.0);
		reduce.push(grad[0],g[i]);
	}
	err/=N;
	errp/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - LOSS - GRAD",str)<<"\n";
	std::cout<<"loss = "<<loss<<"\n";
	std::cout<<"N = "<<N<<"\n";
	std::cout<<"d = "<<delta<<"\n";
	std::cout<<"x = ["<<xmin<<" : "<<xmax<<" : "<<dx<<"]\n";
	//std::cout<<"time - std = "<<time_std.count()<<"\n";
	//std::cout<<"time - new = "<<time_new.count()<<"\n";
	std::cout<<"err - abs = "<<err<<"\n";
	std::cout<<"err - (%) = "<<errp<<"\n";
	std::cout<<"corr - m  = "<<reduce.m()<<"\n";
	std::cout<<"corr - r2 = "<<reduce.r2()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void plot_loss(opt::Loss loss, const char* file){
	const double xmin=-6.0;
	const double xmax=6.0;
	const int N=300;
	const double dx=(xmax-xmin)/(N-1.0);
	const double delta=(1.0*std::rand())/RAND_MAX*(1.0-0.1)+0.1;
	std::cout<<"loss = "<<loss<<" delta = "<<delta<<"\n";
	//const double delta=1.0;
	FILE* writer=fopen(file,"w");
	Eigen::VectorXd value=Eigen::VectorXd::Constant(1,0.0);
	Eigen::VectorXd target=Eigen::VectorXd::Constant(1,0.0);
	Eigen::VectorXd grad=Eigen::VectorXd::Constant(1,0.0);
	if(writer!=NULL){
		for(int i=0; i<N; ++i){
			value[0]=xmin+i*dx;
			const double f=opt::Loss::error2(loss,delta,value,target,grad);
			fprintf(writer,"%f %f %f\n",value[0],f,grad[0]);
		}
		fclose(writer);
		writer=NULL;
	}
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
	
	//==== mae ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - LOSS - MAE",str)<<"\n";
	test_grad(opt::Loss::MAE);
	plot_loss(opt::Loss::MAE,"loss_mae.dat");
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== mse ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - LOSS - MSE",str)<<"\n";
	test_grad(opt::Loss::MSE);
	plot_loss(opt::Loss::MSE,"loss_mse.dat");
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== huber ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - LOSS - HUBER",str)<<"\n";
	test_grad(opt::Loss::HUBER);
	plot_loss(opt::Loss::HUBER,"loss_huber.dat");
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== asinh ====
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - LOSS - ASINH",str)<<"\n";
	test_grad(opt::Loss::ASINH);
	plot_loss(opt::Loss::ASINH,"loss_asinh.dat");
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	
	//==== free memory ====
	delete[] str;
	
	return 0;
}
