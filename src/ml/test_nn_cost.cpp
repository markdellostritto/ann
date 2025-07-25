// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// str
#include "str/print.hpp"
// math
#include "math/reduce.hpp"
// ml
#include "ml/nn.hpp"

void test_unit_nn_quad(){
	//time
	clock_t start,stop;
	double time=0;
	
	//error
	const int N=1000;
	const double deltaf=1.0/100.0;
	double erra=0,errp=0;
	Reduce<2> reduce;
	
	//neural network
	NN::Cost cost;
	NN::ANN nn,nn_copy;
	NN::ANNP annp;
	//neural configuration
	const int nlayer=3;
	std::vector<int> nh(nlayer);
	const int nInp=4;
	const int nOut=3;
	nh[0]=7; nh[1]=5; nh[2]=4;
	annp.sigma_b()=1.0e-3;
	annp.sigma_w()=1.0e-0;
	annp.init()=NN::Init::HE;
	annp.neuron()=NN::Neuron::TANH;
	
	//init rand
	std::srand(std::time(NULL));
	
	//loop over all samples
	for(int iter=0; iter<N; ++iter){
		//resize the nn
		nn.resize(annp,nInp,nh,nOut);
		cost.resize(nn);
		
		//set input/output scaling
		nn.inpw()=Eigen::VectorXd::Random(nInp);
		nn.inpb()=Eigen::VectorXd::Random(nInp);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		
		//initialize the input nodes
		for(int n=0; n<nn.nInp(); ++n) nn.inp()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		
		//execute the network, compute analytic gradient
		nn.fpbp();
		start=std::clock();
		const Eigen::VectorXd outN=nn.out();
		const Eigen::VectorXd outT=nn.out()+Eigen::VectorXd::Random(nOut)*1.0/10.0;
		const Eigen::VectorXd dcdo=(outN-outT);
		cost.grad(nn,dcdo);
		const Eigen::VectorXd gradN=cost.grad();
		stop=std::clock();
		time+=((double)(stop-start))/CLOCKS_PER_SEC*1e9/N;
		
		//compute brute force gradient
		Eigen::VectorXd gradB=Eigen::VectorXd::Zero(nn.size());
		nn_copy=nn;
		int count=0;
		for(int l=0; l<nn.nlayer(); ++l){
			for(int n=0; n<nn.b(l).size(); ++n){
				const double delta=nn.b(l)[n]*deltaf;
				//point 1
				nn_copy.b(l)[n]=nn.b(l)[n]-delta;
				nn_copy.fpbp();
				const Eigen::VectorXd outCN1=nn_copy.out();
				double c1=0.5*(outCN1-outT).dot(outCN1-outT);
				//point 2
				nn_copy.b(l)[n]=nn.b(l)[n]+delta;
				nn_copy.fpbp();
				const Eigen::VectorXd outCN2=nn_copy.out();
				double c2=0.5*(outCN2-outT).dot(outCN2-outT);
				//reset
				nn_copy.b(l)[n]=nn.b(l)[n];
				//gradient
				gradB[count++]=0.5*(c2-c1)/delta;
			}
		}
		for(int l=0; l<nn.nlayer(); ++l){
			for(int k=0; k<nn.w(l).cols(); ++k){
				for(int j=0; j<nn.w(l).rows(); ++j){
					const double delta=nn.w(l)(j,k)*deltaf;
					//point 1
					nn_copy.w(l)(j,k)=nn.w(l)(j,k)-delta;
					nn_copy.fpbp();
					const Eigen::VectorXd outCN1=nn_copy.out();
					double c1=0.5*(outCN1-outT).dot(outCN1-outT);
					//point 2
					nn_copy.w(l)(j,k)=nn.w(l)(j,k)+delta;
					nn_copy.fpbp();
					const Eigen::VectorXd outCN2=nn_copy.out();
					double c2=0.5*(outCN2-outT).dot(outCN2-outT);
					//reset
					nn_copy.w(l)(j,k)=nn.w(l)(j,k);
					//gradient
					gradB[count++]=0.5*(c2-c1)/delta;
				}
			}
		}
		for(int i=0; i<nn.size(); ++i){
			reduce.push(gradB[i],gradN[i]);
		}
		
		//compute error
		//std::cout<<"gradN = "<<gradN.transpose()<<"\n";
		//std::cout<<"gradB = "<<gradB.transpose()<<"\n";
		erra+=(gradN-gradB).norm();
		errp+=(gradN-gradB).norm()/gradB.norm()*100.0;
	}
	
	//compute the error
	erra/=N;
	errp/=N;
	time/=N;
	
	//print the results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - ANN - COST",str)<<"\n";
	std::cout<<"neuron = "<<annp.neuron()<<"\n";
	std::cout<<"config = "<<nn.nInp()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"nsample= "<<N<<"\n";
	std::cout<<"delta  = "<<deltaf<<"\n";
	std::cout<<"erra   = "<<erra<<"\n";
	std::cout<<"errp   = "<<errp<<"\n";
	std::cout<<"r2     = "<<reduce.r2()<<"\n";
	std::cout<<"m      = "<<reduce.m()<<"\n";
	std::cout<<"time   = "<<time<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_unit_nn_single(){
	
	//time
	clock_t start,stop;
	double time=0;
	
	//error
	const int N=1000;
	double erra=0,errp=0;
	Reduce<2> reduce;
	
	//nn
	NN::Cost cost;
	NN::ANN nn,nn_copy;
	NN::ANNP annp;
	const int nlayer=3;
	std::vector<int> nh(nlayer);
	const int nInp=4;
	const int nOut=1;
	nh[0]=7; nh[1]=5; nh[2]=4;
	annp.sigma_b()=1.0e-3;
	annp.sigma_w()=1.0e-0;
	annp.init()=NN::Init::HE;
	annp.neuron()=NN::Neuron::TANH;
	
	//init rand
	std::srand(std::time(NULL));
	
	//loop over all samples
	for(int iter=0; iter<N; ++iter){
		//resize the nn
		nn.resize(annp,nInp,nh,nOut);
		cost.resize(nn);
		
		//set input/output scaling
		nn.inpw()=Eigen::VectorXd::Random(nInp);
		nn.inpb()=Eigen::VectorXd::Random(nInp);
		nn.outw()=Eigen::VectorXd::Random(nOut);
		nn.outb()=Eigen::VectorXd::Random(nOut);
		
		//initialize the input nodes
		for(int n=0; n<nn.nInp(); ++n) nn.inp()[n]=(1.0*std::rand())/RAND_MAX-0.5;
		
		//execute network
		nn.fpbp();
		const Eigen::VectorXd outN=nn.out();
		const Eigen::VectorXd outT=nn.out()+Eigen::VectorXd::Random(nOut)*1.0/10.0;
		
		//compute analytic gradient
		const Eigen::VectorXd dcdo=(outN-outT);
		cost.grad(nn,dcdo);
		const Eigen::VectorXd gradN=cost.grad();
		
		//compute brute force gradient
		const Eigen::VectorXd identity=Eigen::VectorXd::Constant(nOut,1);
		cost.grad(nn,identity);
		const Eigen::VectorXd gradB=cost.grad()*dcdo[0];
		
		//compute error
		for(int i=0; i<nn.size(); ++i){
			reduce.push(gradB[i],gradN[i]);
		}
		//std::cout<<"gradN = "<<gradN.transpose()<<"\n";
		//std::cout<<"gradB = "<<gradB.transpose()<<"\n";
		erra+=(gradN-gradB).norm();
		errp+=(gradN-gradB).norm()/gradB.norm()*100.0;
	}
	
	//compute the error
	erra/=N;
	errp/=N;
	time/=N;
	
	//print the results
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - ANN - COST - SINGLE OUTPUT",str)<<"\n";
	std::cout<<"neuron = "<<annp.neuron()<<"\n";
	std::cout<<"config = "<<nn.nInp()<<" "; for(int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOut()<<"\n";
	std::cout<<"erra   = "<<erra<<"\n";
	std::cout<<"errp   = "<<errp<<"\n";
	std::cout<<"r2     = "<<reduce.r2()<<"\n";
	std::cout<<"m      = "<<reduce.m()<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	test_unit_nn_quad();
	test_unit_nn_single();
	
	delete[] str;
	
	return 0;
}