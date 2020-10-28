#define EIGEN_NO_DEBUG
//#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <Eigen/Dense>

int main(int argc, char* argv[]){
	//**** global variables ****
	//timing
	clock_t start,stop;
	double time;
	//testing
	static const unsigned int N=100000;
	
	//**** M*V ****
	{
		start=std::clock();
		Eigen::MatrixXd M;
		Eigen::VectorXd v,a;
		M.resize(17,17);
		v.resize(17);
		a.resize(17);
		for(unsigned int i=0; i<N; ++i){
			v=Eigen::VectorXd::Random(17);
			M=Eigen::MatrixXd::Random(17,17);
			a.noalias()=M*v;
		}
		stop=std::clock();
		time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"test - time - M*V = "<<time<<"\n";
	}
	
	//**** A1*A2 ****
	{
		start=std::clock();
		Eigen::MatrixXd A1,A2,M;
		A1.resize(17,17);
		A2.resize(17,17);
		M.resize(17,17);
		for(unsigned int i=0; i<N; ++i){
			A1=Eigen::MatrixXd::Random(17,17);
			A2=Eigen::MatrixXd::Random(17,17);
			M.noalias()=A1*A2;
		}
		stop=std::clock();
		time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"test - time - A1*A2 = "<<time<<"\n";
	}
	
	//**** A*A^T ****
	{
		start=std::clock();
		Eigen::MatrixXd A,M;
		A.resize(17,17);
		M.resize(17,17);
		for(unsigned int i=0; i<N; ++i){
			A=Eigen::MatrixXd::Random(17,17);
			M.noalias()=A*A.transpose();
		}
		stop=std::clock();
		time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"test - time - A*A^T = "<<time<<"\n";
	}
}