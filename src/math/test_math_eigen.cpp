// c
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
// c++
#include <iostream>
#include <vector>
// eigen
#include <Eigen/Dense>
//str
#include "str/print.hpp"

using namespace std::chrono;

void test_mat_mult(){
	std::srand(std::time(NULL));
	//constants
	const int N=1000;
	//test
	double error=0.0;
	for(int i=0; i<N; ++i){
		Eigen::Vector3d v1=Eigen::Vector3d::Random();
		const Eigen::Matrix3d mat=Eigen::Matrix3d::Random();
		const Eigen::Vector3d v2=mat*v1;
		v1=mat*v1;
		error+=(v2-v1).norm();
	}
	error/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - MAT - MULT",str)<<"\n";
	std::cout<<"N     = "<<N<<"\n";
	std::cout<<"error = "<<error<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

int main(int argc, char* argv[]){
	
	test_mat_mult();
	
	Eigen::Vector3d vec=Eigen::Vector3d::Random();
	
	std::cout<<"vec = "<<vec.transpose()<<"\n";
	std::cout<<"arr = "<<1.0/vec.array()<<"\n";
	
	return 0;
}