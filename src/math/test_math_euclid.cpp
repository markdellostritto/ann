// c
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
// c++
#include <iostream>
#include <vector>
//math
#include "math/euclid.hpp"
//str
#include "str/print.hpp"
//util
#include "util/time.hpp"

using namespace std::chrono;

void test_rotate(){
	std::srand(std::time(NULL));
	//constants
	const int N=1000;
	//test
	double error_unit=0.0;
	double error_norm=0.0;
	for(int i=0; i<N; ++i){
		const Eigen::Vector3d v1=Eigen::Vector3d::Random();
		const Eigen::Vector3d v2=Eigen::Vector3d::Random();
		const Eigen::Matrix3d mat=math::euclid::rotate(v1,v2);
		const Eigen::Vector3d v3=mat*v1;
		error_unit+=(v3/v3.norm()-v2/v2.norm()).norm();
		error_norm+=fabs(v1.norm()-v3.norm());
	}
	error_unit/=N;
	error_norm/=N;
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - ROTATE",str)<<"\n";
	std::cout<<"error - unit = "<<error_unit<<"\n";
	std::cout<<"error - norm = "<<error_norm<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

int main(int argc, char* argv[]){
	
	test_rotate();
	
	return 0;
}