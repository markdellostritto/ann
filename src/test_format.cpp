// c libaries
#include <ctime>
#include <cstring>
// c++ libaries
#include <iostream>
// ann - math
#include "accumulator.hpp"
// ann - eigen
#include "eigen.hpp"
// ann - string
#include "string.hpp"
// ann - structure
#include "structure.hpp"
// ann - format
#include "ann.hpp"
#include "xyz.hpp"
// ann - units
#include "units.hpp"
// ann - compiler
#include "compiler.hpp"
// ann - print
#include "print.hpp"

static Structure make_struc(){
	//generate NaCl crystal
	double a0=5.6199998856;
	AtomType atomT; 
	atomT.name=true; atomT.an=false; atomT.index=false; atomT.type=false;
	atomT.charge=true; atomT.posn=true; atomT.symm=true;
	Structure strucg;
	const int natoms=8;
	strucg.resize(natoms,atomT);
	const Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	strucg.init(lv);
	strucg.posn(0)<<0.000000000,0.000000000,0.000000000;
	strucg.posn(1)<<0.000000000,0.500000000,0.500000000;
	strucg.posn(2)<<0.500000000,0.000000000,0.500000000;
	strucg.posn(3)<<0.500000000,0.500000000,0.000000000;
	strucg.posn(4)<<0.500000000,0.500000000,0.500000000;
	strucg.posn(5)<<0.500000000,0.000000000,0.000000000;
	strucg.posn(6)<<0.000000000,0.500000000,0.000000000;
	strucg.posn(7)<<0.000000000,0.000000000,0.500000000;
	for(int i=0; i<8; ++i) strucg.posn(i)*=a0;
	for(int i=0; i<4; ++i) strucg.charge(i)=1;
	for(int i=4; i<8; ++i) strucg.charge(i)=-1;
	for(int i=0; i<4; ++i) strucg.name(i)="Na";
	for(int i=4; i<8; ++i) strucg.name(i)="Cl";
	for(int i=0; i<8; ++i) strucg.symm(i)=Eigen::VectorXd::Random(8);
	strucg.energy()=-4.5892768;
	//return structure
	return strucg;
}

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	//ann
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("ANN",str)<<"\n";
	{
		Structure struc1,struc2;
		struc1=make_struc();
		//print structure
		std::cout<<"*************************************************************************\n";
		std::cout<<struc1<<"\n";
		for(int i=0; i<8; ++i){
			std::cout<<struc1.name(i)<<" "<<struc1.charge(i)<<" "<<struc1.posn(i).transpose()<<" "<<struc1.symm(i).transpose()<<"\n";
		}
		std::cout<<"*************************************************************************\n";
		//write to file
		ANN::write("test_format.ann",struc1.atomType(),struc1);
		//read from file
		ANN::read("test_format.ann",struc1.atomType(),struc2);
		//write structures to array
		const int size1=serialize::nbytes(struc1);
		const int size2=serialize::nbytes(struc2);
		std::cout<<"*************************************************************************\n";
		std::cout<<struc2<<"\n";
		for(int i=0; i<8; ++i){
			std::cout<<struc2.name(i)<<" "<<struc2.charge(i)<<" "<<struc2.posn(i).transpose()<<" "<<struc2.symm(i).transpose()<<"\n";
		}
		std::cout<<"*************************************************************************\n";
		std::cout<<"err - size = "<<std::abs(size1-size2)<<"\n";
		if(size1==size2){
			char* arr1=new char[size1];
			char* arr2=new char[size2];
			serialize::pack(struc1,arr1);
			serialize::pack(struc2,arr2);
			std::cout<<"err - byte = "<<std::memcmp(arr1,arr2,size1)<<"\n";
		}
	}
	std::cout<<print::title("ANN",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	//xyz
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("XYZ",str)<<"\n";
	{
		Structure struc1,struc2;
		struc1=make_struc();
		//print structure
		std::cout<<"*************************************************************************\n";
		std::cout<<struc1<<"\n";
		for(int i=0; i<8; ++i){
			std::cout<<struc1.name(i)<<" "<<struc1.charge(i)<<" "<<struc1.posn(i).transpose()<<" "<<struc1.symm(i).transpose()<<"\n";
		}
		std::cout<<"*************************************************************************\n";
		//write to file
		XYZ::write("test_format.xyz",struc1.atomType(),struc1);
		//read from file
		XYZ::read("test_format.xyz",struc1.atomType(),struc2);
		//write structures to array
		const int size1=serialize::nbytes(struc1);
		const int size2=serialize::nbytes(struc2);
		std::cout<<"*************************************************************************\n";
		std::cout<<struc2<<"\n";
		for(int i=0; i<8; ++i){
			std::cout<<struc2.name(i)<<" "<<struc2.charge(i)<<" "<<struc2.posn(i).transpose()<<" "<<struc2.symm(i).transpose()<<"\n";
		}
		std::cout<<"*************************************************************************\n";
		std::cout<<"err - size = "<<std::abs(size1-size2)<<"\n";
		if(size1==size2){
			char* arr1=new char[size1];
			char* arr2=new char[size2];
			serialize::pack(struc1,arr1);
			serialize::pack(struc2,arr2);
			std::cout<<"err - byte = "<<std::memcmp(arr1,arr2,size1)<<"\n";
		}
	}
	std::cout<<print::title("XYZ",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
}