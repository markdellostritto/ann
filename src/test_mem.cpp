// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
#include <iomanip>
// ann - math
#include "math_const.hpp"
#include "math_special.hpp"
#include "accumulator.hpp"
// ann - eigen
#include "eigen.hpp"
// ann - string
#include "string.hpp"
// ann - optimization
#include "optimize.hpp"
// ann - ewald
#include "ewald3D.hpp"
// ann - cutoff
#include "cutoff.hpp"
// ann - symmetry functions
#include "symm_radial_t1.hpp"
#include "symm_radial_g1.hpp"
#include "symm_radial_g2.hpp"
#include "symm_angular_g3.hpp"
#include "symm_angular_g4.hpp"
// ann - neural network
#include "nn.hpp"
// ann - neural network potential
#include "nn_pot.hpp"
// ann - structure
#include "structure.hpp"
#include "cell_list.hpp"
// ann - units
#include "units.hpp"
// ann - compiler
#include "compiler.hpp"
// ann - print
#include "print.hpp"
// ann - random
#include "random.hpp"
// ann - test - unit
#include "test_mem.hpp"

int main(int argc, char* argv[]){
	
	char* str=new char[print::len_buf];
	
	//**********************************************
	// symm
	//**********************************************
	
	std::cout<<print::title("SYMM - T1",str)<<"\n";
	{
		PhiR_T1 obj(1.4268,0.56278905);
		test_mem(obj);
	}
	std::cout<<print::title("SYMM - G1",str)<<"\n";
	{
		PhiR_G1 obj;
		test_mem(obj);
	}
	std::cout<<print::title("SYMM - G2",str)<<"\n";
	{
		PhiR_G2 obj(1.4268,0.56278905);
		test_mem(obj);
	}
	std::cout<<print::title("SYMM - G3",str)<<"\n";
	{
		PhiA_G3 obj(1.4268,2.5,1);
		test_mem(obj);
	}
	std::cout<<print::title("SYMM - G4",str)<<"\n";
	{
		PhiA_G4 obj(1.4268,2.5,1);
		test_mem(obj);
	}
	
	//**********************************************
	// eigen
	//**********************************************
	
	std::cout<<print::title("Eigen Vector 3d",str)<<"\n";
	{
		Eigen::Vector3d obj=Eigen::Vector3d::Random();
		test_mem(obj);
	}
	std::cout<<print::title("Eigen Vector xd",str)<<"\n";
	{
		Eigen::VectorXd obj=Eigen::VectorXd::Random(7);
		test_mem(obj);
	}
	std::cout<<print::title("Eigen Matrix 3d",str)<<"\n";
	{
		Eigen::Matrix3d obj=Eigen::Matrix3d::Random();
		test_mem(obj);
	}
	std::cout<<print::title("Eigen Matrix xd",str)<<"\n";
	{
		Eigen::MatrixXd obj=Eigen::MatrixXd::Random(5,7);
		test_mem(obj);
	}
	
	//**********************************************
	// ewald
	//**********************************************
	
	std::cout<<print::title("Ewald 3D Coulomb",str)<<"\n";
	{
		units::System::type unitsys=units::System::AU;
		units::consts::init(unitsys);
		double a0=5.6199998856;
		if(unitsys==units::System::AU) a0*=units::BOHRpANG;
		AtomType atomT; 
		atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
		atomT.charge=true; atomT.posn=true;
		Structure strucg;
		const int natoms=8;
		strucg.resize(natoms,atomT);
		Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
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
		//ewald
		Ewald3D::Coulomb ewald;
		//test
		test_mem(ewald);
	}
	
	//**********************************************
	// nn
	//**********************************************
	
	std::cout<<print::title("NN",str)<<"\n";
	{
		//local function variables
		NN::Network nn;
		//resize the nn
		nn.idev()=1.0;
		nn.initType()=NN::InitN::HE;
		nn.tfType()=NN::TransferN::TANH;
		std::vector<int> nh(2);
		nh[0]=7; nh[1]=5;
		nn.resize(2,nh,3);
		//test
		test_mem(nn);
	}
	
	//**********************************************
	// nn pot
	//**********************************************
	
	std::cout<<print::title("NNH",str)<<"\n";
	{
		NNH nnh,nnh_copy;
		//initialize neural network
		std::vector<int> nh(4);
		nh[0]=12; nh[1]=8; nh[2]=4; nh[3]=2;
		nnh.nn().idev()=1.0;
		nnh.nn().initType()=NN::InitN::HE;
		nnh.nn().tfType()=NN::TransferN::ARCTAN;
		nnh.nn().seed()=-1;
		nnh.nn().resize(nh);
		//initialize basis
		BasisR basisR; basisR.init_G2(6,cutoff::Name::COS,10.0);
		BasisA basisA; basisA.init_G4(4,cutoff::Name::COS,10.0);
		std::vector<Atom> species(2);
		species[0].name()="Ar";
		species[0].id()=string::hash("Ar");
		species[0].mass()=5.0;
		species[0].energy()=-7.0;
		species[1].name()="Ne";
		species[1].id()=string::hash("Ne");
		species[1].mass()=12.0;
		species[1].energy()=-9.0;
		nnh.atom()=species[0];
		nnh.rc()=10.0;
		nnh.resize(species);
		nnh.basisR(0)=basisR;
		nnh.basisR(1)=basisR;
		nnh.basisA(0,0)=basisA;
		nnh.basisA(0,1)=basisA;
		nnh.basisA(1,1)=basisA;
		nnh.init_input();
		//test
		test_mem(nnh);
	}
	
	std::cout<<print::title("NNPOT",str)<<"\n";
	{
		NNH nnh;
		NNPot nnp;
		//initialize neural network
		std::vector<int> nh(4);
		nh[0]=12; nh[1]=8; nh[2]=4; nh[3]=2;
		nnh.nn().idev()=1.0;
		nnh.nn().initType()=NN::InitN::HE;
		nnh.nn().tfType()=NN::TransferN::ARCTAN;
		nnh.nn().seed()=-1;
		nnh.nn().resize(nh);
		//initialize basis
		BasisR basisR; basisR.init_G2(6,cutoff::Name::COS,10.0);
		BasisA basisA; basisA.init_G4(4,cutoff::Name::COS,10.0);
		std::vector<Atom> species(2);
		species[0].name()="Ar";
		species[0].id()=string::hash("Ar");
		species[0].mass()=5.0;
		species[0].energy()=-7.0;
		species[1].name()="Ne";
		species[1].id()=string::hash("Ne");
		species[1].mass()=12.0;
		species[1].energy()=-9.0;
		nnh.rc()=10.0;
		nnh.atom()=species[0];
		nnh.resize(species);
		nnh.basisR(0)=basisR;
		nnh.basisR(1)=basisR;
		nnh.basisA(0,0)=basisA;
		nnh.basisA(0,1)=basisA;
		nnh.basisA(1,1)=basisA;
		nnh.init_input();
		//resize potential
		nnp.resize(species);
		nnp.nnh(0)=nnh;
		nnp.nnh(1)=nnh;
		//test
		test_mem(nnp);
	}
	
	//**********************************************
	// struc
	//**********************************************
	
	std::cout<<print::title("STRUC",str)<<"\n";
	{
		//generate Ar crystal
		units::System::type unitsys=units::System::METAL;
		units::consts::init(unitsys);
		const double a0=10.512;
		AtomType atomT; 
		atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
		atomT.charge=true; atomT.posn=true; atomT.symm=true;
		Structure struc;
		const int natoms=32;
		struc.resize(natoms,atomT);
		Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
		struc.init(lv);
		struc.energy()=-8.527689257;
		struc.posn(0)<<0,0,0;
		struc.posn(1)<<0,0,5.25600004195;
		struc.posn(2)<<0,5.25600004195,0;
		struc.posn(3)<<0,5.25600004195,5.25600004195;
		struc.posn(4)<<5.25600004195,0,0;
		struc.posn(5)<<5.25600004195,0,5.25600004195;
		struc.posn(6)<<5.25600004195,5.25600004195,0;
		struc.posn(7)<<5.25600004195,5.25600004195,5.25600004195;
		struc.posn(8)<<0,2.628000020975,2.628000020975;
		struc.posn(9)<<0,2.628000020975,7.883999821149;
		struc.posn(0)<<0,7.883999821149,2.628000020975;
		struc.posn(11)<<0,7.883999821149,7.883999821149;
		struc.posn(12)<<5.25600004195,2.628000020975,2.628000020975;
		struc.posn(13)<<5.25600004195,2.628000020975,7.883999821149;
		struc.posn(14)<<5.25600004195,7.883999821149,2.628000020975;
		struc.posn(15)<<5.25600004195,7.883999821149,7.883999821149;
		struc.posn(16)<<2.628000020975,0,2.628000020975;
		struc.posn(17)<<2.628000020975,0,7.883999821149;
		struc.posn(18)<<2.628000020975,5.25600004195,2.628000020975;
		struc.posn(19)<<2.628000020975,5.25600004195,7.883999821149;
		struc.posn(20)<<7.883999821149,0,2.628000020975;
		struc.posn(21)<<7.883999821149,0,7.883999821149;
		struc.posn(22)<<7.883999821149,5.25600004195,2.628000020975;
		struc.posn(23)<<7.883999821149,5.25600004195,7.883999821149;
		struc.posn(24)<<2.628000020975,2.628000020975,0;
		struc.posn(25)<<2.628000020975,2.628000020975,5.25600004195;
		struc.posn(26)<<2.628000020975,7.883999821149,0;
		struc.posn(27)<<2.628000020975,7.883999821149,5.25600004195;
		struc.posn(28)<<7.883999821149,2.628000020975,0;
		struc.posn(29)<<7.883999821149,2.628000020975,5.25600004195;
		struc.posn(30)<<7.883999821149,7.883999821149,0;
		struc.posn(31)<<7.883999821149,7.883999821149,5.25600004195;
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.symm(i)=Eigen::VectorXd::Random(12);
		}
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.charge(i)=1.0;
		}
		//test
		test_mem(struc);
	}
	
	delete[] str;
}