// c
#include <cstdlib>
#include <ctime>
// c++
#include <iostream>
#include <random>
// math
#include "math/const.hpp"
// structure
#include "struc/structure.hpp"
#include "struc/neighbor.hpp"
// str
#include "str/print.hpp"
// chem
#include "chem/units.hpp"
// torch
#include "torch/kspace_london.hpp"
#include "torch/pot_lj_cut.hpp"
#include "torch/pot_lj_long.hpp"

Structure make_struc(){
	const double a0=5.3972587586;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=false; atomT.c6=false;
	Structure struc;
	const int natoms=4;
	struc.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	struc.init(lv);
	struc.posn(0)<<0.0,0.0,0.0;
	struc.posn(1)<<0.0,0.5,0.5;
	struc.posn(2)<<0.5,0.0,0.5;
	struc.posn(3)<<0.5,0.5,0.0;
	for(int i=0; i<natoms; ++i){
		struc.posn(i)*=a0;
		struc.name(i)="Ar";
		struc.type(i)=0;
		struc.index(i)=i;
	}
	Structure ssuper;
	Eigen::Vector3i nlat;
	nlat<<3,3,3;
	Structure::super(struc,ssuper,nlat);
	return ssuper;
}

Structure make_struc(double a, int l){
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=false; atomT.c6=false;
	Structure struc;
	const int natoms=4;
	struc.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a;
	struc.init(lv);
	struc.posn(0)<<0.0,0.0,0.0;
	struc.posn(1)<<0.0,0.5,0.5;
	struc.posn(2)<<0.5,0.0,0.5;
	struc.posn(3)<<0.5,0.5,0.0;
	for(int i=0; i<natoms; ++i){
		struc.posn(i)*=a;
		struc.name(i)="Ar";
		struc.type(i)=0;
		struc.index(i)=i;
	}
	Structure ssuper;
	Eigen::Vector3i nlat;
	nlat<<l,l,l;
	Structure::super(struc,ssuper,nlat);
	return ssuper;
}

void test_energy_london(){
	//begin
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - LONDON - LONG - ENERGY",str)<<"\n";
	//random
	std::default_random_engine generator(std::time(NULL));
	std::uniform_real_distribution<double> udist(0.0,1.0);
	//potentials
	std::srand(std::time(NULL));
	int N=10;
	const double dr=0.3;
	std::vector<double> ecut(N);
	std::vector<double> elong(N);
	ptnl::PotLJCut pCut;
	ptnl::PotLJLong pLong;
	//make structure
	const double a=(5.5-5.0)*udist(generator)+5.0;
	const int l=3;
	Structure struc=make_struc(a,l);
	std::cout<<"constructing FCC Lattice\n";
	std::cout<<"structure:\n";
	std::cout<<"\ta = "<<a<<"\n";
	std::cout<<"\tl = "<<l<<"\n";
	std::cout<<"\tn = "<<struc.nAtoms()<<"\n";
	//set potential parameters
	std::cout<<"setting potential parameters\n";
	const double rcl=12;
	const double rcc=rcl*15;
	const double e=(0.1-0.0)*udist(generator)+0.0;
	const double s=(5.0-1.0)*udist(generator)+1.0;
	const double s6=s*s*s*s*s*s;
	const double c6=4.0*e*s6;
	Eigen::MatrixXd b=Eigen::MatrixXd::Constant(1,1,c6);
	//set potential - cut
	std::cout<<"setting potential - cut\n";
	pCut.rc()=rcc;
	pCut.resize(1);
	pCut.e()(0,0)=e;
	pCut.s()(0,0)=s;
	pCut.init();
	//set potential - long
	std::cout<<"setting potential - long\n";
	pLong.rc()=rcl;
	pLong.resize(1);
	pLong.e()(0,0)=e;
	pLong.s()(0,0)=s;
	pLong.s6()(0,0)=s6;
	pLong.ksl().prec()=1e-12;
	pLong.ksl().rc()=rcl;
	pLong.ksl().init(struc,b);
	pLong.init();
	//print potential
	std::cout<<"k-space prec = "<<pLong.ksl().prec()<<"\n";
	std::cout<<"rcut - long  = "<<rcl<<"\n";
	std::cout<<"rcut - short = "<<rcc<<"\n";
	std::cout<<"epsilon      = "<<e<<"\n";
	std::cout<<"sigma        = "<<s<<"\n";
	//compute energy
	std::cout<<"computing energy - random perturbations\n";
	std::cout<<"N - struc = "<<N<<"\n";
	NeighborList ncut; ncut.rc()=rcc;
	NeighborList nlong; nlong.rc()=rcl;
	std::cout<<"# ecut elong\n";
	for(int n=0; n<N; ++n){
		Structure tmp=struc;
		for(int i=0; i<struc.nAtoms(); ++i){
			tmp.posn(i).noalias()+=Eigen::Vector3d::Random()*dr;
		}
		ncut.build(tmp,rcc);
		nlong.build(tmp,rcl);
		ecut[n]=pCut.energy(tmp,ncut);
		elong[n]=pLong.energy(tmp,nlong);
		std::cout<<ecut[n]<<" "<<elong[n]<<"\n";
	}
	//compute error
	double error=0;
	for(int i=0; i<N; ++i){
		error+=std::fabs(ecut[i]-elong[i]);
	}
	error/=(N*struc.nAtoms());
	//print
	std::cout<<"error (energy/atom) = "<<error<<"\n";
	//end
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_london_prec(){
	//begin
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - LONDON - LONG - PREC",str)<<"\n";
	//random
	std::default_random_engine generator(std::time(NULL));
	std::uniform_real_distribution<double> udist(0.0,1.0);
	//variables
	std::srand(std::time(NULL));
	ptnl::PotLJLong pLong;
	//make structure
	const double a=(5.5-5.0)*udist(generator)+5.0;
	const int l=3;
	Structure struc=make_struc(a,l);
	std::cout<<"constructing FCC Lattice\n";
	std::cout<<"structure:\n";
	std::cout<<"\ta = "<<a<<"\n";
	std::cout<<"\tl = "<<l<<"\n";
	std::cout<<"\tn = "<<struc.nAtoms()<<"\n";
	//set potential parameters
	std::cout<<"setting potential parameters\n";
	const double rcl=12;
	const double e=(0.1-0.0)*udist(generator)+0.0;
	const double s=(5.0-1.0)*udist(generator)+1.0;
	std::cout<<"epsilon = "<<e<<"\n";
	std::cout<<"sigma   = "<<s<<"\n";
	std::cout<<"rcut    = "<<rcl<<"\n";
	//precision
	const int N=12;
	double prec[N]={1e-2,1e-3,1e-4,1e-5,1e-6,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14};
	std::vector<double> elong(N);
	std::cout<<"# prec elong\n";
	for(int n=0; n<N; ++n){
		//set potential - long
		pLong.rc()=rcl;
		pLong.resize(1);
		pLong.e()(0,0)=e;
		pLong.s()(0,0)=s;
		pLong.ksl().prec()=prec[n];
		pLong.ksl().rc()=rcl;
		pLong.init();
		//compute energy
		NeighborList nlong; nlong.rc()=rcl;
		nlong.build(struc,rcl);
		elong[n]=pLong.energy(struc,nlong);
		std::cout<<prec[n]<<" "<<elong[n]<<"\n";
	}
	//end
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_london_cut(){
	//begin
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - LONDON - CUT - CUT",str)<<"\n";
	//random
	std::default_random_engine generator(std::time(NULL));
	std::uniform_real_distribution<double> udist(0.0,1.0);
	//variables
	ptnl::PotLJCut pCut;
	//make structure
	const double a=(5.5-5.0)*udist(generator)+5.0;
	const int l=3;
	Structure struc=make_struc(a,l);
	std::cout<<"constructing FCC Lattice\n";
	std::cout<<"structure:\n";
	std::cout<<"\ta = "<<a<<"\n";
	std::cout<<"\tl = "<<l<<"\n";
	std::cout<<"\tn = "<<struc.nAtoms()<<"\n";
	//set potential parameters
	std::cout<<"setting potential parameters\n";
	const double e=(0.1-0.0)*udist(generator)+0.0;
	const double s=(5.0-1.0)*udist(generator)+1.0;
	std::cout<<"epsilon = "<<e<<"\n";
	std::cout<<"sigma   = "<<s<<"\n";
	//precision
	const int N=50;
	std::vector<double> rcut(N);
	for(int n=0; n<N; ++n){
		rcut[n]=10.0+2.0*n;
	}
	std::vector<double> ecut(N);
	std::cout<<"# rcut ecut\n";
	for(int n=0; n<N; ++n){
		//set potential - long
		pCut.rc()=rcut[n];
		pCut.resize(1);
		pCut.e()(0,0)=e;
		pCut.s()(0,0)=s;
		pCut.init();
		//compute energy
		NeighborList nlong; nlong.rc()=rcut[n];
		nlong.build(struc,rcut[n]);
		ecut[n]=pCut.energy(struc,nlong);
		std::cout<<rcut[n]<<" "<<ecut[n]<<"\n";
	}
	//end
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_time_cut(){
	//variables
	std::srand(std::time(NULL));
	int N=100;
	const double dr=0.3;
	std::vector<double> ecut(N);
	ptnl::PotLJCut pCut;
	//make structure
	Structure struc=make_struc();
	std::cout<<struc<<"\n";
	//set potential
	std::cout<<"setting potential\n";
	const double rc=20;
	const double e=0.011;
	const double s=3.345;
	pCut.rc()=rc;
	pCut.resize(1);
	pCut.e()(0,0)=0.011;
	pCut.s()(0,0)=s;
	pCut.init();
	//compute energy
	std::cout<<"computing energy\n";
	clock_t start_=std::clock();
	NeighborList ncut; ncut.rc()=rc;
	for(int n=0; n<N; ++n){
		Structure tmp=struc;
		for(int i=0; i<struc.nAtoms(); ++i){
			tmp.posn(i).noalias()+=Eigen::Vector3d::Random()*dr;
		}
		ncut.build(tmp,rc);
		ecut[n]=pCut.energy(tmp,ncut);
	}
	clock_t stop_=std::clock();
	double time_=((double)stop_-start_)/CLOCKS_PER_SEC;
	std::cout<<"time - cut = "<<time_<<"\n";
}

void test_time_long(){
	//variables
	std::srand(std::time(NULL));
	int N=100;
	const double dr=0.3;
	std::vector<double> elong(N);
	ptnl::PotLJLong pLong;
	//make structure
	Structure struc=make_struc();
	std::cout<<struc<<"\n";
	//set potential
	std::cout<<"setting potential\n";
	const double rc=9;
	const double e=0.011;
	const double s=3.345;
	pLong.resize(1);
	pLong.rc()=rc;
	pLong.e()(0,0)=e;
	pLong.s()(0,0)=s;
	pLong.ksl().prec()=1e-12;
	pLong.ksl().rc()=rc;
	pLong.init();
	//compute energy
	std::cout<<"computing energy\n";
	clock_t start_=std::clock();
	NeighborList nlong; nlong.rc()=rc;
	for(int n=0; n<N; ++n){
		Structure tmp=struc;
		for(int i=0; i<struc.nAtoms(); ++i){
			tmp.posn(i).noalias()+=Eigen::Vector3d::Random()*dr;
		}
		nlong.build(tmp,rc);
		elong[n]=pLong.energy(tmp,nlong);
	}
	clock_t stop_=std::clock();
	double time_=((double)stop_-start_)/CLOCKS_PER_SEC;
	std::cout<<"time - long = "<<time_<<"\n";
}

int main(int argc, char* argv[]){
	
	test_energy_london();
	test_london_prec();
	test_london_cut();
	//test_time_cut();
	//test_time_long();
	
	return 0;
}