// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// structure
#include "struc/structure.hpp"
#include "struc/neighbor.hpp"
#include "struc/cell_list.hpp"
// units
#include "chem/units.hpp"
// str
#include "str/print.hpp"

void test_neighbor_super(){
	//units
	units::System unitsys=units::System::METAL;
	units::consts::init(unitsys);
	//rand
	std::srand(std::time(NULL));
	
	//atom type
	AtomType atomT; 
	atomT.name=true; atomT.an=false; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=false;
	//structure parameters
	const double a0=4.0;
	const double skew=0.2;
	const int ntypes=3;
	const int napt=5;
	const int natoms=ntypes*napt;
	//neighbor parameters
	const double rc=12.0;
	const int nsup=std::ceil(rc/a0);
	const Eigen::Vector3i nlat=Eigen::Vector3i::Constant(nsup)*2;
	const int nlatp=nlat.prod();
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"nlat = "<<nlat.transpose()<<"\n";
	std::cout<<"nlatp = "<<nlatp<<"\n";
	
	//make the structure
	Structure struc;
	struc.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	lv(0,0)+=((1.0*std::rand())/RAND_MAX-0.5)*0.1*a0;
	lv(1,1)+=((1.0*std::rand())/RAND_MAX-0.5)*0.1*a0;
	lv(2,2)+=((1.0*std::rand())/RAND_MAX-0.5)*0.1*a0;
	lv.noalias()+=Eigen::Matrix3d::Random()*skew;
	struc.init(lv);
	Eigen::Matrix3d lvi=lv.inverse();
	
	//set the positions
	int c=0;
	for(int i=0; i<ntypes; ++i){
		for(int j=0; j<napt; ++j){
			struc.type(c)=i;
			struc.index(c)=c;
			Eigen::Vector3d posn;
			posn[0]=(1.0*std::rand())/RAND_MAX;
			posn[1]=(1.0*std::rand())/RAND_MAX;
			posn[2]=(1.0*std::rand())/RAND_MAX;
			struc.posn(c).noalias()=lv*posn;
			c++;
		}
	}
	
	std::cout<<struc<<"\n";
	for(int i=0; i<struc.nAtoms(); ++i){
		std::cout<<struc.posn(i).transpose()<<"\n";
	}
	
	//construct neighbor list
	NeighborList nlist;
	nlist.build(struc,rc);
	
	//make super cell
	Structure super;
	Structure::super(struc,super,nlat);
	
	//make super list
	NeighborList slist;
	slist.build(super,rc);
	
	//compare distances - self
	std::cout<<"searching for identical distance vectors in the neighbor list\n";
	int errc=0;
	for(int n=0; n<struc.nAtoms(); ++n){
		for(int i=0; i<nlist.size(n); ++i){
			for(int j=i+1; j<nlist.size(n); ++j){
				if((nlist.neigh(n,i).r()-nlist.neigh(n,j).r()).norm()<1e-6) errc++;
			}
		}
	}
	std::cout<<"err - self = "<<errc<<"\n";
	
	//compare distances - super
	int errm=0;
	for(int n=0; n<struc.nAtoms(); ++n){
		for(int i=0; i<nlist.size(n); ++i){
			for(int k=0; k<nlatp; ++k){
				bool match=false;
				for(int j=0; j<slist.size(n+k*struc.nAtoms()); ++j){
					if((nlist.neigh(n,i).r()-slist.neigh(n+k*struc.nAtoms(),j).r()).norm()<1e-6) match=true;
				}
				if(!match) errm++;
			}
		}
	}
	std::cout<<"err - match = "<<errm<<"\n";
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("TEST - NEIGHBOR - SUPER",str)<<"\n";
	test_neighbor_super();
	std::cout<<print::title("TEST - NEIGHBOR - SUPER",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}