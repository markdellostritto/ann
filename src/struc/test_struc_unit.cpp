// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// structure
#include "struc/structure.hpp"
// units
#include "chem/units.hpp"
// str
#include "str/print.hpp"

//**********************************************
// LJ
//**********************************************

struct LJ{
private:
	double eps_,sigma_;
public:
	//==== constructors/destructors ====
	LJ(){}
	LJ(double eps, double sigma):eps_(eps),sigma_(sigma){}
	~LJ(){}
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& sigma(){return sigma_;}
	const double& sigma()const{return sigma_;}
	
	//==== operators ====
	double operator()(double r){
		const double x=sigma_/r;
		const double x6=x*x*x*x*x*x;
		return 4.0*eps_*(x6*x6-x6);
	}
};

//**********************************************
// structure
//**********************************************

void test_unit_struc(){
	char* buf=new char[print::len_buf];
	std::cout<<print::buf(buf,print::char_buf)<<"\n";
	//generate Ar crystal
	units::System unitsys=units::System::METAL;
	units::consts::init(unitsys);
	const double a0=10.512;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=true;
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
	std::cout<<"struc = "<<struc<<"\n";
	/*for(int i=0; i<struc.nAtoms(); ++i){
		std::cout<<struc.name(i)<<struc.index(i)+1<<" "<<struc.posn(i).transpose()<<"\n";
	}
	for(int i=0; i<struc.nAtoms(); ++i){
		std::cout<<struc.name(i)<<struc.index(i)+1<<" "<<struc.symm(i).transpose()<<"\n";
	}*/
	//pack
	int size=serialize::nbytes(struc);
	std::cout<<"size = "<<size<<"\n";
	char* memarr=new char[size];
	std::cout<<"packing structure\n";
	serialize::pack(struc,memarr);
	//unpack
	Structure struc_new;
	std::cout<<"unpacking structure\n";
	serialize::unpack(struc_new,memarr);
	std::cout<<"struc_new = "<<struc_new<<"\n";
	/*for(int i=0; i<struc_new.nAtoms(); ++i){
		std::cout<<struc_new.name(i)<<struc_new.index(i)+1<<" "<<struc_new.posn(i).transpose()<<"\n";
	}
	for(int i=0; i<struc_new.nAtoms(); ++i){
		std::cout<<struc_new.name(i)<<struc_new.index(i)+1<<" "<<struc_new.symm(i).transpose()<<"\n";
	}*/
	//compute error
	double err_posn=0;
	for(int i=0; i<32; ++i){
		err_posn+=(struc.posn(i)-struc_new.posn(i)).norm();
	}
	double err_symm=0;
	for(int i=0; i<32; ++i){
		err_symm+=(struc.symm(i)-struc_new.symm(i)).norm();
	}
	std::cout<<"err_lv   = "<<(struc.R()-struc_new.R()).norm()<<"\n";
	std::cout<<"err_posn = "<<err_posn<<"\n";
	std::cout<<"err_symm = "<<err_symm<<"\n";
	std::cout<<print::buf(buf,print::char_buf)<<"\n";
	delete[] buf;
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("STRUC",str)<<"\n";
	test_unit_struc();
	std::cout<<print::title("STRUC",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}