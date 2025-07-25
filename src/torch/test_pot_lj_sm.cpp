// str
#include "str/string.hpp"
#include "str/print.hpp"
// chem
#include "chem/units.hpp"
// struc
#include "struc/structure.hpp"
#include "struc/neighbor.hpp"
// torch
#include "torch/pot_lj_sm.hpp"

Structure make_NaCl_pair(double d){
	const double a0=20;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=true; atomT.posn=true; atomT.force=true; atomT.symm=false;
	Structure struc;
	const int natoms=2;
	struc.resize(natoms,atomT);
	const Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	struc.init(lv);
	struc.pe()=0.0;
	struc.posn(0)<<0.0,0.0, 0.5*d;
	struc.posn(1)<<0.0,0.0,-0.5*d;
	struc.name(0)="Na";
	struc.name(1)="Cl";
	struc.type(0)=0;
	struc.type(1)=1;
	struc.charge(0)=1;
	struc.charge(1)=-1;
	return struc;
}

Structure make_NaCl_small(){
	const double a0=5.6199998856;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=true; atomT.posn=true; atomT.symm=false;
	Structure struc;
	const int natoms=2;
	struc.resize(natoms,atomT);
	Eigen::Matrix3d lv;
	lv<<0.5,0.5,0.0,0.5,0.0,0.5,0.0,0.5,0.5;
	lv*=a0;
	struc.init(lv);
	struc.pe()=0.0;
	struc.posn(0)<<0.25,0.25,0.25;
	struc.posn(1)<<-0.25,-0.25,-0.25;
	for(int i=0; i<natoms; ++i) struc.posn(i)*=a0;
	struc.name(0)="Na";
	struc.name(1)="Cl";
	struc.type(0)=0;
	struc.type(1)=1;
	struc.charge(0)=1;
	struc.charge(1)=-1;
	return struc;
}

Structure make_NaCl_large(){
	const double a0=16.8599987030;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=true; atomT.posn=true; atomT.symm=false;
	Structure struc;
	const int natoms=216;
	struc.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	struc.init(lv);
	struc.posn(0)<<0,0,0;
	struc.posn(1)<<0,0,0.333333343;
	struc.posn(2)<<0,0,0.666666687;
	struc.posn(3)<<0,0.333333343,0;
	struc.posn(4)<<0,0.333333343,0.333333343;
	struc.posn(5)<<0,0.333333343,0.666666687;
	struc.posn(6)<<0,0.666666687,0;
	struc.posn(7)<<0,0.666666687,0.333333343;
	struc.posn(8)<<0,0.666666687,0.666666687;
	struc.posn(9)<<0.333333343,0,0;
	struc.posn(10)<<0.333333343,0,0.333333343;
	struc.posn(11)<<0.333333343,0,0.666666687;
	struc.posn(12)<<0.333333343,0.333333343,0;
	struc.posn(13)<<0.333333343,0.333333343,0.333333343;
	struc.posn(14)<<0.333333343,0.333333343,0.666666687;
	struc.posn(15)<<0.333333343,0.666666687,0;
	struc.posn(16)<<0.333333343,0.666666687,0.333333343;
	struc.posn(17)<<0.333333343,0.666666687,0.666666687;
	struc.posn(18)<<0.666666687,0,0;
	struc.posn(19)<<0.666666687,0,0.333333343;
	struc.posn(20)<<0.666666687,0,0.666666687;
	struc.posn(21)<<0.666666687,0.333333343,0;
	struc.posn(22)<<0.666666687,0.333333343,0.333333343;
	struc.posn(23)<<0.666666687,0.333333343,0.666666687;
	struc.posn(24)<<0.666666687,0.666666687,0;
	struc.posn(25)<<0.666666687,0.666666687,0.333333343;
	struc.posn(26)<<0.666666687,0.666666687,0.666666687;
	struc.posn(27)<<0,0.166666672,0.166666672;
	struc.posn(28)<<0,0.166666672,0.5;
	struc.posn(29)<<0,0.166666672,0.833333313;
	struc.posn(30)<<0,0.5,0.166666672;
	struc.posn(31)<<0,0.5,0.5;
	struc.posn(32)<<0,0.5,0.833333313;
	struc.posn(33)<<0,0.833333313,0.166666672;
	struc.posn(34)<<0,0.833333313,0.5;
	struc.posn(35)<<0,0.833333313,0.833333313;
	struc.posn(36)<<0.333333343,0.166666672,0.166666672;
	struc.posn(37)<<0.333333343,0.166666672,0.5;
	struc.posn(38)<<0.333333343,0.166666672,0.833333313;
	struc.posn(39)<<0.333333343,0.5,0.166666672;
	struc.posn(40)<<0.333333343,0.5,0.5;
	struc.posn(41)<<0.333333343,0.5,0.833333313;
	struc.posn(42)<<0.333333343,0.833333313,0.166666672;
	struc.posn(43)<<0.333333343,0.833333313,0.5;
	struc.posn(44)<<0.333333343,0.833333313,0.833333313;
	struc.posn(45)<<0.666666687,0.166666672,0.166666672;
	struc.posn(46)<<0.666666687,0.166666672,0.5;
	struc.posn(47)<<0.666666687,0.166666672,0.833333313;
	struc.posn(48)<<0.666666687,0.5,0.166666672;
	struc.posn(49)<<0.666666687,0.5,0.5;
	struc.posn(50)<<0.666666687,0.5,0.833333313;
	struc.posn(51)<<0.666666687,0.833333313,0.166666672;
	struc.posn(52)<<0.666666687,0.833333313,0.5;
	struc.posn(53)<<0.666666687,0.833333313,0.833333313;
	struc.posn(54)<<0.166666672,0,0.166666672;
	struc.posn(55)<<0.166666672,0,0.5;
	struc.posn(56)<<0.166666672,0,0.833333313;
	struc.posn(57)<<0.166666672,0.333333343,0.166666672;
	struc.posn(58)<<0.166666672,0.333333343,0.5;
	struc.posn(59)<<0.166666672,0.333333343,0.833333313;
	struc.posn(60)<<0.166666672,0.666666687,0.166666672;
	struc.posn(61)<<0.166666672,0.666666687,0.5;
	struc.posn(62)<<0.166666672,0.666666687,0.833333313;
	struc.posn(63)<<0.5,0,0.166666672;
	struc.posn(64)<<0.5,0,0.5;
	struc.posn(65)<<0.5,0,0.833333313;
	struc.posn(66)<<0.5,0.333333343,0.166666672;
	struc.posn(67)<<0.5,0.333333343,0.5;
	struc.posn(68)<<0.5,0.333333343,0.833333313;
	struc.posn(69)<<0.5,0.666666687,0.166666672;
	struc.posn(70)<<0.5,0.666666687,0.5;
	struc.posn(71)<<0.5,0.666666687,0.833333313;
	struc.posn(72)<<0.833333313,0,0.166666672;
	struc.posn(73)<<0.833333313,0,0.5;
	struc.posn(74)<<0.833333313,0,0.833333313;
	struc.posn(75)<<0.833333313,0.333333343,0.166666672;
	struc.posn(76)<<0.833333313,0.333333343,0.5;
	struc.posn(77)<<0.833333313,0.333333343,0.833333313;
	struc.posn(78)<<0.833333313,0.666666687,0.166666672;
	struc.posn(79)<<0.833333313,0.666666687,0.5;
	struc.posn(80)<<0.833333313,0.666666687,0.833333313;
	struc.posn(81)<<0.166666672,0.166666672,0;
	struc.posn(82)<<0.166666672,0.166666672,0.333333343;
	struc.posn(83)<<0.166666672,0.166666672,0.666666687;
	struc.posn(84)<<0.166666672,0.5,0;
	struc.posn(85)<<0.166666672,0.5,0.333333343;
	struc.posn(86)<<0.166666672,0.5,0.666666687;
	struc.posn(87)<<0.166666672,0.833333313,0;
	struc.posn(88)<<0.166666672,0.833333313,0.333333343;
	struc.posn(89)<<0.166666672,0.833333313,0.666666687;
	struc.posn(90)<<0.5,0.166666672,0;
	struc.posn(91)<<0.5,0.166666672,0.333333343;
	struc.posn(92)<<0.5,0.166666672,0.666666687;
	struc.posn(93)<<0.5,0.5,0;
	struc.posn(94)<<0.5,0.5,0.333333343;
	struc.posn(95)<<0.5,0.5,0.666666687;
	struc.posn(96)<<0.5,0.833333313,0;
	struc.posn(97)<<0.5,0.833333313,0.333333343;
	struc.posn(98)<<0.5,0.833333313,0.666666687;
	struc.posn(99)<<0.833333313,0.166666672,0;
	struc.posn(100)<<0.833333313,0.166666672,0.333333343;
	struc.posn(101)<<0.833333313,0.166666672,0.666666687;
	struc.posn(102)<<0.833333313,0.5,0;
	struc.posn(103)<<0.833333313,0.5,0.333333343;
	struc.posn(104)<<0.833333313,0.5,0.666666687;
	struc.posn(105)<<0.833333313,0.833333313,0;
	struc.posn(106)<<0.833333313,0.833333313,0.333333343;
	struc.posn(107)<<0.833333313,0.833333313,0.666666687;
	struc.posn(108)<<0.166666672,0.166666672,0.166666672;
	struc.posn(109)<<0.166666672,0.166666672,0.5;
	struc.posn(110)<<0.166666672,0.166666672,0.833333313;
	struc.posn(111)<<0.166666672,0.5,0.166666672;
	struc.posn(112)<<0.166666672,0.5,0.5;
	struc.posn(113)<<0.166666672,0.5,0.833333313;
	struc.posn(114)<<0.166666672,0.833333313,0.166666672;
	struc.posn(115)<<0.166666672,0.833333313,0.5;
	struc.posn(116)<<0.166666672,0.833333313,0.833333313;
	struc.posn(117)<<0.5,0.166666672,0.166666672;
	struc.posn(118)<<0.5,0.166666672,0.5;
	struc.posn(119)<<0.5,0.166666672,0.833333313;
	struc.posn(120)<<0.5,0.5,0.166666672;
	struc.posn(121)<<0.5,0.5,0.5;
	struc.posn(122)<<0.5,0.5,0.833333313;
	struc.posn(123)<<0.5,0.833333313,0.166666672;
	struc.posn(124)<<0.5,0.833333313,0.5;
	struc.posn(125)<<0.5,0.833333313,0.833333313;
	struc.posn(126)<<0.833333313,0.166666672,0.166666672;
	struc.posn(127)<<0.833333313,0.166666672,0.5;
	struc.posn(128)<<0.833333313,0.166666672,0.833333313;
	struc.posn(129)<<0.833333313,0.5,0.166666672;
	struc.posn(130)<<0.833333313,0.5,0.5;
	struc.posn(131)<<0.833333313,0.5,0.833333313;
	struc.posn(132)<<0.833333313,0.833333313,0.166666672;
	struc.posn(133)<<0.833333313,0.833333313,0.5;
	struc.posn(134)<<0.833333313,0.833333313,0.833333313;
	struc.posn(135)<<0.166666672,0,0;
	struc.posn(136)<<0.166666672,0,0.333333343;
	struc.posn(137)<<0.166666672,0,0.666666687;
	struc.posn(138)<<0.166666672,0.333333343,0;
	struc.posn(139)<<0.166666672,0.333333343,0.333333343;
	struc.posn(140)<<0.166666672,0.333333343,0.666666687;
	struc.posn(141)<<0.166666672,0.666666687,0;
	struc.posn(142)<<0.166666672,0.666666687,0.333333343;
	struc.posn(143)<<0.166666672,0.666666687,0.666666687;
	struc.posn(144)<<0.5,0,0;
	struc.posn(145)<<0.5,0,0.333333343;
	struc.posn(146)<<0.5,0,0.666666687;
	struc.posn(147)<<0.5,0.333333343,0;
	struc.posn(148)<<0.5,0.333333343,0.333333343;
	struc.posn(149)<<0.5,0.333333343,0.666666687;
	struc.posn(150)<<0.5,0.666666687,0;
	struc.posn(151)<<0.5,0.666666687,0.333333343;
	struc.posn(152)<<0.5,0.666666687,0.666666687;
	struc.posn(153)<<0.833333313,0,0;
	struc.posn(154)<<0.833333313,0,0.333333343;
	struc.posn(155)<<0.833333313,0,0.666666687;
	struc.posn(156)<<0.833333313,0.333333343,0;
	struc.posn(157)<<0.833333313,0.333333343,0.333333343;
	struc.posn(158)<<0.833333313,0.333333343,0.666666687;
	struc.posn(159)<<0.833333313,0.666666687,0;
	struc.posn(160)<<0.833333313,0.666666687,0.333333343;
	struc.posn(161)<<0.833333313,0.666666687,0.666666687;
	struc.posn(162)<<0,0.166666672,0;
	struc.posn(163)<<0,0.166666672,0.333333343;
	struc.posn(164)<<0,0.166666672,0.666666687;
	struc.posn(165)<<0,0.5,0;
	struc.posn(166)<<0,0.5,0.333333343;
	struc.posn(167)<<0,0.5,0.666666687;
	struc.posn(168)<<0,0.833333313,0;
	struc.posn(169)<<0,0.833333313,0.333333343;
	struc.posn(170)<<0,0.833333313,0.666666687;
	struc.posn(171)<<0.333333343,0.166666672,0;
	struc.posn(172)<<0.333333343,0.166666672,0.333333343;
	struc.posn(173)<<0.333333343,0.166666672,0.666666687;
	struc.posn(174)<<0.333333343,0.5,0;
	struc.posn(175)<<0.333333343,0.5,0.333333343;
	struc.posn(176)<<0.333333343,0.5,0.666666687;
	struc.posn(177)<<0.333333343,0.833333313,0;
	struc.posn(178)<<0.333333343,0.833333313,0.333333343;
	struc.posn(179)<<0.333333343,0.833333313,0.666666687;
	struc.posn(180)<<0.666666687,0.166666672,0;
	struc.posn(181)<<0.666666687,0.166666672,0.333333343;
	struc.posn(182)<<0.666666687,0.166666672,0.666666687;
	struc.posn(183)<<0.666666687,0.5,0;
	struc.posn(184)<<0.666666687,0.5,0.333333343;
	struc.posn(185)<<0.666666687,0.5,0.666666687;
	struc.posn(186)<<0.666666687,0.833333313,0;
	struc.posn(187)<<0.666666687,0.833333313,0.333333343;
	struc.posn(188)<<0.666666687,0.833333313,0.666666687;
	struc.posn(189)<<0,0,0.166666672;
	struc.posn(190)<<0,0,0.5;
	struc.posn(191)<<0,0,0.833333313;
	struc.posn(192)<<0,0.333333343,0.166666672;
	struc.posn(193)<<0,0.333333343,0.5;
	struc.posn(194)<<0,0.333333343,0.833333313;
	struc.posn(195)<<0,0.666666687,0.166666672;
	struc.posn(196)<<0,0.666666687,0.5;
	struc.posn(197)<<0,0.666666687,0.833333313;
	struc.posn(198)<<0.333333343,0,0.166666672;
	struc.posn(199)<<0.333333343,0,0.5;
	struc.posn(200)<<0.333333343,0,0.833333313;
	struc.posn(201)<<0.333333343,0.333333343,0.166666672;
	struc.posn(202)<<0.333333343,0.333333343,0.5;
	struc.posn(203)<<0.333333343,0.333333343,0.833333313;
	struc.posn(204)<<0.333333343,0.666666687,0.166666672;
	struc.posn(205)<<0.333333343,0.666666687,0.5;
	struc.posn(206)<<0.333333343,0.666666687,0.833333313;
	struc.posn(207)<<0.666666687,0,0.166666672;
	struc.posn(208)<<0.666666687,0,0.5;
	struc.posn(209)<<0.666666687,0,0.833333313;
	struc.posn(210)<<0.666666687,0.333333343,0.166666672;
	struc.posn(211)<<0.666666687,0.333333343,0.5;
	struc.posn(212)<<0.666666687,0.333333343,0.833333313;
	struc.posn(213)<<0.666666687,0.666666687,0.166666672;
	struc.posn(214)<<0.666666687,0.666666687,0.5;
	struc.posn(215)<<0.666666687,0.666666687,0.833333313;
	for(int i=0; i<natoms; ++i){
		struc.posn(i)*=a0;
	}
	for(int i=0; i<natoms/2; ++i){
		struc.name(i)="Na";
	}
	for(int i=natoms/2; i<natoms; ++i){
		struc.name(i)="Cl";
	}
	for(int i=0; i<natoms/2; ++i){
		struc.type(i)=0;
	}
	for(int i=natoms/2; i<natoms; ++i){
		struc.type(i)=1;
	}
	for(int i=0; i<natoms/2; ++i){
		struc.charge(i)=1;
	}
	for(int i=natoms/2; i<natoms; ++i){
		struc.charge(i)=-1;
	}
	return struc;
}

Structure make_NaCl_cluster(double a){
	const double a0=a;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=true; atomT.posn=true; atomT.symm=false;
	Structure struc;
	const int natoms=12;
	struc.resize(natoms,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	struc.init(lv);
	struc.pe()=0;
	struc.posn(0)<< -1.02107138952728,-1.38193765488367,-2.2833169126567;
	struc.posn(1)<< 1.72392286979552,1.01697854170313,-2.49121390786186;
	struc.posn(2)<< -1.14977779358835,1.72715575287346,-0.178779332539615;
	struc.posn(3)<< 1.48300720156817,-1.2272665505338,0.512030813916504;
	struc.posn(4)<< -1.39064978727599,-0.516829138452847,2.82441809552583;
	struc.posn(5)<< 1.35458449240771,1.88187502160585,2.61669126095979;
	struc.posn(6)<< -1.22847627830703,2.04061529778295,2.40633606165964;
	struc.posn(7)<< 1.19209783515698,-0.674653940110196,3.04146632613773;
	struc.posn(8)<< -1.34196153440059,-1.06776147786149,0.280815905396762;
	struc.posn(9)<< 1.6753173534583,1.56779231877372,0.05257636430612;
	struc.posn(10)<<1.56191189918146,-1.54050266734643,-2.07295707986487;
	struc.posn(11)<<-0.858904868469087,1.17453449645013,-2.70806759497883;
	for(int i=0; i<6; ++i) struc.name(i)="Na";
	for(int i=6; i<12; ++i) struc.name(i)="Cl";
	for(int i=0; i<6; ++i) struc.type(i)=0;
	for(int i=6; i<12; ++i) struc.type(i)=1;
	for(int i=0; i<6; ++i) struc.charge(i)=1;
	for(int i=6; i<12; ++i) struc.charge(i)=-1;
	return struc;
}

void test_unit_pair(){
	//constants
	const int npts=200;
	const double dmin=0.5;
	const double dmax=6.0;
	const double dx=(dmax-dmin)/npts;
	const double rcut=dmax+0.1;
	//units
	units::System unitsys=units::System::METAL;
	units::consts::init(unitsys);
	//generate distance
	std::cout<<"generating distances\n";
	std::vector<double> d(npts);
	for(int i=0; i<npts; ++i){
		d[i]=dx*i+dmin;
	}
	//compute energy
	std::cout<<"computing energies\n";
	ptnl::PotLJSm pot;
	pot.rc()=rcut;
	pot.resize(2);
	pot.e()(0,0)=11.0;
	pot.e()(1,1)=17.0;
	pot.s()(0,0)=1.66;
	pot.s()(1,1)=1.02;
	pot.init();
	std::vector<double> pe(npts);
	std::vector<double> force_e(npts);
	for(int i=0; i<npts; ++i){
		Structure struc=make_NaCl_pair(d[i]);
		NeighborList nlist(struc,rcut);
		pe[i]=pot.compute(struc,nlist);
		force_e[i]=struc.force(1)[2];
	}
	//compute forces
	std::cout<<"computing forces\n";
	std::vector<double> force_g(npts);
	force_g[0]=(pe[1]-pe[0])/dx;
	for(int i=1; i<npts-1; ++i){
		force_g[i]=0.5*(pe[i+1]-pe[i-1])/dx;
	}
	force_g[npts-1]=(pe[npts-1]-pe[npts-2])/dx;
	//print
	std::cout<<"d e fe fg\n";
	for(int i=0; i<npts; ++i){
		std::cout<<d[i]<<" "<<pe[i]<<" "<<force_e[i]<<" "<<force_g[i]<<"\n";
	}
}

int main(int argc, char* argv[]){
	
	char* str=new char[print::len_buf];

	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("POT_PAULI",str)<<"\n";
	test_unit_pair();
	std::cout<<print::title("POT_PAULI",str)<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	return 0;
}