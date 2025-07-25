// c libaries
#include <ctime>
// c++ libaries
#include <iostream>
// ann - str
#include "str/print.hpp"
#include "str/string.hpp"
// ann - ml
#include "ml/nn.hpp"
// ann - nnp
#include "nnp/nnp.hpp"
#include "nnp/basis_radial.hpp"
#include "nnp/basis_angular.hpp"
// ann - structure
#include "struc/structure.hpp"
// ann - chem
#include "chem/units.hpp"
// ann - cutoff
#include "cutoff.hpp"

//**********************************************
//
//**********************************************

Structure make_NaCl_small(){
	const double a0=5.6199998856;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=true;
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
	return struc;
}

Structure make_NaCl_large(){
	const double a0=16.8599987030;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=true;
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
	return struc;
}

void test_symm_super_crystal(){
	//species
	std::vector<Type> species(2);
	species[0].name()="Na";
	species[0].id()=string::hash(species[0].name());
	species[0].mass().val()=22.990;
	species[0].mass().flag()=true;
	species[0].energy().val()=-1.68396789;
	species[0].energy().flag()=true;
	species[1].name()="Cl";
	species[1].id()=string::hash(species[1].name());
	species[1].mass().val()=35.45;
	species[1].mass().flag()=true;
	species[1].energy().val()=-2.15869789;
	species[1].energy().flag()=true;
	//nnp
	const double rcut=8.0;
	NNP nnp(species);
	nnp.rc()=rcut;
	NN::ANNP init;
	std::vector<int> nh(3);
	nh[0]=5; nh[1]=3; nh[2]=2;
	init.init()=NN::Init::HE;
	init.seed()=-1;
	init.neuron()=NN::Neuron::SWISH;
	//initialize basis
	std::cout<<"initializing basis\n";
	const int nR=6;
	BasisR basisR(rcut,Cutoff::Name::COS,nR,BasisR::Name::TANH);
	const int nA=4;
	BasisA basisA(rcut,Cutoff::Name::COS,nA,BasisA::Name::GAUSS);
	for(int i=0; i<nR; ++i){
		basisR.rs(i)=1.06;
		basisR.eta(i)=i;
	}
	for(int i=0; i<nA; ++i){
		basisA.eta(i)=i;
		basisA.zeta(i)=i;
		basisA.lambda(i)=1;
	}
	//initialize neural network
	std::cout<<"initializing neural network\n";
	for(int i=0; i<species.size(); ++i){
		nnp.nnh(i).nn().resize(init,nh);
		nnp.nnh(i).type()=species[0];
		nnp.nnh(i).resize(species.size());
		for(int j=0; j<species.size(); ++j){
			nnp.nnh(i).basisR(j)=basisR;
		}
		for(int j=0; j<species.size(); ++j){
			for(int k=j; k<species.size(); ++k){
				nnp.nnh(i).basisA(j,k)=basisA;
			}
		}
		nnp.nnh(i).init_input();
	}
	std::cout<<nnp<<"\n";
	std::cout<<basisR<<"\n";
	std::cout<<basisA<<"\n";
	//make structures
	std::cout<<"making structures\n";
	Structure struc_small=make_NaCl_small();
	Structure struc_large=make_NaCl_large();
	//compute neighbor lists
	std::cout<<"computing neighbor lists\n";
	NeighborList nlist_small(struc_small,rcut);
	NeighborList nlist_large(struc_large,rcut);
	//compute symmetry functions
	std::cout<<"initializing symmetry functions\n";
	NNP::init(nnp,struc_small);
	NNP::init(nnp,struc_large);
	std::cout<<"computing symmetry functions\n";
	NNP::symm(nnp,struc_small,nlist_small);
	NNP::symm(nnp,struc_large,nlist_large);
	/*
	std::cout<<"printing symmetry functions\n";
	for(int i=0; i<struc_small.nAtoms(); ++i){
		std::cout<<"symm - small - "<<i<<" = "<<struc_small.symm(i).transpose()<<"\n";
	}
	for(int i=0; i<struc_large.nAtoms(); ++i){
		std::cout<<"symm - large - "<<i<<" = "<<struc_large.symm(i).transpose()<<"\n";
	}
	*/
	double err_Cl=0,err_Na=0;
	for(int i=0; i<struc_large.nAtoms()/2; ++i){
		err_Na+=(struc_large.symm(i)-struc_small.symm(0)).norm();
	}
	for(int i=struc_large.nAtoms()/2; i<struc_large.nAtoms(); ++i){
		err_Cl+=(struc_large.symm(i)-struc_small.symm(1)).norm();
	}
	std::cout<<"err_Na/atom  = "<<err_Na/struc_large.nAtoms()<<"\n";
	std::cout<<"err_Na/atom  = "<<err_Cl/struc_large.nAtoms()<<"\n";
	std::cout<<"err_tot/atom = "<<(err_Na+err_Cl)/struc_large.nAtoms()<<"\n";
}

void test_symm_super_rand(){
	//species
	std::vector<Type> species(2);
	species[0].name()="Na";
	species[0].id()=string::hash(species[0].name());
	species[0].mass().val()=22.990;
	species[0].mass().flag()=true;
	species[0].energy().val()=-1.68396789;
	species[0].energy().flag()=true;
	species[1].name()="Cl";
	species[1].id()=string::hash(species[1].name());
	species[1].mass().val()=35.45;
	species[1].mass().flag()=true;
	species[1].energy().val()=-2.15869789;
	species[1].energy().flag()=true;
	//nnp
	const double rcut=8.0;
	NNP nnp(species);
	nnp.rc()=rcut;
	NN::ANNP init;
	std::vector<int> nh(3);
	nh[0]=5; nh[1]=3; nh[2]=2;
	init.init()=NN::Init::HE;
	init.seed()=-1;
	init.neuron()=NN::Neuron::SWISH;
	//initialize basis
	std::cout<<"initializing basis\n";
	const int nR=6;
	BasisR basisR(rcut,Cutoff::Name::COS,nR,BasisR::Name::TANH);
	const int nA=4;
	BasisA basisA(rcut,Cutoff::Name::COS,nA,BasisA::Name::GAUSS);
	for(int i=0; i<nR; ++i){
		basisR.rs(i)=1.06;
		basisR.eta(i)=i;
	}
	for(int i=0; i<nA; ++i){
		basisA.eta(i)=i;
		basisA.zeta(i)=i;
		basisA.lambda(i)=1;
	}
	//initialize neural network
	std::cout<<"initializing neural network\n";
	for(int i=0; i<species.size(); ++i){
		nnp.nnh(i).nn().resize(init,nh);
		nnp.nnh(i).type()=species[0];
		nnp.nnh(i).resize(species.size());
		for(int j=0; j<species.size(); ++j){
			nnp.nnh(i).basisR(j)=basisR;
		}
		for(int j=0; j<species.size(); ++j){
			for(int k=j; k<species.size(); ++k){
				nnp.nnh(i).basisA(j,k)=basisA;
			}
		}
		nnp.nnh(i).init_input();
	}
	std::cout<<nnp<<"\n";
	std::cout<<basisR<<"\n";
	std::cout<<basisA<<"\n";
	//make structures
	std::cout<<"making structures\n";
	const double a0=5.6199998856*2;
	const int natoms=8;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=false; atomT.posn=true; atomT.symm=true;
	Structure struc_small;
	Structure struc_large;
	struc_small.resize(natoms,atomT);
	Eigen::Matrix3d lv;
	lv<<0.5,0.5,0.0,0.5,0.0,0.5,0.0,0.5,0.5;
	lv*=a0;
	struc_small.init(lv);
	for(int i=0; i<natoms; ++i){
		struc_small.posn(i)=Eigen::Vector3d::Random()*a0;
		if(i%2==0){
			struc_small.name(i)="Na";
			struc_small.type(i)=0;
		} else {
			struc_small.name(i)="Cl";
			struc_small.type(i)=1;
		}
	}
	const Eigen::Vector3i nlat=Eigen::Vector3i::Constant(3);
	Structure::super(struc_small,struc_large,nlat);
	//compute neighbor lists
	std::cout<<"computing neighbor lists\n";
	NeighborList nlist_small(struc_small,rcut);
	NeighborList nlist_large(struc_large,rcut);
	//compute symmetry functions
	std::cout<<"initializing symmetry functions\n";
	NNP::init(nnp,struc_small);
	NNP::init(nnp,struc_large);
	std::cout<<"computing symmetry functions\n";
	NNP::symm(nnp,struc_small,nlist_small);
	NNP::symm(nnp,struc_large,nlist_large);
	int c=0;
	double err=0;
	for(int i=0; i<nlat[0]; ++i){
		for(int j=0; j<nlat[1]; ++j){
			for(int k=0; k<nlat[2]; ++k){
				for(int n=0; n<struc_small.nAtoms(); ++n){
					//std::cout<<"sm["<<n<<"] = "<<struc_small.symm(n).transpose()<<"\n";
					//std::cout<<"lg["<<n<<"] = "<<struc_large.symm(n).transpose()<<"\n";
					err+=(struc_small.symm(n)-struc_large.symm(c++)).norm();	
				}
			}
		}
	}
	std::cout<<"err = "<<err<<"\n";
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	char* str=new char[print::len_buf];
	
	test_symm_super_crystal();
	test_symm_super_rand();
	delete[] str;
	
	return 0;
}
