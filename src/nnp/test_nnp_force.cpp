// c
#include <cstdlib>
#include <ctime>
// c++
#include <iostream>
// struc
#include "struc/structure.hpp"
// nnp
#include "nnp/nnp.hpp"

Structure& make_random(Structure& struc){
	//resize struc
	const int natoms=20;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; 
    atomT.index=true; atomT.type=true;
	atomT.charge=false; 
	atomT.posn=true; atomT.force=true; 
    atomT.symm=true;
	struc.resize(natoms,atomT);
	
	//set the types
	const int ntypes=3;
	std::vector<std::string> names(ntypes);
	names[0]="Ar";
	names[1]="Ne";
	names[2]="Xe";
	std::vector<double> weights(ntypes);
	
	//resize the lattice
	const double a0=20.0;
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	struc.init(lv);
	
	//set the atomic properties
	for(int i=0; i<natoms; ++i){
		const int type=std::rand()%ntypes;
		struc.type(i)=type;
		struc.name(i)=names[type];
		struc.posn(i)=Eigen::Vector3d::Random()*a0;
	}
	
	//return 
	return struc;
}

void compute_force_num(Structure& struc, NNP& nnp){
	const double eps=1.0e-6;
	Structure struc_p=struc;
	Structure struc_m=struc;
	
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.force(i).setZero();
	}
	
	for(int i=0; i<struc.nAtoms(); ++i){
		//reset positions
		for(int n=0; n<struc.nAtoms(); ++n){
			struc_p.posn(n)=struc.posn(n);
			struc_m.posn(n)=struc.posn(n);
		}
        //compute force on atom i
		for(int j=0; j<3; ++j){
			struc_p.posn(i)=struc.posn(i); struc_p.posn(i)[j]+=eps;
			struc_m.posn(i)=struc.posn(i); struc_m.posn(i)[j]-=eps;
            NeighborList nlist_p(struc_p,nnp.rc());
            NeighborList nlist_m(struc_m,nnp.rc());
		    NNP::symm(nnp,struc_p,nlist_p);
            NNP::symm(nnp,struc_m,nlist_m);
            const double ep=NNP::energy(nnp,struc_p);
            const double em=NNP::energy(nnp,struc_m);
			struc.force(i)[j]=-0.5*(ep-em)/eps;
		}
	}
}

int main(int argc, char* argv[]){
	
	//==== make the structure ====
	std::cout<<"making structures\n";
	Structure struc_a;
	Structure struc_n;
	make_random(struc_a);
	struc_n=struc_a;
    
    //==== set the types ====
	std::cout<<"setting types\n";
	std::vector<Type> types;
	for(int i=0; i<struc_a.nAtoms(); ++i){
		bool match=false;
		for(int n=0; n<types.size(); ++n){
			if(types[n].name()==struc_a.name(i)){
				match=true; break;
			}
		}
		if(!match){
			types.push_back(Type(struc_a.name(i)));
		}
	}
	std::cout<<"Types = \n";
	for(int n=0; n<types.size(); ++n){
		std::cout<<types[n]<<"\n";
	}

    //nnp
	const double rcut=6.0;
	NNP nnp(types);
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
	BasisR basisR(rcut,Cutoff::Name::COS,nR,BasisR::Name::LOGCOSH);
    const int na=0;
	const int nA=2*na;
	BasisA basisA(rcut,Cutoff::Name::COS,nA,BasisA::Name::GAUSS);
	for(int i=0; i<nR; ++i){
		basisR.rs(i)=1.54;
		basisR.eta(i)=i+1;
	}
	for(int i=0; i<na; ++i){
		basisA.eta(i)=2.5279;
		basisA.zeta(i)=1.1390;
		basisA.lambda(i)=1;
	}
    for(int i=na; i<nA; ++i){
		basisA.eta(i)=2.5279;
		basisA.zeta(i)=1.1390;
		basisA.lambda(i)=-1;
	}

    //initialize neural network
	std::cout<<"initializing neural network\n";
	for(int i=0; i<types.size(); ++i){
		nnp.nnh(i).type()=types[i];
		nnp.nnh(i).resize(types.size());
		for(int j=0; j<types.size(); ++j){
			nnp.nnh(i).basisR(j)=basisR;
		}
		for(int j=0; j<types.size(); ++j){
			for(int k=j; k<types.size(); ++k){
				nnp.nnh(i).basisA(j,k)=basisA;
			}
		}
    }
    for(int i=0; i<types.size(); ++i){
		nnp.nnh(i).init_input();
        nnp.nnh(i).nn().resize(init,nnp.nnh(i).nInput(),nh,1);
        nnp.nnh(i).dOdZ().resize(nnp.nnh(i).nn());
	}
	std::cout<<nnp<<"\n";
	std::cout<<basisR<<"\n";
	std::cout<<basisA<<"\n";

    //compute neighbor lists
	std::cout<<"computing neighbor lists\n";
	NeighborList nlist_a(struc_a,rcut);
	NeighborList nlist_n(struc_n,rcut);

    //==== compute the analytical forces ====
	std::cout<<"computing forces - analytical\n";
	NNP::init(nnp,struc_a);
    NNP::symm(nnp,struc_a,nlist_a);
    NNP::compute(nnp,struc_a,nlist_a);
    for(int i=0; i<struc_a.nAtoms(); ++i){
		std::cout<<"fa["<<i<<"] = "<<struc_a.force(i).transpose()<<"\n";
	}

    //==== compute the numerical forces ====
	std::cout<<"computing forces - numerical\n";
	NNP::init(nnp,struc_n);
	NNP::symm(nnp,struc_n,nlist_n);
    compute_force_num(struc_n,nnp);
	for(int i=0; i<struc_n.nAtoms(); ++i){
		std::cout<<"fn["<<i<<"] = "<<struc_n.force(i).transpose()<<"\n";
	}
	
    return 0;
}