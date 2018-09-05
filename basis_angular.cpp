#include "basis_angular.hpp"

//memer variables

const double BasisA::V_CUT=1e-5;

//initialization

void BasisA::init_G3(BasisA& basis, unsigned int nA, CutoffN::type tcut, double rcut){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init_G3(BasisA&,unsigned int,CutoffN::type,double):\n";
	if(nA==0) throw std::invalid_argument("Invalid number of angular functions.");
	basis.phiAN=PhiAN::G3;
	const double s=0.75;
	double eta=6.0/(rcut*rcut);
	basis.fA.resize(nA);
	double lambda=1;
	for(unsigned int i=0; i<nA; ++i){
		double zeta=std::pow(s,std::log(1.0*nA))*i*i+1.0;
		basis.fA[i].reset(new PhiA_G3(tcut,rcut,eta,zeta,lambda));
	}
}

void BasisA::init_G4(BasisA& basis, unsigned int nA, CutoffN::type tcut, double rcut){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init_G4(BasisA&,unsigned int,CutoffN::type,double):\n";
	if(nA==0) throw std::invalid_argument("Invalid number of angular functions.");
	basis.phiAN=PhiAN::G4;
	const double s=0.75;
	double eta=8.0/(rcut*rcut);
	basis.fA.resize(nA);
	double lambda=1;
	for(unsigned int i=0; i<nA; ++i){
		double zeta=std::pow(s,std::log(1.0*nA))*i*i+1.0;
		basis.fA[i].reset(new PhiA_G4(tcut,rcut,eta,zeta,lambda));
	}
}

void BasisA::write(FILE* writer,const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::write(FILE*,const BasisA&):\n";
	//rc,eta,theta0,zeta,lambda,tcut
	std::string str;
	if(basis.phiAN==PhiAN::G3) str="G3";
	else if(basis.phiAN==PhiAN::G4) str="G4";
	fprintf(writer,"BasisA %s %i\n",str.c_str(),basis.fA.size());
	if(basis.phiAN==PhiAN::G3){
		//tcut,rcut,eta,zeta,lambda
		for(unsigned int i=0; i<basis.fA.size(); ++i){
			if(basis.fA[i]->tcut==CutoffN::COS) str="COS";
			else if(basis.fA[i]->tcut==CutoffN::TANH) str="TANH";
			PhiA_G3& g3=static_cast<PhiA_G3&>(*basis.fA[i]);
			fprintf(writer,"\t%s %f %f %f %i\n",str.c_str(),g3.rc,g3.eta,g3.zeta,g3.lambda);
		}
	} else if(basis.phiAN==PhiAN::G4){
		//tcut,rcut,eta,zeta,lambda
		for(unsigned int i=0; i<basis.fA.size(); ++i){
			if(basis.fA[i]->tcut==CutoffN::COS) str="COS";
			else if(basis.fA[i]->tcut==CutoffN::TANH) str="TANH";
			PhiA_G4& g4=static_cast<PhiA_G4&>(*basis.fA[i]);
			fprintf(writer,"\t%s %f %f %f %i\n",str.c_str(),g4.rc,g4.eta,g4.zeta,g4.lambda);
		}
	}
}

void BasisA::read(FILE* reader, BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::read(FILE*, BasisA&):\n";
	//local variables
	char* input=(char*)malloc(sizeof(char)*string::M);
	unsigned int n=0;
	std::vector<std::string> strlist;
	CutoffN::type tcut;
	//reset the basis
	basis.fA.clear();
	//read header
	fgets(input,string::M,reader);
	n=string::substrN(input,string::WS);
	if(n!=3) throw std::runtime_error("Invalid BasisA format.");
	strlist.resize(n);
	strlist[0]=std::string(std::strtok(input,string::WS));
	strlist[1]=std::string(std::strtok(NULL,string::WS));
	strlist[2]=std::string(std::strtok(NULL,string::WS));
	//resize the basis
	basis.fA.resize(std::atoi(strlist[2].c_str()));
	//loop over all basis functions
	if(strlist[1]=="G3"){
		//tcut,rc,eta,zeta,theta0
		for(unsigned int i=0; i<basis.fA.size(); ++i){
			fgets(input,string::M,reader);
			n=string::substrN(input,string::WS);
			if(n!=5) throw std::runtime_error("Invalid G3 format.");
			strlist.resize(n);
			strlist[0]=std::string(string::to_upper(std::strtok(input,string::WS)));
			for(unsigned int j=1; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
			tcut=CutoffN::load(strlist[0].c_str());
			double rc=std::atof(strlist[1].c_str());
			double eta=std::atof(strlist[2].c_str());
			double zeta=std::atof(strlist[3].c_str());
			int lambda=std::atoi(strlist[4].c_str());
			if(tcut==CutoffN::UNKNOWN) std::runtime_error("Invalid G3 cutoff.");
			basis.fA[i]=std::shared_ptr<PhiA>(new PhiA_G3(tcut,rc,eta,zeta,lambda));
		}
	} else if(strlist[1]=="G4"){
		//tcut,rc,eta,zeta,lambda
		for(unsigned int i=0; i<basis.fA.size(); ++i){
			fgets(input,string::M,reader);
			n=string::substrN(input,string::WS);
			if(n!=5) throw std::runtime_error("Invalid G4 format.");
			strlist.resize(n);
			strlist[0]=std::string(string::to_upper(std::strtok(input,string::WS)));
			for(unsigned int j=1; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
			tcut=CutoffN::load(strlist[0].c_str());
			double rc=std::atof(strlist[1].c_str());
			double eta=std::atof(strlist[2].c_str());
			double zeta=std::atof(strlist[3].c_str());
			int lambda=std::atoi(strlist[4].c_str());
			if(tcut==CutoffN::UNKNOWN) std::runtime_error("Invalid G4 cutoff.");
			basis.fA[i]=std::shared_ptr<PhiA>(new PhiA_G4(tcut,rc,eta,zeta,lambda));
		}
	} else throw std::invalid_argument("Invalid angular function type");
	//free local variables
	free(input);
}

std::ostream& operator<<(std::ostream& out, const BasisA& basisA){
	out<<"BasisA "<<basisA.phiAN<<" "<<basisA.fA.size()<<"\n";
	if(basisA.phiAN==PhiAN::G3){
		for(unsigned int i=0; i<basisA.fA.size()-1; ++i) out<<"\t"<<static_cast<PhiA_G3&>(*basisA.fA[i])<<"\n";
		out<<"\t"<<static_cast<PhiA_G3&>(*basisA.fA.back());
	} else if(basisA.phiAN==PhiAN::G4){
		for(unsigned int i=0; i<basisA.fA.size()-1; ++i) out<<"\t"<<static_cast<PhiA_G4&>(*basisA.fA[i])<<"\n";
		out<<"\t"<<static_cast<PhiA_G4&>(*basisA.fA.back());
	}
	return out;
}
