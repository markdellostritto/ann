#include "basis_angular.hpp"

//memer variables

const double BasisA::V_CUT=1e-5;

//constructors/destructors

BasisA::~BasisA(){
	clear();
}

BasisA::BasisA(const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA(const BasisA&):\n";
	fA_=NULL;
	clear();
	phiAN_=basis.phiAN();
	nfA_=basis.nfA();
	if(phiAN_==PhiAN::G3){
		fA_=new PhiA*[nfA_];
		for(unsigned int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G3(static_cast<const PhiA_G3&>(basis.fA(i)));
	} else if(phiAN_==PhiAN::G4){
		fA_=new PhiA*[nfA_];
		for(unsigned int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G4(static_cast<const PhiA_G4&>(basis.fA(i)));
	}
}

BasisA& BasisA::operator=(const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::operator=(const BasisA&):\n";
	clear();
	phiAN_=basis.phiAN();
	nfA_=basis.nfA();
	if(phiAN_==PhiAN::G3){
		fA_=new PhiA*[nfA_];
		for(unsigned int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G3(static_cast<const PhiA_G3&>(basis.fA(i)));
	} else if(phiAN_==PhiAN::G4){
		fA_=new PhiA*[nfA_];
		for(unsigned int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G4(static_cast<const PhiA_G4&>(basis.fA(i)));
	}
	return *this;
}

//initialization

void BasisA::init_G3(unsigned int nA, CutoffN::type tcut, double rcut){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init_G3(unsigned int,CutoffN::type,double):\n";
	if(nA==0) throw std::invalid_argument("Invalid number of angular functions.");
	clear();
	nfA_=nA;
	phiAN_=PhiAN::G3;
	const double s=0.75;
	double eta=6.0/(rcut*rcut);
	fA_=new PhiA*[nfA_];
	double lambda=1;
	for(unsigned int i=0; i<nfA_; ++i){
		double zeta=std::pow(s,std::log(1.0*nfA_))*i*i+1.0;
		fA_[i]=new PhiA_G3(tcut,rcut,eta,zeta,lambda);
	}
}

void BasisA::init_G4(unsigned int nA, CutoffN::type tcut, double rcut){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init_G4(BasisA&,unsigned int,CutoffN::type,double):\n";
	if(nA==0) throw std::invalid_argument("Invalid number of angular functions.");
	clear();
	nfA_=nA;
	phiAN_=PhiAN::G4;
	const double s=0.75;
	double eta=8.0/(rcut*rcut);
	fA_=new PhiA*[nfA_];
	double lambda=1;
	for(unsigned int i=0; i<nfA_; ++i){
		double zeta=std::pow(s,std::log(1.0*nfA_))*i*i+1.0;
		fA_[i]=new PhiA_G4(tcut,rcut,eta,zeta,lambda);
	}
}

void BasisA::write(FILE* writer, const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::write(FILE*):\n";
	//rc,eta,zeta,lambda,tcut
	std::string str;
	if(basis.phiAN()==PhiAN::G3) str="G3";
	else if(basis.phiAN()==PhiAN::G4) str="G4";
	fprintf(writer,"BasisA %s %i\n",str.c_str(),basis.nfA());
	if(basis.phiAN()==PhiAN::G3){
		//tcut,rcut,eta,zeta,lambda
		for(unsigned int i=0; i<basis.nfA(); ++i){
			if(basis.fA(i).tcut==CutoffN::COS) str="COS";
			else if(basis.fA(i).tcut==CutoffN::TANH) str="TANH";
			const PhiA_G3& g3=static_cast<const PhiA_G3&>(basis.fA(i));
			fprintf(writer,"\t%s %f %f %f %i\n",str.c_str(),g3.rc,g3.eta,g3.zeta,g3.lambda);
		}
	} else if(basis.phiAN()==PhiAN::G4){
		//tcut,rcut,eta,zeta,lambda
		for(unsigned int i=0; i<basis.nfA(); ++i){
			if(basis.fA(i).tcut==CutoffN::COS) str="COS";
			else if(basis.fA(i).tcut==CutoffN::TANH) str="TANH";
			const PhiA_G4& g4=static_cast<const PhiA_G4&>(basis.fA(i));
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
	//read header
	fgets(input,string::M,reader);
	n=string::substrN(input,string::WS);
	if(n!=3) throw std::runtime_error("Invalid BasisA format.");
	strlist.resize(n);
	strlist[0]=std::string(std::strtok(input,string::WS));
	strlist[1]=std::string(std::strtok(NULL,string::WS));
	strlist[2]=std::string(std::strtok(NULL,string::WS));
	//resize the basis
	unsigned int nfA=std::atoi(strlist[2].c_str());
	//loop over all basis functions
	if(strlist[1]=="G3"){
		basis.init_G3(nfA,CutoffN::UNKNOWN,1.0);//actual values don't matter, will be overwritten
		//tcut,rc,eta,zeta,theta0
		for(unsigned int i=0; i<basis.nfA(); ++i){
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
			static_cast<PhiA_G3&>(basis.fA(i))=PhiA_G3(tcut,rc,eta,zeta,lambda);
		}
		basis.phiAN()=PhiAN::G3;
	} else if(strlist[1]=="G4"){
		basis.init_G4(nfA,CutoffN::UNKNOWN,1.0);//actual values don't matter, will be overwritten
		//tcut,rc,eta,zeta,lambda
		for(unsigned int i=0; i<basis.nfA(); ++i){
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
			static_cast<PhiA_G4&>(basis.fA(i))=PhiA_G4(tcut,rc,eta,zeta,lambda);
		}
		basis.phiAN()=PhiAN::G4;
	} else throw std::invalid_argument("Invalid angular function type");
	//free local variables
	free(input);
}

std::ostream& operator<<(std::ostream& out, const BasisA& basis){
	out<<"BasisA "<<basis.phiAN_<<" "<<basis.nfA_<<"\n";
	if(basis.phiAN_==PhiAN::G3){
		for(unsigned int i=0; i<basis.nfA_-1; ++i) out<<"\t"<<static_cast<const PhiA_G3&>(*basis.fA_[i])<<"\n";
		out<<"\t"<<static_cast<PhiA_G3&>(*basis.fA_[basis.nfA_-1]);
	} else if(basis.phiAN_==PhiAN::G4){
		for(unsigned int i=0; i<basis.nfA_-1; ++i) out<<"\t"<<static_cast<const PhiA_G4&>(*basis.fA_[i])<<"\n";
		out<<"\t"<<static_cast<PhiA_G4&>(*basis.fA_[basis.nfA_-1]);
	}
	return out;
}

//member functions

void BasisA::clear(){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::clear():\n";
	if(fA_!=NULL){
		for(unsigned int i=0; i<nfA_; ++i) delete fA_[i];
		delete[] fA_;
		fA_=NULL;
	}
	nfA_=0;
	phiAN_=PhiAN::UNKNOWN;
}	

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const BasisA& obj){
		unsigned int N=0;
		N+=sizeof(obj.phiAN());//name of symmetry functions
		N+=sizeof(unsigned int);//number of symmetry functions
		if(obj.phiAN()==PhiAN::G3){
			for(unsigned int i=0; i<obj.nfA(); ++i) N+=nbytes(static_cast<const PhiA_G3&>(obj.fA(i)));
		} else if(obj.phiAN()==PhiAN::G4){
			for(unsigned int i=0; i<obj.nfA(); ++i) N+=nbytes(static_cast<const PhiA_G4&>(obj.fA(i)));
		} else throw std::runtime_error("Invalid angular symmetry function.");
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const BasisA& obj, char* arr){
		unsigned int pos=0,size=obj.nfA();
		std::memcpy(arr+pos,&obj.phiAN(),sizeof(obj.phiAN())); pos+=sizeof(obj.phiAN());
		std::memcpy(arr+pos,&size,sizeof(size)); pos+=sizeof(size);
		if(obj.phiAN()==PhiAN::G3){
			for(unsigned int i=0; i<obj.nfA(); ++i){
				pack(static_cast<const PhiA_G3&>(obj.fA(i)),arr+pos); pos+=nbytes(static_cast<const PhiA_G3&>(obj.fA(i)));
			}
		} else if(obj.phiAN()==PhiAN::G4){
			for(unsigned int i=0; i<obj.nfA(); ++i){
				pack(static_cast<const PhiA_G4&>(obj.fA(i)),arr+pos); pos+=nbytes(static_cast<const PhiA_G4&>(obj.fA(i)));
			}
		} else throw std::runtime_error("Invalid angular symmetry function.");
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(BasisA& obj, const char* arr){
		unsigned int pos=0,size=0;
		std::memcpy(&obj.phiAN(),arr+pos,sizeof(obj.phiAN())); pos+=sizeof(obj.phiAN());
		std::memcpy(&size,arr+pos,sizeof(size)); pos+=sizeof(size);
		if(obj.phiAN()==PhiAN::G3){
			obj.init_G3(size,CutoffN::UNKNOWN,1);//values don't matter, as they will be overwritten
			for(unsigned int i=0; i<obj.nfA(); ++i){
				unpack(static_cast<PhiA_G3&>(obj.fA(i)),arr+pos); pos+=nbytes(static_cast<const PhiA_G3&>(obj.fA(i)));
			}
		} else if(obj.phiAN()==PhiAN::G4){
			obj.init_G4(size,CutoffN::UNKNOWN,1);//values don't matter, as they will be overwritten
			for(unsigned int i=0; i<obj.nfA(); ++i){
				unpack(static_cast<PhiA_G4&>(obj.fA(i)),arr+pos); pos+=nbytes(static_cast<const PhiA_G4&>(obj.fA(i)));
			}
		} else throw std::runtime_error("Invalid angular symmetry function.");
	}
	
}
