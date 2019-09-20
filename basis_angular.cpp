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
	tcut_=basis.tcut();//cutoff type
	rc_=basis.rc();//cutoff value
	norm_=1.0/(0.5*4.0/3.0*num_const::PI*rc_*rc_*rc_);
	norm_=norm_*norm_;
	phiAN_=basis.phiAN();//angular function type
	nfA_=basis.nfA();//number of angular functions
	if(nfA_>0){
		symm_=Eigen::VectorXd::Zero(nfA_);
		if(phiAN_==PhiAN::G3){
			fA_=new PhiA*[nfA_];
			for(unsigned int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G3(static_cast<const PhiA_G3&>(basis.fA(i)));
		} else if(phiAN_==PhiAN::G4){
			fA_=new PhiA*[nfA_];
			for(unsigned int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G4(static_cast<const PhiA_G4&>(basis.fA(i)));
		}
	}
}

BasisA& BasisA::operator=(const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::operator=(const BasisA&):\n";
	clear();
	tcut_=basis.tcut();//cutoff type
	rc_=basis.rc();//cutoff value
	norm_=1.0/(0.5*4.0/3.0*num_const::PI*rc_*rc_*rc_);
	norm_=norm_*norm_;
	phiAN_=basis.phiAN();//angular function type
	nfA_=basis.nfA();//number of angular functions
	if(nfA_>0){
		symm_=Eigen::VectorXd::Zero(nfA_);
		if(phiAN_==PhiAN::G3){
			fA_=new PhiA*[nfA_];
			for(unsigned int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G3(static_cast<const PhiA_G3&>(basis.fA(i)));
		} else if(phiAN_==PhiAN::G4){
			fA_=new PhiA*[nfA_];
			for(unsigned int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G4(static_cast<const PhiA_G4&>(basis.fA(i)));
		}
	}
	return *this;
}

//initialization

void BasisA::init_G3(unsigned int nA, CutoffN::type tcut, double rcut){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init_G3(unsigned int,CutoffN::type,double):\n";
	if(nA==0) throw std::invalid_argument("Invalid number of angular functions.");
	clear();
	//set basis parameters
	tcut_=tcut;//cutoff type
	rc_=rcut;//cutoff value
	norm_=1.0/(0.5*4.0/3.0*num_const::PI*rc_*rc_*rc_);
	norm_=norm_*norm_;
	phiAN_=PhiAN::G3;//angular function type
	nfA_=nA;//angular function type
	if(nfA_>0){
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfA_);
		const double s=0.75;
		const double eta=1.0/(3.0*rcut*rcut);
		fA_=new PhiA*[nfA_];
		const double lambda=1;
		for(unsigned int i=0; i<nfA_; ++i){
			double zeta=std::pow((i+1.0),std::log(i+2.0)/std::log(nfA_/(nfA_+1.0)+2.0));
			fA_[i]=new PhiA_G3(eta,zeta,lambda);
		}
	}
}

void BasisA::init_G4(unsigned int nA, CutoffN::type tcut, double rcut){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init_G4(BasisA&,unsigned int,CutoffN::type,double):\n";
	if(nA==0) throw std::invalid_argument("Invalid number of angular functions.");
	clear();
	//set basis parameters
	tcut_=tcut;//cutoff type
	rc_=rcut;//cutoff value
	norm_=1.0/(0.5*4.0/3.0*num_const::PI*rc_*rc_*rc_);
	norm_=norm_*norm_;
	phiAN_=PhiAN::G4;//angular function type
	nfA_=nA;//angular function type
	if(nfA_>0){
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfA_);
		const double s=0.75;
		const double eta=1.0/(2.0*rcut*rcut);
		fA_=new PhiA*[nfA_];
		const double lambda=1;
		for(unsigned int i=0; i<nfA_; ++i){
			double zeta=std::pow((i+1.0),std::log(i+2.0)/std::log(nfA_/(nfA_+1.0)+2.0));
			fA_[i]=new PhiA_G4(eta,zeta,lambda);
		}
	}
}

void BasisA::write(const char* filename, const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::write(const char*,const BasisA&):\n";
	FILE* writer=fopen(filename,"w");
	if(writer!=NULL){
		BasisA::write(writer,basis);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("I/O ERROR: Could not write BasisA to file: \"")+std::string(filename)+std::string("\""));
}

void BasisA::read(const char* filename, BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::read(const char*,const BasisA&):\n";
	FILE* reader=fopen(filename,"r");
	if(reader!=NULL){
		BasisA::read(reader,basis);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("I/O ERROR: Could not read BasisA from file: \"")+std::string(filename)+std::string("\""));
}

void BasisA::write(FILE* writer, const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::write(FILE*):\n";
	std::string str_phian,str_tcut,str;
	if(basis.tcut()==CutoffN::COS) str_tcut="COS";
	else if(basis.tcut()==CutoffN::TANH) str_tcut="TANH";
	if(basis.phiAN()==PhiAN::G3) str_phian="G3";
	else if(basis.phiAN()==PhiAN::G4) str_phian="G4";
	fprintf(writer,"BasisA %s %f %s %i\n",str_tcut.c_str(),basis.rc(),str_phian.c_str(),basis.nfA());
	if(basis.phiAN()==PhiAN::G3){
		//tcut,rcut,eta,zeta,lambda
		for(unsigned int i=0; i<basis.nfA(); ++i){
			const PhiA_G3& g3=static_cast<const PhiA_G3&>(basis.fA(i));
			fprintf(writer,"\tG3 %f %f %i\n",g3.eta,g3.zeta,g3.lambda);
		}
	} else if(basis.phiAN()==PhiAN::G4){
		//tcut,rcut,eta,zeta,lambda
		for(unsigned int i=0; i<basis.nfA(); ++i){
			const PhiA_G4& g4=static_cast<const PhiA_G4&>(basis.fA(i));
			fprintf(writer,"\tG4 %f %f %i\n",g4.eta,g4.zeta,g4.lambda);
		}
	}
}

void BasisA::read(FILE* reader, BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::read(FILE*, BasisA&):\n";
	//local variables
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	//split header
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	if(strlist.size()!=5) throw std::runtime_error("Invalid BasisR format.");
	//read header
	const CutoffN::type tcut=CutoffN::read(strlist[1].c_str());
	const double rc=std::atof(strlist[2].c_str());
	unsigned int nfa=std::atoi(strlist[4].c_str());
	if(nfa>0) basis.symm()=Eigen::VectorXd::Zero(nfa);
	//loop over all basis functions
	if(strlist[3]=="G3"){
		basis.init_G3(nfa,tcut,rc);
		//G3,eta,zeta,lambda
		for(unsigned int i=0; i<basis.nfA(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+3) throw std::runtime_error("Invalid G3 format.");
			const double eta=std::atof(strlist[1].c_str());
			const double zeta=std::atof(strlist[2].c_str());
			const int lambda=std::atoi(strlist[3].c_str());
			if(tcut==CutoffN::UNKNOWN) std::runtime_error("Invalid G3 cutoff.");
			static_cast<PhiA_G3&>(basis.fA(i))=PhiA_G3(eta,zeta,lambda);
		}
		basis.phiAN()=PhiAN::G3;
	} else if(strlist[3]=="G4"){
		basis.init_G4(nfa,tcut,rc);
		//G4,eta,zeta,lambda
		for(unsigned int i=0; i<basis.nfA(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+3) throw std::runtime_error("Invalid G4 format.");
			const double eta=std::atof(strlist[1].c_str());
			const double zeta=std::atof(strlist[2].c_str());
			const int lambda=std::atoi(strlist[3].c_str());
			if(tcut==CutoffN::UNKNOWN) std::runtime_error("Invalid G4 cutoff.");
			static_cast<PhiA_G4&>(basis.fA(i))=PhiA_G4(eta,zeta,lambda);
		}
		basis.phiAN()=PhiAN::G4;
	} else throw std::invalid_argument("Invalid angular function type");
	//free local variables
	delete[] input;
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
	tcut_=CutoffN::UNKNOWN;
	rc_=0;
	norm_=0;
	phiAN_=PhiAN::UNKNOWN;
	nfA_=0;
}	

void BasisA::symm(double cos, const double d[3]){
	const double c[3]={
		CutoffF::funcs[tcut_](d[0],rc_),
		CutoffF::funcs[tcut_](d[1],rc_),
		CutoffF::funcs[tcut_](d[2],rc_)
	};
	for(int na=nfA_-1; na>=0; --na){
		symm_[na]=fA_[na]->val(cos,d,c)*norm_;
	}
}

void BasisA::force(double* fij, double* fik, double cos, const double d[3], const double* dEdG){
	//d[3]=(rij,rik,rjk)
	//cos=rij*rik/(|rij|*|rik|)
	/*double amp,angle;
	const double c[3]={
		CutoffF::funcs[tcut_](d[0],rc_),//cut(rij)
		CutoffF::funcs[tcut_](d[1],rc_),//cut(rik)
		CutoffF::funcs[tcut_](d[2],rc_) //cut(rjk)
	};
	const double g[2]={
		CutoffFD::funcs[tcut_](d[0],rc_),//cut'(rij)
		CutoffFD::funcs[tcut_](d[1],rc_) //cut'(rik)
	};
	for(int na=nfA_-1; na>=0; --na){
		//gradient - cosine - central atom
		amp=-0.5*fA_[na]->grad_angle(cos)*fA_[na]->dist(d,c)*dEdG[na]*norm_;
		fij[0]+=amp/d[0]*(-cos);
		fij[1]+=amp/d[1];
		fik[0]+=amp/d[1]*(-cos);
		fik[1]+=amp/d[0];
		//gradient distance - central atom
		amp=-0.5*fA_[na]->angle(cos)*dEdG[na]*norm_;
		fij[0]+=amp*fA_[na]->grad_dist_0(d,c,g[0]);
		fik[0]+=amp*fA_[na]->grad_dist_1(d,c,g[1]);
	}*/
	double amp,angle,gangle,dist;
	const double c[3]={
		CutoffF::funcs[tcut_](d[0],rc_),//cut(rij)
		CutoffF::funcs[tcut_](d[1],rc_),//cut(rik)
		CutoffF::funcs[tcut_](d[2],rc_) //cut(rjk)
	};
	const double g[3]={
		CutoffFD::funcs[tcut_](d[0],rc_),//cut'(rij)
		CutoffFD::funcs[tcut_](d[1],rc_),//cut'(rik)
		CutoffFD::funcs[tcut_](d[2],rc_) //cut'(rjk)
	};
	fij[0]=0; fij[1]=0;
	fik[0]=0; fik[1]=0;
	double gradd[3];//grad w.r.t. distance (not angle)
	for(int na=nfA_-1; na>=0; --na){
		fA_[na]->compute_angle(cos,angle,gangle);
		fA_[na]->compute_dist(d,c,g,dist,gradd);
		//gradient - cosine - central atom
		amp=-0.5*gangle*dist*dEdG[na]*norm_;
		fij[0]+=amp/d[0]*(-cos);
		fij[1]+=amp/d[1];
		fik[0]+=amp/d[1]*(-cos);
		fik[1]+=amp/d[0];
		//gradient distance - central atom
		amp=-0.5*angle*dEdG[na]*norm_;
		fij[0]+=amp*gradd[0];
		fik[0]+=amp*gradd[1];
	}
}

//operators

bool operator==(const BasisA& basis1, const BasisA& basis2){
	if(basis1.nfA()!=basis2.nfA()) return false;
	else if(basis1.phiAN()!=basis2.phiAN()) return false;
	else if(basis1.tcut()!=basis2.tcut()) return false;
	else if(basis1.rc()!=basis2.rc()) return false;
	else {
		unsigned int nfA=basis1.nfA();
		PhiAN::type phiAN=basis1.phiAN();
		if(phiAN==PhiAN::G3){
			for(unsigned int i=0; i<nfA; ++i){
				if(static_cast<const PhiA_G3&>(basis1.fA(i))!=static_cast<const PhiA_G3&>(basis2.fA(i))) return false;
			}
		} else if(phiAN==PhiAN::G4){
			for(unsigned int i=0; i<nfA; ++i){
				if(static_cast<const PhiA_G4&>(basis1.fA(i))!=static_cast<const PhiA_G4&>(basis2.fA(i))) return false;
			}
		}
		return true;
	}
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const BasisA& obj){
		unsigned int N=0;
		N+=sizeof(obj.tcut());//name of cutoff function
		N+=sizeof(obj.rc());//cutoff value
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
		unsigned int pos=0,nfA=obj.nfA();
		std::memcpy(arr+pos,&obj.tcut(),sizeof(obj.tcut())); pos+=sizeof(obj.tcut());//name of cutoff function
		std::memcpy(arr+pos,&obj.rc(),sizeof(obj.rc())); pos+=sizeof(obj.rc());//cutoff value
		std::memcpy(arr+pos,&obj.phiAN(),sizeof(obj.phiAN())); pos+=sizeof(obj.phiAN());//name of symmetry functions
		std::memcpy(arr+pos,&nfA,sizeof(nfA)); pos+=sizeof(nfA);//number of symmetry functions
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
		unsigned int pos=0,nfA=0;
		double rc=0;
		PhiAN::type phiAN=PhiAN::UNKNOWN;
		CutoffN::type cutN=CutoffN::UNKNOWN;
		std::memcpy(&cutN,arr+pos,sizeof(cutN)); pos+=sizeof(cutN);//name of cutoff function
		std::memcpy(&rc,arr+pos,sizeof(rc)); pos+=sizeof(rc);//cutoff value
		std::memcpy(&phiAN,arr+pos,sizeof(phiAN)); pos+=sizeof(phiAN);//name of symmetry functions
		std::memcpy(&nfA,arr+pos,sizeof(nfA)); pos+=sizeof(nfA);//number of symmetry functions
		if(phiAN==PhiAN::G3){
			obj.init_G3(nfA,cutN,rc);
			for(unsigned int i=0; i<obj.nfA(); ++i){
				unpack(static_cast<PhiA_G3&>(obj.fA(i)),arr+pos); pos+=nbytes(static_cast<const PhiA_G3&>(obj.fA(i)));
			}
		} else if(phiAN==PhiAN::G4){
			obj.init_G4(nfA,cutN,rc);
			for(unsigned int i=0; i<obj.nfA(); ++i){
				unpack(static_cast<PhiA_G4&>(obj.fA(i)),arr+pos); pos+=nbytes(static_cast<const PhiA_G4&>(obj.fA(i)));
			}
		} else throw std::runtime_error("Invalid angular symmetry function.");
	}
	
}
