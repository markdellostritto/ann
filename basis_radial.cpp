#include "basis_radial.hpp"

//member variables

const double BasisR::V_CUT=1e-5;

//constructors/destructors

BasisR::~BasisR(){
	clear();
}	

BasisR::BasisR(const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR(const BasisR&):\n";
	fR_=NULL;
	clear();
	tcut_=basis.tcut();//cutoff type
	rc_=basis.rc();//cutoff value
	norm_=1.0/(0.5*4.0/3.0*num_const::PI*rc_*rc_*rc_);
	phiRN_=basis.phiRN();//radial function type
	nfR_=basis.nfR();//number of radial functions
	if(basis.nfR()>0){
		symm_=Eigen::VectorXd::Zero(nfR_);
		if(phiRN_==PhiRN::G1){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G1(static_cast<const PhiR_G1&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::G2){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G2(static_cast<const PhiR_G2&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::T1){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_T1(static_cast<const PhiR_T1&>(basis.fR(i)));
		}
	}
}

//operators

BasisR& BasisR::operator=(const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::operator=(const BasisR&):\n";
	clear();
	tcut_=basis.tcut();//cutoff type
	rc_=basis.rc();//cutoff value
	norm_=1.0/(0.5*4.0/3.0*num_const::PI*rc_*rc_*rc_);
	phiRN_=basis.phiRN();//radial function type
	nfR_=basis.nfR();//number of radial functions
	if(basis.nfR()>0){
		symm_=Eigen::VectorXd::Zero(nfR_);
		if(phiRN_==PhiRN::G1){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G1(static_cast<const PhiR_G1&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::G2){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G2(static_cast<const PhiR_G2&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::T1){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_T1(static_cast<const PhiR_T1&>(basis.fR(i)));
		}
	}
}

std::ostream& operator<<(std::ostream& out, const BasisR& basis){
	out<<"BasisR "<<basis.tcut_<<" "<<basis.rc_<<" "<<basis.phiRN_<<" "<<basis.nfR_<<"\n";
	if(basis.phiRN_==PhiRN::G1){
		for(unsigned int i=0; i<basis.nfR_-1; ++i) out<<"\t"<<static_cast<const PhiR_G1&>(*basis.fR_[i])<<"\n";
		out<<"\t"<<static_cast<PhiR_G1&>(*basis.fR_[basis.nfR_-1]);
	} else if(basis.phiRN_==PhiRN::G2){
		for(unsigned int i=0; i<basis.nfR_-1; ++i) out<<"\t"<<static_cast<const PhiR_G2&>(*basis.fR_[i])<<"\n";
		out<<"\t"<<static_cast<PhiR_G2&>(*basis.fR_[basis.nfR_-1]);
	} else if(basis.phiRN_==PhiRN::T1){
		for(unsigned int i=0; i<basis.nfR_-1; ++i) out<<"\t"<<static_cast<const PhiR_T1&>(*basis.fR_[i])<<"\n";
		out<<"\t"<<static_cast<PhiR_T1&>(*basis.fR_[basis.nfR_-1]);
	}
	return out;
}

//initialization

void BasisR::init_G1(unsigned int nR, CutoffN::type tcut, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_G1(unsigned int,CutoffN::type,double,double):\n";
	if(rcut<=0) throw std::invalid_argument("Invalid radial cutoff.");
	clear();
	//set basis parameters
	phiRN_=PhiRN::G1;//radial function type
	nfR_=nR;//number of radial functions
	tcut_=tcut;//cutoff type
	rc_=rcut;//cutoff value
	norm_=1.0/(0.5*4.0/3.0*num_const::PI*rc_*rc_*rc_);
	if(nfR_>0){
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfR_);
		fR_=new PhiR*[nfR_];
		for(unsigned int i=0; i<nfR_; ++i){
			fR_[i]=new PhiR_G1();
		}
	}
}

void BasisR::init_G2(unsigned int nR, CutoffN::type tcut, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_G2(unsigned int,CutoffN::type,double):\n";
	if(rcut<=0) throw std::invalid_argument("Invalid radial cutoff.");
	clear();
	//set basis parameters
	phiRN_=PhiRN::G2;//radial function type
	nfR_=nR;//number of radial functions
	tcut_=tcut;//cutoff type
	rc_=rcut;//cutoff value
	norm_=1.0/(0.5*4.0/3.0*num_const::PI*rc_*rc_*rc_);
	if(nfR_>0){
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfR_);
		fR_=new PhiR*[nfR_];
		const double dr=rcut/(nfR_-1.0);
		for(unsigned int i=0; i<nfR_; ++i){
			const double eta=-std::log(V_CUT)/(4.0*dr*dr);
			const double rs=i*dr;
			fR_[i]=new PhiR_G2(rs,eta);
		}
	}
}

void BasisR::init_T1(unsigned int nR, CutoffN::type tcut, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_T1(unsigned int,CutoffN::type,double):\n";
	if(rcut<=0) throw std::invalid_argument("Invalid radial cutoff.");
	clear();
	//set basis parameters
	phiRN_=PhiRN::T1;//radial function type
	nfR_=nR;//number of radial functions
	tcut_=tcut;//cutoff type
	rc_=rcut;//cutoff value
	norm_=1.0/(0.5*4.0/3.0*num_const::PI*rc_*rc_*rc_);
	if(nfR_>0){
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfR_);
		fR_=new PhiR*[nfR_];
		const double dr=rcut/(nfR_-1.0);
		for(unsigned int i=0; i<nfR_; ++i){
			const double eta=-std::log(V_CUT)/(4.0*dr*dr);
			const double rs=i*dr;
			fR_[i]=new PhiR_T1(rs,eta);
		}
	}
}

//loading/printing

void BasisR::write(const char* filename, const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::write(const char*,const BasisR&):\n";
	FILE* writer=fopen(filename,"w");
	if(writer!=NULL){
		BasisR::write(writer,basis);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("I/O ERROR: Could not write BasisR to file: \"")+std::string(filename)+std::string("\""));
}

void BasisR::read(const char* filename, BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::read(const char*,const BasisR&):\n";
	FILE* reader=fopen(filename,"r");
	if(reader!=NULL){
		BasisR::read(reader,basis);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("I/O ERROR: Could not read BasisR from file: \"")+std::string(filename)+std::string("\""));
}
	
void BasisR::write(FILE* writer,const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::write(FILE*,const BasisR&):\n";
	std::string str_phirn,str_tcut,str;
	if(basis.tcut()==CutoffN::COS) str_tcut="COS";
	else if(basis.tcut()==CutoffN::TANH) str_tcut="TANH";
	if(basis.phiRN()==PhiRN::G1) str_phirn="G1";
	else if(basis.phiRN()==PhiRN::G2) str_phirn="G2";
	else if(basis.phiRN()==PhiRN::T1) str_phirn="T1";
	fprintf(writer,"BasisR %s %f %s %i\n",str_tcut.c_str(),basis.rc(),str_phirn.c_str(),basis.nfR());
	if(basis.phiRN()==PhiRN::G1){
		//tcut,rc
		for(unsigned int i=0; i<basis.nfR(); ++i){
			fprintf(writer,"\tG1\n");
		}
	} else if(basis.phiRN()==PhiRN::G2){
		//tcut,rc,rs,eta
		for(unsigned int i=0; i<basis.nfR(); ++i){
			const PhiR_G2& g2=static_cast<const PhiR_G2&>(basis.fR(i));
			fprintf(writer,"\tG2 %f %f\n",g2.rs,g2.eta);
		}
	} else if(basis.phiRN()==PhiRN::T1){
		//tcut,rc,rs,eta
		for(unsigned int i=0; i<basis.nfR(); ++i){
			const PhiR_T1& t1=static_cast<const PhiR_T1&>(basis.fR(i));
			fprintf(writer,"\tG2 %f %f\n",t1.rs,t1.eta);
		}
	}
}

void BasisR::read(FILE* reader, BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::read(FILE*, BasisR&):\n";
	//local variables
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	//split header
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	if(strlist.size()!=5) throw std::runtime_error("Invalid BasisR format.");
	//read header
	const CutoffN::type tcut=CutoffN::read(strlist[1].c_str());
	const double rc=std::atof(strlist[2].c_str());
	unsigned int nfr=std::atoi(strlist[4].c_str());
	if(nfr>0) basis.symm()=Eigen::VectorXd::Zero(nfr);
	//loop over all basis functions
	if(strlist[3]=="G1"){
		basis.init_G1(nfr,tcut,rc);
		//G1
		for(unsigned int i=0; i<basis.nfR(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+0) throw std::runtime_error("Invalid G1 format.");
			static_cast<PhiR_G1&>(basis.fR(i))=PhiR_G1();
		}
		basis.phiRN()=PhiRN::G1;
	} else if(strlist[3]=="G2"){
		basis.init_G2(nfr,tcut,rc);
		//G2,rs,eta
		for(unsigned int i=0; i<basis.nfR(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+2) throw std::runtime_error("Invalid G2 format.");
			double rs=std::atof(strlist[1].c_str());
			double eta=std::atof(strlist[2].c_str());
			static_cast<PhiR_G2&>(basis.fR(i))=PhiR_G2(rs,eta);
		}
		basis.phiRN()=PhiRN::G2;
	} else if(strlist[3]=="T1"){
		basis.init_T1(nfr,tcut,rc);
		//T1,rs,eta
		for(unsigned int i=0; i<basis.nfR(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+2) throw std::runtime_error("Invalid T1 format.");
			double rs=std::atof(strlist[1].c_str());
			double eta=std::atof(strlist[2].c_str());
			static_cast<PhiR_T1&>(basis.fR(i))=PhiR_T1(rs,eta);
		}
		basis.phiRN()=PhiRN::T1;
	} else throw std::invalid_argument("Invalid radial function type");
	//free local variables
	delete[] input;
}

//member functions

void BasisR::clear(){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::clear():\n";
	if(fR_!=NULL){
		for(unsigned int i=0; i<nfR_; ++i) delete fR_[i];
		delete[] fR_;
		fR_=NULL;
	}
	rc_=0;
	norm_=0;
	tcut_=CutoffN::UNKNOWN;
	nfR_=0;
	phiRN_=PhiRN::UNKNOWN;
}

void BasisR::symm(double dr){
	const double cut=CutoffF::funcs[tcut_](dr,rc_);
	for(int i=nfR_-1; i>=0; --i){
		symm_[i]=fR_[i]->val(dr,cut)*norm_;
	}
}

double BasisR::force(double dr, const double* dEdG){
	double amp=0;
	const double cut=CutoffF::funcs[tcut_](dr,rc_);
	const double gcut=CutoffFD::funcs[tcut_](dr,rc_);
	for(int i=nfR_-1; i>=0; --i){
		amp-=dEdG[i]*fR_[i]->grad(dr,cut,gcut);
	}
	return amp*norm_;
}

//operators

bool operator==(const BasisR& basis1, const BasisR& basis2){
	if(basis1.nfR()!=basis2.nfR()) return false;
	else if(basis1.phiRN()!=basis2.phiRN()) return false;
	else if(basis1.tcut()!=basis2.tcut()) return false;
	else if(basis1.rc()!=basis2.rc()) return false;
	else {
		unsigned int nfR=basis1.nfR();
		PhiRN::type phiRN=basis1.phiRN();
		if(phiRN==PhiRN::G1){
			for(unsigned int i=0; i<nfR; ++i){
				if(static_cast<const PhiR_G1&>(basis1.fR(i))!=static_cast<const PhiR_G1&>(basis2.fR(i))) return false;
			}
		} else if(phiRN==PhiRN::G2){
			for(unsigned int i=0; i<nfR; ++i){
				if(static_cast<const PhiR_G2&>(basis1.fR(i))!=static_cast<const PhiR_G2&>(basis2.fR(i))) return false;
			}
		} else if(phiRN==PhiRN::T1){
			for(unsigned int i=0; i<nfR; ++i){
				if(static_cast<const PhiR_T1&>(basis1.fR(i))!=static_cast<const PhiR_T1&>(basis2.fR(i))) return false;
			}
		}
		return true;
	}
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const BasisR& obj){
		unsigned int N=0;
		N+=sizeof(obj.tcut());//name of cutoff
		N+=sizeof(obj.rc());//cutoff value
		N+=sizeof(obj.phiRN());//name of symmetry functions
		N+=sizeof(unsigned int);//number of symmetry functions
		if(obj.phiRN()==PhiRN::G1){
			for(unsigned int i=0; i<obj.nfR(); ++i) N+=nbytes(static_cast<const PhiR_G1&>(obj.fR(i)));
		} else if(obj.phiRN()==PhiRN::G2){
			for(unsigned int i=0; i<obj.nfR(); ++i) N+=nbytes(static_cast<const PhiR_G2&>(obj.fR(i)));
		} else if(obj.phiRN()==PhiRN::T1){
			for(unsigned int i=0; i<obj.nfR(); ++i) N+=nbytes(static_cast<const PhiR_T1&>(obj.fR(i)));
		} else throw std::runtime_error("Invalid radial symmetry function.");
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const BasisR& obj, char* arr){
		unsigned int pos=0,nfR=obj.nfR();
		std::memcpy(arr+pos,&obj.tcut(),sizeof(obj.tcut())); pos+=sizeof(obj.tcut());
		std::memcpy(arr+pos,&obj.rc(),sizeof(obj.rc())); pos+=sizeof(obj.rc());
		std::memcpy(arr+pos,&obj.phiRN(),sizeof(obj.phiRN())); pos+=sizeof(obj.phiRN());
		std::memcpy(arr+pos,&nfR,sizeof(nfR)); pos+=sizeof(nfR);
		if(obj.phiRN()==PhiRN::G1){
			for(unsigned int i=0; i<obj.nfR(); ++i){
				pack(static_cast<const PhiR_G1&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_G1&>(obj.fR(i)));
			}
		} else if(obj.phiRN()==PhiRN::G2){
			for(unsigned int i=0; i<obj.nfR(); ++i){
				pack(static_cast<const PhiR_G2&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_G2&>(obj.fR(i)));
			}
		} else if(obj.phiRN()==PhiRN::T1){
			for(unsigned int i=0; i<obj.nfR(); ++i){
				pack(static_cast<const PhiR_T1&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_T1&>(obj.fR(i)));
			}
		} else throw std::runtime_error("Invalid radial symmetry function.");
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(BasisR& obj, const char* arr){
		unsigned int pos=0,nfR=0;
		double rc=0;
		CutoffN::type cutN=CutoffN::UNKNOWN;
		PhiRN::type phiRN=PhiRN::UNKNOWN;
		std::memcpy(&cutN,arr+pos,sizeof(cutN)); pos+=sizeof(cutN);
		std::memcpy(&rc,arr+pos,sizeof(rc)); pos+=sizeof(rc);
		std::memcpy(&phiRN,arr+pos,sizeof(phiRN)); pos+=sizeof(phiRN);
		std::memcpy(&nfR,arr+pos,sizeof(nfR)); pos+=sizeof(nfR);
		if(phiRN==PhiRN::G1){
			obj.init_G1(nfR,cutN,rc);
			for(unsigned int i=0; i<obj.nfR(); ++i){
				unpack(static_cast<PhiR_G1&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_G1&>(obj.fR(i)));
			}
		} else if(phiRN==PhiRN::G2){
			obj.init_G2(nfR,cutN,rc);
			for(unsigned int i=0; i<obj.nfR(); ++i){
				unpack(static_cast<PhiR_G2&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_G2&>(obj.fR(i)));
			}
		} else if(phiRN==PhiRN::T1){
			obj.init_T1(nfR,cutN,rc);
			for(unsigned int i=0; i<obj.nfR(); ++i){
				unpack(static_cast<PhiR_T1&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_T1&>(obj.fR(i)));
			}
		} else throw std::runtime_error("Invalid radial symmetry function.");
	}
	
}
