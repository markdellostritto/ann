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
	if(basis.nfR()>0){
		phiRN_=basis.phiRN();
		nfR_=basis.nfR();
		if(phiRN_==PhiRN::G1){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G1(static_cast<const PhiR_G1&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::G2){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G2(static_cast<const PhiR_G2&>(basis.fR(i)));
		}
	}
}

//operators

BasisR& BasisR::operator=(const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::operator=(const BasisR&):\n";
	clear();
	if(basis.nfR()>0){
		phiRN_=basis.phiRN();
		nfR_=basis.nfR();
		if(phiRN_==PhiRN::G1){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G1(static_cast<const PhiR_G1&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::G2){
			fR_=new PhiR*[nfR_];
			for(unsigned int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G2(static_cast<const PhiR_G2&>(basis.fR(i)));
		}
	}
}

std::ostream& operator<<(std::ostream& out, const BasisR& basis){
	out<<"BasisR "<<basis.phiRN_<<" "<<basis.nfR_<<"\n";
	if(basis.phiRN_==PhiRN::G1){
		for(unsigned int i=0; i<basis.nfR_-1; ++i) out<<"\t"<<static_cast<const PhiR_G1&>(*basis.fR_[i])<<"\n";
		out<<"\t"<<static_cast<PhiR_G1&>(*basis.fR_[basis.nfR_-1]);
	} else if(basis.phiRN_==PhiRN::G2){
		for(unsigned int i=0; i<basis.nfR_-1; ++i) out<<"\t"<<static_cast<const PhiR_G2&>(*basis.fR_[i])<<"\n";
		out<<"\t"<<static_cast<PhiR_G2&>(*basis.fR_[basis.nfR_-1]);
	}
	return out;
}

//initialization

void BasisR::init_G1(CutoffN::type tcut, double rmin, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_G1(unsigned int,CutoffN::type,double,double):\n";
	if(rcut<=0) throw std::invalid_argument("Invalid radial cutoff.");
	clear();
	phiRN_=PhiRN::G1;
	nfR_=1;
	fR_=new PhiR*[1];
	fR_[0]=new PhiR_G1(tcut,rcut);
}

void BasisR::init_G2(unsigned int nR, CutoffN::type tcut, double rmin, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_G2(BasisR&,unsigned int,CutoffN::type,double):\n";
	if(nR==0) throw std::invalid_argument("Invalid number of radial functions.");
	if(rcut<=0) throw std::invalid_argument("Invalid radial cutoff.");
	clear();
	phiRN_=PhiRN::G2;
	nfR_=nR;
	fR_=new PhiR*[nfR_];
	double dr=(rcut-rmin)/(nfR_-1.0);
	for(unsigned int i=0; i<nfR_; ++i){
		double eta=-std::log(V_CUT)/(4.0*dr*dr);
		double rs=rmin+i*dr;
		fR_[i]=new PhiR_G2(tcut,rcut,rs,eta);
	}
}

//loading/printing

void BasisR::write(FILE* writer,const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::write(FILE*,const BasisR&):\n";
	std::string str;
	if(basis.phiRN()==PhiRN::G1) str="G1";
	else if(basis.phiRN()==PhiRN::G2) str="G2";
	fprintf(writer,"BasisR %s %i\n",str.c_str(),basis.nfR());
	if(basis.phiRN()==PhiRN::G1){
		//tcut,rc
		if(basis.fR(0).tcut==CutoffN::COS) str="COS";
		else if(basis.fR(0).tcut==CutoffN::TANH) str="TANH";
		fprintf(writer,"\t%s %f\n",str.c_str(),basis.fR(0).rc);
	} else if(basis.phiRN()==PhiRN::G2){
		//tcut,rc,rs,eta
		for(unsigned int i=0; i<basis.nfR(); ++i){
			if(basis.fR(i).tcut==CutoffN::COS) str="COS";
			else if(basis.fR(i).tcut==CutoffN::TANH) str="TANH";
			const PhiR_G2& g2=static_cast<const PhiR_G2&>(basis.fR(i));
			fprintf(writer,"\t%s %f %f %f\n",str.c_str(),g2.rc,g2.rs,g2.eta);
		}
	}
}

void BasisR::read(FILE* reader, BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::read(FILE*, BasisR&):\n";
	//local variables
	char* input=(char*)malloc(sizeof(char)*string::M);
	unsigned int n=0;
	std::vector<std::string> strlist;
	CutoffN::type tcut;
	//read header
	fgets(input,string::M,reader);
	n=string::substrN(input,string::WS);
	if(n!=3) throw std::runtime_error("Invalid BasisR format.");
	strlist.resize(n);
	strlist[0]=std::string(std::strtok(input,string::WS));
	strlist[1]=std::string(std::strtok(NULL,string::WS));
	strlist[2]=std::string(std::strtok(NULL,string::WS));
	//resize the basis
	unsigned int size=std::atoi(strlist[2].c_str());
	//loop over all basis functions
	if(strlist[1]=="G1"){
		basis.init_G1(CutoffN::COS,1.0,0.5);//actual values won't matter as they'll be overwritten
		//tcut,rc
		for(unsigned int i=0; i<basis.nfR(); ++i){
			fgets(input,string::M,reader);
			n=string::substrN(input,string::WS);
			if(n!=2) throw std::runtime_error("Invalid G1 format.");
			strlist.resize(n);
			strlist[0]=std::string(string::to_upper(std::strtok(input,string::WS)));
			for(unsigned int j=1; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
			if(strlist[0]=="COS") tcut=CutoffN::COS;
			else if(strlist[0]=="TANH") tcut=CutoffN::TANH;
			else throw std::runtime_error("Invalid G1 cutoff.");
			double rc=std::atof(strlist[1].c_str());
			static_cast<PhiR_G1&>(basis.fR(i))=PhiR_G1(tcut,rc);
		}
		basis.phiRN()=PhiRN::G1;
	} else if(strlist[1]=="G2"){
		basis.init_G2(size,CutoffN::COS,1.0,0.5);//actual values won't matter as they'll be overwritten
		//tcut,rc,rs,eta
		for(unsigned int i=0; i<basis.nfR(); ++i){
			fgets(input,string::M,reader);
			n=string::substrN(input,string::WS);
			if(n!=4) throw std::runtime_error("Invalid G2 format.");
			strlist.resize(n);
			strlist[0]=std::string(string::to_upper(std::strtok(input,string::WS)));
			for(unsigned int j=1; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
			if(strlist[0]=="COS") tcut=CutoffN::COS;
			else if(strlist[0]=="TANH") tcut=CutoffN::TANH;
			else throw std::runtime_error("Invalid G2 cutoff.");
			double rc=std::atof(strlist[1].c_str());
			double rs=std::atof(strlist[2].c_str());
			double eta=std::atof(strlist[3].c_str());
			static_cast<PhiR_G2&>(basis.fR(i))=PhiR_G2(tcut,rc,rs,eta);
		}
		basis.phiRN()=PhiRN::G2;
	} else throw std::invalid_argument("Invalid radial function type");
	//free local variables
	free(input);
}

//member functions

void BasisR::clear(){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::clear():\n";
	if(fR_!=NULL){
		for(unsigned int i=0; i<nfR_; ++i) delete fR_[i];
		delete[] fR_;
		fR_=NULL;
	}
	nfR_=0;
	phiRN_=PhiRN::UNKNOWN;
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const BasisR& obj){
		unsigned int N=0;
		N+=sizeof(obj.phiRN());//name of symmetry functions
		N+=sizeof(unsigned int);//number of symmetry functions
		if(obj.phiRN()==PhiRN::G1){
			for(unsigned int i=0; i<obj.nfR(); ++i) N+=nbytes(static_cast<const PhiR_G1&>(obj.fR(i)));
		} else if(obj.phiRN()==PhiRN::G2){
			for(unsigned int i=0; i<obj.nfR(); ++i) N+=nbytes(static_cast<const PhiR_G2&>(obj.fR(i)));
		} else throw std::runtime_error("Invalid radial symmetry function.");
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const BasisR& obj, char* arr){
		unsigned int pos=0,size=obj.nfR();
		std::memcpy(arr+pos,&obj.phiRN(),sizeof(obj.phiRN())); pos+=sizeof(obj.phiRN());
		std::memcpy(arr+pos,&size,sizeof(size)); pos+=sizeof(size);
		if(obj.phiRN()==PhiRN::G1){
			for(unsigned int i=0; i<obj.nfR(); ++i){
				pack(static_cast<const PhiR_G1&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_G1&>(obj.fR(i)));
			}
		} else if(obj.phiRN()==PhiRN::G2){
			for(unsigned int i=0; i<obj.nfR(); ++i){
				pack(static_cast<const PhiR_G2&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_G2&>(obj.fR(i)));
			}
		} else throw std::runtime_error("Invalid radial symmetry function.");
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(BasisR& obj, const char* arr){
		unsigned int pos=0,size=0;
		std::memcpy(&obj.phiRN(),arr+pos,sizeof(obj.phiRN())); pos+=sizeof(obj.phiRN());
		std::memcpy(&size,arr+pos,sizeof(size)); pos+=sizeof(size);
		if(obj.phiRN()==PhiRN::G1){
			obj.init_G1(CutoffN::COS,1.0,0.5);//actual values won't matter as they'll be overwritten
			for(unsigned int i=0; i<obj.nfR(); ++i){
				unpack(static_cast<PhiR_G1&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_G1&>(obj.fR(i)));
			}
		} else if(obj.phiRN()==PhiRN::G2){
			obj.init_G2(size,CutoffN::COS,1.0,0.5);//actual values won't matter as they'll be overwritten
			for(unsigned int i=0; i<obj.nfR(); ++i){
				unpack(static_cast<PhiR_G2&>(obj.fR(i)),arr+pos); pos+=nbytes(static_cast<const PhiR_G2&>(obj.fR(i)));
			}
		} else throw std::runtime_error("Invalid radial symmetry function.");
	}
	
}
