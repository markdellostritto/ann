// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// ann - symmetry functions
#include "symm_radial_g1.hpp"
#include "symm_radial_g2.hpp"
#include "symm_radial_t1.hpp"
// ann - string
#include "string.hpp"
// ann - basis - radial
#include "basis_radial.hpp"

//==== using statements ====

using math::constant::PI;

//==== member variables ====

const double BasisR::V_CUT=1e-5;

//==== constructors/destructors ====

/**
* destructor
*/
BasisR::~BasisR(){
	clear();
}	

/**
* copy constructor
* @param basis - the basis to be copied
*/
BasisR::BasisR(const BasisR& basis):normT_(NormN::UNKNOWN),phiRN_(PhiRN::UNKNOWN),fR_(NULL),cutoff_(NULL){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR(const BasisR&):\n";
	fR_=NULL;
	clear();
	if(basis.cutoff()!=NULL){
		switch(basis.cutoff()->name()){
			case cutoff::Name::COS: cutoff_=new cutoff::Cos(basis.rc()); break;
			case cutoff::Name::TANH: cutoff_=new cutoff::Tanh(basis.rc()); break;
			default: throw std::invalid_argument("Invalid cutoff function");
		}
	}
	rc_=basis.rc();//cutoff value
	normT_=basis.normT();
	phiRN_=basis.phiRN();//radial function type
	nfR_=basis.nfR();//number of radial functions
	if(basis.nfR()>0){
		norm_=norm(normT_,rc_);
		symm_=Eigen::VectorXd::Zero(nfR_);
		if(phiRN_==PhiRN::G1){
			fR_=new PhiR*[nfR_];
			for(int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G1(static_cast<const PhiR_G1&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::G2){
			fR_=new PhiR*[nfR_];
			for(int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G2(static_cast<const PhiR_G2&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::T1){
			fR_=new PhiR*[nfR_];
			for(int i=0; i<nfR_; ++i) fR_[i]=new PhiR_T1(static_cast<const PhiR_T1&>(basis.fR(i)));
		}
	}
}

//==== operators ====

/**
* assignment operator
* @param basis - the basis to be copied
* @return the object assigned
*/
BasisR& BasisR::operator=(const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::operator=(const BasisR&):\n";
	clear();
	if(basis.cutoff()!=NULL){
		switch(basis.cutoff()->name()){
			case cutoff::Name::COS: cutoff_=new cutoff::Cos(basis.rc()); break;
			case cutoff::Name::TANH: cutoff_=new cutoff::Tanh(basis.rc()); break;
			default: throw std::invalid_argument("Invalid cutoff function"); break;
		}
	}
	rc_=basis.rc();//cutoff value
	normT_=basis.normT();
	phiRN_=basis.phiRN();//radial function type
	nfR_=basis.nfR();//number of radial functions
	if(basis.nfR()>0){
		norm_=norm(normT_,rc_);
		symm_=Eigen::VectorXd::Zero(nfR_);
		if(phiRN_==PhiRN::G1){
			fR_=new PhiR*[nfR_];
			for(int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G1(static_cast<const PhiR_G1&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::G2){
			fR_=new PhiR*[nfR_];
			for(int i=0; i<nfR_; ++i) fR_[i]=new PhiR_G2(static_cast<const PhiR_G2&>(basis.fR(i)));
		} else if(phiRN_==PhiRN::T1){
			fR_=new PhiR*[nfR_];
			for(int i=0; i<nfR_; ++i) fR_[i]=new PhiR_T1(static_cast<const PhiR_T1&>(basis.fR(i)));
		}
	}
	return *this;
}

/**
* print basis
* @param out - the output stream
* @param basis - the basis to print
* @return the output stream
*/
std::ostream& operator<<(std::ostream& out, const BasisR& basis){
	if(basis.cutoff()!=NULL) out<<"BasisR "<<basis.normT_<<" "<<basis.cutoff_->name()<<" "<<basis.rc_<<" "<<basis.phiRN_<<" "<<basis.nfR_;
	else out<<"BasisR UNKNOWN "<<basis.rc_<<" "<<basis.phiRN_<<" "<<basis.nfR_;
	if(basis.phiRN_==PhiRN::G1){
		for(int i=0; i<basis.nfR_; ++i) out<<"\n\t"<<static_cast<const PhiR_G1&>(*basis.fR_[i]);
	} else if(basis.phiRN_==PhiRN::G2){
		for(int i=0; i<basis.nfR_; ++i) out<<"\n\t"<<static_cast<const PhiR_G2&>(*basis.fR_[i]);
	} else if(basis.phiRN_==PhiRN::T1){
		for(int i=0; i<basis.nfR_; ++i) out<<"\n\t"<<static_cast<const PhiR_T1&>(*basis.fR_[i]);
	}
	return out;
}

/**
* equality operator
* @param basis1 - basis - first
* @param basis2 - basis - second
* @return the equality of the arguments
*/
bool operator==(const BasisR& basis1, const BasisR& basis2){
	if(basis1.nfR()!=basis2.nfR()) return false;
	else if(basis1.phiRN()!=basis2.phiRN()) return false;
	else if(basis1.normT()!=basis2.normT()) return false;
	else if(basis1.cutoff()->name()!=basis2.cutoff()->name()) return false;
	else if(basis1.rc()!=basis2.rc()) return false;
	else {
		const int nfR=basis1.nfR();
		PhiRN::type phiRN=basis1.phiRN();
		if(phiRN==PhiRN::G1){
			for(int i=0; i<nfR; ++i){
				if(static_cast<const PhiR_G1&>(basis1.fR(i))!=static_cast<const PhiR_G1&>(basis2.fR(i))) return false;
			}
		} else if(phiRN==PhiRN::G2){
			for(int i=0; i<nfR; ++i){
				if(static_cast<const PhiR_G2&>(basis1.fR(i))!=static_cast<const PhiR_G2&>(basis2.fR(i))) return false;
			}
		} else if(phiRN==PhiRN::T1){
			for(int i=0; i<nfR; ++i){
				if(static_cast<const PhiR_T1&>(basis1.fR(i))!=static_cast<const PhiR_T1&>(basis2.fR(i))) return false;
			}
		}
		return true;
	}
}

//==== initialization ====

/**
* basis initialization - G1
* @param nR - the number of radial basis functions
* @param tcut - the type of cutoff functions
* @param rcut - the cutoff distance
*/
void BasisR::init_G1(int nR, NormN::type normT, cutoff::Name::type tcut, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_G1(int,cutoff::Name::type,double,double):\n";
	if(rcut<=0) throw std::invalid_argument("BasisR::init_G1(int,cutoff::Name::type,double): Invalid radial cutoff.");
	clear();
	//set basis parameters
	normT_=normT;//normalization scheme
	phiRN_=PhiRN::G1;//radial function type
	nfR_=nR;//number of radial functions
	switch(tcut){
		case cutoff::Name::COS: cutoff_=new cutoff::Cos(rcut); break;
		case cutoff::Name::TANH: cutoff_=new cutoff::Tanh(rcut); break;
		default: throw std::invalid_argument("Invalid cutoff function");
	}
	rc_=rcut;//cutoff value
	if(nfR_>0){
		norm_=norm(normT_,rc_);
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfR_);
		fR_=new PhiR*[nfR_];
		for(int i=0; i<nfR_; ++i){
			fR_[i]=new PhiR_G1();
		}
	}
}

/**
* basis initialization - G2
* @param nR - the number of radial basis functions
* @param tcut - the type of cutoff functions
* @param rcut - the cutoff distance
*/
void BasisR::init_G2(int nR, NormN::type normT, cutoff::Name::type tcut, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_G2(int,cutoff::Name::type,double):\n";
	if(rcut<=0) throw std::invalid_argument("BasisR::init_G2(int,cutoff::Name::type,double): Invalid radial cutoff.");
	clear();
	//set basis parameters
	normT_=normT;//normalization scheme
	phiRN_=PhiRN::G2;//radial function type
	nfR_=nR;//number of radial functions
	switch(tcut){
		case cutoff::Name::COS: cutoff_=new cutoff::Cos(rcut); break;
		case cutoff::Name::TANH: cutoff_=new cutoff::Tanh(rcut); break;
		default: throw std::invalid_argument("Invalid cutoff function");
	}
	rc_=rcut;//cutoff value
	norm_=norm(normT_,rc_);
	if(nfR_>0){
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfR_);
		fR_=new PhiR*[nfR_];
		const double dr=rcut/(nfR_-1.0);
		for(int i=0; i<nfR_; ++i){
			const double eta=-std::log(V_CUT)/(4.0*dr*dr);
			const double rs=i*dr;
			fR_[i]=new PhiR_G2(rs,eta);
		}
	}
}

/**
* basis initialization - T1
* @param nR - the number of radial basis functions
* @param tcut - the type of cutoff functions
* @param rcut - the cutoff distance
*/
void BasisR::init_T1(int nR, NormN::type normT, cutoff::Name::type tcut, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_T1(int,cutoff::Name::type,double):\n";
	if(rcut<=0) throw std::invalid_argument("BasisR::init_T1(int,cutoff::Name::type,double): Invalid radial cutoff.");
	clear();
	//set basis parameters
	normT_=normT;//normalization scheme
	phiRN_=PhiRN::T1;//radial function type
	nfR_=nR;//number of radial functions
	switch(tcut){
		case cutoff::Name::COS: cutoff_=new cutoff::Cos(rcut); break;
		case cutoff::Name::TANH: cutoff_=new cutoff::Tanh(rcut); break;
		default: throw std::invalid_argument("Invalid cutoff function");
	}
	rc_=rcut;//cutoff value
	if(nfR_>0){
		norm_=norm(normT_,rc_);
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfR_);
		fR_=new PhiR*[nfR_];
		const double dr=rcut/(nfR_-1.0);
		for(int i=0; i<nfR_; ++i){
			const double eta=-std::log(V_CUT)/(4.0*dr*dr);
			const double rs=i*dr;
			fR_[i]=new PhiR_T1(rs,eta);
		}
	}
}

//==== reading/writing ====

/**
* write basis to file
* @param filename - the file to which we will write the basis
* @param basis - the basis to be written
*/
void BasisR::write(const char* filename, const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::write(const char*,const BasisR&):\n";
	FILE* writer=fopen(filename,"w");
	if(writer!=NULL){
		BasisR::write(writer,basis);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("BasisR::write(const char*,const BasisR&): Could not write BasisR to file: \"")+std::string(filename)+std::string("\""));
}

/**
* read basis from file
* @param filename - the file which we will read the basis from
* @param basis - the basis to be read
*/
void BasisR::read(const char* filename, BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::read(const char*,const BasisR&):\n";
	FILE* reader=fopen(filename,"r");
	if(reader!=NULL){
		BasisR::read(reader,basis);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("BasisR::read(const char*,BasisR&): Could not read BasisR from file: \"")+std::string(filename)+std::string("\""));
}
	
/**
* write basis to file
* @param writer - file pointer
* @param basis - the basis to be written
*/
void BasisR::write(FILE* writer,const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::write(FILE*,const BasisR&):\n";
	const char* str_tcut=cutoff::Name::name(basis.cutoff()->name());
	const char* str_norm=NormN::name(basis.normT());
	const char* str_phirn=PhiRN::name(basis.phiRN());
	fprintf(writer,"BasisR %s %s %f %s %i\n",str_norm,str_tcut,basis.rc(),str_phirn,basis.nfR());
	if(basis.phiRN()==PhiRN::G1){
		//tcut,rc
		for(int i=0; i<basis.nfR(); ++i){
			fprintf(writer,"\tG1\n");
		}
	} else if(basis.phiRN()==PhiRN::G2){
		//tcut,rc,rs,eta
		for(int i=0; i<basis.nfR(); ++i){
			const PhiR_G2& g2=static_cast<const PhiR_G2&>(basis.fR(i));
			fprintf(writer,"\tG2 %f %f\n",g2.rs,g2.eta);
		}
	} else if(basis.phiRN()==PhiRN::T1){
		//tcut,rc,rs,eta
		for(int i=0; i<basis.nfR(); ++i){
			const PhiR_T1& t1=static_cast<const PhiR_T1&>(basis.fR(i));
			fprintf(writer,"\tT1 %f %f\n",t1.rs,t1.eta);
		}
	}
}

/**
* read basis from file
* @param writer - file pointer
* @param basis - the basis to be read
*/
void BasisR::read(FILE* reader, BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::read(FILE*, BasisR&):\n";
	//local variables
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	//split header
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	if(strlist.size()!=6) throw std::runtime_error("BasisR::read(FILE*,BasisR&): Invalid BasisR format.");
	//read header
	const NormN::type normT=NormN::read(strlist[1].c_str());
	const cutoff::Name::type tcut=cutoff::Name::read(strlist[2].c_str());
	const double rc=std::atof(strlist[3].c_str());
	const std::string phiRNstr=strlist[4];
	const int nfr=std::atoi(strlist[5].c_str());
	if(nfr>0) basis.symm()=Eigen::VectorXd::Zero(nfr);
	//loop over all basis functions
	if(phiRNstr=="G1"){
		basis.init_G1(nfr,normT,tcut,rc);
		//G1
		for(int i=0; i<basis.nfR(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+0) throw std::runtime_error("BasisR::read(FILE*,BasisR&): Invalid G1 format.");
			static_cast<PhiR_G1&>(basis.fR(i))=PhiR_G1();
		}
		basis.phiRN()=PhiRN::G1;
	} else if(phiRNstr=="G2"){
		basis.init_G2(nfr,normT,tcut,rc);
		//G2,rs,eta
		for(int i=0; i<basis.nfR(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+2) throw std::runtime_error("BasisR::read(FILE*,BasisR&): Invalid G2 format.");
			const double rs=std::atof(strlist[1].c_str());
			const double eta=std::atof(strlist[2].c_str());
			static_cast<PhiR_G2&>(basis.fR(i))=PhiR_G2(rs,eta);
		}
		basis.phiRN()=PhiRN::G2;
	} else if(phiRNstr=="T1"){
		basis.init_T1(nfr,normT,tcut,rc);
		//T1,rs,eta
		for(int i=0; i<basis.nfR(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+2) throw std::runtime_error("BasisR::read(FILE*,BasisR&): Invalid T1 format.");
			const double rs=std::atof(strlist[1].c_str());
			const double eta=std::atof(strlist[2].c_str());
			static_cast<PhiR_T1&>(basis.fR(i))=PhiR_T1(rs,eta);
		}
		basis.phiRN()=PhiRN::T1;
	} else throw std::invalid_argument("BasisR::read(FILE*,BasisR&): Invalid radial function type");
	//free local variables
	delete[] input;
}

//==== member functions ====

/**
* clear basis
*/
void BasisR::clear(){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::clear():\n";
	if(fR_!=NULL){
		for(int i=0; i<nfR_; ++i) delete fR_[i];
		delete[] fR_;
		fR_=NULL;
	}
	if(cutoff_!=NULL){
		delete cutoff_;
		cutoff_=NULL;
	}
	rc_=0;
	norm_=0;
	nfR_=0;
	normT_=NormN::UNKNOWN;
	phiRN_=PhiRN::UNKNOWN;
}

/**
* compute symmetry functions
* @param dr - the distance between the central atom and a neighboring atom
*/
void BasisR::symm(double dr){
	const double cut=cutoff_->val(dr);
	for(int i=0; i<nfR_; ++i){
		symm_[i]=fR_[i]->val(dr,cut)*norm_;
	}
}

/**
* compute force
* @param dr - the distance between the central atom and a neighboring atom
* @param dEdG - gradient of energy w.r.t. the inputs
*/
double BasisR::force(double dr, const double* dEdG)const{
	double amp=0;
	const double cut=cutoff_->val(dr);
	const double gcut=cutoff_->grad(dr);
	for(int i=0; i<nfR_; ++i){
		amp-=dEdG[i]*fR_[i]->grad(dr,cut,gcut);
	}
	return amp*norm_;
}

//==== static functions ====

/**
* compute the normalization constant
* @param rc - the cutoff radius
*/
double BasisR::norm(NormN::type normT, double rc){
	double tmp=0;
	switch(normT){
		case NormN::UNIT: tmp=1.0; break;
		case NormN::VOL: tmp=1.0/(0.5*4.0/3.0*(PI*PI-6.0)/PI*rc*rc*rc); break;
		default: throw std::invalid_argument("BasisR::norm(NormT::type,double): invalid normalization scheme.");
	}
	return tmp;
}

//==== serialization ====

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisR& obj){
		int N=0;
		cutoff::Name::type tcut=cutoff::Name::UNKNOWN;
		if(obj.cutoff()!=NULL) tcut=obj.cutoff()->name();
		N+=sizeof(obj.normT());//name of normalization scheme
		N+=sizeof(tcut);//name of cutoff
		N+=sizeof(obj.rc());//cutoff value
		N+=sizeof(obj.phiRN());//name of symmetry functions
		N+=sizeof(int);//number of symmetry functions
		if(obj.phiRN()==PhiRN::G1){
			for(int i=0; i<obj.nfR(); ++i) N+=nbytes(static_cast<const PhiR_G1&>(obj.fR(i)));
		} else if(obj.phiRN()==PhiRN::G2){
			for(int i=0; i<obj.nfR(); ++i) N+=nbytes(static_cast<const PhiR_G2&>(obj.fR(i)));
		} else if(obj.phiRN()==PhiRN::T1){
			for(int i=0; i<obj.nfR(); ++i) N+=nbytes(static_cast<const PhiR_T1&>(obj.fR(i)));
		} else throw std::runtime_error("nbytes(const BasisR&): Invalid radial symmetry function.");
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisR& obj, char* arr){
		int pos=0,nfR=obj.nfR();
		cutoff::Name::type tcut=cutoff::Name::UNKNOWN;
		if(obj.cutoff()!=NULL) tcut=obj.cutoff()->name();
		std::memcpy(arr+pos,&obj.normT(),sizeof(obj.normT())); pos+=sizeof(obj.normT());
		std::memcpy(arr+pos,&tcut,sizeof(tcut)); pos+=sizeof(tcut);
		std::memcpy(arr+pos,&obj.rc(),sizeof(obj.rc())); pos+=sizeof(obj.rc());
		std::memcpy(arr+pos,&obj.phiRN(),sizeof(obj.phiRN())); pos+=sizeof(obj.phiRN());
		std::memcpy(arr+pos,&nfR,sizeof(nfR)); pos+=sizeof(nfR);
		if(obj.phiRN()==PhiRN::G1){
			for(int i=0; i<obj.nfR(); ++i){
				pos+=pack(static_cast<const PhiR_G1&>(obj.fR(i)),arr+pos);
			}
		} else if(obj.phiRN()==PhiRN::G2){
			for(int i=0; i<obj.nfR(); ++i){
				pos+=pack(static_cast<const PhiR_G2&>(obj.fR(i)),arr+pos);
			}
		} else if(obj.phiRN()==PhiRN::T1){
			for(int i=0; i<obj.nfR(); ++i){
				pos+=pack(static_cast<const PhiR_T1&>(obj.fR(i)),arr+pos);
			}
		} else throw std::runtime_error("pack(const BasisR&,char*): Invalid radial symmetry function.");
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisR& obj, const char* arr){
		int pos=0,nfR=0;
		double rc=0;
		NormN::type normT=NormN::UNKNOWN;
		cutoff::Name::type cutN=cutoff::Name::UNKNOWN;
		PhiRN::type phiRN=PhiRN::UNKNOWN;
		std::memcpy(&normT,arr+pos,sizeof(normT)); pos+=sizeof(normT);
		std::memcpy(&cutN,arr+pos,sizeof(cutN)); pos+=sizeof(cutN);
		std::memcpy(&rc,arr+pos,sizeof(rc)); pos+=sizeof(rc);
		std::memcpy(&phiRN,arr+pos,sizeof(phiRN)); pos+=sizeof(phiRN);
		std::memcpy(&nfR,arr+pos,sizeof(nfR)); pos+=sizeof(nfR);
		if(phiRN==PhiRN::G1){
			obj.init_G1(nfR,normT,cutN,rc);
			for(int i=0; i<obj.nfR(); ++i){
				pos+=unpack(static_cast<PhiR_G1&>(obj.fR(i)),arr+pos);
			}
		} else if(phiRN==PhiRN::G2){
			obj.init_G2(nfR,normT,cutN,rc);
			for(int i=0; i<obj.nfR(); ++i){
				pos+=unpack(static_cast<PhiR_G2&>(obj.fR(i)),arr+pos);
			}
		} else if(phiRN==PhiRN::T1){
			obj.init_T1(nfR,normT,cutN,rc);
			for(int i=0; i<obj.nfR(); ++i){
				pos+=unpack(static_cast<PhiR_T1&>(obj.fR(i)),arr+pos);
			}
		} else throw std::runtime_error("unpack(BasisR&,const char*): Invalid radial symmetry function.");
		return pos;
	}
	
}
