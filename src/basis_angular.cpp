// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// ann- symmetry functions
#include "symm_angular_g3.hpp"
#include "symm_angular_g4.hpp"
// ann - string
#include "string.hpp"
// ann - basis - angular
#include "basis_angular.hpp"

//==== using statements ====

using math::constant::PI;

//==== member variables ====

const double BasisA::V_CUT=1e-5;

//==== constructors/destructors ====

/**
* destructor
*/
BasisA::~BasisA(){
	clear();
}

/**
* copy constructor
* @param basis - the basis to be copied
*/
BasisA::BasisA(const BasisA& basis):normT_(NormN::UNKNOWN),phiAN_(PhiAN::UNKNOWN),fA_(NULL),cutoff_(NULL){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA(const BasisA&):\n";
	fA_=NULL;
	clear();
	if(basis.cutoff()!=NULL){
		switch(basis.cutoff()->name()){//cutoff
			case cutoff::Name::COS: cutoff_=new cutoff::Cos(basis.rc()); break;
			case cutoff::Name::TANH: cutoff_=new cutoff::Tanh(basis.rc()); break;
			default: throw std::invalid_argument("Invalid cutoff function");
		}
	}
	rc_=basis.rc();//cutoff value
	normT_=basis.normT();
	phiAN_=basis.phiAN();//angular function type
	nfA_=basis.nfA();//number of angular functions
	if(nfA_>0){
		norm_=norm(normT_,rc_);
		symm_=Eigen::VectorXd::Zero(nfA_);
		if(phiAN_==PhiAN::G3){
			fA_=new PhiA*[nfA_];
			for(int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G3(static_cast<const PhiA_G3&>(basis.fA(i)));
		} else if(phiAN_==PhiAN::G4){
			fA_=new PhiA*[nfA_];
			for(int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G4(static_cast<const PhiA_G4&>(basis.fA(i)));
		}
	}
}

/**
* assignment operator
* @param basis - the basis to be copied
* @return the object assigned
*/
BasisA& BasisA::operator=(const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::operator=(const BasisA&):\n";
	clear();
	if(basis.cutoff()!=NULL){
		switch(basis.cutoff()->name()){//cutoff
			case cutoff::Name::COS: cutoff_=new cutoff::Cos(basis.rc()); break;
			case cutoff::Name::TANH: cutoff_=new cutoff::Tanh(basis.rc()); break;
			default: throw std::invalid_argument("Invalid cutoff function");
		}
	}
	rc_=basis.rc();//cutoff value
	normT_=basis.normT();
	phiAN_=basis.phiAN();//angular function type
	nfA_=basis.nfA();//number of angular functions
	if(nfA_>0){
		norm_=norm(normT_,rc_);
		symm_=Eigen::VectorXd::Zero(nfA_);
		if(phiAN_==PhiAN::G3){
			fA_=new PhiA*[nfA_];
			for(int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G3(static_cast<const PhiA_G3&>(basis.fA(i)));
		} else if(phiAN_==PhiAN::G4){
			fA_=new PhiA*[nfA_];
			for(int i=0; i<nfA_; ++i) fA_[i]=new PhiA_G4(static_cast<const PhiA_G4&>(basis.fA(i)));
		}
	}
	return *this;
}

//==== initialization ====

/**
* basis initialization - G3
* @param nA - the number of angular basis functions
* @param tcut - the type of cutoff functions
* @param rcut - the cutoff distance
*/
void BasisA::init_G3(int nA, NormN::type normT, cutoff::Name::type tcut, double rcut){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init_G3(int,cutoff::Name::type,double):\n";
	clear();
	//set basis parameters
	normT_=normT;//normalization scheme
	phiAN_=PhiAN::G3;//angular function type
	nfA_=nA;//number of angular functions
	switch(tcut){//cutoff
		case cutoff::Name::COS: cutoff_=new cutoff::Cos(rcut); break;
		case cutoff::Name::TANH: cutoff_=new cutoff::Tanh(rcut); break;
		default: throw std::invalid_argument("Invalid cutoff function");
	}
	rc_=rcut;//cutoff value
	if(nfA_>0){
		norm_=norm(normT_,rc_);
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfA_);
		const double eta=1.0/(3.0*rcut*rcut);
		fA_=new PhiA*[nfA_];
		const double lambda=1;
		for(int i=0; i<nfA_; ++i){
			const double zeta=std::pow((i+1.0),std::log(i+2.0)/std::log(nfA_/(nfA_+1.0)+2.0));
			fA_[i]=new PhiA_G3(eta,zeta,lambda);
		}
	}
}

/**
* basis initialization - G4
* @param nA - the number of angular basis functions
* @param tcut - the type of cutoff functions
* @param rcut - the cutoff distance
*/
void BasisA::init_G4(int nA, NormN::type normT, cutoff::Name::type tcut, double rcut){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init_G4(BasisA&,int,cutoff::Name::type,double):\n";
	clear();
	//set basis parameters
	normT_=normT;//normalization scheme
	phiAN_=PhiAN::G4;//angular function type
	nfA_=nA;//number of angular functions
	switch(tcut){//cutoff
		case cutoff::Name::COS: cutoff_=new cutoff::Cos(rcut); break;
		case cutoff::Name::TANH: cutoff_=new cutoff::Tanh(rcut); break;
		default: throw std::invalid_argument("Invalid cutoff function");
	}
	rc_=rcut;//cutoff value
	if(nfA_>0){
		norm_=norm(normT_,rc_);
		//generate symmetry functions
		symm_=Eigen::VectorXd::Zero(nfA_);
		const double eta=1.0/(2.0*rcut*rcut);
		fA_=new PhiA*[nfA_];
		const double lambda=1;
		for(int i=0; i<nfA_; ++i){
			const double zeta=std::pow((i+1.0),std::log(i+2.0)/std::log(nfA_/(nfA_+1.0)+2.0));
			fA_[i]=new PhiA_G4(eta,zeta,lambda);
		}
	}
}

//==== reading/writing ====

/**
* write basis to file
* @param filename - the file to which we will write the basis
* @param basis - the basis to be written
*/
void BasisA::write(const char* filename, const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::write(const char*,const BasisA&):\n";
	FILE* writer=fopen(filename,"w");
	if(writer!=NULL){
		BasisA::write(writer,basis);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("BasisA::write(const char*,const BasisA&): Could not write BasisA to file: \"")+std::string(filename)+std::string("\""));
}

/**
* read basis from file
* @param filename - the file which we will read the basis from
* @param basis - the basis to be read
*/
void BasisA::read(const char* filename, BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::read(const char*,const BasisA&):\n";
	FILE* reader=fopen(filename,"r");
	if(reader!=NULL){
		BasisA::read(reader,basis);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("BasisA::read(const char*,BasisA&): Could not read BasisA from file: \"")+std::string(filename)+std::string("\""));
}

/**
* write basis to file
* @param writer - file pointer
* @param basis - the basis to be written
*/
void BasisA::write(FILE* writer, const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::write(FILE*):\n";
	const char* str_norm=NormN::name(basis.normT());
	const char* str_tcut=cutoff::Name::name(basis.cutoff()->name());
	const char* str_phian=PhiAN::name(basis.phiAN());
	fprintf(writer,"BasisA %s %s %f %s %i\n",str_norm,str_tcut,basis.rc(),str_phian,basis.nfA());
	if(basis.phiAN()==PhiAN::G3){
		//tcut,rcut,eta,zeta,lambda
		for(int i=0; i<basis.nfA(); ++i){
			const PhiA_G3& g3=static_cast<const PhiA_G3&>(basis.fA(i));
			fprintf(writer,"\tG3 %f %f %i\n",g3.eta,g3.zeta,g3.lambda);
		}
	} else if(basis.phiAN()==PhiAN::G4){
		//tcut,rcut,eta,zeta,lambda
		for(int i=0; i<basis.nfA(); ++i){
			const PhiA_G4& g4=static_cast<const PhiA_G4&>(basis.fA(i));
			fprintf(writer,"\tG4 %f %f %i\n",g4.eta,g4.zeta,g4.lambda);
		}
	}
}

/**
* read basis from file
* @param writer - file pointer
* @param basis - the basis to be read
*/
void BasisA::read(FILE* reader, BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::read(FILE*, BasisA&):\n";
	//local variables
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	//split header
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	if(strlist.size()!=6) throw std::runtime_error("BasisA::read(FILE*,BasisA&): Invalid BasisR format.");
	//read header
	const NormN::type normT=NormN::read(strlist[1].c_str());
	const cutoff::Name::type tcut=cutoff::Name::read(strlist[2].c_str());
	const double rc=std::atof(strlist[3].c_str());
	const std::string phiANstr=strlist[4];
	const int nfa=std::atoi(strlist[5].c_str());
	if(nfa>0) basis.symm()=Eigen::VectorXd::Zero(nfa);
	//loop over all basis functions
	if(phiANstr=="G3"){
		basis.init_G3(nfa,normT,tcut,rc);
		//G3,eta,zeta,lambda
		for(int i=0; i<basis.nfA(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+3) throw std::runtime_error("BasisA::read(FILE*,BasisA&): Invalid G3 format.");
			const double eta=std::atof(strlist[1].c_str());
			const double zeta=std::atof(strlist[2].c_str());
			const int lambda=std::atoi(strlist[3].c_str());
			static_cast<PhiA_G3&>(basis.fA(i))=PhiA_G3(eta,zeta,lambda);
		}
		basis.phiAN()=PhiAN::G3;
	} else if(phiANstr=="G4"){
		basis.init_G4(nfa,normT,tcut,rc);
		//G4,eta,zeta,lambda
		for(int i=0; i<basis.nfA(); ++i){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			if(strlist.size()!=1+3) throw std::runtime_error("BasisA::read(FILE*,BasisA&): Invalid G4 format.");
			const double eta=std::atof(strlist[1].c_str());
			const double zeta=std::atof(strlist[2].c_str());
			const int lambda=std::atoi(strlist[3].c_str());
			static_cast<PhiA_G4&>(basis.fA(i))=PhiA_G4(eta,zeta,lambda);
		}
		basis.phiAN()=PhiAN::G4;
	} else throw std::invalid_argument("BasisA::read(FILE*,BasisA&): Invalid angular function type");
	//free local variables
	delete[] input;
}

//==== operators ====

bool operator==(const BasisA& basis1, const BasisA& basis2){
	if(basis1.nfA()!=basis2.nfA()) return false;
	else if(basis1.normT()!=basis2.normT()) return false;
	else if(basis1.phiAN()!=basis2.phiAN()) return false;
	else if(basis1.cutoff()->name()!=basis2.cutoff()->name()) return false;
	else if(basis1.rc()!=basis2.rc()) return false;
	else {
		const int nfA=basis1.nfA();
		PhiAN::type phiAN=basis1.phiAN();
		if(phiAN==PhiAN::G3){
			for(int i=0; i<nfA; ++i){
				if(static_cast<const PhiA_G3&>(basis1.fA(i))!=static_cast<const PhiA_G3&>(basis2.fA(i))) return false;
			}
		} else if(phiAN==PhiAN::G4){
			for(int i=0; i<nfA; ++i){
				if(static_cast<const PhiA_G4&>(basis1.fA(i))!=static_cast<const PhiA_G4&>(basis2.fA(i))) return false;
			}
		}
		return true;
	}
}

std::ostream& operator<<(std::ostream& out, const BasisA& basis){
	if(basis.cutoff_!=NULL) out<<"BasisA "<<basis.normT_<<" "<<basis.cutoff_->name()<<" "<<basis.rc_<<" "<<basis.phiAN_<<" "<<basis.nfA_;
	else out<<"BasisA UNKNOWN "<<basis.normT_<<" "<<basis.rc_<<" "<<basis.phiAN_<<" "<<basis.nfA_;
	if(basis.phiAN_==PhiAN::G3){
		for(int i=0; i<basis.nfA_; ++i) out<<"\n\t"<<static_cast<const PhiA_G3&>(*basis.fA_[i]);
	} else if(basis.phiAN_==PhiAN::G4){
		for(int i=0; i<basis.nfA_; ++i) out<<"\n\t"<<static_cast<const PhiA_G4&>(*basis.fA_[i]);
	}
	return out;
}

//==== member functions ====

/**
* clear basis
*/
void BasisA::clear(){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::clear():\n";
	if(fA_!=NULL){
		for(int i=0; i<nfA_; ++i) delete fA_[i];
		delete[] fA_;
		fA_=NULL;
	}
	if(cutoff_!=NULL){
		delete cutoff_;
		cutoff_=NULL;
	}
	rc_=0;
	norm_=0;
	nfA_=0;
	normT_=NormN::UNKNOWN;
	phiAN_=PhiAN::UNKNOWN;
}	

/**
* compute symmetry functions
* @param dr - the distance between the central atom and a neighboring atom
*/
void BasisA::symm(double cos, const double d[3]){
	const double c[3]={
		cutoff_->val(d[0]),//cut(rij)
		cutoff_->val(d[1]),//cut(rik)
		cutoff_->val(d[2]) //cut(rjk)
	};
	for(int na=0; na<nfA_; ++na){
		symm_[na]=fA_[na]->val(cos,d,c)*norm_;
	}
}

/**
* compute force
* @param dr - the distance between the central atom and a neighboring atom
* @param dEdG - gradient of energy w.r.t. the inputs
*/
void BasisA::force(double& phi, double* eta, double cos, const double d[3], const double* dEdG)const{
	double fangle,gangle,fdist;
	double gradd[3];//grad w.r.t. distance (not angle)
	const double c[3]={
		cutoff_->val(d[0]),//cut(rij)
		cutoff_->val(d[1]),//cut(rik)
		cutoff_->val(d[2]) //cut(rjk)
	};
	const double g[3]={
		cutoff_->grad(d[0]),//cut'(rij)
		cutoff_->grad(d[1]),//cut'(rik)
		cutoff_->grad(d[2]) //cut'(rjk)
	};
	phi=0;
	eta[0]=0;
	eta[1]=0;
	eta[2]=0;
	for(int na=0; na<nfA_; ++na){
		//compute
		fA_[na]->compute_angle(cos,fangle,gangle);
		fA_[na]->compute_dist(d,c,g,fdist,gradd);
		//compute phi
		phi-=dEdG[na]*fdist*gangle;
		//compute eta
		fangle*=dEdG[na];
		eta[0]-=fangle*gradd[0];
		eta[1]-=fangle*gradd[1];
		eta[2]-=fangle*gradd[2];
	}
	phi*=norm_;
	eta[0]*=norm_;
	eta[1]*=norm_;
	eta[2]*=norm_;
}

//==== static functions ====

/**
* compute the normalization constant
* @param rc - the cutoff radius
*/
double BasisA::norm(NormN::type normT, double rc){
	double tmp=0;
	switch(normT){
		case NormN::UNIT: tmp=1.0; break;
		case NormN::VOL: tmp=1.0/(0.5*4.0/3.0*(PI*PI-6.0)/PI*rc*rc*rc); tmp*=tmp; break;
		default: throw std::invalid_argument("BasisA::norm(NormT::type,double): invalid normalization scheme.");
	}
	return tmp;
}

//==== serialization ====

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisA& obj){
		int N=0;
		cutoff::Name::type tcut=cutoff::Name::UNKNOWN;
		if(obj.cutoff()!=NULL) tcut=obj.cutoff()->name();
		N+=sizeof(obj.normT());//name of normalization scheme
		N+=sizeof(tcut);//name of cutoff function
		N+=sizeof(obj.rc());//cutoff value
		N+=sizeof(obj.phiAN());//name of symmetry functions
		N+=sizeof(int);//number of symmetry functions
		if(obj.phiAN()==PhiAN::G3){
			for(int i=0; i<obj.nfA(); ++i) N+=nbytes(static_cast<const PhiA_G3&>(obj.fA(i)));
		} else if(obj.phiAN()==PhiAN::G4){
			for(int i=0; i<obj.nfA(); ++i) N+=nbytes(static_cast<const PhiA_G4&>(obj.fA(i)));
		} else throw std::runtime_error("nbytes(const BasisA&): Invalid angular symmetry function.");
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisA& obj, char* arr){
		int pos=0,nfA=obj.nfA();
		cutoff::Name::type tcut=cutoff::Name::UNKNOWN;
		if(obj.cutoff()!=NULL) tcut=obj.cutoff()->name();
		std::memcpy(arr+pos,&obj.normT(),sizeof(obj.normT())); pos+=sizeof(obj.normT());
		std::memcpy(arr+pos,&tcut,sizeof(tcut)); pos+=sizeof(tcut);//name of cutoff function
		std::memcpy(arr+pos,&obj.rc(),sizeof(obj.rc())); pos+=sizeof(obj.rc());//cutoff value
		std::memcpy(arr+pos,&obj.phiAN(),sizeof(obj.phiAN())); pos+=sizeof(obj.phiAN());//name of symmetry functions
		std::memcpy(arr+pos,&nfA,sizeof(nfA)); pos+=sizeof(nfA);//number of symmetry functions
		if(obj.phiAN()==PhiAN::G3){
			for(int i=0; i<obj.nfA(); ++i){
				pos+=pack(static_cast<const PhiA_G3&>(obj.fA(i)),arr+pos);
			}
		} else if(obj.phiAN()==PhiAN::G4){
			for(int i=0; i<obj.nfA(); ++i){
				pos+=pack(static_cast<const PhiA_G4&>(obj.fA(i)),arr+pos);
			}
		} else throw std::runtime_error("pack(const BasisA&,char*): Invalid angular symmetry function.");
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisA& obj, const char* arr){
		int pos=0,nfA=0;
		double rc=0;
		NormN::type normT=NormN::UNKNOWN;
		PhiAN::type phiAN=PhiAN::UNKNOWN;
		cutoff::Name::type cutN=cutoff::Name::UNKNOWN;
		std::memcpy(&normT,arr+pos,sizeof(normT)); pos+=sizeof(normT);
		std::memcpy(&cutN,arr+pos,sizeof(cutN)); pos+=sizeof(cutN);//name of cutoff function
		std::memcpy(&rc,arr+pos,sizeof(rc)); pos+=sizeof(rc);//cutoff value
		std::memcpy(&phiAN,arr+pos,sizeof(phiAN)); pos+=sizeof(phiAN);//name of symmetry functions
		std::memcpy(&nfA,arr+pos,sizeof(nfA)); pos+=sizeof(nfA);//number of symmetry functions
		if(phiAN==PhiAN::G3){
			obj.init_G3(nfA,normT,cutN,rc);
			for(int i=0; i<obj.nfA(); ++i){
				pos+=unpack(static_cast<PhiA_G3&>(obj.fA(i)),arr+pos);
			}
		} else if(phiAN==PhiAN::G4){
			obj.init_G4(nfA,normT,cutN,rc);
			for(int i=0; i<obj.nfA(); ++i){
				pos+=unpack(static_cast<PhiA_G4&>(obj.fA(i)),arr+pos);
			}
		} else throw std::runtime_error("unpack(BasisA&,const char*): Invalid angular symmetry function.");
		return pos;
	}
	
}
