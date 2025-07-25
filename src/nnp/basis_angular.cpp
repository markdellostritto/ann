// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// math
#include "math/special.hpp"
// string
#include "str/string.hpp"
#include "str/token.hpp"
// basis - angular
#include "nnp/basis_angular.hpp"

//==== using statements ====

using math::constant::PI;
using math::special::fmexp;

//*****************************************
// BasisA::Name - angular function names
//*****************************************

BasisA::Name BasisA::Name::read(const char* str){
	if(std::strcmp(str,"GAUSS")==0) return BasisA::Name::GAUSS;
	else if(std::strcmp(str,"GAUSS2")==0) return BasisA::Name::GAUSS2;
	else if(std::strcmp(str,"SECH")==0) return BasisA::Name::SECH;
	else if(std::strcmp(str,"STUDENT3")==0) return BasisA::Name::STUDENT3;
	else if(std::strcmp(str,"STUDENT4")==0) return BasisA::Name::STUDENT4;
	else if(std::strcmp(str,"STUDENT5")==0) return BasisA::Name::STUDENT5;
	else return BasisA::Name::NONE;
}

const char* BasisA::Name::name(const BasisA::Name& name){
	switch(name){
		case BasisA::Name::GAUSS: return "GAUSS";
		case BasisA::Name::GAUSS2: return "GAUSS2";
		case BasisA::Name::SECH: return "SECH";
		case BasisA::Name::STUDENT3: return "STUDENT3";
		case BasisA::Name::STUDENT4: return "STUDENT4";
		case BasisA::Name::STUDENT5: return "STUDENT5";
		default: return "NONE";
	}
}

std::ostream& operator<<(std::ostream& out, const BasisA::Name& name){
	switch(name){
		case BasisA::Name::GAUSS: out<<"GAUSS"; break;
		case BasisA::Name::GAUSS2: out<<"GAUSS2"; break;
		case BasisA::Name::SECH: out<<"SECH"; break;
		case BasisA::Name::STUDENT3: out<<"STUDENT3"; break;
		case BasisA::Name::STUDENT4: out<<"STUDENT4"; break;
		case BasisA::Name::STUDENT5: out<<"STUDENT5"; break;
		default: out<<"NONE"; break;
	}
	return out;
}

//==== constructors/destructors ====

/**
* constructor
*/
BasisA::BasisA(double rc, Cutoff::Name cutname, int size, BasisA::Name name):Basis(rc,cutname,size){
	if(name==BasisA::Name::NONE) throw std::invalid_argument("BasisA(rc,Cutoff::Name,int,BasisA::Name): invalid angular function type");
	else name_=name;
	resize(size);
	cutoff_=Cutoff(cutname,rc);
}

/**
* destructor
*/
BasisA::~BasisA(){
	clear();
}

//==== operators ====

/**
* print basis
* @param out - the output stream
* @param basis - the basis to print
* @return the output stream
*/
std::ostream& operator<<(std::ostream& out, const BasisA& basis){
	out<<"BasisA "<<basis.cutoff().name()<<" "<<basis.cutoff().rc()<<" "<<basis.name_<<" "<<basis.size_;
	for(int i=0; i<basis.size(); ++i){
		out<<"\n\t"<<basis.eta_[i]<<" "<<basis.zeta_[i]<<" "<<basis.lambda_[i]<<" "<<basis.rflag_[i];
	}
	return out;
}

//==== reading/writing ====

/**
* write basis to file
* @param writer - file pointer
* @param basis - the basis to be written
*/
void BasisA::write(FILE* writer, const BasisA& basis){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::write(FILE*):\n";
	const char* str_tcut=Cutoff::Name::name(basis.cutoff().name());
	const char* str_phian=BasisA::Name::name(basis.name());
	fprintf(writer,"BasisA %s %f %s %i\n",str_tcut,basis.cutoff().rc(),str_phian,basis.size());
	for(int i=0; i<basis.size(); ++i){
		fprintf(writer,"\t%f %f %i %i\n",basis.eta(i),basis.zeta(i),basis.lambda(i),basis.rflag(i));
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
	//read header
	Token token(fgets(input,string::M,reader),string::WS); token.next();
	const Cutoff::Name cutname=Cutoff::Name::read(token.next().c_str());
	const double rc=std::atof(token.next().c_str());
	const BasisA::Name name=BasisA::Name::read(token.next().c_str());
	const int size=std::atoi(token.next().c_str());
	//initialize
	basis=BasisA(rc,cutname,size,name);
	//read parameters
	for(int i=0; i<basis.size(); ++i){
		token.read(fgets(input,string::M,reader),string::WS);
		basis.eta(i)=std::atof(token.next().c_str());
		basis.zeta(i)=std::atof(token.next().c_str());
		basis.lambda(i)=std::atoi(token.next().c_str());
	}
	basis.init();
	//free local variables
	delete[] input;
}

//==== member functions ====

/**
* clear basis
*/
void BasisA::clear(){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::clear():\n";
	Basis::clear();
	name_=BasisA::Name::NONE;
	eta_.clear();
	ieta2_.clear();
	zeta_.clear();
	lambdaf_.clear();
	fdampr_.clear();
	gdampr_.clear();
	etar_.clear();
	ietar2_.clear();
	lambda_.clear();
	rflag_.clear();
}	

void BasisA::resize(int size){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::resize(int):\n";
	Basis::resize(size);
	if(size_>0){
		eta_.resize(size_);
		ieta2_.resize(size_);
		zeta_.resize(size_);
		lambdaf_.resize(size_);
		fdampr_.resize(size_);
		gdampr_.resize(size_);
		etar_.resize(size_);
		ietar2_.resize(size_);
		lambda_.resize(size_);
		rflag_.resize(size_);
	}
}

void BasisA::init(){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::init():\n";
	for(int n=0; n<size_; ++n){
		ieta2_[n]=1.0/(eta_[n]*eta_[n]);
		lambdaf_[n]=0.5*lambda_[n];
	}
	if(size_>0){
		int flag=0;
		rflag_[0]=flag;
		etar_.clear();
		etar_.push_back(eta_[0]);
		for(int n=1; n<size_; ++n){
			if(std::fabs(eta_[n]-eta_[n-1])<1.0e-6){
				rflag_[n]=flag;
			} else {
				rflag_[n]=++flag;
				etar_.push_back(eta_[n]);
			}
		}
		flag++;
		ietar2_.resize(flag);
		for(int n=0; n<flag; ++n){
			ietar2_[n]=1.0/(etar_[n]*etar_[n]);
		}
		fdampr_.resize(flag);
		gdampr_.resize(flag);
	}
}

/**
* compute symmetry functions
* @param cos - the cosine of the triple
* @param dr - the triple distances: dr={rij,rik,rjk} with i at the vertex
*/
double BasisA::symmf(double cos, const double dr[2], double eta, double zeta, int lambda)const{
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::symm(double,const double*):\n";
	const double c[2]={
		cutoff_.cutf(dr[0]),//cut(rij)
		cutoff_.cutf(dr[1])//cut(rik)
	};
	double rval=0;
	const double hcos=0.5*cos;
	const double cprod=c[0]*c[1];
	const double ieta2=1.0/(eta*eta);
	const double r2s=dr[0]*dr[0]+dr[1]*dr[1];
	switch(name_){
		case BasisA::Name::GAUSS:{
			rval=cprod*pow(fabs(0.5+lambda*hcos),zeta)*fmexp(-ieta2*r2s);
		} break;
		case BasisA::Name::GAUSS2:{
			const double fexp=fmexp(-0.5*PI*ieta2*r2s);
			rval=cprod*pow(fabs(0.5+lambda*hcos),zeta)*2.0*fexp/(1.0+fexp);
		} break;
		case BasisA::Name::SECH:{
			const double fexp=fmexp(-0.5*PI*ieta2*r2s);
			rval=cprod*pow(fabs(0.5+lambda*hcos),zeta)*2.0*fexp/(1.0+fexp*fexp);
		} break;
		case BasisA::Name::STUDENT3:{
			rval=cprod*pow(fabs(0.5+lambda*hcos),zeta)*math::special::powint(1.0/sqrt(1.0+ieta2*r2s),4);
		} break;
		case BasisA::Name::STUDENT4:{
			rval=cprod*pow(fabs(0.5+lambda*hcos),zeta)*math::special::powint(1.0/sqrt(1.0+ieta2*r2s),5);
		} break;
		case BasisA::Name::STUDENT5:{
			rval=cprod*pow(fabs(0.5+lambda*hcos),zeta)*math::special::powint(1.0/sqrt(1.0+ieta2*r2s),6);
		} break;
		default:
			throw std::invalid_argument("BasisA::symm(double): Invalid symmetry function.");
		break;
	}
	return rval;
}

/**
* compute force
* @param phi - stores angular gradients
* @param eta - stores radial gradients
* @param cos - the cosine of the triple
* @param dr - the triple distances: r={rij,rik,rjk} with i at the vertex
* @param dEdG - gradient of energy w.r.t. the inputs
*/
void BasisA::symmd(double& fphi, double* feta, double cos, const double dr[2], double eta, double zeta, int lambda)const{
	//compute cutoffs
	const double c[2]={
		cutoff_.cutf(dr[0]),//cut(rij)
		cutoff_.cutf(dr[1])//cut(rik)
	};
	const double g[2]={
		cutoff_.cutg(dr[0]),//cut'(rij)
		cutoff_.cutg(dr[1])//cut'(rik)
	};
	//compute phi, eta
	fphi=0;
	feta[0]=0;
	feta[1]=0;
	const double lambdaf=0.5*lambda;
	const double ieta2=1.0/(eta*eta);
	const double r2s=dr[0]*dr[0]+dr[1]*dr[1];
	const double dij=2.0*dr[0]*c[0];
	const double dik=2.0*dr[1]*c[1];
	switch(name_){
		case BasisA::Name::GAUSS:{
			//compute angular values
			const double cw=fabs(0.5+lambdaf*cos);
			const double gangle=pow(cw,zeta-1.0)*fmexp(-ieta2*r2s);
			const double fangle=cw*gangle;
			//compute phi
			fphi=zeta*lambdaf*gangle;
			//compute eta
			feta[0]=fangle*(-dij*ieta2+g[0]);
			feta[1]=fangle*(-dik*ieta2+g[1]);
		} break;
		case BasisA::Name::GAUSS2:{
			const double fexp=fmexp(-0.5*PI*ieta2*r2s);
			const double den=1.0/(1.0+fexp);
			//compute angular values
			const double cw=fabs(0.5+lambdaf*cos);
			const double gangle=pow(cw,zeta-1.0)*2.0*fexp*den;
			const double fangle=cw*gangle;
			//compute phi
			fphi=zeta*lambdaf*gangle;
			//compute eta
			feta[0]=fangle*(-dij*ieta2*0.5*PI*den+g[0]);
			feta[1]=fangle*(-dik*ieta2*0.5*PI*den+g[1]);
		} break;
		case BasisA::Name::SECH:{
			//compute distance values
			const double fexp=fmexp(-0.5*PI*ieta2*r2s);
			const double fexp2=fexp*fexp;
			const double den=1.0/(1.0+fexp2);
			const double ftanh=(1.0-fexp2)*den;
			const double fsech=2.0*fexp*den;
			//compute angular values
			const double cw=fabs(0.5+lambdaf*cos);
			const double gangle=pow(cw,zeta-1.0)*fsech;
			const double fangle=cw*gangle;
			//compute phi
			fphi=zeta*lambdaf*gangle;
			//compute eta
			feta[0]=fangle*(-dij*0.5*PI*ieta2*ftanh+g[0]);
			feta[1]=fangle*(-dik*0.5*PI*ieta2*ftanh+g[1]);
		} break;
		case BasisA::Name::STUDENT3:{
			//compute angular values
			const double den=1.0/sqrt(1.0+ieta2*r2s);
			const double cw=fabs(0.5+lambdaf*cos);
			const double gangle=pow(cw,zeta-1.0)*math::special::powint(den,4);
			const double fangle=cw*gangle;
			//compute phi
			fphi=zeta*lambdaf*gangle;
			//compute eta
			feta[0]=fangle*(-dij*ieta2*4.0/2.0*den*den+g[0]);
			feta[1]=fangle*(-dik*ieta2*4.0/2.0*den*den+g[1]);
		} break;
		case BasisA::Name::STUDENT4:{
			//compute angular values
			const double den=1.0/sqrt(1.0+ieta2*r2s);
			const double cw=fabs(0.5+lambdaf*cos);
			const double gangle=pow(cw,zeta-1.0)*math::special::powint(den,5);
			const double fangle=cw*gangle;
			//compute phi
			fphi=zeta*lambdaf*gangle;
			//compute eta
			feta[0]=fangle*(-dij*ieta2*5.0/2.0*den*den+g[0]);
			feta[1]=fangle*(-dik*ieta2*5.0/2.0*den*den+g[1]);
		} break;
		case BasisA::Name::STUDENT5:{
			//compute angular values
			const double den=1.0/sqrt(1.0+ieta2*r2s);
			const double cw=fabs(0.5+lambdaf*cos);
			const double gangle=pow(cw,zeta-1.0)*math::special::powint(den,6);
			const double fangle=cw*gangle;
			//compute phi
			fphi=zeta*lambdaf*gangle;
			//compute eta
			feta[0]=fangle*(-dij*ieta2*6.0/2.0*den*den+g[0]);
			feta[1]=fangle*(-dik*ieta2*6.0/2.0*den*den+g[1]);
		} break;
		default:
			throw std::invalid_argument("BasisA::symm(double): Invalid symmetry function.");
		break;
	}
	//normalize
	fphi*=c[0]*c[1];
	feta[0]*=c[1];
	feta[1]*=c[0];
}

/**
* compute symmetry functions
* @param cos - the cosine of the triple
* @param dr - the triple distances: dr={rij,rik,rjk} with i at the vertex
*/
void BasisA::symm(const std::vector<double>& dr, const std::vector<double>& cos){
	if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"BasisA::symm(double,const double*):\n";
	const double cprod=cutoff_.cutf(dr[0])*cutoff_.cutf(dr[1]);//cut(rij)*cut(rik)
	const double r2s=dr[0]*dr[0]+dr[1]*dr[1];
	switch(name_){
		case BasisA::Name::GAUSS:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=cprod*fmexp(-ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//symm_[i]=cprod*pow(fabs(0.5+lambdaf_[i]*cos[0]),zeta_[i])*fmexp(-ieta2_[i]*r2s);
				symm_[i]=pow(fabs(0.5+lambdaf_[i]*cos[0]),zeta_[i])*fdampr_[rflag_[i]];
			}
		} break;
		case BasisA::Name::GAUSS2:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fmexp(-0.5*PI*ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//symm_[i]=cprod*pow(fabs(0.5+lambdaf_[i]*cos[0]),zeta_[i])*fmexp(-ieta2_[i]*r2s);
				symm_[i]=cprod*pow(fabs(0.5+lambdaf_[i]*cos[0]),zeta_[i])*2.0*fdampr_[rflag_[i]]/(1.0+fdampr_[rflag_[i]]);
			}
		} break;
		case BasisA::Name::SECH:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fmexp(-0.5*PI*ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//const double fexp=fmexp(-ieta2_[i]*r2s);
				const double fexp=fdampr_[rflag_[i]];
				symm_[i]=cprod*pow(fabs(0.5+lambdaf_[i]*cos[0]),zeta_[i])*2.0*fexp/(1.0+fexp*fexp);
			}
		} break;
		case BasisA::Name::STUDENT3:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=cprod*math::special::powint(1.0/sqrt(1.0+ietar2_[j]*r2s),4);
			for(int i=0; i<size_; ++i){
				symm_[i]=pow(fabs(0.5+lambdaf_[i]*cos[0]),zeta_[i])*fdampr_[rflag_[i]];
			}
		} break;
		case BasisA::Name::STUDENT4:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=cprod*math::special::powint(1.0/sqrt(1.0+ietar2_[j]*r2s),5);
			for(int i=0; i<size_; ++i){
				symm_[i]=pow(fabs(0.5+lambdaf_[i]*cos[0]),zeta_[i])*fdampr_[rflag_[i]];
			}
		} break;
		case BasisA::Name::STUDENT5:{
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=cprod*math::special::powint(1.0/sqrt(1.0+ietar2_[j]*r2s),6);
			for(int i=0; i<size_; ++i){
				symm_[i]=pow(fabs(0.5+lambdaf_[i]*cos[0]),zeta_[i])*fdampr_[rflag_[i]];
			}
		} break;
		default:
			throw std::invalid_argument("BasisA::symm(double): Invalid symmetry function.");
		break;
	}
}

/**
* compute force
* @param phi - stores angular gradients
* @param eta - stores radial gradients
* @param cos - the cosine of the triple
* @param dr - the triple distances: r={rij,rik,rjk} with i at the vertex
* @param dEdG - gradient of energy w.r.t. the inputs
*/
void BasisA::force(const std::vector<double>& dr, const std::vector<double>& cos, double& phi, double* eta, const double* dEdG){
	//compute cutoffs
	const double c[2]={
		cutoff_.cutf(dr[0]),//cut(rij)
		cutoff_.cutf(dr[1])//cut(rik)
	};
	const double g[2]={
		cutoff_.cutg(dr[0]),//cut'(rij)
		cutoff_.cutg(dr[1])//cut'(rik)
	};
	//compute phi, eta
	phi=0;
	eta[0]=0;
	eta[1]=0;
	const double r2s=dr[0]*dr[0]+dr[1]*dr[1];
	switch(name_){
		case BasisA::Name::GAUSS:{
			const double dij=2.0*dr[0]*c[0];
			const double dik=2.0*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fmexp(-ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute angular values
				const double cw=fabs(0.5+lambdaf_[i]*cos[0]);
				//const double gangle=dEdG[i]*pow(cw,zeta_[i]-1.0)*fmexp(-ieta2_[i]*r2s);
				const double gangle=dEdG[i]*pow(cw,zeta_[i]-1.0)*fdampr_[rflag_[i]];
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]+g[1]);
			}
		} break;
		case BasisA::Name::GAUSS2:{
			const double dij=PI*dr[0]*c[0];
			const double dik=PI*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fmexp(-0.5*PI*ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				const double fexp=fdampr_[rflag_[i]];
				const double den=1.0/(1.0+fexp);
				//compute angular values
				const double cw=fabs(0.5+lambdaf_[i]*cos[0]);
				const double gangle=dEdG[i]*pow(cw,zeta_[i]-1.0)*2.0*fexp*den;
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*den+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*den+g[1]);
			}
		} break;
		case BasisA::Name::SECH:{
			const double dij=PI*dr[0]*c[0];
			const double dik=PI*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=fmexp(-0.5*PI*ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute distance values
				//const double fexp=fmexp(-ieta2_[i]*r2s);
				const double fexp=fdampr_[rflag_[i]];
				const double fexp2=fexp*fexp;
				const double den=1.0/(1.0+fexp2);
				const double ftanh=(1.0-fexp2)*den;
				const double fsech=2.0*fexp*den;
				//compute angular values
				const double cw=fabs(0.5+lambdaf_[i]*cos[0]);
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*fsech;
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*ftanh+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*ftanh+g[1]);
			}
		} break;
		case BasisA::Name::STUDENT3:{
			const double dij=4.0*dr[0]*c[0];
			const double dik=4.0*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=1.0/sqrt(1.0+ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute angular values
				const double den=fdampr_[rflag_[i]];
				const double cw=fabs(0.5+lambdaf_[i]*cos[0]);
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*math::special::powint(den,4);
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*den*den+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*den*den+g[1]);
			}
		} break;
		case BasisA::Name::STUDENT4:{
			const double dij=5.0*dr[0]*c[0];
			const double dik=5.0*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=1.0/sqrt(1.0+ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute angular values
				const double den=fdampr_[rflag_[i]];
				const double cw=fabs(0.5+lambdaf_[i]*cos[0]);
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*math::special::powint(den,5);
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*den*den+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*den*den+g[1]);
			}
		} break;
		case BasisA::Name::STUDENT5:{
			const double dij=6.0*dr[0]*c[0];
			const double dik=6.0*dr[1]*c[1];
			for(int j=0; j<etar_.size(); ++j) fdampr_[j]=1.0/sqrt(1.0+ietar2_[j]*r2s);
			for(int i=0; i<size_; ++i){
				//compute angular values
				const double den=fdampr_[rflag_[i]];
				const double cw=fabs(0.5+lambdaf_[i]*cos[0]);
				const double gangle=pow(cw,zeta_[i]-1.0)*dEdG[i]*math::special::powint(den,6);
				const double fangle=cw*gangle;
				//compute phi
				phi-=zeta_[i]*lambdaf_[i]*gangle;
				//compute eta
				eta[0]-=fangle*(-dij*ieta2_[i]*den*den+g[0]);
				eta[1]-=fangle*(-dik*ieta2_[i]*den*den+g[1]);
			}
		} break;
		default:
			throw std::invalid_argument("BasisA::force(double&,double*,double,const double[3],const double*)const: Invalid symmetry function.");
		break;
	}
	//normalize
	phi*=c[0]*c[1];
	eta[0]*=c[1];
	eta[1]*=c[0];
}

//==== serialization ====

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisA& obj){
		if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"nbytes(const BasisA&):\n";
		int size=0;
		size+=sizeof(obj.size());//number of symmetry functions
		size+=sizeof(obj.name());//name of symmetry functions
		size+=nbytes(obj.cutoff());
		const int s=obj.size();
		size+=sizeof(double)*s;//eta
		size+=sizeof(double)*s;//zeta
		size+=sizeof(int)*s;//lambda
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisA& obj, char* arr){
		if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"pack(const BasisA&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.size(),sizeof(obj.size())); pos+=sizeof(obj.size());
		std::memcpy(arr+pos,&obj.name(),sizeof(obj.name())); pos+=sizeof(obj.name());
		pos+=pack(obj.cutoff(),arr+pos);
		const int size=obj.size();
		if(size>0){
			std::memcpy(arr+pos,obj.eta().data(),size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(arr+pos,obj.zeta().data(),size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(arr+pos,obj.lambda().data(),size*sizeof(int)); pos+=size*sizeof(int);
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisA& obj, const char* arr){
		if(BASIS_ANGULAR_PRINT_FUNC>0) std::cout<<"unpack(BasisA&,const char*):\n";
		int pos=0;
		int size=0;
		BasisA::Name name=BasisA::Name::NONE;
		std::memcpy(&size,arr+pos,sizeof(size)); pos+=sizeof(size);
		std::memcpy(&name,arr+pos,sizeof(BasisA::Name)); pos+=sizeof(BasisA::Name);
		pos+=unpack(obj.cutoff(),arr+pos);
		obj=BasisA(obj.cutoff().rc(),obj.cutoff().name(),size,name);
		if(size>0){
			std::memcpy(obj.eta().data(),arr+pos,size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(obj.zeta().data(),arr+pos,size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(obj.lambda().data(),arr+pos,size*sizeof(int)); pos+=size*sizeof(int);
		}
		obj.init();
		return pos;
	}
	
}
