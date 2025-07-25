// c libraries
#include <cstring>
#include <cstdio>
// c++ libraries
#include <iostream>
#include <vector>
// math
#include "math/special.hpp"
#include "math/const.hpp"
// str
#include "str/string.hpp"
#include "str/token.hpp"
// basis - radial
#include "nnp/basis_radial.hpp"

//==== using statements ====

using math::special::fmexp;
using math::constant::Rad2;

//*****************************************
// BasisR::Name - radial function names
//*****************************************

BasisR::Name BasisR::Name::read(const char* str){
	if(std::strcmp(str,"GAUSSIAN")==0) return BasisR::Name::GAUSSIAN;
	else if(std::strcmp(str,"SECH")==0) return BasisR::Name::SECH;
	else if(std::strcmp(str,"LOGISTIC")==0) return BasisR::Name::LOGISTIC;
	else if(std::strcmp(str,"TANH")==0) return BasisR::Name::TANH;
	else if(std::strcmp(str,"LOGCOSH")==0) return BasisR::Name::LOGCOSH;
	else if(std::strcmp(str,"LOGCOSH2")==0) return BasisR::Name::LOGCOSH2;
	else return BasisR::Name::NONE;
}

const char* BasisR::Name::name(const BasisR::Name& name){
	switch(name){
		case BasisR::Name::GAUSSIAN: return "GAUSSIAN";
		case BasisR::Name::SECH: return "SECH";
		case BasisR::Name::LOGISTIC: return "LOGISTIC";
		case BasisR::Name::TANH: return "TANH";
		case BasisR::Name::LOGCOSH: return "LOGCOSH";
		case BasisR::Name::LOGCOSH2: return "LOGCOSH2";
		default: return "NONE";
	}
}

std::ostream& operator<<(std::ostream& out, const BasisR::Name& name){
	switch(name){
		case BasisR::Name::GAUSSIAN: out<<"GAUSSIAN"; break;
		case BasisR::Name::SECH: out<<"SECH"; break;
		case BasisR::Name::LOGISTIC: out<<"LOGISTIC"; break;
		case BasisR::Name::TANH: out<<"TANH"; break;
		case BasisR::Name::LOGCOSH: out<<"LOGCOSH"; break;
		case BasisR::Name::LOGCOSH2: out<<"LOGCOSH2"; break;
		default: out<<"NONE"; break;
	}
	return out;
}

//*****************************************
// BasisR - radial basis
//*****************************************

//==== constructors/destructors ====

/**
* constructor
*/
BasisR::BasisR(double rc, Cutoff::Name cutname, int size, BasisR::Name name):Basis(rc,cutname,size){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR(rc,Cutoff::Name,int,BasisR::Name):\n";
	if(name==BasisR::Name::NONE) throw std::invalid_argument("BasisR(rc,Cutoff::Name,int,BasisR::Name): invalid radial function type");
	else name_=name;
	resize(size);
	cutoff_=Cutoff(cutname,rc);
}

/**
* destructor
*/
BasisR::~BasisR(){
	clear();
}


//==== operators ====

/**
* print basis
* @param out - the output stream
* @param basis - the basis to print
* @return the output stream
*/
std::ostream& operator<<(std::ostream& out, const BasisR& basis){
	out<<"BasisR "<<basis.cutoff().name()<<" "<<basis.cutoff().rc()<<" "<<basis.name_<<" "<<basis.size_;
	for(int i=0; i<basis.size(); ++i){
		out<<"\n\t"<<basis.rs_[i]<<" "<<basis.eta_[i]<<" ";
	}
	return out;
}

//==== reading/writing ====

/**
* write basis to file
* @param writer - file pointer
* @param basis - the basis to be written
*/
void BasisR::write(FILE* writer,const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::write(FILE*,const BasisR&):\n";
	const char* str_tcut=Cutoff::Name::name(basis.cutoff().name());
	const char* str_phirn=BasisR::Name::name(basis.name());
	fprintf(writer,"BasisR %s %f %s %i\n",str_tcut,basis.cutoff().rc(),str_phirn,basis.size());
	for(int i=0; i<basis.size(); ++i){
		fprintf(writer,"\t%f %f\n",basis.rs(i),basis.eta(i));
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
	//read header
	Token token(fgets(input,string::M,reader),string::WS); token.next();
	const Cutoff::Name cutname=Cutoff::Name::read(token.next().c_str());
	const double rc=std::atof(token.next().c_str());
	const BasisR::Name name=BasisR::Name::read(token.next().c_str());
	const int size=std::atoi(token.next().c_str());
	//initialize
	basis=BasisR(rc,cutname,size,name);
	//read parameters
	for(int i=0; i<basis.size(); ++i){
		token.read(fgets(input,string::M,reader),string::WS);
		basis.rs(i)=std::atof(token.next().c_str());
		basis.eta(i)=std::atof(token.next().c_str());
	}
	//free local variables
	delete[] input;
}

//==== member functions ====

/**
* clear basis
*/
void BasisR::clear(){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::clear():\n";
	Basis::clear();
	name_=BasisR::Name::NONE;
	rs_.clear();
	eta_.clear();
}

/**
* resize symmetry function and parameter arrays
* @param size - the total number of symmetry functions/parameters
*/
void BasisR::resize(int size){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::resize(int):\n";
	Basis::resize(size);
	if(size_>0){
		rs_.resize(size_);
		eta_.resize(size_);
	}
}

/**
* compute symmetry function - function value
* @param dr - the distance between the central atom and a neighboring atom
*/
double BasisR::symmf(double dr, double eta, double rs)const{
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::symmf(double,double,double)const:\n";
	const double cutf=cutoff_.cutf(dr);
	double rval=0;
	switch(name_){
		case BasisR::Name::GAUSSIAN:{
			const double arg=eta*(dr-rs);
			rval=fmexp(-arg*arg)*cutf;
		} break;
		case BasisR::Name::SECH:{
			rval=math::special::sech(eta*(dr-rs))*cutf;
		} break;
		case BasisR::Name::LOGISTIC:{
			const double sechf=math::special::sech(eta*(dr-rs));
			rval=sechf*sechf*cutf;
		} break;
		case BasisR::Name::TANH:{
			rval=0.5*(tanh(-eta*(dr-rs))+1.0)*cutf;
		} break;
		case BasisR::Name::LOGCOSH:{
			rval=0.5*log1p(fmexp(-2.0*eta*(dr-rs)))*cutf;
		} break;
		case BasisR::Name::LOGCOSH2:{
			const double arg=-eta*(dr-rs);
			const double fexp=fmexp(2.0*Rad2*arg);
			rval=(arg==0.0)?0.5*cutf/Rad2:arg*fexp/(fexp-1.0)*cutf;
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double): Invalid symmetry function.");
		break;
	}
	return rval;
}

/**
* compute symmetry function - derivative value
* @param dr - the distance between the central atom and a neighboring atom
*/
double BasisR::symmd(double dr, double eta, double rs)const{
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::symmd(double,double,double)const:\n";
	double rval=0;
	const double cutf=cutoff_.cutf(dr);
	const double cutg=cutoff_.cutg(dr);
	switch(name_){
		case BasisR::Name::GAUSSIAN:{
			const double arg=eta*(dr-rs);
			rval=fmexp(-arg*arg)*(-2.0*eta*arg*cutf+cutg);
		} break;
		case BasisR::Name::SECH:{
			/*const double sechf=math::special::sech(eta*(dr-rs));
			const double tanhf=math::special::tanh(eta*(dr-rs));
			rval=sechf*(-1.0*eta*tanhf*cutf+cutg);*/
			const double expf=fmexp(-eta*(dr-rs));
			const double expf2=expf*expf;
			const double den=1.0/(1.0+expf2);
			rval=2.0*expf*den*(-1.0*eta*(1.0-expf2)*den*cutf+cutg);
		} break;
		case BasisR::Name::LOGISTIC:{
			const double sechf=math::special::sech(eta*(dr-rs));
			const double tanhf=math::special::tanh(eta*(dr-rs));
			rval=sechf*sechf*(-2.0*eta*tanhf*cutf+cutg);
		} break;
		case BasisR::Name::TANH:{
			const double coshf=math::special::cosh(-eta*(dr-rs));
			const double tanhf=math::special::tanh(-eta*(dr-rs));
			rval=0.5*(-eta*(1.0-tanhf*tanhf)*cutf+(1.0+tanhf)*cutg);
		} break;
		case BasisR::Name::LOGCOSH:{
			const double fexp=fmexp(-2.0*eta*(dr-rs));
			rval=-eta*fexp/(1.0+fexp)*cutf+0.5*log1p(fexp)*cutg;
		} break;
		case BasisR::Name::LOGCOSH2:{
			const double arg=-eta*(dr-rs);
			const double fexp=fmexp(2.0*Rad2*arg);
			const double den=1.0/(fexp-1.0);
			rval=(arg==0.0)?0.5*(-eta*cutg+cutf/Rad2):fexp*(eta*(1.0+2.0*Rad2*arg-fexp)*den*cutf+arg*cutg)*den;
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double): Invalid symmetry function.");
		break;
	}
	return rval;
}

/**
* compute symmetry functions
* @param dr - the distance between the central atom and a neighboring atom
*/
void BasisR::symm(const std::vector<double>& dr){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::symm(double):\n";
	const double cutf=cutoff_.cutf(dr[0]);
	switch(name_){
		case BasisR::Name::GAUSSIAN:{
			for(int i=0; i<size_; ++i){
				const double arg=eta_[i]*(dr[0]-rs_[i]);
				symm_[i]=fmexp(-arg*arg)*cutf;
			}
		} break;
		case BasisR::Name::SECH:{
			for(int i=0; i<size_; ++i){
				//symm_[i]=math::special::sech(eta_[i]*(dr[0]-rs_[i]))*cutf;
				const double expf=fmexp(-eta_[i]*(dr[0]-rs_[i]));
				symm_[i]=2.0*expf/(1.0+expf*expf)*cutf;
			}
		} break;
		case BasisR::Name::LOGISTIC:{
			for(int i=0; i<size_; ++i){
				//const double sechf=math::special::sech(eta_[i]*(dr[0]-rs_[i]));
				//symm_[i]=sechf*sechf*cutf;
				const double expf=fmexp(-2.0*eta_[i]*(dr[0]-rs_[i]));
				symm_[i]=4.0*expf/((1.0+expf)*(1.0+expf))*cutf;
			}
		} break;
		case BasisR::Name::TANH:{
			for(int i=0; i<size_; ++i){
				symm_[i]=0.5*(tanh(-eta_[i]*(dr[0]-rs_[i]))+1.0)*cutf;
			}
		} break;
		case BasisR::Name::LOGCOSH:{
			for(int i=0; i<size_; ++i){
				symm_[i]=0.5*log1p(fmexp(-2.0*eta_[i]*(dr[0]-rs_[i])))*cutf;
			}
		} break;
		case BasisR::Name::LOGCOSH2:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr[0]-rs_[i]);
				const double fexp=fmexp(2.0*Rad2*arg);
				symm_[i]=(arg==0.0)?0.5*cutf/Rad2:arg*fexp/(fexp-1.0)*cutf;
			}
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double): Invalid symmetry function.");
		break;
	}
}

/**
* compute force
* @param dr - the distance between the central atom and a neighboring atom
* @param dEdG - gradient of energy w.r.t. the inputs
*/
double BasisR::force(const std::vector<double>& dr, const double* dEdG)const{
	double amp=0;
	const double cutf=cutoff_.cutf(dr[0]);
	const double cutg=cutoff_.cutg(dr[0]);
	switch(name_){
		case BasisR::Name::GAUSSIAN:{
			for(int i=0; i<size_; ++i){
				const double arg=eta_[i]*(dr[0]-rs_[i]);
				amp-=dEdG[i]*fmexp(-arg*arg)*(-2.0*eta_[i]*arg*cutf+cutg);
			}
		} break;
		case BasisR::Name::SECH:{
			for(int i=0; i<size_; ++i){
				/*const double sechf=math::special::sech(eta_[i]*(dr[0]-rs_[i]));
				const double tanhf=math::special::tanh(eta_[i]*(dr[0]-rs_[i]));
				amp-=dEdG[i]*sechf*(-1.0*eta_[i]*tanhf*cutf+cutg);*/
				const double expf=fmexp(-eta_[i]*(dr[0]-rs_[i]));
				const double expf2=expf*expf;
				const double den=1.0/(1.0+expf2);
				amp-=dEdG[i]*2.0*expf*den*(-eta_[i]*(1.0-expf2)*den*cutf+cutg);
			}
		} break;
		case BasisR::Name::LOGISTIC:{
			for(int i=0; i<size_; ++i){
				/*const double sechf=math::special::sech(eta_[i]*(dr[0]-rs_[i]));
				const double tanhf=math::special::tanh(eta_[i]*(dr[0]-rs_[i]));
				amp-=dEdG[i]*sechf*sechf*(-2.0*eta_[i]*tanhf*cutf+cutg);*/
				const double fexp2=fmexp(-2.0*eta_[i]*(dr[0]-rs_[i]));
				const double den=1.0/(1.0+fexp2);
				const double tanhf=(1.0-fexp2)*den;
				const double sech2f=4.0*fexp2*den*den;
				amp-=dEdG[i]*sech2f*(-2.0*eta_[i]*tanhf*cutf+cutg);
			}
		} break;
		case BasisR::Name::TANH:{
			for(int i=0; i<size_; ++i){
				const double tanhf=tanh(-eta_[i]*(dr[0]-rs_[i]));
				amp-=dEdG[i]*0.5*(-eta_[i]*(1.0-tanhf*tanhf)*cutf+(1.0+tanhf)*cutg);
			}
		} break;
		case BasisR::Name::LOGCOSH:{
			for(int i=0; i<size_; ++i){
				const double fexp=fmexp(-2.0*eta_[i]*(dr[0]-rs_[i]));
				amp-=dEdG[i]*(-eta_[i]*fexp/(1.0+fexp)*cutf+0.5*log1p(fexp)*cutg);
			}
		} break;
		case BasisR::Name::LOGCOSH2:{
			for(int i=0; i<size_; ++i){
				const double arg=-eta_[i]*(dr[0]-rs_[i]);
				const double fexp=fmexp(2.0*Rad2*arg);
				const double den=1.0/(fexp-1.0);
				const double val=(arg==0.0)?0.5*(-eta_[i]*cutg+cutf/Rad2):fexp*(eta_[i]*(1.0+2.0*Rad2*arg-fexp)*den*cutf+arg*cutg)*den;
				amp-=dEdG[i]*val;
			}
		} break;
		default:
			throw std::invalid_argument("BasisR::symm(double): Invalid symmetry function.");
		break;
	}
	return amp;
}

//==== serialization ====

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const BasisR& obj){
		if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"nbytes(const BasisR&):\n";
		int size=0;
		size+=sizeof(obj.size());//number of symmetry functions
		size+=sizeof(obj.name());//name of symmetry functions
		size+=nbytes(obj.cutoff());
		const int s=obj.size();
		size+=sizeof(double)*s;//rs
		size+=sizeof(double)*s;//eta
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const BasisR& obj, char* arr){
		if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"pack(const BasisR&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.size(),sizeof(obj.size())); pos+=sizeof(obj.size());//number of symmetry functions
		std::memcpy(arr+pos,&obj.name(),sizeof(obj.name())); pos+=sizeof(obj.name());//name of symmetry functions
		pos+=pack(obj.cutoff(),arr+pos);
		const int size=obj.size();
		if(size>0){
			std::memcpy(arr+pos,obj.rs().data(),size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(arr+pos,obj.eta().data(),size*sizeof(double)); pos+=size*sizeof(double);
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(BasisR& obj, const char* arr){
		if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"unpack(BasisR&,const char*):\n";
		int pos=0; int size=0;
		BasisR::Name name=BasisR::Name::NONE;
		std::memcpy(&size,arr+pos,sizeof(size)); pos+=sizeof(size);
		std::memcpy(&name,arr+pos,sizeof(BasisR::Name)); pos+=sizeof(BasisR::Name);
		pos+=unpack(obj.cutoff(),arr+pos);
		obj=BasisR(obj.cutoff().rc(),obj.cutoff().name(),size,name);
		if(size>0){
			std::memcpy(obj.rs().data(),arr+pos,size*sizeof(double)); pos+=size*sizeof(double);
			std::memcpy(obj.eta().data(),arr+pos,size*sizeof(double)); pos+=size*sizeof(double);
		}
		return pos;
	}
	
}
