// c++
#include <iostream>
#include <stdexcept>
// string
#include "str/string.hpp"
#include "str/token.hpp"
#include "str/print.hpp"
// basis - radial
#include "nnp/basis_radial.hpp"
// basis - angular
#include "nnp/basis_angular.hpp"

class Mix{
public:
	enum Type{
		MAX,
		AMEAN,
		GMEAN,
		HMEAN,
		QMEAN,
		UNKNOWN
	};
	//constructor
	Mix():t_(Type::UNKNOWN){}
	Mix(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Mix read(const char* str);
	static const char* name(const Mix& mix);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};

std::ostream& operator<<(std::ostream& out, const Mix& mix){
	switch(mix){
		case Mix::MAX: out<<"MAX"; break;
		case Mix::AMEAN: out<<"AMEAN"; break;
		case Mix::GMEAN: out<<"GMEAN"; break;
		case Mix::HMEAN: out<<"HMEAN"; break;
		case Mix::QMEAN: out<<"QMEAN"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Mix::name(const Mix& mix){
	switch(mix){
		case Mix::MAX: return "MAX";
		case Mix::AMEAN: return "AMEAN";
		case Mix::GMEAN: return "GMEAN";
		case Mix::HMEAN: return "HMEAN";
		case Mix::QMEAN: return "QMEAN";
		default: return "UNKNOWN";
	}
}

Mix Mix::read(const char* str){
	if(std::strcmp(str,"MAX")==0) return Mix::MAX;
	else if(std::strcmp(str,"AMEAN")==0) return Mix::AMEAN;
	else if(std::strcmp(str,"GMEAN")==0) return Mix::GMEAN;
	else if(std::strcmp(str,"HMEAN")==0) return Mix::HMEAN;
	else if(std::strcmp(str,"QMEAN")==0) return Mix::QMEAN;
	else return Mix::UNKNOWN;
}

int main(int argc, char* argv[]){
	//file i/o
		FILE* reader=NULL;
		char* pfile=new char[string::M];
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		Token token;
	//cutoff
		double rcut=0;
		Cutoff::Name cutname;
	//types
		int ntypes=0;
		int nra=0;
		std::vector<double> radr;
		std::vector<std::vector<double> > rada;
		std::vector<double> rcutl;
		std::vector<std::string> types;
	//basis - radial
		PhiRN phiRN;
		int nR=0;
		std::vector<double> eta;
	//basis - angular
		PhiAN phiAN;
		int nA=0;
		std::vector<double> zeta;
		std::vector<int> lambda;
		Mix mix_radial;
		Mix mix_angular;
	//misc
		bool error=false;
	
	try{
		//======== print title ========
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::title("MAKE BASIS",strbuf,' ')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		
		//======== check the arguments ========
		if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
		
		//======== load the parameter file ========
		std::cout<<"reading parameter file\n";
		std::strcpy(pfile,argv[1]);
		
		//======== open the parameter file ========
		std::cout<<"opening parameter file\n";
		reader=fopen(pfile,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+pfile);
		
		//======== read in the parameters ========
		std::cout<<"reading parameters\n";
		//cutoff
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS); token.next();
		cutname=Cutoff::Name::read(token.next().c_str());
		rcut=std::atof(token.next().c_str());
		std::cout<<"cutoff = "<<cutname<<" "<<rcut<<"\n";
		//mixing - radial
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS); token.next();
		mix_radial=Mix::read(string::to_upper(token.next()).c_str());
		std::cout<<"mix - radial = "<<mix_radial<<"\n";
		//mixing - angular
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS); token.next();
		mix_angular=Mix::read(string::to_upper(token.next()).c_str());
		std::cout<<"mix - angular = "<<mix_angular<<"\n";
		//ntypes
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS); token.next();
		ntypes=std::atoi(token.next().c_str());
		std::cout<<"ntypes = "<<ntypes<<"\n";
		if(ntypes<=0) throw std::invalid_argument("Invalid number of types.\n");
		//nra
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS); token.next();
		nra=std::atoi(token.next().c_str());
		std::cout<<"nra = "<<ntypes<<"\n";
		if(nra<=0) throw std::invalid_argument("Invalid number of angular radii.\n");
		//types
		for(int i=0; i<ntypes; ++i){
			token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
			types.push_back(token.next());
			radr.push_back(std::atof(token.next().c_str()));
			rada.push_back(std::vector<double>(nra,0.0));
			for(int j=0; j<nra; ++j){
				rada.back()[j]=std::atof(token.next().c_str());
			}
			if(!token.end()) rcutl.push_back(std::atof(token.next().c_str()));
			else rcutl.push_back(rcut);
			std::cout<<"type["<<i<<"] = "<<types.back()<<" "<<radr.back()<<" ";
			for(int j=0; j<nra; ++j) std::cout<<rada.back()[j]<<" ";
			std::cout<<" "<<rcutl.back()<<"\n";
		}
		//radial basis
		std::cout<<"reading radial basis\n";
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS); token.next();
		phiRN=PhiRN::read(token.next().c_str());
		nR=std::atoi(token.next().c_str());
		for(int i=0; i<nR; ++i){
			token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
			eta.push_back(std::atof(token.next().c_str()));
		}
		//angular basis
		std::cout<<"reading angular basis\n";
		token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS); token.next();
		phiAN=PhiAN::read(token.next().c_str());
		nA=std::atoi(token.next().c_str());
		for(int i=0; i<nA; ++i){
			token.read(string::trim_right(fgets(input,string::M,reader),string::COMMENT),string::WS);
			zeta.push_back(std::atof(token.next().c_str()));
			lambda.push_back(std::atoi(token.next().c_str()));
		}
		
		//======== close file ========
		std::cout<<"closing parameter file\n";
		fclose(reader);
		reader=NULL;
		
		if(mix_radial==Mix::UNKNOWN) throw std::invalid_argument("Invalid radial mixing scheme.\n");
		if(mix_angular==Mix::UNKNOWN) throw std::invalid_argument("Invalid angular mixing scheme.\n");
		
		//======== write basis files ========
		for(int i=0; i<ntypes; ++i){
			std::string fname="basis_"+types[i];
			FILE* writer=fopen(fname.c_str(),"w");
			if(writer==NULL) throw std::invalid_argument("Could not open basis file.");
			fprintf(writer,"cut %6.4f\n",rcut);
			fprintf(writer,"nspecies %i\n",ntypes);
			fprintf(writer,"basis %s\n",types[i].c_str());
			//radial basis
			for(int j=0; j<ntypes; ++j){
				double rc=0.0;
				switch(mix_radial){
					case Mix::MAX:{
						rc=(rcutl[i]>rcutl[j])?rcutl[i]:rcutl[j];
					} break;
					case Mix::AMEAN:{
						rc=0.5*(rcutl[i]+rcutl[j]);
					} break;
					case Mix::GMEAN:{
						rc=sqrt(rcutl[i]*rcutl[j]);
					} break;
					case Mix::HMEAN:{
						rc=2.0/(1.0/rcutl[i]+1.0/rcutl[j]);
					} break;
					case Mix::QMEAN:{
						rc=sqrt(0.5*(rcutl[i]*rcutl[i]+rcutl[j]*rcutl[j]));
					} break;
					default: rc=0;
				}
				fprintf(writer,"basis_radial %s\n",types[j].c_str());
				fprintf(writer,"BasisR %s %6.4f %s %i\n",
					Cutoff::Name::name(cutname),
					rc,PhiRN::name(phiRN),nR
				);
				double rs=0;
				switch(mix_radial){
					case Mix::MAX:{
						rs=(radr[i]>radr[j])?radr[i]:radr[j];
					} break;
					case Mix::AMEAN:{
						rs=0.5*(radr[i]+radr[j]);
					} break;
					case Mix::GMEAN:{
						rs=sqrt(radr[i]*radr[j]);
					} break;
					case Mix::HMEAN:{
						rs=2.0/(1.0/radr[i]+1.0/radr[j]);
					} break;
					case Mix::QMEAN:{
						rs=sqrt(0.5*(radr[i]*radr[i]+radr[j]*radr[j]));
					} break;
					default: rs=0;
				}
				for(int n=0; n<nR; ++n){
					fprintf(writer,"\t%5.3f %5.3f\n",rs,eta[n]);
				}
			}
			//angular basis
			for(int j=0; j<ntypes; ++j){
				for(int k=j; k<ntypes; ++k){
					double rc1=0.0;
					double rc2=0.0;
					double rc=0;
					switch(mix_angular){
						case Mix::MAX:{
							rc1=(rcutl[i]>rcutl[j])?rcutl[i]:rcutl[j];
							rc2=(rcutl[i]>rcutl[k])?rcutl[k]:rcutl[k];
							rc=(rc1>rc2)?rc1:rc2;
						} break;
						case Mix::AMEAN:{
							rc1=0.5*(rcutl[i]+rcutl[j]);
							rc2=0.5*(rcutl[i]+rcutl[k]);
							rc=0.5*(rc1+rc2);
						} break;
						case Mix::GMEAN:{
							rc1=sqrt(rcutl[i]*rcutl[j]);
							rc2=sqrt(rcutl[i]*rcutl[k]);
							rc=sqrt(rc1*rc2);
						} break;
						case Mix::HMEAN:{
							rc1=2.0/(1.0/rcutl[i]+1.0/rcutl[j]);
							rc2=2.0/(1.0/rcutl[i]+1.0/rcutl[k]);
							rc=2.0/(1.0/rc1+1.0/rc2);
						} break;
						case Mix::QMEAN:{
							rc1=sqrt(0.5*(rcutl[i]*rcutl[i]+rcutl[j]*rcutl[j]));
							rc2=sqrt(0.5*(rcutl[i]*rcutl[i]+rcutl[k]*rcutl[k]));
							rc=sqrt(0.5*(rc1*rc1+rc2*rc2));
						} break;
						default:{
							rc1=0;
							rc2=0;
							rc=0;
						} break;
					}
					fprintf(writer,"basis_angular %s %s\n",types[j].c_str(),types[k].c_str());
					fprintf(writer,"BasisA %s %6.4f %s %i\n",
						Cutoff::Name::name(cutname),
						rc,PhiAN::name(phiAN),nA*nra
					);
					for(int l=0; l<nra; ++l){
						double reta=0;
						switch(mix_angular){
							case Mix::MAX:{
								reta=rada[i][l]+((rada[j][l]>rada[k][l])?rada[j][l]:rada[k][l]); 
							} break;
							case Mix::AMEAN: {
								reta=rada[i][l]+0.5*(rada[j][l]+rada[k][l]); 
							} break;
							case Mix::GMEAN: {
								reta=rada[i][l]+sqrt(rada[j][l]*rada[k][l]); 
							} break;
							case Mix::HMEAN: {
								reta=rada[i][l]+2.0/(1.0/rada[j][l]+1.0/rada[k][l]); 
							} break;
							case Mix::QMEAN:{
								reta=rada[i][l]+sqrt(0.5*(rada[j][l]*rada[j][l]+rada[k][l]*rada[k][l]));
							} break;
							default: reta=0.0;
						}
						for(int n=0; n<nA; ++n){
							fprintf(writer,"\t%5.3f %5.3f %i\n",reta,zeta[n],lambda[n]);
						}
					}
				}
			}
			//close file
			fclose(writer);
			writer=NULL;
		}
		
	}catch(std::exception& e){
		std::cout<<"ERROR in make_basis(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] pfile;
	delete[] input;
	delete[] strbuf;
	
	if(error) return 1;
	return 0;
}