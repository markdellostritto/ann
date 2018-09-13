#include "basis_radial.hpp"

//member variables

const double BasisR::V_CUT=1e-5;

//initialization

void BasisR::init_G1(BasisR& basis, CutoffN::type tcut, double rmin, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_G1(BasisR&,unsigned int,CutoffN::type,double,double):\n";
	if(rcut<=0) throw std::invalid_argument("Invalid radial cutoff.");
	basis.phiRN=PhiRN::G1;
	basis.fR.resize(1);
	basis.fR[0].reset(new PhiR_G1(tcut,rcut));
}

void BasisR::init_G2(BasisR& basis, unsigned int nR, CutoffN::type tcut, double rmin, double rcut){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::init_G2(BasisR&,unsigned int,CutoffN::type,double):\n";
	if(nR==0) throw std::invalid_argument("Invalid number of radial functions.");
	if(rcut<=0) throw std::invalid_argument("Invalid radial cutoff.");
	basis.phiRN=PhiRN::G2;
	basis.fR.resize(nR);
	double dr=(rcut-rmin)/(nR-1.0);
	for(unsigned int i=0; i<nR; ++i){
		double eta=-std::log(V_CUT)/(4.0*dr*dr);
		double rs=rmin+i*dr;
		basis.fR[i].reset(new PhiR_G2(tcut,rcut,eta,rs));
	}
}

//loading/printing

void BasisR::write(FILE* writer,const BasisR& basis){
	if(BASIS_RADIAL_PRINT_FUNC>0) std::cout<<"BasisR::write(FILE*,const BasisR&):\n";
	std::string str;
	if(basis.phiRN==PhiRN::G1) str="G1";
	else if(basis.phiRN==PhiRN::G2) str="G2";
	fprintf(writer,"BasisR %s %i\n",str.c_str(),basis.fR.size());
	if(basis.phiRN==PhiRN::G1){
		//tcut,rc
		if(basis.fR[0]->tcut==CutoffN::COS) str="COS";
		else if(basis.fR[0]->tcut==CutoffN::TANH) str="TANH";
		fprintf(writer,"\t%s %f\n",str.c_str(),basis.fR[0]->rc);
	} else if(basis.phiRN==PhiRN::G2){
		//tcut,rc,rs,eta
		for(unsigned int i=0; i<basis.fR.size(); ++i){
			if(basis.fR[i]->tcut==CutoffN::COS) str="COS";
			else if(basis.fR[i]->tcut==CutoffN::TANH) str="TANH";
			PhiR_G2& g2=static_cast<PhiR_G2&>(*basis.fR[i]);
			fprintf(writer,"\t%s %f %f %f\n",str.c_str(),g2.rc,g2.eta,g2.rs);
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
	//reset the basis
	basis.fR.clear();
	//read header
	fgets(input,string::M,reader);
	n=string::substrN(input,string::WS);
	if(n!=3) throw std::runtime_error("Invalid BasisR format.");
	strlist.resize(n);
	strlist[0]=std::string(std::strtok(input,string::WS));
	strlist[1]=std::string(std::strtok(NULL,string::WS));
	strlist[2]=std::string(std::strtok(NULL,string::WS));
	//resize the basis
	basis.fR.resize(std::atoi(strlist[2].c_str()));
	//loop over all basis functions
	if(strlist[1]=="G1"){
		//tcut,rc
		for(unsigned int i=0; i<basis.fR.size(); ++i){
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
			basis.fR[i]=std::shared_ptr<PhiR>(new PhiR_G1(tcut,rc));
		}
		basis.phiRN=PhiRN::G1;
	} else if(strlist[1]=="G2"){
		//tcut,rc,rs,eta
		for(unsigned int i=0; i<basis.fR.size(); ++i){
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
			basis.fR[i]=std::shared_ptr<PhiR>(new PhiR_G2(tcut,rc,rs,eta));
		}
		basis.phiRN=PhiRN::G2;
	} else throw std::invalid_argument("Invalid radial function type");
	//free local variables
	free(input);
}

std::ostream& operator<<(std::ostream& out, const BasisR& basisR){
	out<<"BasisR "<<basisR.phiRN<<" "<<basisR.fR.size()<<"\n";
	if(basisR.phiRN==PhiRN::G1){
		for(unsigned int i=0; i<basisR.fR.size()-1; ++i) out<<"\t"<<static_cast<PhiR_G1&>(*basisR.fR[i])<<"\n";
		out<<"\t"<<static_cast<PhiR_G1&>(*basisR.fR.back());
	} else if(basisR.phiRN==PhiRN::G2){
		for(unsigned int i=0; i<basisR.fR.size()-1; ++i) out<<"\t"<<static_cast<PhiR_G2&>(*basisR.fR[i])<<"\n";
		out<<"\t"<<static_cast<PhiR_G2&>(*basisR.fR.back());
	}
	return out;
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const BasisR& obj){
		unsigned int N=0;
		N+=sizeof(obj.phiRN);//name of symmetry functions
		N+=sizeof(unsigned int);//number of symmetry functions
		if(obj.phiRN==PhiRN::G1){
			for(unsigned int i=0; i<obj.fR.size(); ++i) N+=nbytes(dynamic_cast<PhiR_G1&>(*obj.fR[i]));
		} else if(obj.phiRN==PhiRN::G2){
			for(unsigned int i=0; i<obj.fR.size(); ++i) N+=nbytes(dynamic_cast<PhiR_G2&>(*obj.fR[i]));
		} else throw std::runtime_error("Invalid radial symmetry function.");
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const BasisR& obj, char* arr){
		unsigned int pos=0,size=obj.fR.size();
		std::memcpy(arr+pos,&obj.phiRN,sizeof(obj.phiRN)); pos+=sizeof(obj.phiRN);
		std::memcpy(arr+pos,&size,sizeof(size)); pos+=sizeof(size);
		if(obj.phiRN==PhiRN::G1){
			for(unsigned int i=0; i<obj.fR.size(); ++i){
				pack(dynamic_cast<const PhiR_G1&>(*obj.fR[i]),arr+pos); pos+=nbytes(dynamic_cast<const PhiR_G1&>(*obj.fR[i]));
			}
		} else if(obj.phiRN==PhiRN::G2){
			for(unsigned int i=0; i<obj.fR.size(); ++i){
				pack(dynamic_cast<const PhiR_G2&>(*obj.fR[i]),arr+pos); pos+=nbytes(dynamic_cast<const PhiR_G2&>(*obj.fR[i]));
			}
		} else throw std::runtime_error("Invalid radial symmetry function.");
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(BasisR& obj, const char* arr){
		unsigned int pos=0,size=0;
		std::memcpy(&obj.phiRN,arr+pos,sizeof(obj.phiRN)); pos+=sizeof(obj.phiRN);
		std::memcpy(&size,arr+pos,sizeof(size)); pos+=sizeof(size);
		obj.fR.resize(size);
		if(obj.phiRN==PhiRN::G1){
			for(unsigned int i=0; i<obj.fR.size(); ++i){
				obj.fR[i].reset(new PhiR_G1());
				unpack(dynamic_cast<PhiR_G1&>(*obj.fR[i]),arr+pos); pos+=nbytes(dynamic_cast<const PhiR_G1&>(*obj.fR[i]));
			}
		} else if(obj.phiRN==PhiRN::G2){
			for(unsigned int i=0; i<obj.fR.size(); ++i){
				obj.fR[i].reset(new PhiR_G2());
				unpack(dynamic_cast<PhiR_G2&>(*obj.fR[i]),arr+pos); pos+=nbytes(dynamic_cast<const PhiR_G2&>(*obj.fR[i]));
			}
		} else throw std::runtime_error("Invalid radial symmetry function.");
	}
	
}
