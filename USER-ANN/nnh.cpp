// c libraries
#include <cmath>
#include <cstdio>
// c++ libraries
#include <iostream>
// ann - math
#include "math_const_ann.h"
// ann - print
#include "print.h"
// ann - nnh
#include "nnh.h"

//************************************************************
// NEURAL NETWORK HAMILTONIAN
//************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const NNH& nnh){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	//hamiltonian
	out<<"ATOM     = "<<nnh.atom_<<"\n";
	out<<"R_CUT    = "<<nnh.rc_<<"\n";
	//species
	out<<"NSPECIES = "<<nnh.nspecies_<<"\n";
	out<<"ATOMS    = \n";
	for(int i=0; i<nnh.nspecies_; ++i) std::cout<<"\t"<<nnh.species_[i]<<"\n";
	//potential parameters
	out<<"N_INPUT  = "; std::cout<<nnh.nInput_<<" "; std::cout<<"\n";
	out<<"N_INPUTR = "; std::cout<<nnh.nInputR_<<" "; std::cout<<"\n";
	out<<"N_INPUTA = "; std::cout<<nnh.nInputA_<<" "; std::cout<<"\n";
	out<<nnh.nn_<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

//misc

void NNH::defaults(){
	//hamiltonian
		atom_.clear();
		nn_.clear();
		rc_=0;
	//interacting species
		nspecies_=0;
		species_.clear();
	//basis for pair/triple interactions
		basisR_.clear();
		basisA_.clear();
	//network configuration
		nInput_=0;
		nInputR_=0;
		nInputA_=0;
		offsetR_.clear();
		offsetA_.clear();
}

//resizing

void NNH::resize(const std::vector<AtomANN>& species){
	if(species.size()==0) throw std::invalid_argument("NNH::resize(const std::vector<AtomANN>&): invalid number of species.");
	nspecies_=species.size();
	basisR_.resize(nspecies_);
	basisA_.resize(nspecies_);
	species_.resize(nspecies_);
	offsetR_.resize(nspecies_);
	offsetA_.resize(nspecies_);
	species_=species;
	for(int i=0; i<nspecies_; ++i){
		map_.add(string::hash(species_[i].name()),i);
	}
}

void NNH::init_input(){
	//radial inputs
	nInputR_=0;
	for(int i=0; i<nspecies_; ++i){
		nInputR_+=basisR_[i].nfR();
	}
	//radial offsets
	offsetR_[0]=0;
	for(int i=1; i<nspecies_; ++i){
		offsetR_[i]=offsetR_[i-1]+basisR_[i-1].nfR();
	}
	//angular inputs
	nInputA_=0;
	for(int i=0; i<nspecies_; ++i){
		for(int j=i; j<nspecies_; ++j){
			nInputA_+=basisA_(j,i).nfA();
		}
	}
	//angular offsets
	offsetA_[0]=0;
	for(int i=1; i<basisA_.size(); ++i){
		offsetA_[i]=offsetA_[i-1]+basisA_[i-1].nfA();
	}
	//total number of inputs
	nInput_=nInputR_+nInputA_;
}

//output

double NNH::energy(const Eigen::VectorXd& symm){
	return nn_.execute(symm)[0]+atom_.energy();
}

//reading/writing - all

void NNH::write(const std::string& filename)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::write(const std::string&):\n";
	FILE* writer=NULL;
	writer=fopen(filename.c_str(),"w");
	if(writer!=NULL){
		write(writer);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("NNH::write(const std::string&): Could not write to nnh file: \"")+filename+std::string("\""));
}

void NNH::write(FILE* writer)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::write(FILE*):\n";
	//==== write the header ====
	fprintf(writer,"ann\n");
	//==== write the global cutoff ====
	fprintf(writer,"cut %f\n",rc_);
	//==== write the central species ====
	fprintf(writer,"%s %f %f %f\n",atom_.name().c_str(),atom_.mass(),atom_.energy(),atom_.charge());
	//==== write the number of species ====
	fprintf(writer,"nspecies %i\n",nspecies_);
	//==== write all species ====
	for(int i=0; i<nspecies_; ++i){
		fprintf(writer,"%s %f %f\n",species_[i].name().c_str(),species_[i].mass(),species_[i].energy());
	}
	//==== write the radial basis ====
	for(int j=0; j<nspecies_; ++j){
		fprintf(writer,"basis_radial %s\n",species_[j].name().c_str());
		BasisR::write(writer,basisR_[j]);
	}
	//==== write the angular basis ====
	for(int j=0; j<nspecies_; ++j){
		for(int k=j; k<nspecies_; ++k){
			fprintf(writer,"basis_angular %s %s\n",species_[j].name().c_str(),species_[k].name().c_str());
			BasisA::write(writer,basisA_(j,k));
		}
	}
	//==== write the neural network ====
	NN::Network::write(writer,nn_);
}

void NNH::read(const std::string& filename){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::read(const std::string&):\n";
	FILE* reader=NULL;
	reader=fopen(filename.c_str(),"r");
	if(reader!=NULL){
		read(reader);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("NNH::read(const std::string&): Could not open nnpot file: \"")+filename+std::string("\""));
}

void NNH::read(FILE* reader){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::read(FILE*):\n";
	//==== local function variables ====
	std::vector<std::string> strlist;
	char* input=new char[string::M];
	//==== reader in header ====
	fgets(input,string::M,reader);
	//==== read in global cutoff ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const double rc=std::atof(std::strtok(NULL,string::WS));
	if(rc<=0) throw std::invalid_argument("NNH::read(FILE*): invalid cutoff.");
	else rc_=rc;
	//==== read the central atom ====
	AtomANN::read(fgets(input,string::M,reader),atom_);
	//==== read the number of species ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const int nspecies=std::atoi(std::strtok(NULL,string::WS));
	//==== read all species ====
	std::vector<AtomANN> species(nspecies);
	for(int i=0; i<nspecies; ++i){
		AtomANN::read(fgets(input,string::M,reader),species[i]);
	}
	//==== resize the hamiltonian ====
	resize(species);
	//==== read the radial basis ====
	for(int i=0; i<nspecies_; ++i){
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int jj=index(strlist[1]);
		BasisR::read(reader,basisR_[jj]);
	}
	//==== read the angular basis ====
	for(int i=0; i<nspecies_; ++i){
		for(int j=i; j<nspecies_; ++j){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const int jj=index(strlist[1]);
			const int kk=index(strlist[2]);
			BasisA::read(reader,basisA_(jj,kk));
		}
	}
	//==== read the neural network ====
	NN::Network::read(reader,nn_);
	//==== set the number of inputs and offsets ====
	init_input();
	//======== free local variables ========
	delete[] input;
}

//reading/writing - basis

void NNH::write_basis(const std::string& filename)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::write_basis(const std::string&):\n";
	FILE* writer=NULL;
	writer=fopen(filename.c_str(),"w");
	if(writer!=NULL){
		write_basis(writer);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("NNH::write_basis(const std::string&): Could not write to nnh file: \"")+filename+std::string("\""));
}

void NNH::write_basis(FILE* writer)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::write_basis(FILE*):\n";
	//==== write the header ====
	fprintf(writer,"ann\n");
	//==== write the global cutoff ====
	fprintf(writer,"cut %f\n",rc_);
	//==== write the central species ====
	fprintf(writer,"%s %f %f %f\n",atom_.name().c_str(),atom_.mass(),atom_.energy(),atom_.charge());
	//==== write the number of species ====
	fprintf(writer,"nspecies %i\n",nspecies_);
	//==== write all species ====
	for(int i=0; i<nspecies_; ++i){
		fprintf(writer,"%s %f %f\n",species_[i].name().c_str(),species_[i].mass(),species_[i].energy());
	}
	//==== write the radial basis ====
	for(int j=0; j<nspecies_; ++j){
		fprintf(writer,"basis_radial %s\n",species_[j].name().c_str());
		BasisR::write(writer,basisR_[j]);
	}
	//==== write the angular basis ====
	for(int j=0; j<nspecies_; ++j){
		for(int k=j; k<nspecies_; ++k){
			fprintf(writer,"basis_angular %s %s\n",species_[j].name().c_str(),species_[k].name().c_str());
			BasisA::write(writer,basisA_(j,k));
		}
	}
}

void NNH::read_basis(const std::string& filename){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::read_basis(const std::string&):\n";
	FILE* reader=NULL;
	reader=fopen(filename.c_str(),"r");
	if(reader!=NULL){
		read_basis(reader);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("NNH::read_basis(const std::string&): Could not open nnpot file: \"")+filename+std::string("\""));
}

void NNH::read_basis(FILE* reader){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::read_basis(FILE*):\n";
	//==== local function variables ====
	std::vector<std::string> strlist;
	char* input=new char[string::M];
	//==== reader in header ====
	fgets(input,string::M,reader);
	//==== read in global cutoff ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const double rc=std::atof(std::strtok(NULL,string::WS));
	if(rc<=0) throw std::invalid_argument("NNH::read(FILE*): invalid cutoff.");
	else rc_=rc;
	//==== read the central atom ====
	AtomANN::read(fgets(input,string::M,reader),atom_);
	//==== read the number of species ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const int nspecies=std::atoi(std::strtok(NULL,string::WS));
	//==== read all species ====
	std::vector<AtomANN> species(nspecies);
	for(int i=0; i<nspecies; ++i){
		AtomANN::read(fgets(input,string::M,reader),species[i]);
	}
	//==== resize the hamiltonian ====
	resize(species);
	//==== read the radial basis ====
	for(int i=0; i<nspecies_; ++i){
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int jj=index(strlist[1]);
		BasisR::read(reader,basisR_[jj]);
	}
	//==== read the angular basis ====
	for(int i=0; i<nspecies_; ++i){
		for(int j=i; j<nspecies_; ++j){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const int jj=index(strlist[1]);
			const int kk=index(strlist[2]);
			BasisA::read(reader,basisA_(jj,kk));
		}
	}
	//==== set the number of inputs and offsets ====
	init_input();
	//======== free local variables ========
	delete[] input;
}

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNH& obj){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"nbytes(const NNH&):\n";
	int size=0;
	//hamiltonian
	size+=nbytes(obj.atom());
	size+=nbytes(obj.nn());
	size+=nbytes(obj.rc());
	//species
	size+=nbytes(obj.nspecies());//nspecies_
	for(int i=0; i<obj.nspecies(); ++i){
		size+=nbytes(obj.species(i));//species_
	}
	size+=nbytes(obj.map());
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		size+=nbytes(obj.basisR(j));
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			size+=nbytes(obj.basisA(j,k));
		}
	}
	//return the size
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNH& obj, char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"pack(const NNH&,char*):\n";
	int pos=0;
	//hamiltonian
	pos+=pack(obj.atom(),arr+pos);
	pos+=pack(obj.nn(),arr+pos);
	pos+=pack(obj.rc(),arr+pos);
	//species
	pos+=pack(obj.nspecies(),arr+pos);
	for(int i=0; i<obj.nspecies(); ++i){
		pos+=pack(obj.species(i),arr+pos);
	}
	pos+=pack(obj.map(),arr+pos);
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		pos+=pack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			pos+=pack(obj.basisA(j,k),arr+pos);
		}
	}
	//return bytes written
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNH& obj, const char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"unpack(NNH&,const char*):\n";
	int pos=0;
	//hamiltonian
	pos+=unpack(obj.atom(),arr+pos);
	pos+=unpack(obj.nn(),arr+pos);
	pos+=unpack(obj.rc(),arr+pos);
	//species
	int nspecies=0;
	pos+=unpack(nspecies,arr+pos);
	std::vector<AtomANN> species(nspecies);
	for(int i=0; i<nspecies; ++i){
		pos+=unpack(species[i],arr+pos);
	}
	obj.resize(species);
	pos+=unpack(obj.map(),arr+pos);
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		pos+=unpack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			pos+=unpack(obj.basisA(j,k),arr+pos);
		}
	}
	//intialize the inputs and offsets
	obj.init_input();
	//return bytes read
	return pos;
}

}
