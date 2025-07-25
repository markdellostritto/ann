// mpi
#include <mpi.h>
// c libraries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// c++ libraries
#include <iostream>
#include <exception>
#include <algorithm>
#include <random>
#include <chrono>
// structure
#include "struc/structure.hpp"
#include "struc/neighbor.hpp"
// format
#include "format/file_struc.hpp"
#include "format/format.hpp"
// math
#include "math/reduce.hpp"
#include "math/corr.hpp"
#include "math/special.hpp"
#include "math/eigen.hpp"
// string
#include "str/string.hpp"
#include "str/token.hpp"
#include "str/print.hpp"
#include "str/parse.hpp"
// chem
#include "chem/units.hpp"
#include "chem/alias.hpp"
// thread
#include "thread/comm.hpp"
#include "thread/dist.hpp"
#include "thread/mpif.hpp"
// util
#include "util/compiler.hpp"
#include "util/time.hpp"
// torch
#include "torch/pot.hpp"
#include "torch/pot_gauss_long.hpp"
#include "torch/pot_ldamp_long.hpp"
#include "torch/pot_ldamp_cut.hpp"
#include "torch/pot_pauli.hpp"
// nnpte
#include "nnp/nnpqa3.hpp"

static bool compare_pair(const std::pair<int,double>& p1, const std::pair<int,double>& p2){
	return p1.first<p2.first;
}

using math::special::mod;
using math::constant::LOG2;

//************************************************************
// MPI Communicators
//************************************************************

thread::Comm WORLD;// all processors
thread::Comm BATCH;// subgroup for each element of the batch
thread::Comm BEAD;

//************************************************************
// serialization
//************************************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNPTE& obj){
   if(NNPTE_PRINT_FUNC>0) std::cout<<"nbytes(const NNPTE&)\n";
	int size=0;
	//nnp
		size+=nbytes(obj.nnp_);
	//input/output
		size+=nbytes(obj.file_params_);
		size+=nbytes(obj.file_error_);
		size+=nbytes(obj.file_ann_);
		size+=nbytes(obj.file_restart_);
	//flags
		size+=sizeof(bool);//restart
		size+=sizeof(bool);//reset
		size+=sizeof(bool);//wparams
	//optimization
		size+=nbytes(obj.batch_);
		size+=nbytes(obj.obj_);
		size+=nbytes(obj.iter_);
		size+=nbytes(obj.algo_);
		size+=nbytes(obj.decay_);
		size+=sizeof(PreScale);//prescale_
		size+=sizeof(PreBias);//prebias_
		size+=sizeof(double);//inscale_
		size+=sizeof(double);//inbias_
		size+=sizeof(double);//delta_
	//quantum
		size+=sizeof(int);
		size+=sizeof(double);//eps_
		size+=sizeof(double);//rho_
		size+=sizeof(double);//beta1i_
		size+=sizeof(double);//beta2i_
		size+=nbytes(obj.mgrad_);
		size+=nbytes(obj.mgrad2_);
		size+=nbytes(obj.mgradq_);
		size+=nbytes(obj.mgradq2_);
		size+=nbytes(obj.roots_bead_);
	//return the size
		return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNPTE& obj, char* arr){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"pack(const NNPTE&,char*)\n";
	int pos=0;
	//nnp
		pos+=pack(obj.nnp_,arr+pos);
	//input/output
		pos+=pack(obj.file_params_,arr+pos);
		pos+=pack(obj.file_error_,arr+pos);
		pos+=pack(obj.file_ann_,arr+pos);
		pos+=pack(obj.file_restart_,arr+pos);
	//flags
		std::memcpy(arr+pos,&obj.restart_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.reset_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.wparams_,sizeof(bool)); pos+=sizeof(bool);
	//optimization
		pos+=pack(obj.batch_,arr+pos);
		pos+=pack(obj.obj_,arr+pos);
		pos+=pack(obj.iter_,arr+pos);
		pos+=pack(obj.algo_,arr+pos);
		pos+=pack(obj.decay_,arr+pos);
		std::memcpy(arr+pos,&obj.prescale_,sizeof(PreScale)); pos+=sizeof(PreScale);
		std::memcpy(arr+pos,&obj.prebias_,sizeof(PreBias)); pos+=sizeof(PreBias);
		std::memcpy(arr+pos,&obj.inscale_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.inbias_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.delta_,sizeof(double)); pos+=sizeof(double);
	//quantum
		std::memcpy(arr+pos,&obj.nqstep(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.rho(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.beta1i_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.beta2i_,sizeof(double)); pos+=sizeof(double);
		pos+=pack(obj.mgrad_,arr+pos);
		pos+=pack(obj.mgrad2_,arr+pos);
		pos+=pack(obj.mgradq_,arr+pos);
		pos+=pack(obj.mgradq2_,arr+pos);
		pos+=pack(obj.roots_bead_,arr+pos);
	//return bytes written
		return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNPTE& obj, const char* arr){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"unpack(const NNPTE&,char*)\n";
	int pos=0;
	//nnp
		pos+=unpack(obj.nnp_,arr+pos);
	//input/output
		pos+=unpack(obj.file_params_,arr+pos);
		pos+=unpack(obj.file_error_,arr+pos);
		pos+=unpack(obj.file_ann_,arr+pos);
		pos+=unpack(obj.file_restart_,arr+pos);
	//flags
		std::memcpy(&obj.restart_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.reset_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.wparams_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
	//optimization
		pos+=unpack(obj.batch_,arr+pos);
		pos+=unpack(obj.obj_,arr+pos);
		pos+=unpack(obj.iter_,arr+pos);
		pos+=unpack(obj.algo_,arr+pos);
		pos+=unpack(obj.decay_,arr+pos);
		std::memcpy(&obj.prescale_,arr+pos,sizeof(PreScale)); pos+=sizeof(PreScale);
		std::memcpy(&obj.prebias_,arr+pos,sizeof(PreBias)); pos+=sizeof(PreBias);
		std::memcpy(&obj.inscale_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.inbias_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.delta_,arr+pos,sizeof(double)); pos+=sizeof(double);
		obj.deltai()=1.0/obj.delta();
		obj.delta2()=obj.delta()*obj.delta();
	//quantum
		std::memcpy(&obj.nqstep(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.rho(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.beta1i_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.beta2i_,arr+pos,sizeof(double)); pos+=sizeof(double);
		pos+=unpack(obj.mgrad_,arr+pos);
		pos+=unpack(obj.mgrad2_,arr+pos);
		pos+=unpack(obj.mgradq_,arr+pos);
		pos+=unpack(obj.mgradq2_,arr+pos);
		pos+=unpack(obj.roots_bead_,arr+pos);
	//return bytes read
		return pos;
}
	
}

//************************************************************
// PreScale
//************************************************************

std::ostream& operator<<(std::ostream& out, const PreScale& prescale){
	switch(prescale){
		case PreScale::NONE: out<<"NONE"; break;
		case PreScale::DEV: out<<"DEV"; break;
		case PreScale::MINMAX: out<<"MINMAX"; break;
		case PreScale::MAX: out<<"MAX"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* PreScale::name(const PreScale& prescale){
	switch(prescale){
		case PreScale::NONE: return "NONE";
		case PreScale::DEV: return "DEV";
		case PreScale::MINMAX: return "MINMAX";
		case PreScale::MAX: return "MAX";
		default: return "UNKNOWN";
	}
}

PreScale PreScale::read(const char* str){
	if(std::strcmp(str,"NONE")==0) return PreScale::NONE;
	else if(std::strcmp(str,"DEV")==0) return PreScale::DEV;
	else if(std::strcmp(str,"MINMAX")==0) return PreScale::MINMAX;
	else if(std::strcmp(str,"MAX")==0) return PreScale::MAX;
	else return PreScale::UNKNOWN;
}

//************************************************************
// PreBias
//************************************************************

std::ostream& operator<<(std::ostream& out, const PreBias& prebias){
	switch(prebias){
		case PreBias::NONE: out<<"NONE"; break;
		case PreBias::MEAN: out<<"MEAN"; break;
		case PreBias::MID: out<<"MID"; break;
		case PreBias::MIN: out<<"MIN"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* PreBias::name(const PreBias& prebias){
	switch(prebias){
		case PreBias::NONE: return "NONE";
		case PreBias::MEAN: return "MEAN";
		case PreBias::MID: return "MID";
		case PreBias::MIN: return "MIN";
		default: return "UNKNOWN";
	}
}

PreBias PreBias::read(const char* str){
	if(std::strcmp(str,"NONE")==0) return PreBias::NONE;
	else if(std::strcmp(str,"MEAN")==0) return PreBias::MEAN;
	else if(std::strcmp(str,"MID")==0) return PreBias::MID;
	else if(std::strcmp(str,"MIN")==0) return PreBias::MIN;
	else return PreBias::UNKNOWN;
}

//************************************************************
// Mode
//************************************************************

std::ostream& operator<<(std::ostream& out, const Mode& mode){
	switch(mode){
		case Mode::TRAIN: out<<"TRAIN"; break;
		case Mode::TEST: out<<"TEST"; break;
		case Mode::SYMM: out<<"SYMM"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Mode::name(const Mode& mode){
	switch(mode){
		case Mode::TRAIN: return "TRAIN";
		case Mode::TEST: return "TEST";
		case Mode::SYMM: return "SYMM";
		default: return "UNKNOWN";
	}
}

Mode Mode::read(const char* str){
	if(std::strcmp(str,"TRAIN")==0) return Mode::TRAIN;
	else if(std::strcmp(str,"TEST")==0) return Mode::TEST;
	else if(std::strcmp(str,"SYMM")==0) return Mode::SYMM;
	else return Mode::UNKNOWN;
}

//************************************************************
// NNPTE - Neural Network Potential - Optimization
//************************************************************

const double NNPTE::beta1_=0.9;
const double NNPTE::beta2_=0.999;

std::ostream& operator<<(std::ostream& out, const NNPTE& nnpte){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NNPTE",str)<<"\n";
	//files
	out<<"file_params  = "<<nnpte.file_params_<<"\n";
	out<<"file_error   = "<<nnpte.file_error_<<"\n";
	out<<"file_ann     = "<<nnpte.file_ann_<<"\n";
	out<<"file_restart = "<<nnpte.file_restart_<<"\n";
	//flags
	out<<"restart      = "<<nnpte.restart_<<"\n";
	out<<"reset        = "<<nnpte.reset_<<"\n";
	out<<"wparams      = "<<nnpte.wparams_<<"\n";
	//optimization
	out<<"batch        = "<<nnpte.batch_<<"\n";
	out<<"decay        = "<<nnpte.decay_<<"\n";
	out<<"n_print      = "<<nnpte.iter().nPrint()<<"\n";
	out<<"n_write      = "<<nnpte.iter().nWrite()<<"\n";
	out<<"max          = "<<nnpte.iter().max()<<"\n";
	out<<"stop         = "<<nnpte.iter().stop()<<"\n";
	out<<"loss         = "<<nnpte.iter().loss()<<"\n";
	out<<"tol          = "<<nnpte.iter().tol()<<"\n";
	out<<"gamma        = "<<nnpte.obj().gamma()<<"\n";
	out<<"delta        = "<<nnpte.delta()<<"\n";
	//quantum adam
	out<<"rho          = "<<nnpte.rho()<<"\n";
	out<<"eps          = "<<nnpte.eps()<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

NNPTE::NNPTE(){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNP::NNPTE():\n";
	defaults();
};

void NNPTE::defaults(){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNP::defaults():\n";
	//nnp
		nTypes_=0;
	//input/output
		file_params_="nnp_params.dat";
		file_error_="nnp_error";
		file_restart_="nnpte";
		file_ann_="ann";
	//flags
		restart_=false;
		reset_=false;
		wparams_=false;
	//optimization
		delta_=1.0;
		delta2_=1.0;
		deltai_=1.0;
	//quantum adam
		eps_=1.0e-16;
		rho_=0.0;
		beta1i_=beta1_;
		beta2i_=beta2_;
		roots_bead_.clear();
	//error
		error_[0]=0;//loss - train
		error_[1]=0;//loss - val
		error_[2]=0;//rmse - train
		error_[3]=0;//rmse - val
}

void NNPTE::clear(){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNP::clear():\n";
	//elements
		nTypes_=0;
		gElement_.clear();
		pElement_.clear();
	//nnp
		nnp_.clear();
	//optimization
		batch_.clear();
		obj_.clear();
	//error
		error_[0]=0;//loss - train
		error_[1]=0;//loss - val
		error_[2]=0;//rmse - train
		error_[3]=0;//rmse - val
}

void NNPTE::write_restart(const char* file){
	if(NNPTE_PRINT_FUNC>1) std::cout<<"NNPTE::write_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* writer=NULL;
	bool error=false;
	try{
		//open file
		writer=fopen(file,"wb");
		if(writer==NULL) throw std::runtime_error(std::string("NNPTE::write_restart(const char*): Could not open file: ")+file);
		//allocate buffer
		const int nBytes=serialize::nbytes(*this);
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTE::write_restart(const char*): Could not allocate memory.");
		//write to buffer
		serialize::pack(*this,arr);
		//write to file
		const int nWrite=fwrite(arr,sizeof(char),nBytes,writer);
		if(nWrite!=nBytes) throw std::runtime_error("NNPTE::write_restart(const char*): Write error.");
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(writer); writer=NULL;
	}catch(std::exception& e){
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(writer!=NULL) fclose(writer);
	if(error) throw std::runtime_error("NNPTE::write_restart(const char*): Failed to write");
}

void NNPTE::read_restart(const char* file){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNPTE::read_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* reader=NULL;
	bool error=false;
	try{
		//open file
		reader=fopen(file,"rb");
		if(reader==NULL) throw std::runtime_error(std::string("NNPTE::read_restart(const char*): Could not open file: ")+std::string(file));
		//find size
		std::fseek(reader,0,SEEK_END);
		const int nBytes=std::ftell(reader);
		std::fseek(reader,0,SEEK_SET);
		//allocate buffer
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTE::read_restart(const char*): Could not allocate memory.");
		//read from file
		const int nRead=fread(arr,sizeof(char),nBytes,reader);
		if(nRead!=nBytes) throw std::runtime_error("NNPTE::read_restart(const char*): Read error.");
		//read from buffer
		serialize::unpack(*this,arr);
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(reader); reader=NULL;
	}catch(std::exception& e){
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(reader!=NULL) fclose(reader);
	if(error) throw std::runtime_error("NNPTE::read_restart(const char*): Failed to read");
}

void NNPTE::train(int batchSize, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNPTE::train(NNP&,std::vector<Structure>&,int):\n";
	//====== local function variables ======
	//statistics
		std::vector<int> N;//total number of inputs for each element
		std::vector<Eigen::VectorXd> avg_in;//average of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> max_in;//max of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> min_in;//min of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> dev_in;//average of the stddev for each element (nnp_.nSpecies_ x nInput_)
	//timing
		Clock clock;
	//random
		std::default_random_engine generator;

	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"training NN potential\n";
	
	//====== check the parameters ======
	if(batchSize<=0) throw std::invalid_argument("NNPTE::train(int): Invalid batch size.");
	if(struc_train.size()==0) throw std::invalid_argument("NNPTE::train(int): No training data provided.");
	if(struc_val.size()==0) throw std::invalid_argument("NNPTE::train(int): No validation data provided.");
	
	//====== get the number of structures ======
	double nBatchF=(1.0*batchSize)/BATCH.size();
	double nTrainF=(1.0*struc_train.size())/BATCH.size();
	double nValF=(1.0*struc_val.size())/BATCH.size();
	MPI_Allreduce(MPI_IN_PLACE,&nBatchF,1,MPI_DOUBLE,MPI_SUM,BEAD.mpic());
	MPI_Allreduce(MPI_IN_PLACE,&nTrainF,1,MPI_DOUBLE,MPI_SUM,BEAD.mpic());
	MPI_Allreduce(MPI_IN_PLACE,&nValF,1,MPI_DOUBLE,MPI_SUM,BEAD.mpic());
	const int nBatch=std::round(nBatchF);
	const int nTrain=std::round(nTrainF);
	const int nVal=std::round(nValF);
	
	//====== set the distributions over the atoms ======
	dist_atomt.resize(struc_train.size());
	dist_atomv.resize(struc_val.size());
	for(int i=0; i<struc_train.size(); ++i) dist_atomt[i].init(BATCH.size(),BATCH.rank(),struc_train[i].nAtoms());
	for(int i=0; i<struc_val.size(); ++i) dist_atomv[i].init(BATCH.size(),BATCH.rank(),struc_val[i].nAtoms());
	
	//====== initialize the random number generator ======
	rngen_=std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
	
	//====== resize the optimization data ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing the optimization data\n";
	//set the number of types
	nTypes_=nnp_.ntypes();
	//resize per-element arrays
	pElement_.resize(nTypes_);
	gElement_.resize(nTypes_);
	grad_.resize(nTypes_);
	for(int n=0; n<nTypes_; ++n){
		const int nn_size=nnp_.nnh(n).nn().size();
		pElement_[n]=Eigen::VectorXd::Zero(nn_size);
		gElement_[n]=Eigen::VectorXd::Zero(nn_size);
		grad_[n]=Eigen::VectorXd::Zero(nn_size);
	}
	//resize gradient objects
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing gradient data\n";
	cost_.resize(nTypes_);
	for(int n=0; n<nTypes_; ++n){
		cost_[n].resize(nnp_.nnh(n).nn());
	}
	
	//====== compute the number of atoms of each element ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing the number of atoms of each element\n";
	std::vector<double> nAtoms_(nTypes_,0);
	for(int i=0; i<struc_train.size(); ++i){
		for(int j=0; j<struc_train[i].nAtoms(); ++j){
			++nAtoms_[struc_train[i].type(j)];
		}
	}
	MPI_Allreduce(MPI_IN_PLACE,nAtoms_.data(),nTypes_,MPI_DOUBLE,MPI_SUM,BEAD.mpic());
	for(int i=0; i<nTypes_; ++i) nAtoms_[i]/=BATCH.size();
	if(NNPTE_PRINT_DATA>-1 && WORLD.rank()==0){
		char* strbuf=new char[print::len_buf];
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("ATOM - DATA",strbuf)<<"\n";
		for(int i=0; i<nTypes_; ++i){
			const std::string& name=nnp_.nnh(i).type().name();
			const int index=nnp_.index(nnp_.nnh(i).type().name());
			std::cout<<name<<"("<<index<<") - "<<(int)nAtoms_[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		delete[] strbuf;
	}
	
	//====== set the indices and batch size ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting indices and batch\n";
	batch_.resize(batchSize,struc_train.size());
	
	//====== collect input statistics ======
	//resize arrays
	N.resize(nTypes_);
	max_in.resize(nTypes_);
	min_in.resize(nTypes_);
	avg_in.resize(nTypes_);
	dev_in.resize(nTypes_);
	for(int n=0; n<nTypes_; ++n){
		const int nInput=nnp_.nnh(n).nInput();
		max_in[n]=Eigen::VectorXd::Constant(nInput,-1.0*std::numeric_limits<double>::max());
		min_in[n]=Eigen::VectorXd::Constant(nInput,1.0*std::numeric_limits<double>::max());
		avg_in[n]=Eigen::VectorXd::Zero(nInput);
		dev_in[n]=Eigen::VectorXd::Zero(nInput);
	}
	//compute the total number
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the total number\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			++N[struc_train[n].type(i)];
		}
	}
	for(int i=0; i<nTypes_; ++i){
		double Nloc=(1.0*N[i])/BATCH.size();//normalize by the size of the BATCH group
		MPI_Allreduce(MPI_IN_PLACE,&Nloc,1,MPI_DOUBLE,MPI_SUM,BEAD.mpic());
		N[i]=static_cast<int>(std::round(Nloc));
	}
	//compute the max/min
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the max/min\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		for(int i=0; i<strucl.nAtoms(); ++i){
			const int index=strucl.type(i);
			for(int k=0; k<nnp_.nnh(index).nInput(); ++k){
				if(strucl.symm(i)[k]>max_in[index][k]) max_in[index][k]=strucl.symm(i)[k];
				if(strucl.symm(i)[k]<min_in[index][k]) min_in[index][k]=strucl.symm(i)[k];
			}
		}
	}
	for(int i=0; i<nTypes_; ++i){
		MPI_Allreduce(MPI_IN_PLACE,min_in[i].data(),min_in[i].size(),MPI_DOUBLE,MPI_MIN,BEAD.mpic());
		MPI_Allreduce(MPI_IN_PLACE,max_in[i].data(),max_in[i].size(),MPI_DOUBLE,MPI_MAX,BEAD.mpic());
	}
	//compute the average
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the average\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			avg_in[struc_train[n].type(i)].noalias()+=struc_train[n].symm(i);
		}
	}
	for(int i=0; i<nTypes_; ++i){
		avg_in[i]/=BATCH.size();//normalize by the size of the BATCH group
		MPI_Allreduce(MPI_IN_PLACE,avg_in[i].data(),avg_in[i].size(),MPI_DOUBLE,MPI_SUM,BEAD.mpic());
		avg_in[i]/=N[i];
	}
	//compute the stddev
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the stddev\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		for(int i=0; i<strucl.nAtoms(); ++i){
			const int index=strucl.type(i);
			dev_in[index].noalias()+=(avg_in[index]-strucl.symm(i)).cwiseProduct(avg_in[index]-strucl.symm(i));
		}
	}
	for(int i=0; i<dev_in.size(); ++i){
		for(int j=0; j<dev_in[i].size(); ++j){
			dev_in[i][j]/=BATCH.size();//normalize by the size of the BATCH group
			MPI_Allreduce(MPI_IN_PLACE,&dev_in[i][j],1,MPI_DOUBLE,MPI_SUM,BEAD.mpic());
			dev_in[i][j]=sqrt(dev_in[i][j]/(N[i]-1.0));
		}
	}
	
	//====== precondition the input ======
	std::vector<Eigen::VectorXd> inpb_(nTypes_);//input bias
	std::vector<Eigen::VectorXd> inpw_(nTypes_);//input weight
	for(int n=0; n<nTypes_; ++n){
		inpb_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),0.0);
		inpw_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),1.0);
	}
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"pre-conditioning input\n";
	switch(prebias_){
		case PreBias::MEAN:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=-1.0*avg_in[n];
			}
		} break;
		case PreBias::MID:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=-0.5*(max_in[n]+min_in[n]);
			}
		} break;
		case PreBias::MIN:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=-min_in[n];
			}
		} break;
		case PreBias::NONE:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),0.0);
			}
		}break;
		case PreBias::UNKNOWN:
			throw std::invalid_argument("Invalid input bias.");
		break;
	}
	for(int n=0; n<nTypes_; ++n){
		inpb_[n].noalias()+=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),inbias_);
	}
	switch(prescale_){
		case PreScale::DEV:{
			for(int n=0; n<nTypes_; ++n){
				for(int i=0; i<inpw_[n].size(); ++i){
					if(dev_in[n][i]==0) inpw_[n][i]=1.0;
					else inpw_[n][i]=1.0/dev_in[n][i];
				}
			}
		} break;
		case PreScale::MINMAX:{
			for(int n=0; n<nTypes_; ++n){
				for(int i=0; i<inpw_[n].size(); ++i){
					const double s=(max_in[n][i]-min_in[n][i]);
					if(s>0) inpw_[n][i]=1.0/s;
					else inpw_[n][i]=1.0;
				}
			}
		} break;
		case PreScale::MAX:{
			for(int n=0; n<nTypes_; ++n){
				for(int i=0; i<inpw_[n].size(); ++i){
					const double s=max_in[n][i];
					if(s>0) inpw_[n][i]=1.0/s;
					else inpw_[n][i]=1.0;
				}
			}
		} break;
		case PreScale::NONE:{
			for(int n=0; n<nTypes_; ++n){
				inpw_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),1.0);
			}
		}break;
		case PreScale::UNKNOWN:
			throw std::invalid_argument("Invalid input scaling.");
		break;
	}
	for(int n=0; n<nTypes_; ++n){
		inpw_[n]*=inscale_;
	}
	
	//====== set the bias for each of the species ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting the bias for each species\n";
	for(int n=0; n<nTypes_; ++n){
		NN::ANN& nn_=nnp_.nnh(n).nn();
		for(int i=0; i<nn_.nInp(); ++i) nn_.inpb()[i]=inpb_[n][i];
		for(int i=0; i<nn_.nInp(); ++i) nn_.inpw()[i]=inpw_[n][i];
		nn_.outb()[0]=0.0;
		nn_.outw()[0]=1.0;
	}
	
	//====== initialize the optimization data - standard ======
	const int nParams=nnp_.size();
	wm1_=Eigen::VectorXd::Zero(nParams);
	wp1_=Eigen::VectorXd::Zero(nParams);
	if(restart_){
		//restart
		if(WORLD.rank()==0) std::cout<<"restarting optimization\n";
		if(nParams!=obj_.dim()) throw std::runtime_error(
			std::string("NNPTE::train(int): Network has ")
			+std::to_string(nParams)+std::string(" while opt has ")
			+std::to_string(obj_.dim())+std::string(" parameters.")
		);
	} else {
		//from scratch
		if(WORLD.rank()==0) std::cout<<"starting from scratch\n";
		//resize the optimization objects
		obj_.resize(nParams);
		//load random initial values in the per-element arrays
		for(int n=0; n<nTypes_; ++n){
			nnp_.nnh(n).nn()>>pElement_[n];
			gElement_[n]=Eigen::VectorXd::Random(nnp_.nnh(n).nn().size())*1e-6;
		}
		//load initial values from per-element arrays into global arrays
		int count=0;
		for(int n=0; n<nTypes_; ++n){
			for(int m=0; m<pElement_[n].size(); ++m){
				obj_.p()[count]=pElement_[n][m];
				obj_.g()[count]=gElement_[n][m];
				++count;
			}
		}
		//adam - standard
		mgrad_=Eigen::VectorXd::Zero(nParams);
		mgrad2_=Eigen::VectorXd::Zero(nParams);
		//adam - quantum
		mgradq_=Eigen::VectorXd::Zero(nParams);
		mgradq2_=Eigen::VectorXd::Zero(nParams);
	}
	
	//====== print input statistics and bias ======
	if(NNPTE_PRINT_DATA>-1 && WORLD.rank()==0){
		char* strbuf=new char[print::len_buf];
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("OPT - DATA",strbuf)<<"\n";
		std::cout<<"N-PARAMS    = \n\t"<<nParams<<"\n";
		std::cout<<"AVG - INPUT = \n"; for(int i=0; i<avg_in.size(); ++i) std::cout<<"\t"<<avg_in[i].transpose()<<"\n";
		std::cout<<"MAX - INPUT = \n"; for(int i=0; i<max_in.size(); ++i) std::cout<<"\t"<<max_in[i].transpose()<<"\n";
		std::cout<<"MIN - INPUT = \n"; for(int i=0; i<min_in.size(); ++i) std::cout<<"\t"<<min_in[i].transpose()<<"\n";
		std::cout<<"DEV - INPUT = \n"; for(int i=0; i<dev_in.size(); ++i) std::cout<<"\t"<<dev_in[i].transpose()<<"\n";
		std::cout<<"PRE-BIAS    = \n"; for(int i=0; i<inpb_.size(); ++i) std::cout<<"\t"<<inpb_[i].transpose()<<"\n";
		std::cout<<"PRE-SCALE   = \n"; for(int i=0; i<inpw_.size(); ++i) std::cout<<"\t"<<inpw_[i].transpose()<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		delete[] strbuf;
	}
	
	//====== execute the optimization ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"executing the optimization\n";
	//optimization variables
	const double nBatchi_=1.0/nBatch;
	const double nVali_=1.0/nVal;
	//bcast parameters
	MPI_Bcast(obj_.p().data(),obj_.p().size(),MPI_DOUBLE,0,BEAD.mpic());
	//allocate status vectors
	std::vector<int> step;
	std::vector<double> gamma,loss_t,loss_v,rmse_t,rmse_v;
	std::vector<Eigen::VectorXd> params;
	if(BEAD.rank()==0){
		int size=iter_.max()/iter_.nPrint();
		if(size==0) ++size;
		step.resize(size);
		gamma.resize(size);
		loss_t.resize(size);
		loss_v.resize(size);
		rmse_t.resize(size);
		rmse_v.resize(size);
		params.resize(size);
	}
	//print status header to standard output
	if(WORLD.rank()==0) printf("opt gamma loss_t loss_v rmse_t rmse_v\n");
	//start the clock
	clock.begin();
	//begin optimization
	Eigen::VectorXd gtot_=Eigen::VectorXd::Zero(obj_.dim());
	const bool evn=(BEAD.color()%2==0);
	const bool odd=(BEAD.color()%2==1);
	const int nf=roots_bead_[mod(BEAD.color()+1,BEAD.ncomm())];
	const int nb=roots_bead_[mod(BEAD.color()-1,BEAD.ncomm())];
	for(int iter=0; iter<iter_.max(); ++iter){
		double error_sum_[4]={0.0,0.0,0.0,0.0};
		//compute the error and gradient
		//error(obj_.p(),struc_train,struc_val);
		error2(obj_.p(),struc_train,struc_val);
		//pack the gradient
		int count=0;
		for(int n=0; n<nTypes_; ++n){
			std::memcpy(gtot_.data()+count,gElement_[n].data(),gElement_[n].size()*sizeof(double));
			count+=gElement_[n].size();
		}
		//accumulate gradient and error
		obj_.g().setZero();
		MPI_Reduce(gtot_.data(),obj_.g().data(),gtot_.size(),MPI_DOUBLE,MPI_SUM,0,BEAD.mpic());
		MPI_Reduce(error_,error_sum_,4,MPI_DOUBLE,MPI_SUM,0,BEAD.mpic());
		if(BEAD.rank()==0){
			//compute error averaged over the batch
			error_[0]=error_sum_[0]*nBatchi_;//loss - train
			error_[1]=error_sum_[1]*nVali_;//loss - val
			error_[2]=sqrt(error_sum_[2]*nBatchi_);//rmse - train
			error_[3]=sqrt(error_sum_[3]*nVali_);//rmse - val
			//compute gradient averaged over the batch
			obj_.g()*=nBatchi_;
			//compute the new position
			obj_.val()=error_[0];
			//compute step - adam
			gradq_=2.0*obj_.p();
			const double cfac=obj_.gamma()*sqrt(1.0-beta2i_)/(1.0-beta1i_);
			for(int n=0; n<nParams; ++n){
				//add to the running average of the gradients
				mgrad_[n]*=beta1_; mgrad_[n]+=(1.0-beta1_)*obj_.g()[n];
				//add to the running average of the square of the gradients
				mgrad2_[n]*=beta2_; mgrad2_[n]+=(1.0-beta2_)*obj_.g()[n]*obj_.g()[n];
				//calculate the new position
				obj_.p()[n]-=cfac*mgrad_[n]/(sqrt(mgrad2_[n])+eps_);//adam
				//obj_.p()[n]-=cfac*(beta1_*mgrad_[n]+(1.0-beta1_)*obj_.g()[n])/(sqrt(mgrad2_[n])+eps_);//nadam
			}
			const Eigen::VectorXd& tmp_=obj_.p();
			//communicate parameters - forward
			if(odd) MPI_Recv(wp1_.data(),wp1_.size(),MPI_DOUBLE,nb,0,WORLD.mpic(),MPI_STATUS_IGNORE);
			if(evn) MPI_Send(tmp_.data(),tmp_.size(),MPI_DOUBLE,nf,0,WORLD.mpic());
			if(evn) MPI_Recv(wp1_.data(),wp1_.size(),MPI_DOUBLE,nb,0,WORLD.mpic(),MPI_STATUS_IGNORE);
			if(odd) MPI_Send(tmp_.data(),tmp_.size(),MPI_DOUBLE,nf,0,WORLD.mpic());
			//communicate parameters - backward
			if(odd) MPI_Recv(wm1_.data(),wm1_.size(),MPI_DOUBLE,nf,0,WORLD.mpic(),MPI_STATUS_IGNORE);
			if(evn) MPI_Send(tmp_.data(),tmp_.size(),MPI_DOUBLE,nb,0,WORLD.mpic());
			if(evn) MPI_Recv(wm1_.data(),wm1_.size(),MPI_DOUBLE,nf,0,WORLD.mpic(),MPI_STATUS_IGNORE);
			if(odd) MPI_Send(tmp_.data(),tmp_.size(),MPI_DOUBLE,nb,0,WORLD.mpic());
			//compute quantum gradient
			gradq_.noalias()-=wm1_+wp1_;
			//compute step - quantum 
			const double qfac=cfac*rho_;
			for(int n=0; n<nParams; ++n){
				//add to the running average of the gradients
				mgradq_[n]*=beta1_; mgradq_[n]+=(1.0-beta1_)*gradq_[n];
				//add to the running average of the square of the gradients
				mgradq2_[n]*=beta2_; mgradq2_[n]+=(1.0-beta2_)*gradq_[n]*gradq_[n];
				//calculate the new position
				obj_.p()[n]-=qfac*mgradq_[n]/(sqrt(mgradq2_[n])+eps_);//adam
				//obj_.p()[n]-=qfac*(beta1_*mgradq_[n]+(1.0-beta1_)*gradq_[n])/(sqrt(mgradq2_[n])+eps_);//nadam
			}
			//update beta
			beta1i_*=beta1_;
			beta2i_*=beta2_;
			//print/write error
			if((unsigned int)iter_.step()%iter_.nPrint()==0){
				const int t=(unsigned int)iter/iter_.nPrint();
				step[t]=iter_.count();
				gamma[t]=obj_.gamma();
				loss_t[t]=error_[0];
				loss_v[t]=error_[1];
				rmse_t[t]=error_[2];
				rmse_v[t]=error_[3];
				if(WORLD.rank()==0) printf("%8i %12.10e %12.10e %12.10e %12.10e %12.10e\n",
					step[t],gamma[t],loss_t[t],loss_v[t],rmse_t[t],rmse_v[t]);
			}
			//write the basis and potentials
			if(iter_.step()%iter_.nWrite()==0){
				if(NNPTE_PRINT_STATUS>1) std::cout<<"writing the restart file and potentials\n";
				//write restart file
				const std::string file_restart=file_restart_+"_p"+std::to_string(BEAD.color())+".restart."+std::to_string(iter_.count());
				this->write_restart(file_restart.c_str());
				//write potential file
				const std::string file_ann=file_ann_+"_p"+std::to_string(BEAD.color())+"."+std::to_string(iter_.count());
				NNP::write(file_ann.c_str(),nnp_);
			}
			//compute new step
			obj_.gamma()=decay_.step(obj_,iter_);
		}
		//bcast parameters
		MPI_Bcast(obj_.p().data(),obj_.p().size(),MPI_DOUBLE,0,BEAD.mpic());
		//increment step
		++iter_.step();
		++iter_.count();
	}
	//compute the training time
	clock.end();
	double time_train=clock.duration();
	MPI_Allreduce(MPI_IN_PLACE,&time_train,1,MPI_DOUBLE,MPI_SUM,BEAD.mpic());
	time_train/=BEAD.size();
	MPI_Barrier(BEAD.mpic());
	
	//====== write the error ======
	if(BEAD.rank()==0){
		FILE* writer_error_=NULL;
		std::string file_error=file_error_+"_p"+std::to_string(BEAD.color())+".dat";
		if(!restart_){
			writer_error_=fopen(file_error.c_str(),"w");
			fprintf(writer_error_,"#STEP GAMMA LOSS_TRAIN LOSS_VAL RMSE_TRAIN RMSE_VAL\n");
		} else {
			writer_error_=fopen(file_error.c_str(),"a");
		}
		if(writer_error_==NULL) throw std::runtime_error("NNPQA::train(int): Could not open error record file.");
		for(int t=0; t<step.size(); ++t){
			fprintf(writer_error_,"%6i %12.10e %12.10e %12.10e %12.10e %12.10e\n",step[t],gamma[t],loss_t[t],loss_v[t],rmse_t[t],rmse_v[t]);
		}
		fclose(writer_error_);
		writer_error_=NULL;
	}
	
	//====== write the parameters ======
	if(BEAD.rank()==0 && wparams_){
		FILE* writer_p_=NULL;
		if(!restart_) writer_p_=fopen(file_params_.c_str(),"w");
		else writer_p_=fopen(file_params_.c_str(),"a");
		if(writer_p_==NULL) throw std::runtime_error("NNPTE::train(int): Could not open error record file.");
		for(int t=0; t<step.size(); ++t){
			for(int i=0; i<params[t].size(); ++i){
				fprintf(writer_p_,"%.12f ",params[t][i]);
			}
			fprintf(writer_p_,"\n");
		}
		fclose(writer_p_);
		writer_p_=NULL;
	}
	
	//====== unpack final parameters ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"packing final parameters into neural network\n";
	//unpack from global to per-element arrays
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		for(int m=0; m<pElement_[n].size(); ++m){
			pElement_[n][m]=obj_.p()[count];
			gElement_[n][m]=obj_.g()[count];
			++count;
		}
	}
	//pack from per-element arrays into neural networks
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	if(NNPTE_PRINT_DATA>-1 && WORLD.rank()==0){
		char* strbuf=new char[print::len_buf];
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TRAIN - SUMMARY",strbuf)<<"\n";
		std::cout<<"N-STEP = "<<iter_.step()<<"\n";
		std::cout<<"TIME   = "<<time_train<<"\n";
		if(NNPTE_PRINT_DATA>1){
			std::cout<<"p = "; for(int i=0; i<obj_.p().size(); ++i) std::cout<<obj_.p()[i]<<" "; std::cout<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		delete[] strbuf;
	}
}

void NNPTE::error(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNPTE::error(const Eigen::VectorXd&):\n";
	
	//====== reset the error ======
	error_[0]=0; //loss - training
	error_[1]=0; //loss - validation
	error_[2]=0; //rmse - training
	error_[3]=0; //rmse - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nTypes_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch\n";
	for(int i=0; i<batch_.size(); ++i) batch_[i]=batch_.data((batch_.count()++)%batch_.capacity());
	std::sort(batch_.elements(),batch_.elements()+batch_.size());
	if(batch_.count()>=batch_.capacity()){
		std::shuffle(batch_.data(),batch_.data()+batch_.capacity(),rngen_);
		MPI_Bcast(batch_.data(),batch_.capacity(),MPI_INT,0,BATCH.mpic());
		batch_.count()=0;
	}
	if(NNPTE_PRINT_DATA>1 && WORLD.rank()==0){std::cout<<"batch = "; for(int i=0; i<batch_.size(); ++i) std::cout<<batch_[i]<<" "; std::cout<<"\n";}
	
	//====== compute training error and gradient ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batch_.size(); ++i){
		const int ii=batch_[i];
		//**** compute the energy ****
		double pe=0;
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().fp(struc_train[ii].symm(m));
			//add the atom energy to the total
			pe+=nnp_.nnh(type).nn().out()[0];
		}
		//**** accumulate energy across BATCH communicator ****
		MPI_Allreduce(MPI_IN_PLACE,&pe,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		//**** compute the energy difference normalized by number of atoms ****
		const double norm=1.0/struc_train[ii].nAtoms();
		const double dE=(pe-struc_train[ii].pe())*norm;
		Eigen::VectorXd dcdo=Eigen::VectorXd::Constant(1,norm);
		switch(iter_.loss()){
			case opt::Loss::MSE:{
				error_[0]+=0.5*dE*dE;//loss - train
				dcdo*=dE;
			} break;
			case opt::Loss::MAE:{
				error_[0]+=std::fabs(dE);//loss - train
				dcdo*=math::special::sgn(dE);
			} break;
			case opt::Loss::HUBER:{
				const double arg=dE*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				error_[0]+=delta2_*(sqrtf-1.0);//loss - train
				dcdo*=dE/sqrtf;
			} break;
			case opt::Loss::ASINH:{
				const double arg=dE*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				error_[0]+=delta2_*(1.0-sqrtf+arg*logf);//loss - train
				dcdo*=logf*delta_;
			} break;
			default: break;
		}
		error_[2]+=dE*dE;//rmse - train
		//**** compute the gradient ****
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().fpbp(struc_train[ii].symm(m));
			//compute the gradient 
			gElement_[type].noalias()+=cost_[type].grad(nnp_.nnh(type).nn(),dcdo);
		}
	}
	
	//====== compute validation error ======
	if((unsigned int)iter_.step()%iter_.nPrint()==0 || (unsigned int)iter_.step()%iter_.nWrite()==0){
		if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error\n";
		for(int i=0; i<struc_val.size(); ++i){
			//**** compute the energy ****
			double pe=0;
			for(int n=0; n<dist_atomv[i].size(); ++n){
				//get the index of the atom within the local processor subset
				const int m=dist_atomv[i].index(n);
				//find the element index in the nn potential
				const int type=struc_val[i].type(m);
				//execute the network
				nnp_.nnh(type).nn().fp(struc_val[i].symm(m));
				//add the energy to the total
				pe+=nnp_.nnh(type).nn().out()[0];
			}
			//**** accumulate energy ****
			MPI_Allreduce(MPI_IN_PLACE,&pe,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
			//**** compute error ****
			const double dE=(pe-struc_val[i].pe())/struc_val[i].nAtoms();
			switch(iter_.loss()){
				case opt::Loss::MSE:{
					error_[1]+=0.5*dE*dE;//loss - val
				} break;
				case opt::Loss::MAE:{
					error_[1]+=std::fabs(dE);//loss - val
				} break;
				case opt::Loss::HUBER:{
					const double arg=dE*deltai_;
					error_[1]+=delta2_*(sqrt(1.0+arg*arg)-1.0);//loss - val
				} break;
				case opt::Loss::ASINH:{
					const double arg=dE*deltai_;
					const double sqrtf=sqrt(1.0+arg*arg);
					error_[1]+=delta2_*(1.0-sqrtf+arg*log(arg+sqrtf));//loss - val
				} break;
				default: break;
			}
			error_[3]+=dE*dE;//rmse - val
		}
	}
	
	//====== normalize w.r.t. batch size ======
	//note: we sum these quantities over WORLD, meaning that we are summing over duplicates in each BATCH
	//this normalization step corrects for this double counting
	const double batchsi=1.0/(1.0*BATCH.size());
	error_[0]*=batchsi;//loss - train
	error_[1]*=batchsi;//loss - validation
	error_[2]*=batchsi;//rmse - train
	error_[3]*=batchsi;//rmse - validation
}

void NNPTE::error2(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTE_PRINT_FUNC>0) std::cout<<"NNPTE::error(const Eigen::VectorXd&):\n";
	
	//====== reset the error ======
	error_[0]=0; //loss - training
	error_[1]=0; //loss - validation
	error_[2]=0; //rmse - training
	error_[3]=0; //rmse - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nTypes_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch\n";
	for(int i=0; i<batch_.size(); ++i) batch_[i]=batch_.data((batch_.count()++)%batch_.capacity());
	std::sort(batch_.elements(),batch_.elements()+batch_.size());
	if(batch_.count()>=batch_.capacity()){
		std::shuffle(batch_.data(),batch_.data()+batch_.capacity(),rngen_);
		MPI_Bcast(batch_.data(),batch_.capacity(),MPI_INT,0,BATCH.mpic());
		batch_.count()=0;
	}
	if(NNPTE_PRINT_DATA>1 && WORLD.rank()==0){std::cout<<"batch = "; for(int i=0; i<batch_.size(); ++i) std::cout<<batch_[i]<<" "; std::cout<<"\n";}
	
	//====== compute training error and gradient ======
	if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batch_.size(); ++i){
		//set batch
		const int ii=batch_[i];
		//reset gradients
		for(int n=0; n<nTypes_; ++n) grad_[n].setZero();
		//**** compute the energy ****
		double pe=0;
		const Eigen::VectorXd dcdo=Eigen::VectorXd::Constant(1,1);
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().fpbp(struc_train[ii].symm(m));
			//add the atom energy to the total
			pe+=nnp_.nnh(type).nn().out()[0];
			//compute gradient
			grad_[type].noalias()+=cost_[type].grad(nnp_.nnh(type).nn(),dcdo);
		}
		//**** accumulate energy across BATCH communicator ****
		MPI_Allreduce(MPI_IN_PLACE,&pe,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		//**** compute the energy difference normalized by number of atoms ****
		const double norm=1.0/struc_train[ii].nAtoms();
		const double dE=(pe-struc_train[ii].pe())*norm;
		//**** compute the error and parameter gradients ****
		double gfac=norm;
		switch(iter_.loss()){
			case opt::Loss::MSE:{
				error_[0]+=0.5*dE*dE;//loss - train
				gfac*=dE;
			} break;
			case opt::Loss::MAE:{
				error_[0]+=std::fabs(dE);//loss - train
				gfac*=math::special::sgn(dE);
			} break;
			case opt::Loss::HUBER:{
				const double arg=dE*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				error_[0]+=delta2_*(sqrtf-1.0);//loss - train
				gfac*=dE/sqrtf;
			} break;
			case opt::Loss::ASINH:{
				const double arg=dE*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				error_[0]+=delta2_*(1.0-sqrtf+arg*logf);//loss - train
				gfac*=logf*delta_;
			} break;
			default: break;
		}
		error_[2]+=dE*dE;//rmse - train
		//**** compute the gradient ****
		for(int j=0; j<nTypes_; ++j){
			gElement_[j].noalias()+=grad_[j]*gfac;
		}
	}
	
	//====== compute validation error ======
	if((unsigned int)iter_.step()%iter_.nPrint()==0 || (unsigned int)iter_.step()%iter_.nWrite()==0){
		if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error\n";
		for(int i=0; i<struc_val.size(); ++i){
			//**** compute the energy ****
			double pe=0;
			for(int n=0; n<dist_atomv[i].size(); ++n){
				//get the index of the atom within the local processor subset
				const int m=dist_atomv[i].index(n);
				//find the element index in the nn potential
				const int type=struc_val[i].type(m);
				//execute the network
				nnp_.nnh(type).nn().fp(struc_val[i].symm(m));
				//add the energy to the total
				pe+=nnp_.nnh(type).nn().out()[0];
			}
			//**** accumulate energy ****
			MPI_Allreduce(MPI_IN_PLACE,&pe,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
			//**** compute error ****
			const double dE=(pe-struc_val[i].pe())/struc_val[i].nAtoms();
			switch(iter_.loss()){
				case opt::Loss::MSE:{
					error_[1]+=0.5*dE*dE;//loss - val
				} break;
				case opt::Loss::MAE:{
					error_[1]+=std::fabs(dE);//loss - val
				} break;
				case opt::Loss::HUBER:{
					const double arg=dE*deltai_;
					error_[1]+=delta2_*(sqrt(1.0+arg*arg)-1.0);//loss - val
				} break;
				case opt::Loss::ASINH:{
					const double arg=dE*deltai_;
					const double sqrtf=sqrt(1.0+arg*arg);
					error_[1]+=delta2_*(1.0-sqrtf+arg*log(arg+sqrtf));//loss - val
				} break;
				default: break;
			}
			error_[3]+=dE*dE;//rmse - val
		}
	}
	
	//====== normalize w.r.t. batch size ======
	//note: we sum these quantities over WORLD, meaning that we are summing over duplicates in each BATCH
	//this normalization step corrects for this double counting
	const double batchsi=1.0/(1.0*BATCH.size());
	error_[0]*=batchsi;//loss - train
	error_[1]*=batchsi;//loss - validation
	error_[2]*=batchsi;//rmse - train
	error_[3]*=batchsi;//rmse - validation
}

void NNPTE::read(const char* file, NNPTE& nnpte){
	if(NN_PRINT_FUNC>0) std::cout<<"NNPTE::read(const char*,NNPTE&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		NNPTE::read(reader,nnpte);
		std::fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for reading.\n"));
}

void NNPTE::read(FILE* reader, NNPTE& nnpte){
	//==== local variables ====
	char* input=new char[string::M];
	Token token;
	//==== rewind reader ====
	std::rewind(reader);
	//==== read parameters ====
	while(fgets(input,string::M,reader)!=NULL){
		token.read(string::trim_right(input,string::COMMENT),string::WS);
		if(token.end()) continue;//skip empty lines
		const std::string tag=string::to_upper(token.next());
		//nnp
		if(tag=="R_CUT"){
			nnpte.nnp().rc()=std::atof(token.next().c_str());
		}
		//files
		if(tag=="FILE_ERROR"){
			nnpte.file_error()=token.next();
		} else if(tag=="FILE_PARAMS"){
			nnpte.file_params()=token.next();
		} else if(tag=="FILE_ANN"){
			nnpte.file_ann()=token.next();
		} else if(tag=="FILE_RESTART"){
			nnpte.file_restart()=token.next();
		}
		//flags
		if(tag=="RESTART"){//read restart file
			nnpte.restart()=string::boolean(token.next().c_str());//restarting
		} else if(tag=="RESET"){//whether to precondition the inputs
			nnpte.reset()=string::boolean(token.next().c_str());
		} else if(tag=="WRITE_PARAMS"){
			nnpte.wparams()=string::boolean(token.next().c_str());
		} 
		//optimization
		if(tag=="LOSS"){
			nnpte.iter().loss()=opt::Loss::read(string::to_upper(token.next()).c_str());
		} else if(tag=="STOP"){
			nnpte.iter().stop()=opt::Stop::read(string::to_upper(token.next()).c_str());
		} else if(tag=="MAX_ITER"){
			nnpte.iter().max()=std::atoi(token.next().c_str());
		} else if(tag=="N_PRINT"){
			nnpte.iter().nPrint()=std::atoi(token.next().c_str());
		} else if(tag=="N_WRITE"){
			nnpte.iter().nWrite()=std::atoi(token.next().c_str());
		} else if(tag=="TOL"){
			nnpte.iter().tol()=std::atof(token.next().c_str());
		} else if(tag=="GAMMA"){
			nnpte.obj().gamma()=std::atof(token.next().c_str());
		} else if(tag=="DECAY"){
			nnpte.decay().read(token);
		} else if(tag=="DELTA"){
			nnpte.delta()=std::atof(token.next().c_str());
			nnpte.deltai()=1.0/nnpte.delta();
			nnpte.delta2()=nnpte.delta()*nnpte.delta();
		} else if(tag=="PRESCALE"){
			nnpte.prescale()=PreScale::read(string::to_upper(token.next()).c_str());
		} else if(tag=="PREBIAS"){
			nnpte.prebias()=PreBias::read(string::to_upper(token.next()).c_str());
		} else if(tag=="INSCALE"){
			nnpte.inscale()=std::atof(token.next().c_str());
		} else if(tag=="INBIAS"){
			nnpte.inbias()=std::atof(token.next().c_str());
		} 
		//quantum adam
		if(tag=="RHO"){
			nnpte.rho()=std::atof(token.next().c_str());
		} else if(tag=="EPS"){
			nnpte.eps()=std::atof(token.next().c_str());
		} 
	}
	//==== free local variables ====
	delete[] input;
}

//************************************************************
// MAIN
//************************************************************

int main(int argc, char* argv[]){
	//======== global variables ========
	//units
		units::System unitsys=units::System::UNKNOWN;
	//mode
		Mode mode=Mode::TRAIN;
	//atom format
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=false; atomT.symm=true; atomT.charge=false;
	//flags - compute
		struct Compute{
			bool coul=false;  //compute - external potential - coul
			bool vdw=false;   //compute - external potential - vdw
			bool rep=false;   //compute - external potential - repulsive
			bool force=false; //compute - forces
			bool norm=false;  //compute - energy normalization
			bool zero=false;  //compute - zero point energy
		} compute;
	//flags - writing
		struct Write{
			bool coul=false;   //write - energy - coulomb
			bool vdw=false;    //write - energy - vdw
			bool rep=false;    //write - energy - repulsive
			bool force=false;  //write - force
			bool energy=false; //write - energy
			bool input=false;  //write - inputs
		} write;
	//external potentials
		ptnl::PotGaussLong pot_coul;
		ptnl::PotLDampLong pot_vdw;
		ptnl::PotPauli pot_rep;
	//nn potential - opt
		int nBatch=-1;
		std::vector<Type> types;//unique atomic species
		NNPTE nnpte;//nn potential optimization data
		std::vector<std::vector<int> > nh;//hidden layer configuration
		NN::ANNP annp;//neural network initialization parameters
	//data names
		static const char* const dnames[] = {"TRAINING","VALIDATION","TESTING"};
	//structures - format
		FILE_FORMAT::type format;//format of training data
	//structures - data
		std::vector<int> nstrucs(3,0);
		std::vector<std::vector<std::string> > data(3); //data files
		std::vector<std::vector<std::string> > files(3); //structure files
		std::vector<std::vector<Structure> > strucs(3); //structures
		std::vector<std::vector<int> > indices(3);
		std::vector<Alias> aliases;
	//mpi data distribution
		std::vector<thread::Dist> dist(4);
	//timing
		Clock clock,clock_wall; //time objects
		double time_wall=0;     //total wall time
		std::vector<double> time_energy(3,0.0);
		std::vector<double> time_force(3,0.0);
		std::vector<double> time_symm(3,0.0);
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];//string - input
		char* strbuf=new char[print::len_buf];//string - buffer
		std::string ifile;//file - input
		std::vector<input::Arg> args;
		bool read_pot=false;
		std::string file_pot;
		std::vector<std::string> files_basis;//file - stores basis
	//string
		Token token;
	//beads
		int nbeads=1;
	
	try{
		//************************************************************************************
		// LOADING/INITIALIZATION
		//************************************************************************************
		
		//======== initialize mpi ========
		MPI_Init(&argc,&argv);
		WORLD.mpic()=MPI_COMM_WORLD;
		MPI_Comm_size(WORLD.mpic(),&WORLD.size());
		MPI_Comm_rank(WORLD.mpic(),&WORLD.rank());
		
		//======== start wall clock ========
		if(WORLD.rank()==0) clock_wall.begin();
		
		//======== print title ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::title("ATOMIC NEURAL NETWORK",strbuf,' ')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
		}
		
		//======== print compiler information ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("COMPILER",strbuf)<<"\n";
			std::cout<<"date     = "<<compiler::date()<<"\n";
			std::cout<<"time     = "<<compiler::time()<<"\n";
			std::cout<<"compiler = "<<compiler::name()<<"\n";
			std::cout<<"version  = "<<compiler::version()<<"\n";
			std::cout<<"standard = "<<compiler::standard()<<"\n";
			std::cout<<"arch     = "<<compiler::arch()<<"\n";
			std::cout<<"instr    = "<<compiler::instr()<<"\n";
			std::cout<<"os       = "<<compiler::os()<<"\n";
			std::cout<<"omp      = "<<compiler::omp()<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print mathematical constants ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::printf("PI    = %.15f\n",math::constant::PI);
			std::printf("RadPI = %.15f\n",math::constant::RadPI);
			std::printf("Rad2  = %.15f\n",math::constant::Rad2);
			std::printf("Log2  = %.15f\n",math::constant::LOG2);
			std::printf("Eps<D> = %.15e\n",std::numeric_limits<double>::epsilon());
			std::printf("Min<D> = %.15e\n",std::numeric_limits<double>::min());
			std::printf("Max<D> = %.15e\n",std::numeric_limits<double>::max());
			std::printf("Min<I> = %i\n",std::numeric_limits<int>::min());
			std::printf("Max<I> = %i\n",std::numeric_limits<int>::max());
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print physical constants ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			//std::printf("bohr-r  (A)  = %.12f\n",units::BOHR);
			//std::printf("hartree (eV) = %.12f\n",units::HARTREE);
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== set mpi data ========
		{
			int* ranks=new int[WORLD.size()];
			MPI_Gather(&WORLD.rank(),1,MPI_INT,ranks,1,MPI_INT,0,WORLD.mpic());
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MPI",strbuf)<<"\n";
				std::cout<<"world - size = "<<WORLD.size()<<"\n"<<std::flush;
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<std::flush;
			}
			delete[] ranks;
		}
		
		//======== parse the input ========
		if(WORLD.rank()==0){
			if(NNPTE_PRINT_STATUS>0) std::cout<<"parsing input\n";
			input::parse(argc,argv,args);
			for(int i=0; i<args.size(); ++i){
				if(args[i].key()=="in") ifile=args[i].val(0);
				if(args[i].key()=="nbeads") nbeads=std::atoi(args[i].val(0).c_str());
			}
			if(nbeads<=0) throw std::invalid_argument("Invalid number of beads.");
			if(WORLD.size()%nbeads!=0) throw std::invalid_argument("main(int,char**): invalid number of beads, must divide proc number.");
		}
		//bcast input
		thread::bcast(WORLD.mpic(),0,ifile);
		MPI_Bcast(&nbeads,1,MPI_INT,0,WORLD.mpic());
		
		//======== set the bead groups ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"setting the bead groups\n";
		BEAD=WORLD.split(WORLD.color(WORLD.ncomm(nbeads)));
		const int rank_bead=BEAD.rank();
		const int rank_world=WORLD.rank();
		std::vector<int> ranks_bead(WORLD.size(),-1);
		std::vector<int> ranks_world(WORLD.size(),-1);
		MPI_Allgather(&rank_bead,1,MPI_INT,ranks_bead.data(),1,MPI_INT,WORLD.mpic());
		MPI_Allgather(&rank_world,1,MPI_INT,ranks_world.data(),1,MPI_INT,WORLD.mpic());
		std::vector<int> roots_bead;
		for(int i=0; i<WORLD.size(); ++i){
			if(ranks_bead[i]==0) roots_bead.push_back(ranks_world[i]);
		}
		if(roots_bead.size()!=nbeads) throw std::runtime_error("main(int,char**): invalid number of head ranks, does not match number of beads.");
		if(WORLD.rank()==0){
			std::cout<<std::flush;
			std::cout<<"ranks_world = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<ranks_world[i]<<" "; std::cout<<"\n";
			std::cout<<"ranks_bead  = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<ranks_bead[i]<<" "; std::cout<<"\n";
			std::cout<<"roots_bead  = "; for(int i=0; i<roots_bead.size(); ++i) std::cout<<roots_bead[i]<<" "; std::cout<<"\n";
			std::cout<<std::flush;
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== rank 0 reads from file ========
		if(BEAD.rank()==0){
			
			//======== open the par && WORLD.rank()==0ameter file ========
			if(NNPTE_PRINT_STATUS>0) std::cout<<"opening parameter file\n";
			reader=fopen(ifile.c_str(),"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+ifile);
			
			//======== read in the parameters ========
			if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL){
				token.read(string::trim_right(input,string::COMMENT),string::WS);
				if(token.end()) continue;//skip empty line
				const std::string tag=string::to_upper(token.next());
				std::cout<<"tag = "<<tag<<"\n";
				//general
				if(tag=="UNITS"){//units
					unitsys=units::System::read(string::to_upper(token.next()).c_str());
				} else if(tag=="FORMAT"){//simulation format
					format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
				} 
				//data and execution mode
				if(tag=="MODE"){//mode of calculation
					mode=Mode::read(string::to_upper(token.next()).c_str());
				} else if(tag=="DATA_TRAIN"){//data - training
					data[0].push_back(token.next());
				} else if(tag=="DATA_VAL"){//data - validation
					data[1].push_back(token.next());
				} else if(tag=="DATA_TEST"){//data - testing
					data[2].push_back(token.next());
				}
				//atom
				if(tag=="ATOM"){//atom - name/mass/energy
					//process the string
					const std::string name=token.next();
					const std::string atomtag=string::to_upper(token.next());
					const int id=string::hash(name);
					//look for the atom name in the existing list of atom names
					int index=-1;
					for(int i=0; i<types.size(); ++i){
						if(name==types[i].name()){index=i;break;}
					}
					//if atom is not found, add it
					if(index<0){
						index=types.size();
						types.push_back(Type());
						types.back().name()=name;
						types.back().id()=id;
						files_basis.resize(files_basis.size()+1);
						nh.resize(nh.size()+1);
					}
					//set tag value
					if(atomtag=="MASS"){
						types[index].mass().flag()=true;
						types[index].mass().val()=std::atof(token.next().c_str());
					} else if(atomtag=="CHARGE"){
						types[index].charge().flag()=true;
						types[index].charge().val()=std::atof(token.next().c_str());
						atomT.charge=true;
					} else if(atomtag=="CHI"){
						types[index].chi().flag()=true;
						types[index].chi().val()=std::atof(token.next().c_str());
						atomT.chi=true;
					} else if(atomtag=="ETA"){
						types[index].eta().flag()=true;
						types[index].eta().val()=std::atof(token.next().c_str());
						atomT.eta=true;
					} else if(atomtag=="ENERGY"){
						types[index].energy().flag()=true;
						types[index].energy().val()=std::atof(token.next().c_str());
					} else if(atomtag=="RVDW"){
						types[index].rvdw().flag()=true;
						types[index].rvdw().val()=std::atof(token.next().c_str());
					} else if(atomtag=="RCOV"){
						types[index].rcov().flag()=true;
						types[index].rcov().val()=std::atof(token.next().c_str());
					} else if(atomtag=="C6"){
						types[index].c6().flag()=true;
						types[index].c6().val()=std::atof(token.next().c_str());
					} else if(atomtag=="BASIS"){
						files_basis[index]=token.next();
					} else if(atomtag=="NH"){
						nh[index].clear();
						while(!token.end()) nh[index].push_back(std::atoi(token.next().c_str()));
					}
				} 
				//neural network potential
				if(tag=="READ_POT"){
					file_pot=token.next();
					read_pot=true;
				}
				//batch
				if(tag=="N_BATCH"){//size of the batch
					nBatch=std::atoi(token.next().c_str());
				} 
				//flags - writing
				if(tag=="WRITE"){
					const std::string wtype=string::to_upper(token.next());
					if(wtype=="COUL") write.coul=string::boolean(token.next().c_str());
					else if(wtype=="VDW") write.vdw=string::boolean(token.next().c_str());
					else if(wtype=="REP") write.rep=string::boolean(token.next().c_str());
					else if(wtype=="FORCE") write.force=string::boolean(token.next().c_str());
					else if(wtype=="ENERGY") write.energy=string::boolean(token.next().c_str());
					else if(wtype=="INPUT") write.input=string::boolean(token.next().c_str());
				}
				//flags - compute
				if(tag=="COMPUTE"){
					const std::string ctype=string::to_upper(token.next());
					if(ctype=="COUL") compute.coul=string::boolean(token.next().c_str());
					else if(ctype=="VDW") compute.vdw=string::boolean(token.next().c_str());
					else if(ctype=="REP") compute.rep=string::boolean(token.next().c_str());
					else if(ctype=="FORCE") compute.force=string::boolean(token.next().c_str());
					else if(ctype=="NORM") compute.norm=string::boolean(token.next().c_str());
					else if(ctype=="ZERO") compute.zero=string::boolean(token.next().c_str());
				}
				//potential 
				if(tag=="POT_COUL"){
					token.next();
					pot_coul.read(token);
				} else if(tag=="POT_VDW"){
					token.next();
					pot_vdw.read(token);
				} else if(tag=="POT_REP"){
					token.next();
					pot_rep.read(token);
				}
				//alias
				if(tag=="ALIAS"){
					aliases.push_back(Alias());
					Alias::read(token,aliases.back());
				}
			}
			
			//======== set atom flags =========
			if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting atom flags\n";
			atomT.force=compute.force;
			atomT.charge=compute.coul;
			
			//======== read - nnpte =========
			if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"reading neural network training parameters\n";
			NNPTE::read(reader,nnpte);
			nnpte.roots_bead()=roots_bead;
			
			//======== read - annp =========
			if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"reading neural network parameters\n";
			NN::ANNP::read(reader,annp);
			
			//======== close parameter file ========
			if(NNPTE_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"closing parameter file\n";
			fclose(reader);
			reader=NULL;
			
			//======== (restart == false) ========
			if(!nnpte.restart_){
				//======== (read potential == false) ========
				if(!read_pot){
					//resize the potential
					if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"resizing potential\n";
					nnpte.nnp().resize(types);
					//read basis files
					if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"reading basis files\n";
					if(files_basis.size()!=nnpte.nnp().ntypes()) throw std::runtime_error("main(int,char**): invalid number of basis files.");
					for(int i=0; i<nnpte.nnp().ntypes(); ++i){
						const char* file=files_basis[i].c_str();
						const char* atomName=types[i].name().c_str();
						NNP::read_basis(file,nnpte.nnp(),atomName);
					}
					//initialize the neural network hamiltonians
					if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing neural network hamiltonians\n";
					for(int i=0; i<nnpte.nnp().ntypes(); ++i){
						NNH& nnhl=nnpte.nnp().nnh(i);
						nnhl.type()=types[i];
						nnhl.nn().resize(annp,nnhl.nInput(),nh[i],1);
						nnhl.dOdZ().resize(nnhl.nn());
					}
				}
				//======== (read potential == true) ========
				if(read_pot){
					if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"reading potential\n";
					NNP::read(file_pot.c_str(),nnpte.nnp());
				}
			}
			//======== (restart == true) ========
			if(nnpte.restart_){
				if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"reading restart file\n";
				const std::string file=nnpte.file_restart_+"_p"+std::to_string(BEAD.color())+".restart";
				nnpte.read_restart(file.c_str());
				nnpte.restart()=true;
			}
			
			//========= check the data =========
			if(mode==Mode::TRAIN && data[0].size()==0) throw std::invalid_argument("No data provided - training.");
			if(mode==Mode::TRAIN && data[1].size()==0) throw std::invalid_argument("No data provided - validation.");
			if(mode==Mode::TEST  && data[2].size()==0) throw std::invalid_argument("No data provided - testing.");
			if(mode==Mode::UNKNOWN) throw std::invalid_argument("Invalid calculation mode");
			if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			if(types.size()==0) throw std::invalid_argument("Invalid number of types.");
		
		}
		
		if(WORLD.rank()==0){
			//======== print parameters ========
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"nbeads    = "<<nbeads<<"\n";
			std::cout<<"read_pot  = "<<read_pot<<"\n";
			std::cout<<"atom_type = "<<atomT<<"\n";
			std::cout<<"format    = "<<format<<"\n";
			std::cout<<"units     = "<<unitsys<<"\n";
			std::cout<<"mode      = "<<mode<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DATA FILES",strbuf)<<"\n";
			std::cout<<"data_train = \n"; for(int i=0; i<data[0].size(); ++i) std::cout<<"\t\t"<<data[0][i]<<"\n";
			std::cout<<"data_val   = \n"; for(int i=0; i<data[1].size(); ++i) std::cout<<"\t\t"<<data[1][i]<<"\n";
			std::cout<<"data_test  = \n"; for(int i=0; i<data[2].size(); ++i) std::cout<<"\t\t"<<data[2][i]<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("COMPUTE FLAGS",strbuf)<<"\n";
			std::cout<<"coul  = "<<compute.coul<<"\n";
			std::cout<<"vdw   = "<<compute.vdw<<"\n";
			std::cout<<"rep   = "<<compute.rep<<"\n";
			std::cout<<"force = "<<compute.force<<"\n";
			std::cout<<"norm  = "<<compute.norm<<"\n";
			std::cout<<"zero  = "<<compute.zero<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("WRITE FLAGS",strbuf)<<"\n";
			std::cout<<"coul   = "<<write.coul<<"\n";
			std::cout<<"vdw    = "<<write.vdw<<"\n";
			std::cout<<"rep    = "<<write.rep<<"\n";
			std::cout<<"force  = "<<write.force<<"\n";
			std::cout<<"energy = "<<write.energy<<"\n";
			std::cout<<"inputs = "<<write.input<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("EXTERNAL POTENTIAL",strbuf)<<"\n";
			if(compute.coul) std::cout<<"COUL = "<<pot_coul<<"\n";
			if(compute.vdw)  std::cout<<"VDW  = "<<pot_vdw<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("TYPES",strbuf)<<"\n";
			for(int i=0; i<types.size(); ++i){
				std::cout<<types[i]<<"\n";
			}
			std::cout<<print::title("ALIAS",strbuf)<<"\n";
			for(int i=0; i<aliases.size(); ++i){
				std::cout<<aliases[i]<<"\n";
			}
			std::cout<<annp<<"\n";
			std::cout<<nnpte<<"\n";
			std::cout<<nnpte.nnp()<<"\n";
			
		}
		
		//======== bcast the parameters ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		//general parameters
		MPI_Bcast(&unitsys,1,MPI_INT,0,BEAD.mpic());
		//mode
		MPI_Bcast(&mode,1,MPI_INT,0,BEAD.mpic());
		//atom type
		thread::bcast(BEAD.mpic(),0,atomT);
		thread::bcast(BEAD.mpic(),0,annp);
		//flags - compute
		MPI_Bcast(&compute.coul,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.vdw,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.rep,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.norm,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.zero,1,MPI_C_BOOL,0,WORLD.mpic());
		//flags - writing
		MPI_Bcast(&write.coul,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.vdw,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.rep,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.energy,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.input,1,MPI_C_BOOL,0,WORLD.mpic());
		//external potential
		if(compute.coul) thread::bcast(BEAD.mpic(),0,pot_coul);
		if(compute.vdw) thread::bcast(BEAD.mpic(),0,pot_vdw);
		//batch
		MPI_Bcast(&nBatch,1,MPI_INT,0,BEAD.mpic());
		//structures - format
		MPI_Bcast(&format,1,MPI_INT,0,BEAD.mpic());
		//nnpte
		thread::bcast(BEAD.mpic(),0,nnpte);
		//alias
		int naliases=aliases.size();
		MPI_Bcast(&naliases,1,MPI_INT,0,BEAD.mpic());
		if(BEAD.rank()!=0) aliases.resize(naliases);
		for(int i=0; i<aliases.size(); ++i){
			thread::bcast(BEAD.mpic(),0,aliases[i]);
		}
		
		//======== set the unit system ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"setting the unit system\n";
		units::Consts::init(unitsys);
		
		//************************************************************************************
		// READ DATA
		//************************************************************************************
		
		//======== rank 0 reads the data files (lists of structure files) ========
		if(BEAD.rank()==0){
			if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"reading data\n";
			//==== read data ====
			for(int n=0; n<3; ++n){
				for(int i=0; i<data[n].size(); ++i){
					//open the data file
					if(NNPTE_PRINT_DATA>0) std::cout<<"data["<<n<<"]["<<i<<"]: "<<data[n][i]<<"\n";
					reader=fopen(data[n][i].c_str(),"r");
					if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data[n][i]);
					//read in the data
					while(fgets(input,string::M,reader)!=NULL){
						if(!string::empty(input)) files[n].push_back(std::string(string::trim(input)));
					}
					//close the file
					fclose(reader); reader=NULL;
				}
			}
			//==== print the files ====
			if(NNPTE_PRINT_DATA>1){
				for(int n=0; n<3; ++n){
					if(files[n].size()>0){
						std::cout<<print::buf(strbuf)<<"\n";
						std::cout<<print::title("FILES",strbuf)<<"\n";
						for(int i=0; i<files[n].size(); ++i) std::cout<<"\t"<<files[n][i]<<"\n";
						std::cout<<print::buf(strbuf)<<"\n";
					}
				}
			}
		}
		
		//======== bcast the file names =======
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"bcasting file names\n";
		//bcast names
		for(int n=0; n<3; ++n) thread::bcast(BEAD.mpic(),0,files[n]);
		//set number of structures
		for(int n=0; n<3; ++n) nstrucs[n]=files[n].size();
		//print number of structures
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DATA - SIZE",strbuf)<<"\n";
			std::cout<<"ntrain = "<<nstrucs[0]<<"\n";
			std::cout<<"nval   = "<<nstrucs[1]<<"\n";
			std::cout<<"ntest  = "<<nstrucs[2]<<"\n";
			std::cout<<"nbatch = "<<nBatch<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		//check the batch size
		if(nBatch<=0) throw std::invalid_argument("Invalid batch size.");
		if(nBatch>nstrucs[0]) throw std::invalid_argument("Invalid batch size.");
		
		//======== initializing batch communicator ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing batch communicator\n";
		//split BEAD into BATCH
		BATCH=BEAD.split(BEAD.color(BEAD.ncomm(nBatch)));
		//print batch communicators
		{
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("BATCH COMMUNICATORS",strbuf)<<"\n";
				std::cout<<std::flush;
			}
			MPI_Barrier(WORLD.mpic());
			const int sizeb=serialize::nbytes(BATCH);
			const int sizet=WORLD.size()*serialize::nbytes(BATCH);
			char* arrb=new char[sizeb];
			char* arrt=new char[sizet];
			serialize::pack(BATCH,arrb);
			MPI_Gather(arrb,sizeb,MPI_CHAR,arrt,sizeb,MPI_CHAR,0,WORLD.mpic());
			if(WORLD.rank()==0){
				for(int i=0; i<WORLD.size(); ++i){
					thread::Comm tmp;
					serialize::unpack(tmp,arrt+i*sizeb);
					std::cout<<"BATCH["<<i<<"] = "<<tmp<<"\n";
				}
			}
			delete[] arrb;
			delete[] arrt;
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<std::flush;
			}
			MPI_Barrier(WORLD.mpic());
		}
		
		//======== generate thread distributions ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"generating thread distributions\n";
		//thread dist - divide structures equally among the batch groups
		dist[0].init(BATCH.ncomm(),BATCH.color(),nstrucs[0]);//train
		dist[1].init(BATCH.ncomm(),BATCH.color(),nstrucs[1]);//validation
		dist[2].init(BATCH.ncomm(),BATCH.color(),nstrucs[2]);//test
		dist[3].init(BATCH.ncomm(),BATCH.color(),nBatch);//batch
		//print
		if(WORLD.rank()==0){
			std::string str;
			std::cout<<"thread_dist_train   = "<<thread::Dist::size(str,BATCH.ncomm(),nstrucs[0])<<"\n";
			std::cout<<"thread_dist_val     = "<<thread::Dist::size(str,BATCH.ncomm(),nstrucs[1])<<"\n";
			std::cout<<"thread_dist_test    = "<<thread::Dist::size(str,BATCH.ncomm(),nstrucs[2])<<"\n";
			std::cout<<"thread_dist_batch   = "<<thread::Dist::size(str,BATCH.ncomm(),nBatch)<<"\n";
			std::cout<<"thread_offset_train = "<<thread::Dist::offset(str,BATCH.ncomm(),nstrucs[0])<<"\n";
			std::cout<<"thread_offset_val   = "<<thread::Dist::offset(str,BATCH.ncomm(),nstrucs[1])<<"\n";
			std::cout<<"thread_offset_test  = "<<thread::Dist::offset(str,BATCH.ncomm(),nstrucs[2])<<"\n";
			std::cout<<"thread_offset_batch = "<<thread::Dist::offset(str,BATCH.ncomm(),nBatch)<<"\n";
		}
		
		//======== gen indices (random shuffle) ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"generating indices\n";
		for(int n=0; n<3; ++n){
			indices[n].resize(nstrucs[n],-1);
			for(int i=0; i<indices[n].size(); ++i) indices[n][i]=i;
			std::random_shuffle(indices[n].begin(),indices[n].end());
			thread::bcast(BEAD.mpic(),0,indices[n]);
		}
		
		//======== read the structures ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"reading structures\n";
		for(int n=0; n<3; ++n){
			if(files[n].size()>0){
				//resize structure array
				strucs[n].resize(dist[n].size());
				//rank 0 of batch group reads structures
				if(BATCH.rank()==0){
					for(int i=0; i<dist[n].size(); ++i){
						const std::string& file=files[n][indices[n][dist[n].index(i)]];
						read_struc(file.c_str(),format,atomT,strucs[n][i]);
						if(NNPTE_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<strucs[n][i].pe()<<"\n";
					}
				}
				//broadcast structures to all other procs in the BATCH group
				for(int i=0; i<dist[n].size(); ++i){
					thread::bcast(BATCH.mpic(),0,strucs[n][i]);
				}
			}
		}
		
		//======== apply alias ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"applying aliases\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				Structure& strucl=strucs[n][i];
				for(int j=0; j<strucl.nAtoms(); ++j){
					for(int k=0; k<aliases.size(); ++k){
						for(int l=0; l<aliases[k].labels().size(); ++l){
							if(strucl.name(j)==aliases[k].labels()[l]){
								strucl.name(j)=aliases[k].alias();
							}
						}
					}
				}
			}
		}
		
		//======== check the structures ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"checking the structures\n";
		if(BATCH.rank()==0){
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					const std::string filename=files[n][indices[n][dist[n].index(i)]];
					const Structure& strucl=strucs[n][i];
					if(strucl.nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
					if(std::isinf(strucl.pe())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
					if(strucl.pe()!=strucl.pe()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
					if(std::fabs(strucl.pe())<math::constant::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
					if(compute.force){
						for(int j=0; j<strucl.nAtoms(); ++j){
							const double force=strucl.force(j).squaredNorm();
							if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<strucl.name(j)<<strucl.index(j)<<"\" in \""<<filename<<" has inf force.\n";
							if(force!=force) std::cout<<"WARNING: Atom \""<<strucl.name(j)<<strucl.index(j)<<"\" in \""<<filename<<" has nan force.\n";
						}
					}
					if(NNPTE_PRINT_DATA>1) std::cout<<"\t"<<filename<<" "<<strucl.pe()<<" "<<BEAD.rank()<<"\n";
				}
			}
		}
		MPI_Barrier(WORLD.mpic());
		
		//************************************************************************************
		// ATOM PROPERTIES
		//************************************************************************************
		
		//======== set atom properties ========
		if(WORLD.rank()==0) std::cout<<"setting atomic properties\n";
		
		//======== set the indices ========
		if(WORLD.rank()==0) std::cout<<"setting the indices\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].index(j)=j;
				}
			}
		}
		
		//======== set the types ========
		if(WORLD.rank()==0) std::cout<<"setting the types\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].type(j)=nnpte.nnp_.index(strucs[n][i].name(j));
				}
			}
		}
		
		//======== set the charges ========
		if(atomT.charge){
			if(WORLD.rank()==0) std::cout<<"setting charges\n";
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					for(int j=0; j<strucs[n][i].nAtoms(); ++j){
						strucs[n][i].charge(j)=nnpte.nnp_.nnh(strucs[n][i].type(j)).type().charge().val();
					}
				}
			}
		}
		
		//======== set the electronegativities ========
		if(atomT.chi){
			if(WORLD.rank()==0) std::cout<<"setting electronegativities\n";
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					for(int j=0; j<strucs[n][i].nAtoms(); ++j){
						strucs[n][i].chi(j)=nnpte.nnp_.nnh(strucs[n][i].type(j)).type().chi().val();
					}
				}
			}
		}
		
		//======== set the idempotentials ========
		if(atomT.eta){
			if(WORLD.rank()==0) std::cout<<"setting idempotentials\n";
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					for(int j=0; j<strucs[n][i].nAtoms(); ++j){
						strucs[n][i].eta(j)=nnpte.nnp_.nnh(strucs[n][i].type(j)).type().eta().val();
					}
				}
			}
		}
		
		//************************************************************************************
		// EXTERNAL POTENTIALS
		//************************************************************************************
		
		if(compute.coul && compute.vdw) throw std::invalid_argument("Can't have charge and vdw interactions.");
		
		//======== compute coulomb energies ========
		if(compute.coul){
			if(WORLD.rank()==0) std::cout<<"computing coulomb energies\n";
			for(int n=0; n<3; ++n){
				std::vector<double> ecoul(dist[n].size(),std::numeric_limits<double>::max());
				for(int i=BATCH.rank(); i<dist[n].size(); i+=BATCH.size()){
					NeighborList nlist(strucs[n][i],pot_coul.rc());
					ecoul[i]=pot_coul.energy(strucs[n][i],nlist);
				}
				MPI_Allreduce(MPI_IN_PLACE,ecoul.data(),ecoul.size(),MPI_DOUBLE,MPI_MIN,BATCH.mpic());
				for(int i=0; i<dist[n].size(); ++i){
					strucs[n][i].ecoul()=ecoul[i];
					strucs[n][i].pe()-=ecoul[i];
				}
			}
		}
		
		//======== compute vdw energies ========
		if(compute.vdw){
			if(WORLD.rank()==0) std::cout<<"computing vdw energies\n";
			Reduce<1> ralpha,rerrer,rerrek;
			Reduce<1> rNK;
			std::vector<Reduce<1> > rnk(3);
			//compute parameters
			pot_vdw.resize(nnpte.nnp().ntypes());
			pot_vdw.ksl().rc()=pot_vdw.rc();
			pot_vdw.ksl().prec()=pot_vdw.prec();
			for(int i=0; i<nnpte.nnp().ntypes(); ++i){
				const double ri=nnpte.nnp().nnh(i).type().rvdw().val();
				const double ci=nnpte.nnp().nnh(i).type().c6().val();
				pot_vdw.rvdw()(i,i)=ri;
				pot_vdw.c6()(i,i)=ci;
			}
			pot_vdw.init();
			if(WORLD.rank()==0){
				for(int i=0; i<nnpte.nnp().ntypes(); ++i){
					const std::string ni=nnpte.nnp().nnh(i).type().name();
					for(int j=0; j<nnpte.nnp().ntypes(); ++j){
						const std::string nj=nnpte.nnp().nnh(j).type().name();
						std::cout<<"c6("<<ni<<","<<nj<<") = "<<pot_vdw.c6()(i,j)<<"\n";
						std::cout<<"rvdw("<<ni<<","<<nj<<") = "<<pot_vdw.rvdw()(i,j)<<"\n";
					}
				}
			}
			//compute energy
			for(int n=0; n<3; ++n){
				std::vector<double> evdwl(dist[n].size(),std::numeric_limits<double>::max());
				for(int i=BATCH.rank(); i<dist[n].size(); i+=BATCH.size()){
					NeighborList nlist(strucs[n][i],pot_vdw.rc());
					evdwl[i]=pot_vdw.energy(strucs[n][i],nlist);
					ralpha.push(pot_vdw.ksl().alpha());
					rerrer.push(pot_vdw.ksl().errEr());
					rerrek.push(pot_vdw.ksl().errEk());
					rnk[0].push(pot_vdw.ksl().nk()[0]*1.0);
					rnk[1].push(pot_vdw.ksl().nk()[1]*1.0);
					rnk[2].push(pot_vdw.ksl().nk()[2]*1.0);
					rNK.push(pot_vdw.ksl().nk().prod());
				}
				MPI_Allreduce(MPI_IN_PLACE,evdwl.data(),evdwl.size(),MPI_DOUBLE,MPI_MIN,BATCH.mpic());
				for(int i=0; i<dist[n].size(); ++i){
					strucs[n][i].evdwl()=evdwl[i];
					strucs[n][i].pe()-=evdwl[i];
				}
			}
			if(BEAD.rank()==0){
				std::cout<<"alpha = "<<ralpha.avg()<<" "<<ralpha.min()<<" "<<ralpha.max()<<" "<<ralpha.dev()<<"\n";
				std::cout<<"errEr = "<<rerrer.avg()<<" "<<rerrer.min()<<" "<<rerrer.max()<<" "<<rerrer.dev()<<"\n";
				std::cout<<"errEk = "<<rerrek.avg()<<" "<<rerrek.min()<<" "<<rerrek.max()<<" "<<rerrek.dev()<<"\n";
				std::cout<<"nk[0] = "<<rnk[0].avg()<<" "<<rnk[0].min()<<" "<<rnk[0].max()<<" "<<rnk[0].dev()<<"\n";
				std::cout<<"nk[1] = "<<rnk[1].avg()<<" "<<rnk[1].min()<<" "<<rnk[1].max()<<" "<<rnk[1].dev()<<"\n";
				std::cout<<"nk[2] = "<<rnk[2].avg()<<" "<<rnk[2].min()<<" "<<rnk[2].max()<<" "<<rnk[2].dev()<<"\n";
				std::cout<<"nk    = "<<rNK.avg()<<" "<<rNK.min()<<" "<<rNK.max()<<" "<<rNK.dev()<<"\n";
			}
		}
		
		//************************************************************************************
		// SET INPUTS
		//************************************************************************************
		
		//======== initialize the symmetry functions ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				NNP::init(nnpte.nnp_,strucs[n][i]);
			}
		}
		
		//======== compute the symmetry functions ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"setting the inputs (symmetry functions)\n";
		for(int n=0; n<3; ++n){
			clock.begin();
			if(dist[n].size()>0){
				//compute symmetry functions
				for(int i=BATCH.rank(); i<dist[n].size(); i+=BATCH.size()){
					if(NNPTE_PRINT_STATUS>0) std::cout<<"structure-train["<<i<<"]\n";
					NeighborList nlist(strucs[n][i],nnpte.nnp_.rc());
					NNP::symm(nnpte.nnp_,strucs[n][i],nlist);
				}
				MPI_Barrier(BATCH.mpic());
				//bcast symmetry functions
				for(int i=0; i<BATCH.size(); ++i){
					const int root=i;
					for(int j=root; j<dist[n].size(); j+=BATCH.size()){
						thread::bcast(BATCH.mpic(),root,strucs[n][j]);
					}
				}
				MPI_Barrier(BATCH.mpic());
			}
			clock.end();
			time_symm[n]=clock.duration();
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== print the memory ========
		{
			//compute memory
			int meml[3]={0,0,0};
			for(int n=0; n<3; ++n) for(int i=0; i<dist[n].size(); ++i) meml[n]+=serialize::nbytes(strucs[n][i]);
			//allocate arrays
			std::vector<std::vector<int> > mem(3,std::vector<int>(WORLD.size(),0));
			//gather memory
			for(int n=0; n<3; ++n) MPI_Gather(&meml[n],1,MPI_INT,mem[n].data(),1,MPI_INT,0,WORLD.mpic());
			//compute total
			std::vector<double> memt(3,0.0);
			for(int n=0; n<3; ++n) for(int i=0; i<WORLD.size(); ++i) memt[n]+=mem[n][i];
			//print
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MEMORY",strbuf)<<"\n";
				std::cout<<"memory unit - MB\n";
				std::cout<<"mem - train - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<(1.0*mem[0][i])/1e6<<" "; std::cout<<"\n";
				std::cout<<"mem - val   - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<(1.0*mem[1][i])/1e6<<" "; std::cout<<"\n";
				std::cout<<"mem - test  - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<(1.0*mem[2][i])/1e6<<" "; std::cout<<"\n";
				std::cout<<"mem - train - tot = "<<(1.0*memt[0])/1e6<<"\n";
				std::cout<<"mem - val   - tot = "<<(1.0*memt[1])/1e6<<"\n";
				std::cout<<"mem - test  - tot = "<<(1.0*memt[2])/1e6<<"\n";
				std::cout<<print::buf(strbuf)<<"\n";
			}
		}
		
		//************************************************************************************
		// TRAINING
		//************************************************************************************
		
		//======== subtract ground-state energies ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"subtracting ground-state energies\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].pe()-=nnpte.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
				}
			}
		}
		
		//======== train the nn potential ========
		if(mode==Mode::TRAIN){
			if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"training the nn potential\n";
			nnpte.train(dist[3].size(),strucs[0],strucs[1]);
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== add ground-state energies ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"adding ground-state energies\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].pe()+=nnpte.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
				}
			}
		}
		
		//************************************************************************************
		// EVALUATION
		//************************************************************************************
		
		//======== statistical data - energies/forces/errors ========
		std::vector<double> kendall(3,0);
		std::vector<Reduce<1> > r1_energy(3);
		std::vector<Reduce<2> > r2_energy(3);
		std::vector<Reduce<1> > r1_force(3);
		std::vector<std::vector<Reduce<2> > > r2_force(3,std::vector<Reduce<2> >(3));
		
		//======== compute the final energies ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final energies\n";
		for(int n=0; n<3; ++n){
			if(dist[n].size()>0){
				std::vector<double> energy_n(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_n_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<int> natoms(nstrucs[n],0); std::vector<int> natoms_t(nstrucs[n],0);
				//compute energies
				clock.begin();
				for(int i=0; i<dist[n].size(); ++i){
					if(NNPTE_PRINT_STATUS>0) std::cout<<"structure["<<BEAD.rank()<<"]["<<i<<"]\i";
					energy_r[dist[n].index(i)]=strucs[n][i].pe();
					energy_n[dist[n].index(i)]=NNP::energy(nnpte.nnp_,strucs[n][i]);
					natoms[dist[n].index(i)]=strucs[n][i].nAtoms();
				}
				clock.end();
				time_energy[n]=clock.duration();
				if(compute.zero){
					for(int i=0; i<dist[n].size(); ++i){
						for(int j=0; j<strucs[n][i].nAtoms(); ++j){
							energy_r[dist[n].index(i)]-=nnpte.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
							energy_n[dist[n].index(i)]-=nnpte.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
						}
					}
				}
				MPI_Reduce(energy_r.data(),energy_r_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,BEAD.mpic());
				MPI_Reduce(energy_n.data(),energy_n_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,BEAD.mpic());
				MPI_Reduce(natoms.data(),natoms_t.data(),nstrucs[n],MPI_INT,MPI_MAX,0,BEAD.mpic());
				//accumulate statistics
				for(int i=0; i<nstrucs[n]; ++i){
					r1_energy[n].push(std::fabs(energy_r_t[i]-energy_n_t[i])/natoms_t[i]);
					r2_energy[n].push(energy_r_t[i]/natoms_t[i],energy_n_t[i]/natoms_t[i]);
				}
				kendall[n]=math::corr::kendall(energy_r_t,energy_n_t);
				//normalize
				if(compute.norm){
					for(int i=0; i<nstrucs[n]; ++i) energy_r_t[i]/=natoms_t[i];
					for(int i=0; i<nstrucs[n]; ++i) energy_n_t[i]/=natoms_t[i];
				}
				//write energies
				if(write.energy && BEAD.rank()==0){
					std::string file;
					switch(n){
						case 0: file="nnp_energy_train_p"; break;
						case 1: file="nnp_energy_val_p"; break;
						case 2: file="nnp_energy_test_p"; break;
						default: file="ERROR.dat"; break;
					}
					file+=std::to_string(BEAD.color()); file+=".dat";
					FILE* writer=fopen(file.c_str(),"w");
					if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
					else{
						std::vector<std::pair<int,double> > energy_r_pair(nstrucs[n]);
						std::vector<std::pair<int,double> > energy_n_pair(nstrucs[n]);
						for(int i=0; i<nstrucs[n]; ++i){
							energy_r_pair[i].first=indices[n][i];
							energy_r_pair[i].second=energy_r_t[i];
							energy_n_pair[i].first=indices[n][i];
							energy_n_pair[i].second=energy_n_t[i];
						}
						std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
						std::sort(energy_n_pair.begin(),energy_n_pair.end(),compare_pair);
						fprintf(writer,"#STRUCTURE ENERGY_REF ENERGY_NN\n");
						for(int i=0; i<nstrucs[n]; ++i){
							fprintf(writer,"%s %f %f\n",files[n][i].c_str(),energy_r_pair[i].second,energy_n_pair[i].second);
						}
						fclose(writer); writer=NULL;
					}
				}
			}
		}
		
		//======== compute the final coulomb energies ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final coulomb energies\n";
		for(int n=0; n<3; ++n){
			if(write.coul && dist[n].size()>0){
				std::vector<double> energy_r(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<int> natoms(nstrucs[n],0); std::vector<int> natoms_t(nstrucs[n],0);
				//compute energies
				clock.begin();
				for(int i=0; i<dist[n].size(); ++i){
					if(NNPTE_PRINT_STATUS>0) std::cout<<"structure-train["<<BEAD.rank()<<"]["<<i<<"]\n";
					energy_r[dist[n].index(i)]=strucs[n][i].ecoul();
					natoms[dist[n].index(i)]=strucs[n][i].nAtoms();
				}
				clock.end();
				MPI_Reduce(energy_r.data(),energy_r_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,BEAD.mpic());
				MPI_Reduce(natoms.data(),natoms_t.data(),nstrucs[n],MPI_INT,MPI_MAX,0,BEAD.mpic());
				//normalize
				if(compute.norm){
					for(int i=0; i<nstrucs[n]; ++i) energy_r_t[i]/=natoms_t[i];
				}
				//write energies
				if(write.energy && BEAD.rank()==0){
					std::string file;
					switch(n){
						case 0: file="nnp_ewald_train_p"; break;
						case 1: file="nnp_ewald_val_p"; break;
						case 2: file="nnp_ewald_test_p"; break;
						default: file="ERROR.dat"; break;
					}
					file+=std::to_string(BEAD.color()); file+=".dat";
					FILE* writer=fopen(file.c_str(),"w");
					if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
					else{
						std::vector<std::pair<int,double> > energy_r_pair(nstrucs[n]);
						for(int i=0; i<nstrucs[n]; ++i){
							energy_r_pair[i].first=indices[n][i];
							energy_r_pair[i].second=energy_r_t[i];
						}
						std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
						fprintf(writer,"#STRUCTURE ENERGY_EWALD\n");
						for(int i=0; i<nstrucs[n]; ++i){
							fprintf(writer,"%s %f\n",files[n][i].c_str(),energy_r_pair[i].second);
						}
						fclose(writer); writer=NULL;
					}
				}
			}
		}
		
		//======== compute the final forces ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0 && compute.force) std::cout<<"computing final forces\n";
		if(compute.force && write.force){
			for(int n=0; n<3; ++n){
				if(dist[n].size()>0){
					//compute forces
					clock.begin();
					for(int i=0; i<dist[n].size(); ++i){
						if(NNPTE_PRINT_STATUS>0) std::cout<<"structure["<<n<<"]["<<i<<"]\n";
						Structure& struc=strucs[n][i];
						//compute exact forces
						std::vector<Eigen::Vector3d> f_r(struc.nAtoms());
						for(int j=0; j<struc.nAtoms(); ++j) f_r[j]=struc.force(j);
						//compute nn forces
						NeighborList nlist(struc,nnpte.nnp_.rc());
						NNP::force(nnpte.nnp_,struc,nlist);
						std::vector<Eigen::Vector3d> f_n(struc.nAtoms());
						for(int j=0; j<struc.nAtoms(); ++j) f_n[j]=struc.force(j);
						//compute statistics
						if(BATCH.rank()==0){
							for(int j=0; j<struc.nAtoms(); ++j){
								r1_force[n].push((f_r[j]-f_n[j]).norm());
								r2_force[n][0].push(f_r[j][0],f_n[j][0]);
								r2_force[n][1].push(f_r[j][1],f_n[j][1]);
								r2_force[n][2].push(f_r[j][2],f_n[j][2]);
							}
						}
					}
					clock.end();
					time_force[n]=clock.duration();
					//accumulate statistics
					std::vector<Reduce<1> > r1fv(BEAD.size());
					thread::gather(r1_force[n],r1fv,BEAD.mpic());
					if(BEAD.rank()==0) for(int i=1; i<BEAD.size(); ++i) r1_force[n]+=r1fv[i];
					for(int i=0; i<3; ++i){
						std::vector<Reduce<2> > r2fv(BEAD.size());
						thread::gather(r2_force[n][i],r2fv,BEAD.mpic());
						if(BEAD.rank()==0) for(int j=1; j<BEAD.size(); ++j) r2_force[n][i]+=r2fv[j];
					}
				}
			}
		}
		
		//======== write the inputs ========
		if(write.input){
			for(int nn=0; nn<3; ++nn){
				if(dist[nn].size()>0){
					std::string file;
					switch(nn){
						case 0: file="nnp_inputs_train.dat"; break;
						case 1: file="nnp_inputs_val.dat"; break;
						case 2: file="nnp_inputs_test.dat"; break;
						default: file="ERROR.dat"; break;
					}
					for(int ii=0; ii<BEAD.size(); ++ii){
						if(BEAD.rank()==ii){
							FILE* writer=NULL;
							if(ii==0) writer=fopen(file.c_str(),"w");
							else writer=fopen(file.c_str(),"a");
							if(writer!=NULL){
								for(int n=0; n<dist[nn].size(); ++n){
									for(int i=0; i<strucs[nn][n].nAtoms(); ++i){
										fprintf(writer,"%s%i ",strucs[nn][n].name(i).c_str(),i);
										for(int j=0; j<strucs[nn][n].symm(i).size(); ++j){
											fprintf(writer,"%f ",strucs[nn][n].symm(i)[j]);
										}
										fprintf(writer,"\n");
									}
								}
								fclose(writer); writer=NULL;
							} else std::cout<<"WARNING: Could not open inputs file for training structures\n";
						}
						MPI_Barrier(BEAD.mpic());
					}
				}
			}
		}
		
		//======== stop the wall clock ========
		if(BEAD.rank()==0) clock_wall.end();
		if(BEAD.rank()==0) time_wall=clock_wall.duration();
		
		//************************************************************************************
		// OUTPUT
		//************************************************************************************
		
		//======== print the timing info ========
		for(int n=0; n<3; ++n){
			MPI_Allreduce(MPI_IN_PLACE,&time_symm[n],1,MPI_DOUBLE,MPI_SUM,BEAD.mpic()); 
			MPI_Allreduce(MPI_IN_PLACE,&time_energy[n],1,MPI_DOUBLE,MPI_SUM,BEAD.mpic()); 
			MPI_Allreduce(MPI_IN_PLACE,&time_force[n],1,MPI_DOUBLE,MPI_SUM,BEAD.mpic());
			time_symm[n]/=BEAD.size();
			time_energy[n]/=BEAD.size();
			time_force[n]/=BEAD.size();
		}
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("TIMING (S)",strbuf)<<"\n";
			if(strucs[0].size()>0){
				std::cout<<"time - symm   - train = "<<time_symm[0]<<"\n";
				std::cout<<"time - energy - train = "<<time_energy[0]<<"\n";
				std::cout<<"time - force  - train = "<<time_force[0]<<"\n";
			}
			if(strucs[1].size()>0){
				std::cout<<"time - symm   - val   = "<<time_symm[1]<<"\n";
				std::cout<<"time - energy - val   = "<<time_energy[1]<<"\n";
				std::cout<<"time - force  - val   = "<<time_force[1]<<"\n";
			}
			if(strucs[2].size()>0){
				std::cout<<"time - symm   - test  = "<<time_symm[2]<<"\n";
				std::cout<<"time - energy - test  = "<<time_energy[2]<<"\n";
				std::cout<<"time - force  - test  = "<<time_force[2]<<"\n";
			}
			std::cout<<"time - wall           = "<<time_wall<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print the error statistics ========
		if(WORLD.rank()==0){
			for(int n=0; n<3; ++n){
				if(nstrucs[n]>0){
					std::cout<<print::buf(strbuf)<<"\n";
					if(n==0) std::cout<<print::title("ERROR - STATISTICS - TRAINING",strbuf)<<"\n";
					else if(n==1) std::cout<<print::title("ERROR - STATISTICS - VALIDATION",strbuf)<<"\n";
					else if(n==2) std::cout<<print::title("ERROR - STATISTICS - TESTING",strbuf)<<"\n";
					std::cout<<"\tERROR - AVG - "<<dnames[n]<<" - ENERGY/ATOM = "<<r1_energy[n].avg()<<"\n";
					std::cout<<"\tERROR - DEV - "<<dnames[n]<<" - ENERGY/ATOM = "<<r1_energy[n].dev()<<"\n";
					std::cout<<"\tERROR - MAX - "<<dnames[n]<<" - ENERGY/ATOM = "<<r1_energy[n].max()<<"\n";
					std::cout<<"\tM/R2 - "<<dnames[n]<<" - ENERGY/ATOM = "<<r2_energy[n].m()<<" "<<r2_energy[n].r2()<<"\n";
					std::cout<<"\tKENDALL - "<<dnames[n]<<" = "<<kendall[n]<<"\n";
					if(compute.force){
					std::cout<<"FORCE:\n";
					std::cout<<"\tERROR - AVG - FORCE - "<<dnames[n]<<" = "<<r1_force[n].avg()<<"\n";
					std::cout<<"\tERROR - DEV - FORCE - "<<dnames[n]<<" = "<<r1_force[n].dev()<<"\n";
					std::cout<<"\tERROR - MAX - FORCE - "<<dnames[n]<<" = "<<r1_force[n].max()<<"\n";
					std::cout<<"\tM  (FX,FY,FZ) = "<<r2_force[n][0].m() <<" "<<r2_force[n][1].m() <<" "<<r2_force[n][2].m() <<"\n";
					std::cout<<"\tR2 (FX,FY,FZ) = "<<r2_force[n][0].r2()<<" "<<r2_force[n][1].r2()<<" "<<r2_force[n][2].r2()<<"\n";
					}
					std::cout<<print::buf(strbuf)<<"\n";
				}
			}
		}
		
		//======== write the nn's ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing the nn's\n";
		if(BEAD.rank()==0){
			const std::string file=nnpte.file_ann_+"_p"+std::to_string(BEAD.color());
			NNP::write(file.c_str(),nnpte.nnp_);
		}
		//======== write restart file ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing restart file\n";
		if(BEAD.rank()==0){
			const std::string file=nnpte.file_restart_+"_p"+std::to_string(BEAD.color())+".restart";
			nnpte.write_restart(file.c_str());
		}
		
		//======== finalize mpi ========
		if(NNPTE_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"finalizing mpi\n";
		std::cout<<std::flush;
		MPI_Comm_free(&BATCH.mpic());
		MPI_Comm_free(&BEAD.mpic());
		MPI_Barrier(WORLD.mpic());
		MPI_Finalize();
	}catch(std::exception& e){
		std::cout<<"ERROR in nnpte::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//======== free local variables ========
	delete[] input;
	delete[] strbuf;
	
	return 0;
}
