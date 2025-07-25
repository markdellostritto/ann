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
// string
#include "str/string.hpp"
#include "str/token.hpp"
#include "str/print.hpp"
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
#include "torch/pot_factory.hpp"
#include "torch/pot_gauss_cut.hpp"
#include "torch/pot_gauss_dsf.hpp"
#include "torch/pot_gauss_long.hpp"
// nnpteq
#include "nnp/nnpteq.hpp"

static const double FCHI=1.0;
using math::constant::LOG2;

static bool compare_pair(const std::pair<int,double>& p1, const std::pair<int,double>& p2){
	return p1.first<p2.first;
}

//************************************************************
// MPI Communicators
//************************************************************

thread::Comm WORLD;
thread::Comm BATCH;

//************************************************************
// serialization
//************************************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNPTEQ& obj){
   if(NNPTEQ_PRINT_FUNC>0) std::cout<<"nbytes(const NNPTEQ&)\n";
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
		size+=sizeof(bool);//pre-conditioning
		size+=sizeof(bool);//wparams
	//charge
		size+=nbytes(obj.qeq_);
	//optimization
		size+=nbytes(obj.batch_);
		size+=nbytes(obj.obj_);
		size+=nbytes(obj.algo_);
		size+=nbytes(obj.decay_);
		size+=sizeof(double);//delta_
	//return the size
		return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNPTEQ& obj, char* arr){
	if(NNPTEQ_PRINT_FUNC>0) std::cout<<"pack(const NNPTEQ&,char*)\n";
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
		std::memcpy(arr+pos,&obj.preCond_,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.wparams_,sizeof(bool)); pos+=sizeof(bool);
	//charge
		pos+=pack(obj.qeq_,arr+pos);
	//optimization
		pos+=pack(obj.batch_,arr+pos);
		pos+=pack(obj.obj_,arr+pos);
		pos+=pack(obj.algo_,arr+pos);
		pos+=pack(obj.decay_,arr+pos);
		std::memcpy(arr+pos,&obj.delta_,sizeof(double)); pos+=sizeof(double);
	//return bytes written
		return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNPTEQ& obj, const char* arr){
	if(NNPTEQ_PRINT_FUNC>0) std::cout<<"unpack(const NNPTEQ&,char*)\n";
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
		std::memcpy(&obj.preCond_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.wparams_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
	//charge
		pos+=unpack(obj.qeq_,arr+pos);
	//optimization
		pos+=unpack(obj.batch_,arr+pos);
		pos+=unpack(obj.obj_,arr+pos);
		pos+=unpack(obj.algo_,arr+pos);
		pos+=unpack(obj.decay_,arr+pos);
		std::memcpy(&obj.delta_,arr+pos,sizeof(double)); pos+=sizeof(double);
		obj.deltai()=1.0/obj.delta();
		obj.delta2()=obj.delta()*obj.delta();
	//return bytes read
		return pos;
}
	
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
// NNPTEQ - Neural Network Potential - Optimization
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPTEQ& nnpteq){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NNPTEQ",str)<<"\n";
	//files
	out<<"FILE_PARAMS  = "<<nnpteq.file_params_<<"\n";
	out<<"FILE_ERROR   = "<<nnpteq.file_error_<<"\n";
	out<<"FILE_ANN     = "<<nnpteq.file_ann_<<"\n";
	out<<"FILE_RESTART = "<<nnpteq.file_restart_<<"\n";
	//flags
	out<<"RESTART      = "<<nnpteq.restart_<<"\n";
	out<<"PRE-COND     = "<<nnpteq.preCond_<<"\n";
	out<<"WPARAMS      = "<<nnpteq.wparams_<<"\n";
	//qeq
	out<<"FCHI         = "<<FCHI<<"\n";
	out<<"QEQ          = "<<nnpteq.qeq_<<"\n";
	//optimization
	out<<"BATCH        = "<<nnpteq.batch_<<"\n";
	out<<"ALGO         = "<<nnpteq.algo_<<"\n";
	out<<"DECAY        = "<<nnpteq.decay_<<"\n";
	out<<"N_PRINT      = "<<nnpteq.obj().nPrint()<<"\n";
	out<<"N_WRITE      = "<<nnpteq.obj().nWrite()<<"\n";
	out<<"MAX          = "<<nnpteq.obj().max()<<"\n";
	out<<"STOP         = "<<nnpteq.obj().stop()<<"\n";
	out<<"LOSS         = "<<nnpteq.obj().loss()<<"\n";
	out<<"TOL          = "<<nnpteq.obj().tol()<<"\n";
	out<<"GAMMA        = "<<nnpteq.obj().gamma()<<"\n";
	out<<"DELTA        = "<<nnpteq.delta()<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

NNPTEQ::NNPTEQ(){
	if(NNPTEQ_PRINT_FUNC>0) std::cout<<"NNP::NNPTEQ():\n";
	defaults();
};

void NNPTEQ::defaults(){
	if(NNPTEQ_PRINT_FUNC>0) std::cout<<"NNP::defaults():\n";
	//nnp
		nTypes_=0;
	//input/output
		file_params_="nnp_params.dat";
		file_error_="nnp_error.dat";
		file_restart_="nnpteq.restart";
		file_ann_="ann";
	//flags
		restart_=false;
		preCond_=false;
		wparams_=false;
	//optimization
		delta_=1.0;
		deltai_=1.0;
	//error
		error_[0]=0;
		error_[1]=0;
		error_[2]=0;
		error_[3]=0;
}

void NNPTEQ::clear(){
	if(NNPTEQ_PRINT_FUNC>0) std::cout<<"NNP::clear():\n";
	//element
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

void NNPTEQ::write_restart(const char* file){
	if(NNPTEQ_PRINT_FUNC>1) std::cout<<"NNPTEQ::write_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* writer=NULL;
	bool error=false;
	try{
		//open file
		writer=fopen(file,"wb");
		if(writer==NULL) throw std::runtime_error(std::string("NNPTEQ::write_restart(const char*): Could not open file: ")+file);
		//allocate buffer
		const int nBytes=serialize::nbytes(*this);
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTEQ::write_restart(const char*): Could not allocate memory.");
		//write to buffer
		serialize::pack(*this,arr);
		//write to file
		const int nWrite=fwrite(arr,sizeof(char),nBytes,writer);
		if(nWrite!=nBytes) throw std::runtime_error("NNPTEQ::write_restart(const char*): Write error.");
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
	if(error) throw std::runtime_error("NNPTEQ::write_restart(const char*): Failed to write");
}

void NNPTEQ::read_restart(const char* file){
	if(NNPTEQ_PRINT_FUNC>0) std::cout<<"NNPTEQ::read_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* reader=NULL;
	bool error=false;
	try{
		//open file
		reader=fopen(file,"rb");
		if(reader==NULL) throw std::runtime_error(std::string("NNPTEQ::read_restart(const char*): Could not open file: ")+std::string(file));
		//find size
		std::fseek(reader,0,SEEK_END);
		const int nBytes=std::ftell(reader);
		std::fseek(reader,0,SEEK_SET);
		//allocate buffer
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTEQ::read_restart(const char*): Could not allocate memory.");
		//read from file
		const int nRead=fread(arr,sizeof(char),nBytes,reader);
		if(nRead!=nBytes) throw std::runtime_error("NNPTEQ::read_restart(const char*): Read error.");
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
	if(error) throw std::runtime_error("NNPTEQ::read_restart(const char*): Failed to read");
}

void NNPTEQ::train(int batchSize, std::vector<Structure>& struc_train, std::vector<Structure>& struc_val){
	if(NNPTEQ_PRINT_FUNC>0) std::cout<<"NNPTEQ::train(NNP&,std::vector<Structure>&,int):\n";
	//====== local function variables ======
		char* strbuf=new char[print::len_buf];
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
		
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"training NN potential\n";
	
	//====== check the parameters ======
	if(batchSize<=0) throw std::invalid_argument("NNPTEQ::train(int): Invalid batch size.");
	if(struc_train.size()==0) throw std::invalid_argument("NNPTEQ::train(int): No training data provided.");
	if(struc_val.size()==0) throw std::invalid_argument("NNPTEQ::train(int): No validation data provided.");
	
	//====== get the number of structures ======
	double nBatchF=(1.0*batchSize)/BATCH.size();
	double nTrainF=(1.0*struc_train.size())/BATCH.size();
	double nValF=(1.0*struc_val.size())/BATCH.size();
	MPI_Allreduce(MPI_IN_PLACE,&nBatchF,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	MPI_Allreduce(MPI_IN_PLACE,&nTrainF,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	MPI_Allreduce(MPI_IN_PLACE,&nValF,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
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
	
	//====== compute the ewald parameters ====
	if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing ewald parameters\n";
	bt_.resize(struc_train.size());
	xt_.resize(struc_train.size());
	AIt_.resize(struc_train.size());
	for(int n=0; n<struc_train.size(); ++n){
		NeighborList nlist(struc_train[n],qeq_.pot()->rc());
		qeq_.qt(struc_train[n],nlist);
		bt_[n]=qeq_.b();
		xt_[n]=qeq_.x();
		AIt_[n]=qeq_.A().inverse();
	}
	bv_.resize(struc_val.size());
	xv_.resize(struc_val.size());
	AIv_.resize(struc_val.size());
	for(int n=0; n<struc_val.size(); ++n){
		NeighborList nlist(struc_val[n],qeq_.pot()->rc());
		qeq_.qt(struc_val[n],nlist);
		bv_[n]=qeq_.b();
		xv_[n]=qeq_.x();
		AIv_[n]=qeq_.A().inverse();
	}
	
	//====== resize the optimization data ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing the optimization data\n";
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
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing gradient data\n";
	cost_.resize(nTypes_);
	for(int i=0; i<nTypes_; ++i){
		cost_[i].resize(nnp_.nnh(i).nn());
	}
	
	//====== compute the number of atoms of each element ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing the number of atoms of each element\n";
	std::vector<double> nAtoms_(nTypes_,0);
	for(int i=0; i<struc_train.size(); ++i){
		for(int j=0; j<struc_train[i].nAtoms(); ++j){
			++nAtoms_[struc_train[i].type(j)];
		}
	}
	MPI_Allreduce(MPI_IN_PLACE,nAtoms_.data(),nTypes_,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	for(int i=0; i<nTypes_; ++i) nAtoms_[i]/=BATCH.size();
	if(NNPTEQ_PRINT_DATA>-1 && WORLD.rank()==0){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("ATOM - DATA",strbuf)<<"\n";
		for(int i=0; i<nTypes_; ++i){
			const std::string& name=nnp_.nnh(i).type().name();
			const int index=nnp_.index(nnp_.nnh(i).type().name());
			std::cout<<name<<"("<<index<<") - "<<(int)nAtoms_[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
	}
	
	//====== set the indices and batch size ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting indices and batch\n";
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
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the total number\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			++N[struc_train[n].type(i)];
		}
	}
	//accumulate the number
	for(int i=0; i<nTypes_; ++i){
		double Nloc=(1.0*N[i])/BATCH.size();//normalize by the size of the BATCH group
		MPI_Allreduce(MPI_IN_PLACE,&Nloc,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		N[i]=static_cast<int>(std::round(Nloc));
	}
	//compute the max/min
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the max/min\n";
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
	//accumulate the min/max
	for(int i=0; i<nTypes_; ++i){
		MPI_Allreduce(MPI_IN_PLACE,min_in[i].data(),min_in[i].size(),MPI_DOUBLE,MPI_MIN,WORLD.mpic());
		MPI_Allreduce(MPI_IN_PLACE,max_in[i].data(),max_in[i].size(),MPI_DOUBLE,MPI_MIN,WORLD.mpic());
	}
	//compute the average
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the average\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			avg_in[struc_train[n].type(i)].noalias()+=struc_train[n].symm(i);
		}
	}
	//accumulate the average
	for(int i=0; i<nTypes_; ++i){
		avg_in[i]/=BATCH.size();//normalize by the size of the BATCH group
		MPI_Allreduce(MPI_IN_PLACE,avg_in[i].data(),avg_in[i].size(),MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		avg_in[i]/=N[i];
	}
	//compute the stddev
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the stddev\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		for(int i=0; i<strucl.nAtoms(); ++i){
			const int index=strucl.type(i);
			dev_in[index].noalias()+=(avg_in[index]-strucl.symm(i)).cwiseProduct(avg_in[index]-strucl.symm(i));
		}
	}
	//accumulate the stddev
	for(int i=0; i<dev_in.size(); ++i){
		for(int j=0; j<dev_in[i].size(); ++j){
			dev_in[i][j]/=BATCH.size();//normalize by the size of the BATCH group
			MPI_Allreduce(MPI_IN_PLACE,&dev_in[i][j],1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
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
	if(preCond_){
		if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"pre-conditioning input\n";
		//set the preconditioning vectors - bias
		for(int i=0; i<inpb_.size(); ++i){
			inpb_[i]=-1*avg_in[i];
		}
		//set the preconditioning vectors - weight
		for(int i=0; i<inpw_.size(); ++i){
			for(int j=0; j<inpw_[i].size(); ++j){
				if(dev_in[i][j]==0) inpw_[i][j]=1;
				else inpw_[i][j]=1.0/(1.0*dev_in[i][j]);
			}
		}
	}
	
	//====== set the bias for each of the species ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting the bias for each species\n";
	for(int n=0; n<nTypes_; ++n){
		NN::ANN& nn_=nnp_.nnh(n).nn();
		for(int i=0; i<nn_.nInp(); ++i) nn_.inpb()[i]=inpb_[n][i];
		for(int i=0; i<nn_.nInp(); ++i) nn_.inpw()[i]=inpw_[n][i];
		nn_.outb()[0]=0.0;
		nn_.outw()[0]=1.0;
	}
	
	//====== initialize the optimization data ======
	const int nParams=nnp_.size();
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
		algo_->resize(nParams);
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
	}
	
	//====== print input statistics and bias ======
	if(NNPTEQ_PRINT_DATA>-1 && WORLD.rank()==0){
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
	}
	
	//====== execute the optimization ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"executing the optimization\n";
	//optimization variables
	//bool fbreak=false;
	const double nBatchi_=1.0/nBatch;
	const double nVali_=1.0/nVal;
	//bcast parameters
	MPI_Bcast(obj_.p().data(),obj_.p().size(),MPI_DOUBLE,0,WORLD.mpic());
	//allocate status vectors
	std::vector<int> step;
	std::vector<double> gamma,rmse_g,loss_t,loss_v,rmse_t,rmse_v;
	std::vector<Eigen::VectorXd> params;
	if(WORLD.rank()==0){
		int size=obj_.max()/obj_.nPrint();
		if(size==0) ++size;
		step.resize(size);
		gamma.resize(size);
		rmse_g.resize(size);
		loss_t.resize(size);
		loss_v.resize(size);
		rmse_t.resize(size);
		rmse_v.resize(size);
		params.resize(size);
	}
	std::vector<Eigen::VectorXd> gElementT=gElement_;
	//print status header to standard output
	if(WORLD.rank()==0) printf("opt gamma rmse_g loss_t loss_v rmse_t rmse_v\n");
	//start the clock
	clock.begin();
	//begin optimization
	Eigen::VectorXd gtot_=Eigen::VectorXd::Zero(obj_.dim());
	for(int iter=0; iter<obj_.max(); ++iter){
		double error_sum_[4]={0.0,0.0,0.0,0.0};
		//compute the error and gradient
		error(obj_.p(),struc_train,struc_val);
		//pack the gradient
		int count=0;
		for(int n=0; n<nTypes_; ++n){
			std::memcpy(gtot_.data()+count,gElement_[n].data(),gElement_[n].size()*sizeof(double));
			count+=gElement_[n].size();
		}
		//accumulate gradient and error
		obj_.g().setZero();
		MPI_Reduce(gtot_.data(),obj_.g().data(),gtot_.size(),MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		MPI_Reduce(error_,error_sum_,4,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		if(WORLD.rank()==0){
			//compute error averaged over the batch
			error_[0]=error_sum_[0]*nBatchi_;//loss - train
			error_[1]=error_sum_[1]*nVali_;//loss - val
			error_[2]=sqrt(error_sum_[2]*nBatchi_);//rmse - train
			error_[3]=sqrt(error_sum_[3]*nVali_);//rmse - val
			//compute gradient averaged over the batch
			//obj_.g()*=nBatchi_;
			//print/write error
			if(obj_.step()%obj_.nPrint()==0){
				const int t=iter/obj_.nPrint();
				step[t]=obj_.count();
				gamma[t]=obj_.gamma();
				rmse_g[t]=sqrt(obj_.g().squaredNorm()/nParams);
				loss_t[t]=error_[0];
				loss_v[t]=error_[1];
				rmse_t[t]=error_[2];
				rmse_v[t]=error_[3];
				params[t]=obj_.p();
				printf("%8i %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e\n",
					step[t],gamma[t],rmse_g[t],loss_t[t],loss_v[t],rmse_t[t],rmse_v[t]
				);
			}
			//write the basis and potentials
			if(obj_.step()%obj_.nWrite()==0){
				if(NNPTEQ_PRINT_STATUS>1) std::cout<<"writing the restart file and potentials\n";
				//write restart file
				const std::string file_restart=file_restart_+"."+std::to_string(obj_.count());
				this->write_restart(file_restart.c_str());
				//write potential file
				const std::string file_ann=file_ann_+"."+std::to_string(obj_.count());
				NNP::write(file_ann.c_str(),nnp_);
			}
			//compute the new position
			obj_.val()=error_[0];//loss - train
			obj_.gamma()=decay_->step(obj_);
			algo_->step(obj_);
			//compute the difference
			obj_.dv()=std::fabs(obj_.val()-obj_.valOld());
			obj_.dp()=(obj_.p()-obj_.pOld()).norm();
			//set the new "old" values
			obj_.valOld()=obj_.val();//set "old" value
			obj_.pOld()=obj_.p();//set "old" p value
			obj_.gOld()=obj_.g();//set "old" g value
			//check the break condition
			/*switch(obj_.stop()){
				case opt::Stop::FABS: fbreak=(obj_.val()<obj_.tol()); break;
				case opt::Stop::FREL: fbreak=(obj_.dv()<obj_.tol()); break;
				case opt::Stop::XREL: fbreak=(obj_.dp()<obj_.tol()); break;
			}*/
		}
		//bcast parameters
		MPI_Bcast(obj_.p().data(),obj_.p().size(),MPI_DOUBLE,0,WORLD.mpic());
		//bcast break condition
		/*MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,WORLD.mpic());
		if(fbreak) break;*/
		//increment step
		++obj_.step();
		++obj_.count();
	}
	//compute the training time
	clock.end();
	double time_train=clock.duration();
	MPI_Allreduce(MPI_IN_PLACE,&time_train,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	time_train/=WORLD.size();
	MPI_Barrier(WORLD.mpic());
	
	//====== write the error ======
	if(WORLD.rank()==0){
		FILE* writer_error_=NULL;
		if(!restart_){
			writer_error_=fopen(file_error_.c_str(),"w");
			fprintf(writer_error_,"#STEP GAMMA RMSE_GRAD LOSS_TRAIN LOSS_VAL RMSE_TRAIN RMSE_VAL\n");
		} else {
			writer_error_=fopen(file_error_.c_str(),"a");
		}
		if(writer_error_==NULL) throw std::runtime_error("NNPTE::train(int): Could not open error record file.");
		for(int t=0; t<step.size(); ++t){
			fprintf(writer_error_,
				"%6i %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e\n",
				step[t],gamma[t],rmse_g[t],loss_t[t],loss_v[t],rmse_t[t],rmse_v[t]
			);
		}
		fclose(writer_error_);
		writer_error_=NULL;
	}
	
	//====== write the parameters ======
	if(WORLD.rank()==0 && wparams_){
		FILE* writer_p_=NULL;
		if(!restart_) writer_p_=fopen(file_params_.c_str(),"w");
		else writer_p_=fopen(file_params_.c_str(),"a");
		if(writer_p_==NULL) throw std::runtime_error("NNPTEQ::train(int): Could not open error record file.");
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
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"packing final parameters into neural network\n";
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
	
	if(NNPTEQ_PRINT_DATA>-1 && WORLD.rank()==0){
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TRAIN - SUMMARY",strbuf)<<"\n";
		std::cout<<"N-STEP = "<<obj_.step()<<"\n";
		std::cout<<"TIME   = "<<time_train<<"\n";
		if(NNPTEQ_PRINT_DATA>1){
			std::cout<<"p = "; for(int i=0; i<obj_.p().size(); ++i) std::cout<<obj_.p()[i]<<" "; std::cout<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
	}
	
	delete[] strbuf;
}

void NNPTEQ::error(const Eigen::VectorXd& x, std::vector<Structure>& struc_train, std::vector<Structure>& struc_val){
	if(NNPTEQ_PRINT_FUNC>0) std::cout<<"NNPTEQ::error(const Eigen::VectorXd&):\n";
	
	//====== reset the error ======
	error_[0]=0; //loss - training
	error_[1]=0; //loss - validation
	error_[2]=0; //rmse - training
	error_[3]=0; //rmse - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nTypes_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch\n";
	for(int i=0; i<batch_.size(); ++i) batch_[i]=batch_.data((batch_.count()++)%batch_.capacity());
	std::sort(batch_.elements(),batch_.elements()+batch_.size());
	if(batch_.count()>=batch_.capacity()){
		std::shuffle(batch_.data(),batch_.data()+batch_.capacity(),rngen_);
		MPI_Bcast(batch_.data(),batch_.capacity(),MPI_INT,0,BATCH.mpic());
		batch_.count()=0;
	}
	if(NNPTEQ_PRINT_DATA>1 && WORLD.rank()==0){std::cout<<"batch = "; for(int i=0; i<batch_.size(); ++i) std::cout<<batch_[i]<<" "; std::cout<<"\n";}
	
	//====== compute training error and gradient ======
	if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batch_.size(); ++i){
		//set batch
		const int ii=batch_[i];
		//reset the gradients
		for(int j=0; j<nTypes_; ++j) grad_[j].setZero();
		//**** compute the energy ****
		const int nAtoms=struc_train[ii].nAtoms();
		double energyV=0;
		for(int n=0; n<nAtoms; ++n) struc_train[ii].chi(n)=0.0;
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().fp(struc_train[ii].symm(m));
			//add the atom energy to the total
			energyV+=nnp_.nnh(type).nn().out()[0];
			struc_train[ii].chi(m)=nnp_.nnh(type).nn().out()[1]+nnp_.nnh(type).type().chi().val()*FCHI;
		}
		//**** accumulate energy across BATCH communicator ****
		MPI_Allreduce(MPI_IN_PLACE,&energyV,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		//**** accumulate electronegativity across BATCH communicator ****
		MPI_Allreduce(MPI_IN_PLACE,struc_train[ii].chi().data(),nAtoms,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		//**** compute the charges ****
		for(int n=0; n<nAtoms; ++n) bt_[ii][n]=-1.0*struc_train[ii].chi(n);
		xt_[ii].noalias()=AIt_[ii]*bt_[ii];
		for(int n=0; n<nAtoms; ++n) struc_train[ii].charge(n)=xt_[ii][n];
		//**** compute the electrostatic energy ****
		const double energyQ=-0.5*xt_[ii].dot(bt_[ii]);
		//**** compute the total energy and error ****
		const double nAtomsI=1.0/nAtoms;
		const double energyT=energyQ+energyV;
		const double dE=nAtomsI*(energyT-struc_train[ii].energy());
		double gpre=0;
		switch(obj_.loss()){
			case opt::Loss::MSE:{
				error_[0]+=0.5*dE*dE;
				gpre=nAtomsI*dE;
			}break;
			case opt::Loss::MAE:{
				error_[0]+=fabs(dE);
				gpre=math::special::sgn(dE)*nAtomsI;
			}break;
			case opt::Loss::HUBER:{
				const double rad=sqrt(dE*dE/delta2_+1.0);
				error_[0]+=delta2_*(rad-1.0);
				gpre=dE*nAtomsI/rad;
			}break;
			case opt::Loss::ASINH:{
				const double arg=dE*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				error_[0]+=delta2_*(1.0-sqrtf+arg*logf);//loss - train
				gpre=logf*delta_*nAtomsI;
			}break;
			default: break;
		}
		error_[2]+=dE*dE;//rmse - train
		//**** scale and sum atomic gradients ****
		for(int j=0; j<nTypes_; ++j) grad_[j].setZero();
		for(int n=0; n<dist_atomt[ii].size(); ++n){
			//get the index of the atom within the local processor subset
			const int m=dist_atomt[ii].index(n);
			//find the element index in the nn potential
			const int type=struc_train[ii].type(m);
			//execute the network
			nnp_.nnh(type).nn().fpbp(struc_train[ii].symm(m));
			//compute dcdo
			Eigen::VectorXd dcdo(2);
			dcdo<<gpre,0.5*gpre*struc_train[ii].charge(m);
			//compute gradient
			grad_[type].noalias()+=cost_[type].grad(nnp_.nnh(type).nn(),dcdo);
		}
		//**** accumulate gradient across the BATCH communicator ****
		for(int j=0; j<nTypes_; ++j){
			MPI_Allreduce(MPI_IN_PLACE,grad_[j].data(),grad_[j].size(),MPI_DOUBLE,MPI_SUM,BATCH.mpic());
		}
		//**** add gradient to total ****
		for(int j=0; j<nTypes_; ++j){
			gElement_[j].noalias()+=grad_[j];
		}
	}
	
	//====== compute validation error and gradient ======
	if(obj_.step()%obj_.nPrint()==0 || obj_.step()%obj_.nWrite()==0){
		if(NNPTEQ_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error and gradient\n";
		for(int i=0; i<struc_val.size(); ++i){
			//**** compute the energy ****
			const int nAtoms=struc_val[i].nAtoms();
			double energyV=0;
			for(int n=0; n<nAtoms; ++n) struc_val[i].chi(n)=0.0;
			for(int n=0; n<dist_atomv[i].size(); ++n){
				//get the index of the atom within the local processor subset
				const int m=dist_atomv[i].index(n);
				//find the element index in the nn potential
				const int type=struc_val[i].type(m);
				//execute the network
				nnp_.nnh(type).nn().fp(struc_val[i].symm(m));
				//add the energy to the total
				energyV+=nnp_.nnh(type).nn().out()[0]+nnp_.nnh(type).type().energy().val()*0.0;
				struc_val[i].chi(m)=nnp_.nnh(type).nn().out()[1]+nnp_.nnh(type).type().chi().val()*FCHI;
			}
			//**** accumulate energy ****
			MPI_Allreduce(MPI_IN_PLACE,&energyV,1,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
			//**** accumulate electronegativity across BATCH communicator ****
			MPI_Allreduce(MPI_IN_PLACE,struc_val[i].chi().data(),nAtoms,MPI_DOUBLE,MPI_SUM,BATCH.mpic());
			//**** compute the charges ****
			for(int n=0; n<nAtoms; ++n) bv_[i][n]=-1.0*struc_val[i].chi(n);
			xv_[i].noalias()=AIv_[i]*bv_[i];
			for(int n=0; n<nAtoms; ++n) struc_val[i].charge(n)=xv_[i][n];
			//**** compute the electrostatic energy ****
			const double energyQ=-0.5*xv_[i].dot(bv_[i]);
			//**** compute error ****
			const double energyT=energyQ+energyV;
			const double dE=(energyT-struc_val[i].energy())/nAtoms;
			switch(obj_.loss()){
				case opt::Loss::MSE:{
					error_[1]+=0.5*dE*dE;
				} break;
				case opt::Loss::MAE:{
					error_[1]+=std::fabs(dE);
				} break;
				case opt::Loss::HUBER:{
					const double arg=dE*deltai_;
					error_[1]+=delta2_*(sqrt(1.0+arg*arg)-1.0);//loss - val
				} break;
				case opt::Loss::ASINH:{
					const double arg=dE/delta_;
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

void NNPTEQ::read(const char* file, NNPTEQ& nnpteq){
	if(NN_PRINT_FUNC>0) std::cout<<"NNPTEQ::read(const char*,NNPTEQ&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		NNPTEQ::read(reader,nnpteq);
		std::fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for reading.\n"));
}

void NNPTEQ::read(FILE* reader, NNPTEQ& nnpteq){
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
			nnpteq.nnp().rc()=std::atof(token.next().c_str());
		}
		//files
		if(tag=="FILE_ERROR"){
			nnpteq.file_error()=token.next();
		} else if(tag=="FILE_PARAMS"){
			nnpteq.file_params()=token.next();
		} else if(tag=="FILE_ANN"){
			nnpteq.file_ann()=token.next();
		} else if(tag=="FILE_RESTART"){
			nnpteq.file_restart()=token.next();
		}
		//flags
		if(tag=="RESTART"){//read restart file
			nnpteq.restart()=string::boolean(token.next().c_str());//restarting
		} else if(tag=="PRE_COND"){//whether to precondition the inputs
			nnpteq.preCond()=string::boolean(token.next().c_str());
		} else if(tag=="WRITE_PARAMS"){
			nnpteq.wparams()=string::boolean(token.next().c_str());
		} 
		//optimization
		if(tag=="LOSS"){
			nnpteq.obj().loss()=opt::Loss::read(string::to_upper(token.next()).c_str());
		} else if(tag=="STOP"){
			nnpteq.obj().stop()=opt::Stop::read(string::to_upper(token.next()).c_str());
		} else if(tag=="MAX_ITER"){
			nnpteq.obj().max()=std::atoi(token.next().c_str());
		} else if(tag=="N_PRINT"){
			nnpteq.obj().nPrint()=std::atoi(token.next().c_str());
		} else if(tag=="N_WRITE"){
			nnpteq.obj().nWrite()=std::atoi(token.next().c_str());
		} else if(tag=="TOL"){
			nnpteq.obj().tol()=std::atof(token.next().c_str());
		} else if(tag=="GAMMA"){
			nnpteq.obj().gamma()=std::atof(token.next().c_str());
		} else if(tag=="ALGO"){
			opt::algo::read(nnpteq.algo(),token);
		} else if(tag=="DECAY"){
			opt::decay::read(nnpteq.decay(),token);
		} else if(tag=="DELTA"){
			nnpteq.delta()=std::atof(token.next().c_str());
			nnpteq.deltai()=1.0/nnpteq.delta();
			nnpteq.delta2()=nnpteq.delta()*nnpteq.delta();
		} 
		//potential 
		if(tag=="POT_QEQ"){
			ptnl::read(nnpteq.qeq().pot(),token);
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
		atomT.posn=true; atomT.force=false; atomT.symm=true;
		atomT.charge=true; atomT.eta=true; atomT.chi=true;
	//flags - compute
		struct Compute{
			bool coul=false;  //compute - external potential - coul
			bool vdw=false;   //compute - external potential - vdw
			bool force=false; //compute - forces
			bool norm=false;  //compute - energy normalization
			bool zero=false;  //compute - zero point energy
		} compute;
	//flags - writing
		struct Write{
			bool energy=false; //writing - energies
			bool force=false;  //writing - forces
			bool ewald=false;  //writing - ewald energies
			bool input=false;  //writing - inputs
			bool charge=false; //writing - charge
		} write;
	//nn potential - opt
		int nBatch=-1;
		std::vector<Type> types;//unique atomic species
		NNPTEQ nnpteq;//nn potential optimization data
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
		Clock clock,clock_wall;     //time objects
		double time_wall=0;         //total wall time
		std::vector<double> time_energy(3,0.0);
		std::vector<double> time_force(3,0.0);
		std::vector<double> time_symm(3,0.0);
	//file i/o
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		bool read_pot=false;
		std::string file_pot;
		std::vector<std::string> files_basis;//file - stores basis
	//string
		Token token;
		
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
			std::printf("bohr-r  (A)  = %.12f\n",units::BOHR);
			std::printf("hartree (eV) = %.12f\n",units::HARTREE);
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
		
		//======== rank 0 reads from file ========
		if(WORLD.rank()==0){
			
			//======== check the arguments ========
			if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
			
			//======== load the parameter file ========
			if(NNPTEQ_PRINT_STATUS>0) std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//======== open the parameter file ========
			if(NNPTEQ_PRINT_STATUS>0) std::cout<<"opening parameter file\n";
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+paramfile);
			
			//======== read in the parameters ========
			if(NNPTEQ_PRINT_STATUS>-1) std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL){
				token.read(string::trim_right(input,string::COMMENT),string::WS);
				if(token.end()) continue;//skip empty line
				const std::string tag=string::to_upper(token.next());
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
					if(wtype=="ENERGY") write.energy=string::boolean(token.next().c_str());
					else if(wtype=="FORCE") write.force=string::boolean(token.next().c_str());
					else if(wtype=="EWALD") write.ewald=string::boolean(token.next().c_str());
					else if(wtype=="INPUT") write.input=string::boolean(token.next().c_str());
					else if(wtype=="CHARGE") write.charge=string::boolean(token.next().c_str());
				}
				//flags - compute
				if(tag=="COMPUTE"){
					const std::string ctype=string::to_upper(token.next());
					if(ctype=="COUL") compute.coul=string::boolean(token.next().c_str());
					else if(ctype=="VDW") compute.vdw=string::boolean(token.next().c_str());
					else if(ctype=="FORCE") compute.force=string::boolean(token.next().c_str());
					else if(ctype=="NORM") compute.norm=string::boolean(token.next().c_str());
				} 
			}
			
			//======== set atom flags =========
			if(NNPTEQ_PRINT_STATUS>0) std::cout<<"setting atom flags\n";
			atomT.force=compute.force;
			
			//======== read - nnpteq =========
			if(NNPTEQ_PRINT_STATUS>0) std::cout<<"reading neural network training parameters\n";
			NNPTEQ::read(reader,nnpteq);
			
			//======== read - annp =========
			if(NNPTEQ_PRINT_STATUS>0) std::cout<<"reading neural network parameters\n";
			NN::ANNP::read(reader,annp);
			
			//======== close parameter file ========
			if(NNPTEQ_PRINT_STATUS>0) std::cout<<"closing parameter file\n";
			fclose(reader);
			reader=NULL;
			
			//======== (restart == false) ========
			if(!nnpteq.restart_){
				//======== (read potential == false) ========
				if(!read_pot){
					//resize the potential
					if(NNPTEQ_PRINT_STATUS>-1) std::cout<<"resizing potential\n";
					nnpteq.nnp().resize(types);
					//read basis files
					if(NNPTEQ_PRINT_STATUS>-1) std::cout<<"reading basis files\n";
					if(files_basis.size()!=nnpteq.nnp().ntypes()) throw std::runtime_error("main(int,char**): invalid number of basis files.");
					for(int i=0; i<nnpteq.nnp().ntypes(); ++i){
						const char* file=files_basis[i].c_str();
						const char* atomName=types[i].name().c_str();
						NNP::read_basis(file,nnpteq.nnp(),atomName);
					}
					//initialize the neural network hamiltonians
					if(NNPTEQ_PRINT_STATUS>-1) std::cout<<"initializing neural network hamiltonians\n";
					for(int i=0; i<nnpteq.nnp().ntypes(); ++i){
						NNH& nnhl=nnpteq.nnp().nnh(i);
						nnhl.type()=types[i];
						nnhl.nn().resize(annp,nnhl.nInput(),nh[i],2);
						nnhl.dOdZ().resize(nnhl.nn());
					}
				}
				//======== (read potential == true) ========
				if(read_pot){
					if(NNPTEQ_PRINT_STATUS>-1) std::cout<<"reading potential\n";
					NNP::read(file_pot.c_str(),nnpteq.nnp());
				}
			}
			//======== (restart == true) ========
			if(nnpteq.restart_){
				if(NNPTEQ_PRINT_STATUS>-1) std::cout<<"reading restart file\n";
				const std::string file=nnpteq.file_restart_;
				nnpteq.read_restart(file.c_str());
				nnpteq.restart()=true;
			}
			
			//======== print parameters ========
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"read_pot  = "<<read_pot<<"\n";
			std::cout<<"atom_type = "<<atomT<<"\n";
			std::cout<<"format    = "<<format<<"\n";
			std::cout<<"units     = "<<unitsys<<"\n";
			std::cout<<"mode      = "<<mode<<"\n";
			std::cout<<print::title("DATA FILES",strbuf)<<"\n";
			std::cout<<"data_train = \n"; for(int i=0; i<data[0].size(); ++i) std::cout<<"\t\t"<<data[0][i]<<"\n";
			std::cout<<"data_val   = \n"; for(int i=0; i<data[1].size(); ++i) std::cout<<"\t\t"<<data[1][i]<<"\n";
			std::cout<<"data_test  = \n"; for(int i=0; i<data[2].size(); ++i) std::cout<<"\t\t"<<data[2][i]<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("WRITE FLAGS",strbuf)<<"\n";
			std::cout<<"energy = "<<write.energy<<"\n";
			std::cout<<"ewald  = "<<write.ewald<<"\n";
			std::cout<<"inputs = "<<write.input<<"\n";
			std::cout<<"force  = "<<write.force<<"\n";
			std::cout<<"charge = "<<write.charge<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("COMPUTE FLAGS",strbuf)<<"\n";
			std::cout<<"coul  = "<<compute.coul<<"\n";
			std::cout<<"vdw   = "<<compute.vdw<<"\n";
			std::cout<<"force = "<<compute.force<<"\n";
			std::cout<<"norm  = "<<compute.norm<<"\n";
			std::cout<<"zero  = "<<compute.zero<<"\n";
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
			std::cout<<nnpteq<<"\n";
			std::cout<<nnpteq.nnp()<<"\n";
			
			//========= check the data =========
			if(mode==Mode::TRAIN && data[0].size()==0) throw std::invalid_argument("No training data provided.");
			if(mode==Mode::TRAIN && data[1].size()==0) throw std::invalid_argument("No validation data provided.");
			if(mode==Mode::TEST  && data[2].size()==0) throw std::invalid_argument("No test data provided.");
			if(mode==Mode::UNKNOWN) throw std::invalid_argument("Invalid calculation mode");
			if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			if(types.size()==0) throw std::invalid_argument("Invalid number of types.");
			
			//======== initialize the qeq potential ========
			if(NNPTEQ_PRINT_STATUS>-1) std::cout<<"initializing the qeq potential\n";
			nnpteq.qeq().pot()->resize(types.size());
			if(nnpteq.qeq().pot()->name()==ptnl::Pot::Name::GAUSS_CUT){
				ptnl::PotGaussCut& pot=static_cast<ptnl::PotGaussCut&>(*nnpteq.qeq().pot());
				for(int i=0; i<types.size(); ++i){
					pot.radius(i)=types[i].rcov().val();
					pot.f(i)=1;
				}
			}
			if(nnpteq.qeq().pot()->name()==ptnl::Pot::Name::GAUSS_DSF){
				ptnl::PotGaussDSF& pot=static_cast<ptnl::PotGaussDSF&>(*nnpteq.qeq().pot());
				for(int i=0; i<types.size(); ++i){
					pot.radius(i)=types[i].rcov().val();
					pot.f(i)=1;
				}
			}
			if(nnpteq.qeq().pot()->name()==ptnl::Pot::Name::GAUSS_LONG){
				ptnl::PotGaussLong& pot=static_cast<ptnl::PotGaussLong&>(*nnpteq.qeq().pot());
				for(int i=0; i<types.size(); ++i){
					pot.radius(i)=types[i].rcov().val();
					pot.f(i)=1;
				}
			}
			nnpteq.qeq().pot()->init();
			
		}
		
		//======== bcast the parameters ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		//general parameters
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.mpic());
		//mode
		MPI_Bcast(&mode,1,MPI_INT,0,WORLD.mpic());
		//atom type
		thread::bcast(WORLD.mpic(),0,atomT);
		thread::bcast(WORLD.mpic(),0,annp);
		//flags - compute
		MPI_Bcast(&compute.coul,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.vdw,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.norm,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.zero,1,MPI_C_BOOL,0,WORLD.mpic());
		//flags - writing
		MPI_Bcast(&write.energy,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.ewald,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.input,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.charge,1,MPI_C_BOOL,0,WORLD.mpic());
		//nnp_opt
		MPI_Bcast(&nBatch,1,MPI_INT,0,WORLD.mpic());
		//structures - format
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.mpic());
		//nnpteq
		thread::bcast(WORLD.mpic(),0,nnpteq);
		//alias
		int naliases=aliases.size();
		MPI_Bcast(&naliases,1,MPI_INT,0,WORLD.mpic());
		if(WORLD.rank()!=0) aliases.resize(naliases);
		for(int i=0; i<aliases.size(); ++i){
			thread::bcast(WORLD.mpic(),0,aliases[i]);
		}
		
		//======== set the unit system ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"setting the unit system\n";
		units::consts::init(unitsys);
		
		//************************************************************************************
		// READ DATA
		//************************************************************************************
		
		//======== rank 0 reads the data files (lists of structure files) ========
		if(WORLD.rank()==0){
			if(NNPTEQ_PRINT_STATUS>-1) std::cout<<"reading data\n";
			//==== read data ====
			for(int n=0; n<3; ++n){
				for(int i=0; i<data[n].size(); ++i){
					//open the data file
					if(NNPTEQ_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data[n][i]<<"\n";
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
			if(NNPTEQ_PRINT_DATA>1){
				for(int n=0; n<3; ++n){
					if(files[n].size()>0){
						std::cout<<print::buf(strbuf)<<"\n";
						std::cout<<print::title("FILES - TRAIN",strbuf)<<"\n";
						for(int i=0; i<files[n].size(); ++i) std::cout<<"\t"<<files[n][i]<<"\n";
						std::cout<<print::buf(strbuf)<<"\n";
					}
				}
			}
		}
		
		//======== bcast the file names =======
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"bcasting file names\n";
		//bcast names
		for(int n=0; n<3; ++n) thread::bcast(WORLD.mpic(),0,files[n]);
		//set number of structures
		for(int n=0; n<3; ++n) nstrucs[n]=files[n].size();
		//print number of structures
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DATA",strbuf)<<"\n";
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
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing batch communicator\n";
		//split WORLD into BATCH
		BATCH=WORLD.split(WORLD.color(WORLD.ncomm(nBatch)));
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
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"generating thread distributions\n";
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
		for(int n=0; n<3; ++n){
			indices[n].resize(nstrucs[n],-1);
			for(int i=0; i<indices[n].size(); ++i) indices[n][i]=i;
			std::random_shuffle(indices[n].begin(),indices[n].end());
			thread::bcast(WORLD.mpic(),0,indices[n]);
		}
		
		//======== read the structures ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"reading structures\n";
		for(int n=0; n<3; ++n){
			if(files[n].size()>0){
				//resize structure array
				strucs[n].resize(dist[n].size());
				//rank 0 of batch group reads structures
				if(BATCH.rank()==0){
					for(int i=0; i<dist[n].size(); ++i){
						const std::string& file=files[n][indices[n][dist[n].index(i)]];
						read_struc(file.c_str(),format,atomT,strucs[n][i]);
						if(NNPTEQ_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<strucs[n][i].energy()<<"\n";
					}
				}
				//broadcast structures to all other procs in the BATCH group
				for(int i=0; i<dist[n].size(); ++i){
					thread::bcast(BATCH.mpic(),0,strucs[n][i]);
				}
			}
		}
		
		//======== apply alias ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"applying aliases\n";
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
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"checking the structures\n";
		if(BATCH.rank()==0){
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					const std::string filename=files[n][indices[n][dist[n].index(i)]];
					const Structure& strucl=strucs[n][i];
					if(strucl.nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
					if(std::isinf(strucl.energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
					if(strucl.energy()!=strucl.energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
					if(std::fabs(strucl.energy())<math::constant::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
					if(compute.force){
						for(int j=0; j<strucl.nAtoms(); ++j){
							const double force=strucl.force(j).squaredNorm();
							if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<strucl.name(j)<<strucl.index(j)<<"\" in \""<<filename<<" has inf force.\n";
							if(force!=force) std::cout<<"WARNING: Atom \""<<strucl.name(j)<<strucl.index(j)<<"\" in \""<<filename<<" has nan force.\n";
						}
					}
					for(int j=0; j<strucl.nAtoms(); ++j){
						bool match=false;
						for(int k=0; k<nnpteq.nnp_.ntypes(); ++k){
							if(strucl.name(j)==nnpteq.nnp_.nnh(k).type().name()){
								match=true; break;
							}
						}
						if(!match) throw std::runtime_error(std::string("Could not find type for atom \"")+strucl.name(j)+std::string("\""));
					}
					if(NNPTEQ_PRINT_DATA>1) std::cout<<"\t"<<filename<<" "<<strucl.energy()<<" "<<WORLD.rank()<<"\n";
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
					strucs[n][i].type(j)=nnpteq.nnp_.index(strucs[n][i].name(j));
				}
			}
		}
		
		//======== set the charges ========
		if(atomT.charge){
			if(WORLD.rank()==0) std::cout<<"setting charges\n";
			for(int n=0; n<3; ++n){
				for(int i=0; i<dist[n].size(); ++i){
					for(int j=0; j<strucs[n][i].nAtoms(); ++j){
						strucs[n][i].charge(j)=nnpteq.nnp_.nnh(strucs[n][i].type(j)).type().charge().val();
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
						strucs[n][i].chi(j)=nnpteq.nnp_.nnh(strucs[n][i].type(j)).type().chi().val();
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
						strucs[n][i].eta(j)=nnpteq.nnp_.nnh(strucs[n][i].type(j)).type().eta().val();
					}
				}
			}
		}
		
		//************************************************************************************
		// SET INPUTS
		//************************************************************************************
		
		//======== initialize the symmetry functions ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				NNP::init(nnpteq.nnp_,strucs[n][i]);
			}
		}
		
		//======== compute the symmetry functions ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"setting the inputs (symmetry functions)\n";
		for(int n=0; n<3; ++n){
			clock.begin();
			if(dist[n].size()>0){
				//compute symmetry functions
				for(int i=BATCH.rank(); i<dist[n].size(); i+=BATCH.size()){
					if(NNPTEQ_PRINT_STATUS>0) std::cout<<"structure-train["<<n<<"]\n";
					NeighborList nlist(strucs[n][i],nnpteq.nnp_.rc());
					NNP::symm(nnpteq.nnp_,strucs[n][i],nlist);
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
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"subtracting ground-state energies\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].energy()-=nnpteq.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
				}
			}
		}
		
		//======== train the nn potential ========
		if(mode==Mode::TRAIN){
			if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"training the nn potential\n";
			nnpteq.train(dist[3].size(),strucs[0],strucs[1]);
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== add ground-state energies ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"adding ground-state energies\n";
		for(int n=0; n<3; ++n){
			for(int i=0; i<dist[n].size(); ++i){
				for(int j=0; j<strucs[n][i].nAtoms(); ++j){
					strucs[n][i].energy()+=nnpteq.nnp().nnh(strucs[n][i].type(j)).type().energy().val();
				}
			}
		}
		
		//************************************************************************************
		// EVALUATION
		//************************************************************************************
		
		//======== statistical data - energies/forces/errors ========
		std::vector<double> kendall(3,0);
		std::vector<Reduce<1> > r1_energy(3);
		std::vector<Reduce<1> > r1_qtot(3);
		std::vector<Reduce<2> > r2_energy(3);
		std::vector<Reduce<1> > r1_force(3);
		std::vector<std::vector<Reduce<2> > > r2_force(3,std::vector<Reduce<2> >(3));
		std::vector<std::vector<Reduce<1> > > r1_charge(3);
		std::vector<std::vector<Reduce<1> > > r1_chi(3);
		
		//======== compute the final chi ========
		if(WORLD.rank()==0) std::cout<<"computing final chi\n";
		for(int ii=0; ii<3; ++ii){
			if(dist[ii].size()>0){
				r1_chi[ii].resize(nnpteq.nnp().ntypes());
				//compute electronegativity
				for(int nn=0; nn<dist[ii].size(); ++nn){
					for(int n=0; n<strucs[ii][nn].nAtoms(); ++n){
						//find the element index in the nn potential
						const int type=strucs[ii][nn].type(n);
						//execute the network
						nnpteq.nnp().nnh(type).nn().fp(strucs[ii][nn].symm(n));
						//compute the electronegativity
						strucs[ii][nn].chi(n)=nnpteq.nnp().nnh(type).nn().out()[1]+nnpteq.nnp().nnh(type).type().chi().val()*FCHI;
						if(BATCH.rank()==0) r1_chi[ii][type].push(strucs[ii][nn].chi(n));
					}
				}
				//reduce electronegativity
				for(int i=0; i<nnpteq.nnp().ntypes(); ++i){
					std::vector<Reduce<1> > reduceV(WORLD.size());
					thread::gather(r1_chi[ii][i],reduceV,WORLD.mpic());
					Reduce<1> rt; for(int j=0; j<WORLD.size(); ++j) rt+=reduceV[j];
					if(WORLD.rank()==0){
						std::cout<<"chi - ["<<ii<<"]["<<i<<"] = "<<rt.avg()<<" "<<rt.dev()<<" "<<rt.min()<<" "<<rt.max()<<"\n";
					}
				}
			}
		}
		
		//======== compute the final energies ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final energies\n";
		for(int n=0; n<3; ++n){
			if(dist[n].size()>0){
				std::vector<double> energy_n(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_q(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_v(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_n_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_q_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_v_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<int> natoms(nstrucs[n],0); std::vector<int> natoms_t(nstrucs[n],0);
				//compute energies
				clock.begin();
				for(int i=0; i<dist[n].size(); ++i){
					if(NNPTEQ_PRINT_STATUS>0) std::cout<<"structure-train["<<WORLD.rank()<<"]["<<i<<"]\n";
					energy_r[dist[n].index(i)]=strucs[n][i].energy();
					NeighborList nlist(strucs[n][i],nnpteq.qeq().pot()->rc());
					nnpteq.qeq().qt(strucs[n][i],nlist);
					const double energyQ=-0.5*nnpteq.qeq().x().dot(nnpteq.qeq().b());
					const double energyV=NNP::energy(nnpteq.nnp(),strucs[n][i]);
					energy_n[dist[n].index(i)]=energyQ+energyV;
					energy_q[dist[n].index(i)]=energyQ;
					energy_v[dist[n].index(i)]=energyV;
					natoms[dist[n].index(i)]=strucs[n][i].nAtoms();
				}
				clock.end();
				time_energy[n]=clock.duration();
				MPI_Reduce(energy_r.data(),energy_r_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
				MPI_Reduce(energy_n.data(),energy_n_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
				MPI_Reduce(energy_q.data(),energy_q_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
				MPI_Reduce(energy_v.data(),energy_v_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
				MPI_Reduce(natoms.data(),natoms_t.data(),nstrucs[n],MPI_INT,MPI_MAX,0,WORLD.mpic());
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
					for(int i=0; i<nstrucs[n]; ++i) energy_q_t[i]/=natoms_t[i];
					for(int i=0; i<nstrucs[n]; ++i) energy_v_t[i]/=natoms_t[i];
				}
				//write energies
				if(write.energy && WORLD.rank()==0){
					std::string file;
					switch(n){
						case 0: file="nnp_energy_train.dat"; break;
						case 1: file="nnp_energy_val.dat"; break;
						case 2: file="nnp_energy_test.dat"; break;
						default: file="ERROR.dat"; break;
					}
					FILE* writer=fopen(file.c_str(),"w");
					if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
					else{
						std::vector<std::pair<int,double> > energy_r_pair(nstrucs[n]);
						std::vector<std::pair<int,double> > energy_n_pair(nstrucs[n]);
						std::vector<std::pair<int,double> > energy_q_pair(nstrucs[n]);
						std::vector<std::pair<int,double> > energy_v_pair(nstrucs[n]);
						for(int i=0; i<nstrucs[n]; ++i){
							energy_r_pair[i].first=indices[n][i];
							energy_r_pair[i].second=energy_r_t[i];
							energy_n_pair[i].first=indices[n][i];
							energy_n_pair[i].second=energy_n_t[i];
							energy_q_pair[i].first=indices[n][i];
							energy_q_pair[i].second=energy_q_t[i];
							energy_v_pair[i].first=indices[n][i];
							energy_v_pair[i].second=energy_v_t[i];
						}
						std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
						std::sort(energy_n_pair.begin(),energy_n_pair.end(),compare_pair);
						std::sort(energy_q_pair.begin(),energy_q_pair.end(),compare_pair);
						std::sort(energy_v_pair.begin(),energy_v_pair.end(),compare_pair);
						fprintf(writer,"#STRUCTURE ENERGY_REF ENERGY_NN ENERGY_Q ENERGY_V\n");
						for(int i=0; i<nstrucs[n]; ++i){
							fprintf(writer,"%s %f %f %f %f\n",files[n][i].c_str(),
								energy_r_pair[i].second,energy_n_pair[i].second,
								energy_q_pair[i].second,energy_v_pair[i].second
							);
						}
						fclose(writer); writer=NULL;
					}
				}
			}
		}
		
		//======== compute the final charges ========
		if(WORLD.rank()==0) std::cout<<"computing final charges\n";
		for(int ii=0; ii<3; ++ii){
			if(dist[ii].size()>0){
				r1_charge[ii].resize(nnpteq.nnp().ntypes());
				for(int i=0; i<dist[ii].size(); ++i){
					double qtot=0;
					for(int n=0; n<strucs[ii][i].nAtoms(); ++n){
						const int type=nnpteq.nnp().index(strucs[ii][i].name(n));
						if(BATCH.rank()==0) r1_charge[ii][type].push(strucs[ii][i].charge(n));
						qtot+=strucs[ii][i].charge(n);
					}
					if(BATCH.rank()==0) r1_qtot[ii].push(qtot);
				}
				std::vector<Reduce<1> > reduceQ(WORLD.size());
				thread::gather(r1_qtot[ii],reduceQ,WORLD.mpic());
				Reduce<1> rqt; for(int j=0; j<WORLD.size(); ++j) rqt+=reduceQ[j];
				if(WORLD.rank()==0){
					std::cout<<"qtot["<<ii<<"] = "<<rqt.avg()<<" "<<rqt.dev()<<" "<<rqt.min()<<" "<<rqt.max()<<"\n";
				}
				for(int i=0; i<nnpteq.nnp().ntypes(); ++i){
					std::vector<Reduce<1> > reduceV(WORLD.size());
					thread::gather(r1_charge[ii][i],reduceV,WORLD.mpic());
					Reduce<1> rt; for(int j=0; j<WORLD.size(); ++j) rt+=reduceV[j];
					if(WORLD.rank()==0){
						std::cout<<"chg ["<<ii<<"]["<<i<<"] = "<<rt.avg()<<" "<<rt.dev()<<" "<<rt.min()<<" "<<rt.max()<<"\n";
					}
				}
			}
		}
		if(write.charge==true){
			for(int i=0; i<dist[0].size(); ++i){
				std::cout<<files[0][indices[0][dist[0].index(i)]]<<"\n";
				for(int n=0; n<strucs[0][i].nAtoms(); ++n){
					std::cout<<strucs[0][i].name(n)<<" "<<
					strucs[0][i].posn(n)[0]<<" "<<strucs[0][i].posn(n)[1]<<" "<<strucs[0][i].posn(n)[2]<<" "<<
					strucs[0][i].charge(n)<<" "<<strucs[0][i].chi(n)<<" "<<strucs[0][i].eta(n)<<"\n";
				}
			}
		}
		
		//======== compute the final forces ========
		/*
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0 && nnpteq.force_) std::cout<<"computing final forces\n";
		//==== training structures ====
		if(dist_train.size()>0 && nnpteq.force_){
			//compute forces
			clock.begin();
			for(int n=0; n<dist_train.size(); ++n){
				if(NNPTEQ_PRINT_STATUS>0) std::cout<<"structure-train["<<n<<"]\n";
				Structure& struc=struc_train[n];
				//compute exact forces
				std::vector<Eigen::Vector3d> f_r(struc.nAtoms());
				for(int i=0; i<struc.nAtoms(); ++i) f_r[i]=struc.force(i);
				//compute nn forces
				NeighborList nlist(struc,nnpteq.nnp_.rc());
				NNP::force(nnpteq.nnp_,struc,nlist);
				std::vector<Eigen::Vector3d> f_n(struc.nAtoms());
				for(int i=0; i<struc.nAtoms(); ++i) f_n[i]=struc.force(i);
				//compute statistics
				if(BATCH.rank()==0){
					for(int i=0; i<struc.nAtoms(); ++i){
						r1_force_train.push((f_r[i]-f_n[i]).norm());
						r2_force_train[0].push(f_r[i][0],f_n[i][0]);
						r2_force_train[1].push(f_r[i][1],f_n[i][1]);
						r2_force_train[2].push(f_r[i][2],f_n[i][2]);
					}
				}
			}
			clock.end();
			time_force_train=clock.duration();
			//accumulate statistics
			std::vector<Reduce<1> > r1fv(WORLD.size());
			thread::gather(r1_force_train,r1fv,WORLD.mpic());
			if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r1_force_train+=r1fv[i];
			for(int n=0; n<3; ++n){
				std::vector<Reduce<2> > r2fv(WORLD.size());
				thread::gather(r2_force_train[n],r2fv,WORLD.mpic());
				if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r2_force_train[n]+=r2fv[i];
			}
		}
		*/
		
		//======== write the inputs ========
		for(int ii=0; ii<3; ++ii){
			if(dist[ii].size()>0 && write.input){
				std::string file;
				switch(ii){
					case 0: file="nnp_inputs_train.dat"; break;
					case 1: file="nnp_inputs_val.dat"; break;
					case 2: file="nnp_inputs_test.dat"; break;
					default: file="ERROR.dat"; break;
				}
				for(int ii=0; ii<WORLD.size(); ++ii){
					if(WORLD.rank()==ii){
						FILE* writer=NULL;
						if(ii==0) writer=fopen(file.c_str(),"w");
						else writer=fopen(file.c_str(),"a");
						if(writer!=NULL){
							for(int n=0; n<dist[ii].size(); ++n){
								for(int i=0; i<strucs[ii][n].nAtoms(); ++i){
									fprintf(writer,"%s%i ",strucs[ii][n].name(i).c_str(),i);
									for(int j=0; j<strucs[ii][n].symm(i).size(); ++j){
										fprintf(writer,"%f ",strucs[ii][n].symm(i)[j]);
									}
									fprintf(writer,"\n");
								}
							}
							fclose(writer); writer=NULL;
						} else std::cout<<"WARNING: Could not open inputs file for training structures\n";
					}
					MPI_Barrier(WORLD.mpic());
				}
			}
		}
		
		//======== stop the wall clock ========
		if(WORLD.rank()==0) clock_wall.end();
		if(WORLD.rank()==0) time_wall=clock_wall.duration();
		
		//************************************************************************************
		// OUTPUT
		//************************************************************************************
		
		//======== print the timing info ========
		for(int n=0; n<3; ++n){
			MPI_Allreduce(MPI_IN_PLACE,&time_symm[n],1,MPI_DOUBLE,MPI_SUM,WORLD.mpic()); 
			MPI_Allreduce(MPI_IN_PLACE,&time_energy[n],1,MPI_DOUBLE,MPI_SUM,WORLD.mpic()); 
			MPI_Allreduce(MPI_IN_PLACE,&time_force[n],1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
			time_symm[n]/=WORLD.size();
			time_energy[n]/=WORLD.size();
			time_force[n]/=WORLD.size();
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
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing the nn's\n";
		if(WORLD.rank()==0){
			NNP::write(nnpteq.file_ann_.c_str(),nnpteq.nnp_);
		}
		//======== write restart file ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing restart file\n";
		if(WORLD.rank()==0){
			nnpteq.write_restart(nnpteq.file_restart_.c_str());
		}
		
		//======== finalize mpi ========
		if(NNPTEQ_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"finalizing mpi\n";
		std::cout<<std::flush;
		MPI_Comm_free(&BATCH.mpic());
		MPI_Barrier(WORLD.mpic());
		MPI_Finalize();
	}catch(std::exception& e){
		std::cout<<"ERROR in nnpteq::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//======== free local variables ========
	delete[] paramfile;
	delete[] input;
	delete[] strbuf;
	
	return 0;
}
