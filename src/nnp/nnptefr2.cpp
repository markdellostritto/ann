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
#include "torch/pot_gauss_long.hpp"
#include "torch/pot_ldamp_long.hpp"
#include "torch/pot_ldamp_dsf.hpp"
#include "torch/pot_pauli.hpp"
// ml
#include "ml/pca.hpp"
// nnpte
#include "nnp/nnptefr2.hpp"

static bool compare_pair(const std::pair<int,double>& p1, const std::pair<int,double>& p2){
	return p1.first<p2.first;
}

//************************************************************
// MPI Communicators
//************************************************************

thread::Comm WORLD;// all processors

//************************************************************
// serialization
//************************************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNPTEFR& obj){
   if(NNPTEFR_PRINT_FUNC>0) std::cout<<"nbytes(const NNPTEFR&)\n";
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
		size+=nbytes(obj.batcht_);
		size+=nbytes(obj.batchv_);
		size+=nbytes(obj.obj_);
		size+=nbytes(obj.iter_);
		size+=nbytes(obj.algo_);
		size+=nbytes(obj.decay_);
		size+=sizeof(Norm);
		size+=sizeof(PreScale);
		size+=sizeof(PreBias);
		size+=sizeof(Regularization);
		size+=sizeof(double);//inscale_
		size+=sizeof(double);//inbias_
		size+=sizeof(double);//delta_
		size+=sizeof(double);//beta_
		size+=sizeof(double);//betai_
		size+=sizeof(double);//rmse_t_a_
		size+=sizeof(double);//rmse_v_a_
		size+=sizeof(double);//eta
	//return the size
		return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNPTEFR& obj, char* arr){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"pack(const NNPTEFR&,char*)\n";
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
		pos+=pack(obj.batcht_,arr+pos);
		pos+=pack(obj.batchv_,arr+pos);
		pos+=pack(obj.obj_,arr+pos);
		pos+=pack(obj.iter_,arr+pos);
		pos+=pack(obj.algo_,arr+pos);
		pos+=pack(obj.decay_,arr+pos);
		std::memcpy(arr+pos,&obj.norm_,sizeof(Norm)); pos+=sizeof(Norm);
		std::memcpy(arr+pos,&obj.prescale_,sizeof(PreScale)); pos+=sizeof(PreScale);
		std::memcpy(arr+pos,&obj.prebias_,sizeof(PreBias)); pos+=sizeof(PreBias);
		std::memcpy(arr+pos,&obj.reg_,sizeof(Regularization)); pos+=sizeof(Regularization);
		std::memcpy(arr+pos,&obj.inscale_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.inbias_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.delta_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.beta_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.betai_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.rmse_t_a_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.rmse_v_a_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.eta_,sizeof(double)); pos+=sizeof(double);
	//return bytes written
		return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNPTEFR& obj, const char* arr){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"unpack(const NNPTEFR&,char*)\n";
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
		pos+=unpack(obj.batcht_,arr+pos);
		pos+=unpack(obj.batchv_,arr+pos);
		pos+=unpack(obj.obj_,arr+pos);
		pos+=unpack(obj.iter_,arr+pos);
		pos+=unpack(obj.algo_,arr+pos);
		pos+=unpack(obj.decay_,arr+pos);
		std::memcpy(&obj.norm_,arr+pos,sizeof(Norm)); pos+=sizeof(Norm);
		std::memcpy(&obj.prescale_,arr+pos,sizeof(PreScale)); pos+=sizeof(PreScale);
		std::memcpy(&obj.prebias_,arr+pos,sizeof(PreBias)); pos+=sizeof(PreBias);
		std::memcpy(&obj.reg_,arr+pos,sizeof(Regularization)); pos+=sizeof(Regularization);
		std::memcpy(&obj.inscale_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.inbias_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.delta_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.beta_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.betai_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.rmse_t_a_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.rmse_v_a_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.eta_,arr+pos,sizeof(double)); pos+=sizeof(double);
		obj.deltai()=1.0/obj.delta();
		obj.delta2()=obj.delta()*obj.delta();
	//return bytes read
		return pos;
}

}

//************************************************************
// Perturb
//************************************************************

std::ostream& operator<<(std::ostream& out, const Perturb& perturb){
	switch(perturb){
		case Perturb::CONSTANT: out<<"CONSTANT"; break;
		case Perturb::COVALENT: out<<"COVALENT"; break;
		default: out<<"NONE"; break;
	}
	return out;
}

const char* Perturb::name(const Perturb& perturb){
	switch(perturb){
		case Perturb::CONSTANT: return "CONSTANT";
		case Perturb::COVALENT: return "COVALENT";
		default: return "NONE";
	}
}

Perturb Perturb::read(const char* str){
	if(std::strcmp(str,"CONSTANT")==0) return Perturb::CONSTANT;
	else if(std::strcmp(str,"COVALENT")==0) return Perturb::COVALENT;
	else return Perturb::NONE;
}

//************************************************************
// PreScale
//************************************************************

std::ostream& operator<<(std::ostream& out, const PreScale& prescale){
	switch(prescale){
		case PreScale::IDENTITY: out<<"IDENTITY"; break;
		case PreScale::DEV: out<<"DEV"; break;
		case PreScale::MINMAX: out<<"MINMAX"; break;
		case PreScale::MAX: out<<"MAX"; break;
		default: out<<"NONE"; break;
	}
	return out;
}

const char* PreScale::name(const PreScale& prescale){
	switch(prescale){
		case PreScale::IDENTITY: return "IDENTITY";
		case PreScale::DEV: return "DEV";
		case PreScale::MINMAX: return "MINMAX";
		case PreScale::MAX: return "MAX";
		default: return "NONE";
	}
}

PreScale PreScale::read(const char* str){
	if(std::strcmp(str,"IDENTITY")==0) return PreScale::IDENTITY;
	else if(std::strcmp(str,"DEV")==0) return PreScale::DEV;
	else if(std::strcmp(str,"MINMAX")==0) return PreScale::MINMAX;
	else if(std::strcmp(str,"MAX")==0) return PreScale::MAX;
	else return PreScale::NONE;
}

//************************************************************
// PreBias
//************************************************************

std::ostream& operator<<(std::ostream& out, const PreBias& prebias){
	switch(prebias){
		case PreBias::IDENTITY: out<<"IDENTITY"; break;
		case PreBias::MEAN: out<<"MEAN"; break;
		case PreBias::HAR: out<<"HAR"; break;
		case PreBias::RMS: out<<"RMS"; break;
		case PreBias::MID: out<<"MID"; break;
		case PreBias::MIN: out<<"MIN"; break;
		default: out<<"NONE"; break;
	}
	return out;
}

const char* PreBias::name(const PreBias& prebias){
	switch(prebias){
		case PreBias::IDENTITY: return "IDENTITY";
		case PreBias::MEAN: return "MEAN";
		case PreBias::HAR: return "HAR";
		case PreBias::RMS: return "RMS";
		case PreBias::MID: return "MID";
		case PreBias::MIN: return "MIN";
		default: return "NONE";
	}
}

PreBias PreBias::read(const char* str){
	if(std::strcmp(str,"IDENTITY")==0) return PreBias::IDENTITY;
	else if(std::strcmp(str,"MEAN")==0) return PreBias::MEAN;
	else if(std::strcmp(str,"HAR")==0) return PreBias::HAR;
	else if(std::strcmp(str,"RMS")==0) return PreBias::RMS;
	else if(std::strcmp(str,"MID")==0) return PreBias::MID;
	else if(std::strcmp(str,"MIN")==0) return PreBias::MIN;
	else return PreBias::NONE;
}

//************************************************************
// Norm
//************************************************************

std::ostream& operator<<(std::ostream& out, const Norm& norm){
	switch(norm){
		case Norm::IDENTITY: out<<"IDENTITY"; break;
		case Norm::LINEAR: out<<"LINEAR"; break;
		case Norm::SQRT: out<<"SQRT"; break;
		case Norm::CBRT: out<<"CBRT"; break;
		case Norm::LOG: out<<"LOG"; break;
		default: out<<"NONE"; break;
	}
	return out;
}

const char* Norm::name(const Norm& norm){
	switch(norm){
		case Norm::IDENTITY: return "IDENTITY";
		case Norm::LINEAR: return "LINEAR";
		case Norm::SQRT: return "SQRT";
		case Norm::CBRT: return "CBRT";
		case Norm::LOG: return "LOG";
		default: return "NONE";
	}
}

Norm Norm::read(const char* str){
	if(std::strcmp(str,"IDENTITY")==0) return Norm::IDENTITY;
	else if(std::strcmp(str,"LINEAR")==0) return Norm::LINEAR;
	else if(std::strcmp(str,"SQRT")==0) return Norm::SQRT;
	else if(std::strcmp(str,"CBRT")==0) return Norm::CBRT;
	else if(std::strcmp(str,"LOG")==0) return Norm::LOG;
	else return Norm::NONE;
}

//************************************************************
// Mode
//************************************************************

std::ostream& operator<<(std::ostream& out, const Mode& mode){
	switch(mode){
		case Mode::TRAIN: out<<"TRAIN"; break;
		case Mode::TEST: out<<"TEST"; break;
		case Mode::SYMM: out<<"SYMM"; break;
		default: out<<"NONE"; break;
	}
	return out;
}

const char* Mode::name(const Mode& mode){
	switch(mode){
		case Mode::TRAIN: return "TRAIN";
		case Mode::TEST: return "TEST";
		case Mode::SYMM: return "SYMM";
		default: return "NONE";
	}
}

Mode Mode::read(const char* str){
	if(std::strcmp(str,"TRAIN")==0) return Mode::TRAIN;
	else if(std::strcmp(str,"TEST")==0) return Mode::TEST;
	else if(std::strcmp(str,"SYMM")==0) return Mode::SYMM;
	else return Mode::NONE;
}

//************************************************************
// Regularization
//************************************************************

std::ostream& operator<<(std::ostream& out, const Regularization& reg){
	switch(reg){
		case Regularization::NONE: out<<"NONE"; break;
		case Regularization::LASSO: out<<"LASSO"; break;
		case Regularization::RIDGE: out<<"RIDGE"; break;
		case Regularization::HUBER: out<<"HUBER"; break;
		case Regularization::ASINH: out<<"ASINH"; break;
		default: out<<"NONE"; break;
	}
	return out;
}

const char* Regularization::name(const Regularization& reg){
	switch(reg){
		case Regularization::NONE: return "NONE";
		case Regularization::LASSO: return "LASSO";
		case Regularization::RIDGE: return "RIDGE";
		case Regularization::HUBER: return "HUBER";
		case Regularization::ASINH: return "ASINH";
		default: return "NONE";
	}
}

Regularization Regularization::read(const char* str){
	if(std::strcmp(str,"NONE")==0) return Regularization::NONE;
	else if(std::strcmp(str,"LASSO")==0) return Regularization::LASSO;
	else if(std::strcmp(str,"RIDGE")==0) return Regularization::RIDGE;
	else if(std::strcmp(str,"HUBER")==0) return Regularization::HUBER;
	else if(std::strcmp(str,"ASINH")==0) return Regularization::ASINH;
	else return Regularization::NONE;
}

//************************************************************
// NNPTEFR - Neural Network Potential - Optimization
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPTEFR& nnpte){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NNPTEFR",str)<<"\n";
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
	out<<"algo         = "<<nnpte.algo_<<"\n";
	out<<"decay        = "<<nnpte.decay_<<"\n";
	out<<"n_print      = "<<nnpte.iter().nPrint()<<"\n";
	out<<"n_write      = "<<nnpte.iter().nWrite()<<"\n";
	out<<"max          = "<<nnpte.iter().max()<<"\n";
	out<<"stop         = "<<nnpte.iter().stop()<<"\n";
	out<<"loss         = "<<nnpte.iter().loss()<<"\n";
	out<<"tol          = "<<nnpte.iter().tol()<<"\n";
	out<<"gamma        = "<<nnpte.obj().gamma()<<"\n";
	out<<"delta        = "<<nnpte.delta()<<"\n";
	out<<"beta         = "<<nnpte.beta()<<"\n";
	out<<"eta          = "<<nnpte.eta()<<"\n";
	out<<"norm         = "<<nnpte.norm()<<"\n";
	out<<"reg          = "<<nnpte.reg()<<"\n";
	out<<"prescale     = "<<nnpte.prescale()<<"\n";
	out<<"prebias      = "<<nnpte.prebias()<<"\n";
	out<<"inscale      = "<<nnpte.inscale()<<"\n";
	out<<"inbias       = "<<nnpte.inbias()<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

NNPTEFR::NNPTEFR(){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"NNP::NNPTEFR():\n";
	defaults();
};

void NNPTEFR::defaults(){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"NNP::defaults():\n";
	//nnp
		nTypes_=0;
	//input/output
		file_params_="nnp_params.dat";
		file_error_="nnp_error.dat";
		file_restart_="nnpte.restart";
		file_ann_="ann";
	//flags
		restart_=false;
		reset_=false;
		wparams_=false;
	//optimization
		norm_=Norm::NONE;
		prescale_=PreScale::IDENTITY;
		prebias_=PreBias::IDENTITY;
		reg_=Regularization::NONE;
		inscale_=1.0;
		inbias_=0.0;
		delta_=1.0;
		deltai_=1.0;
		delta2_=1.0;
		beta_=0.999;
		betai_=beta_;
		rmse_t_a_=0;
		rmse_v_a_=0;
		eta_=0.0;
		seed_=-1;
	//error
		error_[0]=0;//loss - train
		error_[1]=0;//loss - val
		error_[2]=0;//rmse - train
		error_[3]=0;//rmse - val
}

void NNPTEFR::clear(){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"NNP::clear():\n";
	//elements
		nTypes_=0;
		gElement_.clear();
		pElement_.clear();
	//nnp
		nnp_.clear();
	//optimization
		batcht_.clear();
		batchv_.clear();
		obj_.clear();
		iter_.clear();
		rmse_t_a_=0;
		rmse_v_a_=0;
	//error
		error_[0]=0;//loss - train
		error_[1]=0;//loss - val
		error_[2]=0;//rmse - train
		error_[3]=0;//rmse - val
}

void NNPTEFR::write_restart(const char* file){
	if(NNPTEFR_PRINT_FUNC>1) std::cout<<"NNPTEFR::write_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* writer=NULL;
	bool error=false;
	try{
		//open file
		writer=fopen(file,"wb");
		if(writer==NULL) throw std::runtime_error(std::string("NNPTEFR::write_restart(const char*): Could not open file: ")+file);
		//allocate buffer
		const int nBytes=serialize::nbytes(*this);
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTEFR::write_restart(const char*): Could not allocate memory.");
		//write to buffer
		serialize::pack(*this,arr);
		//write to file
		const int nWrite=fwrite(arr,sizeof(char),nBytes,writer);
		if(nWrite!=nBytes) throw std::runtime_error("NNPTEFR::write_restart(const char*): Write error.");
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
	if(error) throw std::runtime_error("NNPTEFR::write_restart(const char*): Failed to write");
}

void NNPTEFR::read_restart(const char* file){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"NNPTEFR::read_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* reader=NULL;
	bool error=false;
	try{
		//open file
		reader=fopen(file,"rb");
		if(reader==NULL) throw std::runtime_error(std::string("NNPTEFR::read_restart(const char*): Could not open file: ")+std::string(file));
		//find size
		std::fseek(reader,0,SEEK_END);
		const int nBytes=std::ftell(reader);
		std::fseek(reader,0,SEEK_SET);
		//allocate buffer
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("NNPTEFR::read_restart(const char*): Could not allocate memory.");
		//read from file
		const int nRead=fread(arr,sizeof(char),nBytes,reader);
		if(nRead!=nBytes) throw std::runtime_error("NNPTEFR::read_restart(const char*): Read error.");
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
	if(error) throw std::runtime_error("NNPTEFR::read_restart(const char*): Failed to read");
}

void NNPTEFR::train(int batchSize, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"NNPTEFR::train(NNP&,std::vector<Structure>&,int):\n";
	//====== local function variables ======
	//statistics
		std::vector<int> N;//total number of inputs for each element
		std::vector<Eigen::VectorXd> har_in;//average of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> avg_in;//average of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> max_in;//max of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> min_in;//min of the inputs for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> dev_in;//average of the stddev for each element (nnp_.nSpecies_ x nInput_)
		std::vector<Eigen::VectorXd> rms_in;//average of the stddev for each element (nnp_.nSpecies_ x nInput_)
	//timing
		Clock clock;
	
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"training NN potential\n";
	
	//====== initialize random number generator ======
	if(seed_>0) rngen_=std::mt19937(seed_);
	else rngen_=std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
	
	//====== check the parameters ======
	if(batchSize<=0) throw std::invalid_argument("NNPTEFR::train(int): Invalid batch size.");
	if(struc_train.size()==0) throw std::invalid_argument("NNPTEFR::train(int): No training data provided.");
	if(struc_val.size()==0) throw std::invalid_argument("NNPTEFR::train(int): No validation data provided.");
	
	//====== get the number of structures ======
	int nBatch=batchSize;
	int nTrain=struc_train.size();
	int nVal=struc_val.size();
	MPI_Allreduce(MPI_IN_PLACE,&nBatch,1,MPI_INT,MPI_SUM,WORLD.mpic());
	MPI_Allreduce(MPI_IN_PLACE,&nTrain,1,MPI_INT,MPI_SUM,WORLD.mpic());
	MPI_Allreduce(MPI_IN_PLACE,&nVal,1,MPI_INT,MPI_SUM,WORLD.mpic());
	
	//====== set the normalization constants ======
	normt_.resize(struc_train.size());
	normv_.resize(struc_val.size());
	for(int i=0; i<struc_train.size(); ++i){
		const double nAtoms=struc_train[i].nAtoms();
		switch(norm_){
			case Norm::IDENTITY: normt_[i]=1.0; break;
			case Norm::LINEAR: normt_[i]=1.0/nAtoms; break;
			case Norm::SQRT: normt_[i]=1.0/sqrt(nAtoms); break;
			case Norm::CBRT: normt_[i]=1.0/cbrt(nAtoms); break;
			case Norm::LOG: normt_[i]=1.0/(log(nAtoms)+1.0); break;
			default: throw std::invalid_argument("Invalid normalization method."); break;
		}
	}
	for(int i=0; i<struc_val.size(); ++i){
		const double nAtoms=struc_val[i].nAtoms();
		switch(norm_){
			case Norm::IDENTITY: normv_[i]=1.0; break;
			case Norm::LINEAR: normv_[i]=1.0/nAtoms; break;
			case Norm::SQRT: normv_[i]=1.0/sqrt(nAtoms); break;
			case Norm::CBRT: normv_[i]=1.0/cbrt(nAtoms); break;
			case Norm::LOG: normv_[i]=1.0/(log(nAtoms)+1.0); break;
			default: throw std::invalid_argument("Invalid normalization method."); break;
		}
	}
	
	//====== resize the optimization data ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing the optimization data\n";
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
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resizing gradient data\n";
	cost_.resize(nTypes_);
	for(int n=0; n<nTypes_; ++n){
		cost_[n].resize(nnp_.nnh(n).nn());
	}
	dodp_.resize(nTypes_);
	for(int n=0; n<nTypes_; ++n){
		dodp_[n].resize(nnp_.nnh(n).nn());
	}
	
	//====== compute the number of atoms of each element ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing the number of atoms of each element\n";
	std::vector<double> nAtoms_(nTypes_,0);
	for(int i=0; i<struc_train.size(); ++i){
		for(int j=0; j<struc_train[i].nAtoms(); ++j){
			++nAtoms_[struc_train[i].type(j)];
		}
	}
	MPI_Allreduce(MPI_IN_PLACE,nAtoms_.data(),nTypes_,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
	if(NNPTEFR_PRINT_DATA>-1 && WORLD.rank()==0){
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
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting indices and batch\n";
	batcht_.resize(batchSize,struc_train.size());
	batchv_.resize(batchSize,struc_val.size());
	std::shuffle(batcht_.data(),batcht_.data()+batcht_.capacity(),rngen_);
	std::shuffle(batchv_.data(),batchv_.data()+batchv_.capacity(),rngen_);
	
	//====== collect input statistics ======
	//resize arrays
	N.resize(nTypes_,0);
	max_in.resize(nTypes_);
	min_in.resize(nTypes_);
	har_in.resize(nTypes_);
	avg_in.resize(nTypes_);
	dev_in.resize(nTypes_);
	rms_in.resize(nTypes_);
	for(int n=0; n<nTypes_; ++n){
		const int nInput=nnp_.nnh(n).nInput();
		max_in[n]=Eigen::VectorXd::Constant(nInput,-1.0*std::numeric_limits<double>::max());
		min_in[n]=Eigen::VectorXd::Constant(nInput,1.0*std::numeric_limits<double>::max());
		har_in[n]=Eigen::VectorXd::Zero(nInput);
		avg_in[n]=Eigen::VectorXd::Zero(nInput);
		dev_in[n]=Eigen::VectorXd::Zero(nInput);
		rms_in[n]=Eigen::VectorXd::Zero(nInput);
	}
	//compute the total number
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the total number\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			++N[struc_train[n].type(i)];
		}
	}
	//accumulate the number
	MPI_Allreduce(MPI_IN_PLACE,N.data(),nTypes_,MPI_INT,MPI_SUM,WORLD.mpic());
	//compute the max/min
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the max/min\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		for(int i=0; i<strucl.nAtoms(); ++i){
			const int type=strucl.type(i);
			for(int k=0; k<nnp_.nnh(type).nInput(); ++k){
				if(strucl.symm(i)[k]>max_in[type][k]) max_in[type][k]=strucl.symm(i)[k];
				if(strucl.symm(i)[k]<min_in[type][k]) min_in[type][k]=strucl.symm(i)[k];
			}
		}
	}
	for(int i=0; i<nTypes_; ++i){
		MPI_Allreduce(MPI_IN_PLACE,min_in[i].data(),min_in[i].size(),MPI_DOUBLE,MPI_MIN,WORLD.mpic());
		MPI_Allreduce(MPI_IN_PLACE,max_in[i].data(),max_in[i].size(),MPI_DOUBLE,MPI_MAX,WORLD.mpic());
	}
	//compute the harmonic mean
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the average\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			const Eigen::VectorXd tmp=1.0/struc_train[n].symm(i).array();
			har_in[struc_train[n].type(i)].noalias()+=tmp;
		}
	}
	for(int i=0; i<nTypes_; ++i){
		MPI_Allreduce(MPI_IN_PLACE,har_in[i].data(),har_in[i].size(),MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		const Eigen::VectorXd tmp=N[i]/har_in[i].array();
		har_in[i]=tmp;
	}
	//compute the average
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the average\n";
	for(int n=0; n<struc_train.size(); ++n){
		for(int i=0; i<struc_train[n].nAtoms(); ++i){
			avg_in[struc_train[n].type(i)].noalias()+=struc_train[n].symm(i);
		}
	}
	for(int i=0; i<nTypes_; ++i){
		MPI_Allreduce(MPI_IN_PLACE,avg_in[i].data(),avg_in[i].size(),MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		avg_in[i]/=N[i];
	}
	//compute the stddev
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the stddev\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		for(int i=0; i<strucl.nAtoms(); ++i){
			const int type=strucl.type(i);
			dev_in[type].noalias()+=(avg_in[type]-strucl.symm(i)).cwiseProduct(avg_in[type]-strucl.symm(i));
		}
	}
	for(int i=0; i<dev_in.size(); ++i){
		MPI_Allreduce(MPI_IN_PLACE,dev_in[i].data(),dev_in[i].size(),MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		for(int j=0; j<dev_in[i].size(); ++j){
			dev_in[i][j]=sqrt(dev_in[i][j]/(N[i]-1.0));
		}
	}
	//compute the rms
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"compute the rms\n";
	for(int n=0; n<struc_train.size(); ++n){
		const Structure& strucl=struc_train[n];
		for(int i=0; i<strucl.nAtoms(); ++i){
			rms_in[strucl.type(i)].noalias()+=strucl.symm(i).cwiseProduct(strucl.symm(i));
		}
	}
	for(int i=0; i<dev_in.size(); ++i){
		MPI_Allreduce(MPI_IN_PLACE,rms_in[i].data(),rms_in[i].size(),MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		for(int j=0; j<dev_in[i].size(); ++j){
			rms_in[i][j]=sqrt(rms_in[i][j]/N[i]);
		}
	}
	
	//====== precondition the input ======
	std::vector<Eigen::VectorXd> inpb_(nTypes_);//input bias
	std::vector<Eigen::VectorXd> inpw_(nTypes_);//input weight
	for(int n=0; n<nTypes_; ++n){
		inpb_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),0.0);
		inpw_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),1.0);
	}
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"pre-conditioning input\n";
	switch(prebias_){
		case PreBias::MEAN:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=-1.0*avg_in[n];
			}
		} break;
		case PreBias::HAR:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=-1.0*har_in[n];
			}
		} break;
		case PreBias::MID:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=-0.5*(max_in[n]+min_in[n]);
			}
		} break;
		case PreBias::MIN:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=-1.0*min_in[n];
			}
		} break;
		case PreBias::RMS:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=-1.0*rms_in[n];
			}
		} break;
		case PreBias::IDENTITY:{
			for(int n=0; n<nTypes_; ++n){
				inpb_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),0.0);
			}
		}break;
		case PreBias::NONE:
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
					if(max_in[n][i]>0) inpw_[n][i]=1.0/max_in[n][i];
					else inpw_[n][i]=1.0;
				}
			}
		} break;
		case PreScale::IDENTITY:{
			for(int n=0; n<nTypes_; ++n){
				inpw_[n]=Eigen::VectorXd::Constant(nnp_.nnh(n).nInput(),1.0);
			}
		}break;
		case PreScale::NONE:
			throw std::invalid_argument("Invalid input scaling.");
		break;
	}
	for(int n=0; n<nTypes_; ++n){
		inpw_[n]*=inscale_;
	}
	
	//====== set the bias for each of the species ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"setting the bias for each species\n";
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
			std::string("NNPTEFR::train(int): Network has ")
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
		//beta
		betai_=beta_;
		rmse_t_a_=0;
		rmse_v_a_=0;
	}
	//weight mask
	Eigen::VectorXd wm;
	if(reg_!=Regularization::NONE){
		wm=Eigen::VectorXd::Zero(nParams);
		int count=0;
		for(int n=0; n<nTypes_; ++n){
			for(int m=0; m<nnp_.nnh(n).nn().nBias(); ++m){
				wm[count++]=0.0;
			}
			for(int m=0; m<nnp_.nnh(n).nn().nWeight(); ++m){
				wm[count++]=1.0;
			}
		}
	}
	
	//====== print input statistics and bias ======
	if(NNPTEFR_PRINT_DATA>-1 && WORLD.rank()==0){
		char* strbuf=new char[print::len_buf];
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("OPT - DATA",strbuf)<<"\n";
		std::cout<<"N-PARAMS    = \n\t"<<nParams<<"\n";
		std::cout<<"AVG - INPUT = \n"; for(int i=0; i<avg_in.size(); ++i) std::cout<<"\t"<<avg_in[i].transpose()<<"\n";
		std::cout<<"HAR - INPUT = \n"; for(int i=0; i<har_in.size(); ++i) std::cout<<"\t"<<har_in[i].transpose()<<"\n";
		std::cout<<"MAX - INPUT = \n"; for(int i=0; i<max_in.size(); ++i) std::cout<<"\t"<<max_in[i].transpose()<<"\n";
		std::cout<<"MIN - INPUT = \n"; for(int i=0; i<min_in.size(); ++i) std::cout<<"\t"<<min_in[i].transpose()<<"\n";
		std::cout<<"DEV - INPUT = \n"; for(int i=0; i<dev_in.size(); ++i) std::cout<<"\t"<<dev_in[i].transpose()<<"\n";
		std::cout<<"RMS - INPUT = \n"; for(int i=0; i<rms_in.size(); ++i) std::cout<<"\t"<<rms_in[i].transpose()<<"\n";
		std::cout<<"PRE-BIAS    = \n"; for(int i=0; i<inpb_.size(); ++i) std::cout<<"\t"<<inpb_[i].transpose()<<"\n";
		std::cout<<"PRE-SCALE   = \n"; for(int i=0; i<inpw_.size(); ++i) std::cout<<"\t"<<inpw_[i].transpose()<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		delete[] strbuf;
	}
	
	//====== execute the optimization ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"executing the optimization\n";
	//optimization variables
	//bool fbreak=false;
	const double nBatchi_=1.0/nBatch;
	const double nVali_=1.0/nVal;
	//bcast parameters
	MPI_Bcast(obj_.p().data(),obj_.p().size(),MPI_DOUBLE,0,WORLD.mpic());
	//allocate status vectors
	std::vector<int> step;
	std::vector<double> gamma,loss_t,loss_v,rmse_t,rmse_v,rmse_t_a,rmse_v_a;
	const int size=(iter_.max()/iter_.nPrint()==0)?1:iter_.max()/iter_.nPrint();
	if(WORLD.rank()==0){
		step.resize(size);
		gamma.resize(size);
		loss_t.resize(size);
		loss_v.resize(size);
		rmse_t.resize(size);
		rmse_v.resize(size);
		rmse_t_a.resize(size);
		rmse_v_a.resize(size);
	}
	params_.resize(size,Eigen::VectorXd::Zero(nParams));
	//print status header to standard output
	if(WORLD.rank()==0) printf("opt gamma loss_t loss_v rmse_t rmse_v rmse_t_a rmse_v_a\n");
	//start the clock
	clock.begin();
	//begin optimization
	Eigen::VectorXd gtot_=Eigen::VectorXd::Zero(obj_.dim());
	for(int iter=0; iter<iter_.max(); ++iter){
		//compute the error and gradient
		//error_cost(obj_.p(),struc_train,struc_val);
		error_cost_o1(obj_.p(),struc_train,struc_val);
		//error_dodp(obj_.p(),struc_train,struc_val);
		//pack the gradient
		int count=0;
		for(int n=0; n<nTypes_; ++n){
			std::memcpy(gtot_.data()+count,gElement_[n].data(),gElement_[n].size()*sizeof(double));
			count+=gElement_[n].size();
		}
		//accumulate gradient and error
		double error_sum_[4]={0.0,0.0,0.0,0.0};
		obj_.g().setZero();
		MPI_Reduce(gtot_.data(),obj_.g().data(),gtot_.size(),MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		MPI_Reduce(error_,error_sum_,4,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
		if(WORLD.rank()==0){
			//compute error averaged over the batch
			error_[0]=error_sum_[0]*nBatchi_;//loss - train
			error_[1]=error_sum_[1]*nVali_;//loss - val
			error_[2]=sqrt(error_sum_[2]*nBatchi_);//rmse - train
			error_[3]=sqrt(error_sum_[3]*nVali_);//rmse - val
			rmse_t_a_*=beta_; rmse_t_a_+=(1.0-beta_)*error_[2];
			rmse_v_a_*=beta_; rmse_v_a_+=(1.0-beta_)*error_[3];
			//compute gradient averaged over the batch
			obj_.g()*=nBatchi_;
			//print/write error
			if((unsigned int)iter_.step()%iter_.nPrint()==0){
				const double betaii=1.0/(1.0-betai_);
				const int t=(unsigned int)iter/iter_.nPrint();
				step[t]=iter_.count();
				gamma[t]=obj_.gamma();
				loss_t[t]=error_[0];
				loss_v[t]=error_[1];
				rmse_t[t]=error_[2];
				rmse_v[t]=error_[3];
				rmse_t_a[t]=rmse_t_a_*betaii;
				rmse_v_a[t]=rmse_v_a_*betaii;
				params_[t]=obj_.p();
				printf("%8i %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e\n",
					step[t],gamma[t],loss_t[t],loss_v[t],rmse_t[t],rmse_v[t],rmse_t_a[t],rmse_v_a[t]
				);
			}
			//write the basis and potentials
			if((unsigned int)iter_.step()%iter_.nWrite()==0){
				if(NNPTEFR_PRINT_STATUS>1) std::cout<<"writing the restart file and potentials\n";
				//write restart file
				const std::string file_restart=file_restart_+"."+std::to_string(iter_.count());
				this->write_restart(file_restart.c_str());
				//write potential file
				const std::string file_ann=file_ann_+"."+std::to_string(iter_.count());
				NNP::write(file_ann.c_str(),nnp_);
			}
			//compute the new position
			obj_.val()=error_[0];//loss - train
			obj_.gamma()=decay_.step(obj_.gamma(),iter_);
			algo_->step(obj_);
			//weight decay
			switch(reg_){
				case Regularization::NONE: break;
				case Regularization::LASSO:{
					for(int n=0; n<nParams; ++n){
						obj_.p()[n]-=obj_.gamma()*eta_*wm[n]*math::special::sgn(obj_.pOld()[n]);
					}
				} break;
				case Regularization::RIDGE:{
					for(int n=0; n<nParams; ++n){
						obj_.p()[n]-=obj_.gamma()*eta_*wm[n]*obj_.pOld()[n];
					}
				} break;
				case Regularization::HUBER:{
					for(int n=0; n<nParams; ++n){
						obj_.p()[n]-=obj_.gamma()*eta_*wm[n]*obj_.pOld()[n]/sqrt(1.0+obj_.pOld()[n]*obj_.pOld()[n]);
					}
				} break;
				case Regularization::ASINH:{
					for(int n=0; n<nParams; ++n){
						obj_.p()[n]-=obj_.gamma()*eta_*wm[n]*asinh(obj_.pOld()[n]);
					}
				} break;
			}
			//compute the difference
			obj_.dv()=std::fabs(obj_.val()-obj_.valOld());
			obj_.dp()=(obj_.p()-obj_.pOld()).norm();
			//set the new "old" values
			obj_.valOld()=obj_.val();//set "old" value
			obj_.pOld()=obj_.p();//set "old" p value
			obj_.gOld()=obj_.g();//set "old" g value
			//check the break condition
			/*switch(iter_.stop()){
				case opt::Stop::FABS: fbreak=(obj_.val()<iter_.tol()); break;
				case opt::Stop::FREL: fbreak=(obj_.dv()<iter_.tol()); break;
				case opt::Stop::XREL: fbreak=(obj_.dp()<iter_.tol()); break;
			}*/
		}
		//bcast parameters
		MPI_Bcast(obj_.p().data(),obj_.p().size(),MPI_DOUBLE,0,WORLD.mpic());
		//bcast break condition
		/*MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,WORLD.mpic());
		if(fbreak) break;*/
		//increment step
		++iter_.step();
		++iter_.count();
		betai_*=beta_;
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
			fprintf(writer_error_,"#STEP GAMMA LOSS_TRAIN LOSS_VAL RMSE_TRAIN RMSE_VAL RMSE_TRAIN_A RMSE_VAL_A\n");
		} else {
			writer_error_=fopen(file_error_.c_str(),"a");
		}
		if(writer_error_==NULL) throw std::runtime_error("NNPTEFR::train(int): Could not open error record file.");
		for(int t=0; t<step.size(); ++t){
			fprintf(writer_error_,
				"%6i %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e\n",
				step[t],gamma[t],loss_t[t],loss_v[t],rmse_t[t],rmse_v[t],rmse_t_a[t],rmse_v_a[t]
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
		if(writer_p_==NULL) throw std::runtime_error("NNPTEFR::train(int): Could not open error record file.");
		for(int t=0; t<step.size(); ++t){
			for(int i=0; i<params_[t].size(); ++i){
				fprintf(writer_p_,"%.12f ",params_[t][i]);
			}
			fprintf(writer_p_,"\n");
		}
		fclose(writer_p_);
		writer_p_=NULL;
	}

	//====== bcast the parameters ======
	for(int t=0; t<size; ++t){
		MPI_Bcast(params_[t].data(),params_[t].size(),MPI_DOUBLE,0,WORLD.mpic());
	}
	
	//====== unpack final parameters ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"packing final parameters into neural network\n";
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
	
	if(NNPTEFR_PRINT_DATA>-1 && WORLD.rank()==0){
		char* strbuf=new char[print::len_buf];
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TRAIN - SUMMARY",strbuf)<<"\n";
		std::cout<<"N-STEP = "<<iter_.step()<<"\n";
		std::cout<<"TIME   = "<<time_train<<"\n";
		if(NNPTEFR_PRINT_DATA>1){
			std::cout<<"p = "; for(int i=0; i<obj_.p().size(); ++i) std::cout<<obj_.p()[i]<<" "; std::cout<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		delete[] strbuf;
	}
}

void NNPTEFR::error_cost(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"NNPTEFR::error(const Eigen::VectorXd&):\n";
	
	//====== reset the error ======
	error_[0]=0; //loss - training
	error_[1]=0; //loss - validation
	error_[2]=0; //rmse - training
	error_[3]=0; //rmse - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nTypes_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch - training\n";
	for(int i=0; i<batcht_.size(); ++i) batcht_[i]=batcht_.data((unsigned int)(batcht_.count()++)%batcht_.capacity());
	std::sort(batcht_.elements(),batcht_.elements()+batcht_.size());
	if(batcht_.count()>=batcht_.capacity()){
		std::shuffle(batcht_.data(),batcht_.data()+batcht_.capacity(),rngen_);
		batcht_.count()=0;
	}
	if(NNPTEFR_PRINT_DATA>1 && WORLD.rank()==0){std::cout<<"batch = "; for(int i=0; i<batcht_.size(); ++i) std::cout<<batcht_[i]<<" "; std::cout<<"\n";}
	
	//====== compute training error and gradient ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batcht_.size(); ++i){
		const int ii=batcht_[i];
		//**** compute the energy ****
		double pe=0;
		for(int n=0; n<struc_train[ii].nAtoms(); ++n){
			//find the element index in the nn potential
			const int type=struc_train[ii].type(n);
			//execute the network
			nnp_.nnh(type).nn().fp(struc_train[ii].symm(n));
			//add the atom energy to the total
			pe+=nnp_.nnh(type).nn().out()[0];
		}
		//**** compute the energy difference normalized by number of atoms ****
		const double nAtoms=struc_train[ii].nAtoms();
		const double dE=(pe-struc_train[ii].pe());
		const double dU=dE*normt_[ii];
		Eigen::VectorXd dcdo=Eigen::VectorXd::Constant(1,normt_[ii]);
		switch(iter_.loss()){
			case opt::Loss::MSE:{
				error_[0]+=0.5*dU*dU;//loss - train
				dcdo*=dU;
			} break;
			case opt::Loss::MAE:{
				error_[0]+=std::fabs(dU);//loss - train
				dcdo*=math::special::sgn(dU);
			} break;
			case opt::Loss::HUBER:{
				const double arg=dU*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				error_[0]+=delta2_*(sqrtf-1.0);//loss - train
				dcdo*=dU/sqrtf;
			} break;
			case opt::Loss::ASINH:{
				const double arg=dU*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				error_[0]+=delta2_*(1.0-sqrtf+arg*logf);//loss - train
				dcdo*=logf*delta_;
			} break;
			default: break;
		}
		error_[2]+=(dE*dE)/(nAtoms*nAtoms);//rmse - train
		//**** compute the gradient ****
		for(int n=0; n<struc_train[ii].nAtoms(); ++n){
			//find the element index in the nn potential
			const int type=struc_train[ii].type(n);
			//execute the network
			nnp_.nnh(type).nn().fpbp(struc_train[ii].symm(n));
			//compute the gradient 
			gElement_[type].noalias()+=cost_[type].grad(nnp_.nnh(type).nn(),dcdo);
		}
	}
	
	//====== compute validation error ======
	if((unsigned int)iter_.step()%iter_.nPrint()==0 || (unsigned int)iter_.step()%iter_.nWrite()==0){
		if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error\n";
		for(int i=0; i<struc_val.size(); ++i){
			//**** compute the energy ****
			double pe=0;
			for(int n=0; n<struc_val[i].nAtoms(); ++n){
				//find the element index in the nn potential
				const int type=struc_val[i].type(n);
				//execute the network
				nnp_.nnh(type).nn().fp(struc_val[i].symm(n));
				//add the energy to the total
				pe+=nnp_.nnh(type).nn().out()[0];
			}
			//**** compute error ****
			const double nAtoms=struc_val[i].nAtoms();
			const double dE=(pe-struc_val[i].pe());
			const double dU=dE*normv_[i];
			switch(iter_.loss()){
				case opt::Loss::MSE:{
					error_[1]+=0.5*dU*dU;//loss - val
				} break;
				case opt::Loss::MAE:{
					error_[1]+=std::fabs(dU);//loss - val
				} break;
				case opt::Loss::HUBER:{
					const double arg=dU*deltai_;
					error_[1]+=delta2_*(sqrt(1.0+arg*arg)-1.0);//loss - val
				} break;
				case opt::Loss::ASINH:{
					const double arg=dU*deltai_;
					const double sqrtf=sqrt(1.0+arg*arg);
					error_[1]+=delta2_*(1.0-sqrtf+arg*log(arg+sqrtf));//loss - val
				} break;
				default: break;
			}
			error_[3]+=(dE*dE)/(nAtoms*nAtoms);//rmse - val
		}
	}
}

void NNPTEFR::error_cost_o1(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"NNPTEFR::error(const Eigen::VectorXd&):\n";
	
	//====== reset the error ======
	error_[0]=0; //loss - training
	error_[1]=0; //loss - validation
	error_[2]=0; //rmse - training
	error_[3]=0; //rmse - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nTypes_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch\n";
	for(int i=0; i<batcht_.size(); ++i) batcht_[i]=batcht_.data((unsigned int)(batcht_.count()++)%batcht_.capacity());
	std::sort(batcht_.elements(),batcht_.elements()+batcht_.size());
	if(batcht_.count()>=batcht_.capacity()){
		std::shuffle(batcht_.data(),batcht_.data()+batcht_.capacity(),rngen_);
		batcht_.count()=0;
	}
	if(NNPTEFR_PRINT_DATA>1 && WORLD.rank()==0){
		std::cout<<"batch = "; for(int i=0; i<batcht_.size(); ++i) std::cout<<batcht_[i]<<" "; std::cout<<"\n";
	}
	
	//====== compute training error and gradient ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batcht_.size(); ++i){
		//set batch
		const int ii=batcht_[i];
		//reset gradients
		for(int n=0; n<nTypes_; ++n) grad_[n].setZero();
		//**** compute the energy ****
		double pe=0;
		const Eigen::VectorXd dcdo=Eigen::VectorXd::Constant(1,1);
		for(int n=0; n<struc_train[ii].nAtoms(); ++n){
			//find the element index in the nn potential
			const int type=struc_train[ii].type(n);
			//execute the network
			nnp_.nnh(type).nn().fpbp(struc_train[ii].symm(n));
			//add the atom energy to the total
			pe+=nnp_.nnh(type).nn().out()[0];
			//compute gradient
			grad_[type].noalias()+=cost_[type].grad(nnp_.nnh(type).nn(),dcdo);
		}
		//**** compute the energy difference normalized by number of atoms ****
		const double nAtoms=struc_train[ii].nAtoms();
		const double dE=(pe-struc_train[ii].pe());
		const double dU=dE*normt_[ii];
		//**** compute the error and parameter gradients ****
		double gfac=normt_[ii];
		switch(iter_.loss()){
			case opt::Loss::MSE:{
				error_[0]+=0.5*dU*dU;//loss - train
				gfac*=dU;
			} break;
			case opt::Loss::MAE:{
				error_[0]+=std::fabs(dU);//loss - train
				gfac*=math::special::sgn(dU);
			} break;
			case opt::Loss::HUBER:{
				const double arg=dU*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				error_[0]+=delta2_*(sqrtf-1.0);//loss - train
				gfac*=dU/sqrtf;
			} break;
			case opt::Loss::ASINH:{
				const double arg=dU*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				error_[0]+=delta2_*(1.0-sqrtf+arg*logf);//loss - train
				gfac*=logf*delta_;
			} break;
			default: break;
		}
		error_[2]+=(dE*dE)/(nAtoms*nAtoms);//rmse - train
		//**** compute the gradient ****
		for(int j=0; j<nTypes_; ++j){
			gElement_[j].noalias()+=grad_[j]*gfac;
		}
	}
	
	//====== compute validation error ======
	if((unsigned int)iter_.step()%iter_.nPrint()==0 || (unsigned int)iter_.step()%iter_.nWrite()==0){
		if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error\n";
		for(int i=0; i<struc_val.size(); ++i){
			//**** compute the energy ****
			double pe=0;
			for(int n=0; n<struc_val[i].nAtoms(); ++n){
				//find the element index in the nn potential
				const int type=struc_val[i].type(n);
				//execute the network
				nnp_.nnh(type).nn().fp(struc_val[i].symm(n));
				//add the energy to the total
				pe+=nnp_.nnh(type).nn().out()[0];
			}
			//**** compute error ****
			const double nAtoms=struc_val[i].nAtoms();
			const double dE=(pe-struc_val[i].pe());
			const double dU=dE*normv_[i];
			switch(iter_.loss()){
				case opt::Loss::MSE:{
					error_[1]+=0.5*dU*dU;//loss - val
				} break;
				case opt::Loss::MAE:{
					error_[1]+=std::fabs(dU);//loss - val
				} break;
				case opt::Loss::HUBER:{
					const double arg=dU*deltai_;
					error_[1]+=delta2_*(sqrt(1.0+arg*arg)-1.0);//loss - val
				} break;
				case opt::Loss::ASINH:{
					const double arg=dU*deltai_;
					const double sqrtf=sqrt(1.0+arg*arg);
					error_[1]+=delta2_*(1.0-sqrtf+arg*log(arg+sqrtf));//loss - val
				} break;
				default: break;
			}
			error_[3]+=(dE*dE)/(nAtoms*nAtoms);//rmse - val
		}
	}
}

void NNPTEFR::error_dodp(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val){
	if(NNPTEFR_PRINT_FUNC>0) std::cout<<"NNPTEFR::error(const Eigen::VectorXd&):\n";
	
	//====== reset the error ======
	error_[0]=0; //loss - training
	error_[1]=0; //loss - validation
	error_[2]=0; //rmse - training
	error_[3]=0; //rmse - validation
	
	//====== unpack total parameters into element arrays ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking total parameters into element arrays\n";
	int count=0;
	for(int n=0; n<nTypes_; ++n){
		std::memcpy(pElement_[n].data(),x.data()+count,pElement_[n].size()*sizeof(double));
		count+=pElement_[n].size();
	}
	
	//====== unpack arrays into element nn's ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=0; n<nTypes_; ++n) nnp_.nnh(n).nn()<<pElement_[n];
	
	//====== reset the gradients ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"resetting gradients\n";
	for(int n=0; n<nTypes_; ++n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"randomizing the batch\n";
	for(int i=0; i<batcht_.size(); ++i) batcht_[i]=batcht_.data((unsigned int)(batcht_.count()++)%batcht_.capacity());
	std::sort(batcht_.elements(),batcht_.elements()+batcht_.size());
	if(batcht_.count()>=batcht_.capacity()){
		std::shuffle(batcht_.data(),batcht_.data()+batcht_.capacity(),rngen_);
		batcht_.count()=0;
	}
	if(NNPTEFR_PRINT_DATA>1 && WORLD.rank()==0){std::cout<<"batch = "; for(int i=0; i<batcht_.size(); ++i) std::cout<<batcht_[i]<<" "; std::cout<<"\n";}
	
	//====== compute training error and gradient ======
	if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing training error and gradient\n";
	for(int i=0; i<batcht_.size(); ++i){
		const int ii=batcht_[i];
		//**** compute the energy ****
		double pe=0;
		for(int n=0; n<struc_train[ii].nAtoms(); ++n){
			//find the element index in the nn potential
			const int type=struc_train[ii].type(n);
			//execute the network
			nnp_.nnh(type).nn().fp(struc_train[ii].symm(n));
			//add the atom energy to the total
			pe+=nnp_.nnh(type).nn().out()[0];
		}
		//**** compute the energy difference normalized by number of atoms ****
		const double nAtoms=struc_train[ii].nAtoms();
		const double dE=(pe-struc_train[ii].pe());
		const double dU=dE*normt_[ii];
		Eigen::VectorXd dcdo=Eigen::VectorXd::Constant(1,normt_[ii]);
		switch(iter_.loss()){
			case opt::Loss::MSE:{
				error_[0]+=0.5*dU*dU;//loss - train
				dcdo*=dU;
			} break;
			case opt::Loss::MAE:{
				error_[0]+=std::fabs(dU);//loss - train
				dcdo*=math::special::sgn(dU);
			} break;
			case opt::Loss::HUBER:{
				const double arg=dU*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				error_[0]+=delta2_*(sqrtf-1.0);//loss - train
				dcdo*=dU/sqrtf;
			} break;
			case opt::Loss::ASINH:{
				const double arg=dU*deltai_;
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				error_[0]+=delta2_*(1.0-sqrtf+arg*logf);//loss - train
				dcdo*=logf*delta_;
			} break;
			default: break;
		}
		error_[2]+=(dE*dE)/(nAtoms*nAtoms);//rmse - train
		//**** compute the gradient ****
		for(int n=0; n<struc_train[ii].nAtoms(); ++n){
			//find the element index in the nn potential
			const int type=struc_train[ii].type(n);
			//execute the network
			nnp_.nnh(type).nn().fpbp(struc_train[ii].symm(n));
			//compute the gradient 
			dodp_[type].grad(nnp_.nnh(type).nn());
			gElement_[type].noalias()+=dodp_[type].dodp().row(0)*dcdo[0];
		}
	}
	
	//====== compute validation error ======
	if((unsigned int)iter_.step()%iter_.nPrint()==0 || (unsigned int)iter_.step()%iter_.nWrite()==0){
		if(NNPTEFR_PRINT_STATUS>0 && WORLD.rank()==0) std::cout<<"computing validation error\n";
		for(int i=0; i<struc_val.size(); ++i){
			//**** compute the energy ****
			double pe=0;
			for(int n=0; n<struc_val[i].nAtoms(); ++n){
				//find the element index in the nn potential
				const int type=struc_val[i].type(n);
				//execute the network
				nnp_.nnh(type).nn().fp(struc_val[i].symm(n));
				//add the energy to the total
				pe+=nnp_.nnh(type).nn().out()[0];
			}
			//**** compute error ****
			const double nAtoms=struc_val[i].nAtoms();
			const double dE=(pe-struc_val[i].pe());
			const double dU=dE*normv_[i];
			switch(iter_.loss()){
				case opt::Loss::MSE:{
					error_[1]+=0.5*dU*dU;//loss - val
				} break;
				case opt::Loss::MAE:{
					error_[1]+=std::fabs(dU);//loss - val
				} break;
				case opt::Loss::HUBER:{
					const double arg=dU*deltai_;
					error_[1]+=delta2_*(sqrt(1.0+arg*arg)-1.0);//loss - val
				} break;
				case opt::Loss::ASINH:{
					const double arg=dU*deltai_;
					const double sqrtf=sqrt(1.0+arg*arg);
					error_[1]+=delta2_*(1.0-sqrtf+arg*log(arg+sqrtf));//loss - val
				} break;
				default: break;
			}
			error_[3]+=(dE*dE)/(nAtoms*nAtoms);//rmse - val
		}
	}
}

void NNPTEFR::read(const char* file, NNPTEFR& nnpte){
	if(NN_PRINT_FUNC>0) std::cout<<"NNPTEFR::read(const char*,NNPTEFR&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		NNPTEFR::read(reader,nnpte);
		std::fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for reading.\n"));
}

void NNPTEFR::read(FILE* reader, NNPTEFR& nnpte){
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
		} else if(tag=="RESET"){//read restart file
			nnpte.reset()=string::boolean(token.next().c_str());//restarting
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
		} else if(tag=="ALGO"){
			opt::algo::read(nnpte.algo(),token);
		} else if(tag=="DECAY"){
			nnpte.decay().read(token);
		} else if(tag=="DELTA"){
			nnpte.delta()=std::atof(token.next().c_str());
			nnpte.deltai()=1.0/nnpte.delta();
			nnpte.delta2()=nnpte.delta()*nnpte.delta();
		} else if(tag=="BETA"){
			nnpte.beta()=std::atof(token.next().c_str());
			nnpte.betai()=nnpte.beta();
		} else if(tag=="ETA"){
			nnpte.eta()=std::atof(token.next().c_str());
		} else if(tag=="NORM"){
			nnpte.norm()=Norm::read(string::to_upper(token.next()).c_str());
		} else if(tag=="REG"){
			nnpte.reg()=Regularization::read(string::to_upper(token.next()).c_str());
		} else if(tag=="PRESCALE"){
			nnpte.prescale()=PreScale::read(string::to_upper(token.next()).c_str());
		} else if(tag=="PREBIAS"){
			nnpte.prebias()=PreBias::read(string::to_upper(token.next()).c_str());
		} else if(tag=="INSCALE"){
			nnpte.inscale()=std::atof(token.next().c_str());
		} else if(tag=="INBIAS"){
			nnpte.inbias()=std::atof(token.next().c_str());
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
		units::System unitsys=units::System::NONE;
	//mode
		Mode mode=Mode::TRAIN;
	//atom format
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.symm=true;
		atomT.charge=false;
	//flags - compute
		struct Compute{
			bool coul=false;  //compute - external potential - coulomb
			bool vdwl=false;  //compute - external potential - vdw - long
			bool vdws=false;  //compute - external potential - vdw - short
			bool rep=false;   //compute - external potential - repulsive
			bool force=false; //compute - forces
			bool norm=false;  //compute - energy normalization
			bool zero=false;  //compute - zero point energy
			bool pca=false;   //compute - pca
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
		ptnl::PotLDampLong pot_vdw_l;
		ptnl::PotLDampDSF pot_vdw_s;
		ptnl::PotPauli pot_rep;
	//nn potential - opt
		int nBatch=-1;
		std::vector<Type> types;//unique atomic species
		NNPTEFR nnpte;//nn potential optimization data
		std::vector<std::vector<int> > nh;//hidden layer configuration
		NN::ANNP annp;//neural network initialization parameters
	//taylor series
		rng::dist::Name rdist=rng::dist::Name::NONE;
		Perturb perturb=Perturb::NONE;
		double rdelta=0.0;
		int nadd=0;
	//pca
		int ngrid=50;
	//data names
		static const char* const dnames[] = {"TRAINING","VALIDATION","TESTING"};
	//structures - format
		FILE_FORMAT::type format;//format of training data
	//structures - data
		const int nData=3;
		std::vector<int> nstrucs(nData,0);
		std::vector<std::vector<std::string> > data(nData); //data files
		std::vector<std::vector<std::string> > files(nData); //structure files
		std::vector<std::vector<Structure> > strucs_org(nData); //structures
		std::vector<std::vector<Structure> > strucs_tot(nData); //structures
		std::vector<std::vector<int> > indices(nData);
		std::vector<Alias> aliases;
	//mpi data distribution
		std::vector<thread::Dist> dist_struc(nData+1);//nData + batch
	//timing
		Clock clock,clock_wall; //time objects
		double time_wall=0;     //total wall time
		std::vector<double> time_energy(nData,0.0);
		std::vector<double> time_force(nData,0.0);
		std::vector<double> time_symm(nData,0.0);
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
	//random
		int seed_global=-1;
		std::vector<int> seed_local;
		std::mt19937 rngen;
		
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
			std::printf("PI     = %.15f\n",math::constant::PI);
			std::printf("RadPI  = %.15f\n",math::constant::RadPI);
			std::printf("Rad2   = %.15f\n",math::constant::Rad2);
			std::printf("Log2   = %.15f\n",math::constant::LOG2);
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
			std::printf("alpha        = %.12f\n",units::ALPHA);
			std::printf("p/e mass     = %.12f\n",units::MPoME);
			std::printf("bohr-r  (A)  = %.12f\n",units::Bohr2Ang);
			std::printf("hartree (eV) = %.12f\n",units::Eh2Ev);
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
			if(NNPTEFR_PRINT_STATUS>0) std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//======== open the parameter file ========
			if(NNPTEFR_PRINT_STATUS>0) std::cout<<"opening parameter file\n";
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+paramfile);
			
			//======== read in the parameters ========
			if(NNPTEFR_PRINT_STATUS>-1) std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL){
				token.read(string::trim_right(input,string::COMMENT),string::WS);
				if(token.end()) continue;//skip empty line
				const std::string tag=string::to_upper(token.next());
				//general
				if(tag=="UNITS"){//units
					unitsys=units::System::read(string::to_upper(token.next()).c_str());
				} else if(tag=="FORMAT"){//simulation format
					format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
				} else if(tag=="RDELTA"){
					rdelta=std::atof(token.next().c_str());
				} else if(tag=="RDIST"){
					rdist=rng::dist::Name::read(string::to_upper(token.next()).c_str());
				} else if(tag=="PERTURB"){
					perturb=Perturb::read(string::to_upper(token.next()).c_str());
				} else if(tag=="NADD"){
					nadd=std::atoi(token.next().c_str());
				} else if(tag=="NGRID"){
					ngrid=std::atoi(token.next().c_str());
				} else if(tag=="SEED"){
					seed_global=std::atoi(token.next().c_str());
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
					} else if(atomtag=="Z"){
						types[index].z().flag()=true;
						types[index].z().val()=std::atof(token.next().c_str());
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
					else if(ctype=="VDWS") compute.vdws=string::boolean(token.next().c_str());
					else if(ctype=="VDWL") compute.vdwl=string::boolean(token.next().c_str());
					else if(ctype=="REP") compute.rep=string::boolean(token.next().c_str());
					else if(ctype=="FORCE") compute.force=string::boolean(token.next().c_str());
					else if(ctype=="NORM") compute.norm=string::boolean(token.next().c_str());
					else if(ctype=="ZERO") compute.zero=string::boolean(token.next().c_str());
					else if(ctype=="PCA") compute.pca=string::boolean(token.next().c_str());
				}
				//potential 
				if(tag=="POT_COUL"){
					token.next();
					pot_coul.read(token);
				} else if(tag=="POT_VDWL"){
					token.next();
					pot_vdw_l.read(token);
				} else if(tag=="POT_VDWS"){
					token.next();
					pot_vdw_s.read(token);
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
			if(NNPTEFR_PRINT_STATUS>0) std::cout<<"setting atom flags\n";
			atomT.charge=compute.coul;
			
			//======== read - nnpte =========
			if(NNPTEFR_PRINT_STATUS>0) std::cout<<"reading neural network training parameters\n";
			NNPTEFR::read(reader,nnpte);
			
			//======== read - annp =========
			if(NNPTEFR_PRINT_STATUS>0) std::cout<<"reading neural network parameters\n";
			NN::ANNP::read(reader,annp);
			
			//======== close parameter file ========
			if(NNPTEFR_PRINT_STATUS>0) std::cout<<"closing parameter file\n";
			fclose(reader);
			reader=NULL;
			
			//======== (restart == false) ========
			if(!nnpte.restart()){
				//======== (read potential == false) ========
				if(!read_pot){
					//resize the potential
					if(NNPTEFR_PRINT_STATUS>-1) std::cout<<"resizing potential\n";
					nnpte.nnp().resize(types);
					//read basis files
					if(NNPTEFR_PRINT_STATUS>-1) std::cout<<"reading basis files\n";
					if(files_basis.size()!=nnpte.nnp().ntypes()) throw std::runtime_error("main(int,char**): invalid number of basis files.");
					for(int i=0; i<nnpte.nnp().ntypes(); ++i){
						const char* file=files_basis[i].c_str();
						const char* atomName=types[i].name().c_str();
						NNP::read_basis(file,nnpte.nnp(),atomName);
					}
					//initialize the neural network hamiltonians
					if(NNPTEFR_PRINT_STATUS>-1) std::cout<<"initializing neural network hamiltonians\n";
					for(int i=0; i<nnpte.nnp().ntypes(); ++i){
						NNH& nnhl=nnpte.nnp().nnh(i);
						nnhl.nn().resize(annp,nnhl.nInput(),nh[i],1);
						nnhl.dOdZ().resize(nnhl.nn());
					}
				}
				//======== (read potential == true) ========
				if(read_pot){
					if(NNPTEFR_PRINT_STATUS>-1) std::cout<<"reading potential\n";
					NNP::read(file_pot.c_str(),nnpte.nnp());
				}
			}
			//======== (restart == true) ========
			if(nnpte.restart()){
				if(NNPTEFR_PRINT_STATUS>-1) std::cout<<"reading restart file\n";
				//save optimization parameters before reading restart file
				const bool reset=nnpte.reset();
				opt::Decay decay=nnpte.decay();
				opt::Iterator iter=nnpte.iter();
				opt::Objective obj=nnpte.obj();
				const std::string file=nnpte.file_restart_;
				//read restart
				nnpte.read_restart(file.c_str());
				const int count=nnpte.iter().count();//count is never reset
				//reset optimization parameters
				if(reset){
					if(NNPTEFR_PRINT_STATUS>-1) std::cout<<"resetting optimization\n";
					nnpte.decay()=decay;
					nnpte.iter()=iter;
					nnpte.iter().count()=count;
					nnpte.obj().gamma()=obj.gamma();
				}
				//reset the flags for printing/writing
				nnpte.restart()=true;
				nnpte.reset()=reset;
			}
			
			//======== print parameters ========
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"seed      = "<<seed_global<<"\n";
			std::cout<<"read_pot  = "<<read_pot<<"\n";
			std::cout<<"atom_type = "<<atomT<<"\n";
			std::cout<<"format    = "<<format<<"\n";
			std::cout<<"units     = "<<unitsys<<"\n";
			std::cout<<"mode      = "<<mode<<"\n";
			std::cout<<"perturb   = "<<perturb<<"\n";
			std::cout<<"rdist     = "<<rdist<<"\n";
			std::cout<<"rdelta    = "<<rdelta<<"\n";
			std::cout<<"nadd      = "<<nadd<<"\n";
			std::cout<<"ngrid     = "<<ngrid<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DATA FILES",strbuf)<<"\n";
			std::cout<<"data_train = \n"; for(int i=0; i<data[0].size(); ++i) std::cout<<"\t\t"<<data[0][i]<<"\n";
			std::cout<<"data_val   = \n"; for(int i=0; i<data[1].size(); ++i) std::cout<<"\t\t"<<data[1][i]<<"\n";
			std::cout<<"data_test  = \n"; for(int i=0; i<data[2].size(); ++i) std::cout<<"\t\t"<<data[2][i]<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("COMPUTE FLAGS",strbuf)<<"\n";
			std::cout<<"coul  = "<<compute.coul<<"\n";
			std::cout<<"vdws  = "<<compute.vdws<<"\n";
			std::cout<<"vdwl  = "<<compute.vdwl<<"\n";
			std::cout<<"rep   = "<<compute.rep<<"\n";
			std::cout<<"force = "<<compute.force<<"\n";
			std::cout<<"norm  = "<<compute.norm<<"\n";
			std::cout<<"zero  = "<<compute.zero<<"\n";
			std::cout<<"pca   = "<<compute.pca<<"\n";
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
			if(compute.vdws) std::cout<<"VDWS = "<<pot_vdw_s<<"\n";
			if(compute.vdwl) std::cout<<"VDWL = "<<pot_vdw_l<<"\n";
			if(compute.rep)  std::cout<<"REP  = "<<pot_rep<<"\n";
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
			
			//========= check the data =========
			if(mode==Mode::TRAIN && data[0].size()==0) throw std::invalid_argument("No data provided - training.");
			if(mode==Mode::TRAIN && data[1].size()==0) throw std::invalid_argument("No data provided - validation.");
			if(mode==Mode::TEST  && data[2].size()==0) throw std::invalid_argument("No data provided - testing.");
			if(mode==Mode::NONE) throw std::invalid_argument("Invalid calculation mode");
			if(format==FILE_FORMAT::NONE) throw std::invalid_argument("Invalid file format.");
			if(unitsys==units::System::NONE) throw std::invalid_argument("Invalid unit system.");
			if(types.size()==0) throw std::invalid_argument("Invalid number of types.");
			if(nadd<0) throw std::invalid_argument("Number of additional structures must be not be negative.");
			if(nadd>0 && perturb==Perturb::NONE) throw std::invalid_argument("Invalid perturbation.");
			if(nadd>0 && rdist==rng::dist::Name::NONE) throw std::invalid_argument("Invalid radial distribution.");
			if(nadd>0 && rdelta==0) throw std::invalid_argument("Additional structures must have non-zero rdelta.");
		}
		
		//======== bcast the parameters ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		//general parameters
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&nadd,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&perturb,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&rdist,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&rdelta,1,MPI_DOUBLE,0,WORLD.mpic());
		//mode
		MPI_Bcast(&mode,1,MPI_INT,0,WORLD.mpic());
		//atom type
		thread::bcast(WORLD.mpic(),0,atomT);
		thread::bcast(WORLD.mpic(),0,annp);
		//flags - compute
		MPI_Bcast(&compute.coul,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.vdwl,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.vdws,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.rep,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.norm,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.zero,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&compute.pca,1,MPI_C_BOOL,0,WORLD.mpic());
		//flags - writing
		MPI_Bcast(&write.coul,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.vdw,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.rep,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.energy,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&write.input,1,MPI_C_BOOL,0,WORLD.mpic());
		//external potential
		if(compute.coul) thread::bcast(WORLD.mpic(),0,pot_coul);
		if(compute.vdws) thread::bcast(WORLD.mpic(),0,pot_vdw_l);
		if(compute.vdwl) thread::bcast(WORLD.mpic(),0,pot_vdw_s);
		if(compute.rep) thread::bcast(WORLD.mpic(),0,pot_rep);
		//batch
		MPI_Bcast(&nBatch,1,MPI_INT,0,WORLD.mpic());
		//structures - format
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.mpic());
		//nnpte
		thread::bcast(WORLD.mpic(),0,nnpte);
		//alias
		int naliases=aliases.size();
		MPI_Bcast(&naliases,1,MPI_INT,0,WORLD.mpic());
		if(WORLD.rank()!=0) aliases.resize(naliases);
		for(int i=0; i<aliases.size(); ++i){
			thread::bcast(WORLD.mpic(),0,aliases[i]);
		}
		
		//======== set the random seed ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"setting the random seed\n";
		if(seed_global>0) rngen=std::mt19937(seed_global);
		else rngen=std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
		seed_local.resize(WORLD.size());
		for(int i=0; i<WORLD.size(); ++i){
			seed_local[i]=rngen();
		}
		MPI_Bcast(seed_local.data(),seed_local.size(),MPI_INT,0,WORLD.mpic());
		nnpte.seed()=seed_local[WORLD.rank()];
		rngen=std::mt19937(seed_local[WORLD.rank()]);
		
		//======== set the unit system ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"setting the unit system\n";
		units::Consts::init(unitsys);
		
		//************************************************************************************
		// READ DATA
		//************************************************************************************
		
		//======== rank 0 reads the data files (lists of structure files) ========
		if(WORLD.rank()==0){
			if(NNPTEFR_PRINT_STATUS>-1) std::cout<<"reading data\n";
			//==== read data ====
			for(int n=0; n<nData; ++n){
				for(int i=0; i<data[n].size(); ++i){
					//open the data file
					if(NNPTEFR_PRINT_DATA>0) std::cout<<"data["<<n<<"]["<<i<<"]: "<<data[n][i]<<"\n";
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
			if(NNPTEFR_PRINT_DATA>1){
				for(int n=0; n<nData; ++n){
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
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"bcasting file names\n";
		//bcast names
		for(int n=0; n<nData; ++n) thread::bcast(WORLD.mpic(),0,files[n]);
		//set number of structures
		for(int n=0; n<nData; ++n) nstrucs[n]=files[n].size();
		//print number of structures
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DATA - SIZE",strbuf)<<"\n";
			std::cout<<"ntrain - org = "<<nstrucs[0]<<"\n";
			std::cout<<"nval   - org = "<<nstrucs[1]<<"\n";
			std::cout<<"ntest  - org = "<<nstrucs[2]<<"\n";
			std::cout<<"ntrain - tot = "<<nstrucs[0]*(1+nadd)<<"\n";
			std::cout<<"nval   - tot = "<<nstrucs[1]*(1+nadd)<<"\n";
			std::cout<<"ntest  - tot = "<<nstrucs[2]*(1+nadd)<<"\n";
			std::cout<<"nbatch = "<<nBatch<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		//check the batch size
		if(nBatch<=0) throw std::invalid_argument("Invalid batch size.");
		if(nBatch>nstrucs[0]) throw std::invalid_argument("Invalid batch size.");
		
		//======== generate thread distributions ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"generating thread distributions\n";
		//thread dist - divide structures equally among the batch groups
		dist_struc[0].init(WORLD.size(),WORLD.rank(),nstrucs[0]);//train
		dist_struc[1].init(WORLD.size(),WORLD.rank(),nstrucs[1]);//validation
		dist_struc[2].init(WORLD.size(),WORLD.rank(),nstrucs[2]);//test
		dist_struc[3].init(WORLD.size(),WORLD.rank(),nBatch);//batch
		//print
		if(WORLD.rank()==0){
			std::string str;
			std::cout<<"thread_dist_train   = "<<thread::Dist::size(str,WORLD.size(),nstrucs[0])<<"\n";
			std::cout<<"thread_dist_val     = "<<thread::Dist::size(str,WORLD.size(),nstrucs[1])<<"\n";
			std::cout<<"thread_dist_test    = "<<thread::Dist::size(str,WORLD.size(),nstrucs[2])<<"\n";
			std::cout<<"thread_dist_batch   = "<<thread::Dist::size(str,WORLD.size(),nBatch)<<"\n";
			std::cout<<"thread_offset_train = "<<thread::Dist::offset(str,WORLD.size(),nstrucs[0])<<"\n";
			std::cout<<"thread_offset_val   = "<<thread::Dist::offset(str,WORLD.size(),nstrucs[1])<<"\n";
			std::cout<<"thread_offset_test  = "<<thread::Dist::offset(str,WORLD.size(),nstrucs[2])<<"\n";
			std::cout<<"thread_offset_batch = "<<thread::Dist::offset(str,WORLD.size(),nBatch)<<"\n";
		}
		
		//======== gen indices (random shuffle) ========
		for(int n=0; n<nData; ++n){
			indices[n].resize(nstrucs[n],-1);
			for(int i=0; i<indices[n].size(); ++i) indices[n][i]=i;
			std::random_shuffle(indices[n].begin(),indices[n].end());
			thread::bcast(WORLD.mpic(),0,indices[n]);
		}
		
		//======== read the structures ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"reading structures\n";
		for(int n=0; n<nData; ++n){
			if(files[n].size()>0){
				strucs_org[n].resize(dist_struc[n].size());
				for(int i=0; i<dist_struc[n].size(); ++i){
					const std::string& file=files[n][indices[n][dist_struc[n].index(i)]];
					read_struc(file.c_str(),format,atomT,strucs_org[n][i]);
					if(NNPTEFR_PRINT_DATA>1) std::cout<<"\t"<<file<<" "<<strucs_org[n][i].pe()<<"\n";
				}
			}
		}
		
		//======== apply alias ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"applying aliases\n";
		for(int n=0; n<nData; ++n){
			for(int i=0; i<strucs_org[n].size(); ++i){
				Structure& strucl=strucs_org[n][i];
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
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"checking the structures\n";
		for(int n=0; n<nData; ++n){
			for(int i=0; i<dist_struc[n].size(); ++i){
				const std::string filename=files[n][indices[n][dist_struc[n].index(i)]];
				const Structure& strucl=strucs_org[n][i];
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
				for(int j=0; j<strucl.nAtoms(); ++j){
					bool match=false;
					for(int k=0; k<nnpte.nnp().ntypes(); ++k){
						if(strucl.name(j)==nnpte.nnp().nnh(k).type().name()){
							match=true; break;
						}
					}
					if(!match) throw std::runtime_error(std::string("Could not find type for atom \"")+strucl.name(j)+std::string("\""));
				}
				if(NNPTEFR_PRINT_DATA>1) std::cout<<"\t"<<filename<<" "<<strucl.pe()<<" "<<WORLD.rank()<<"\n";
			}
		}
		
		//************************************************************************************
		// ATOM PROPERTIES
		//************************************************************************************
		
		//======== set atom properties ========
		if(WORLD.rank()==0) std::cout<<"setting atomic properties\n";
		
		//======== set the indices ========
		if(WORLD.rank()==0) std::cout<<"setting the indices\n";
		for(int n=0; n<nData; ++n){
			for(int i=0; i<strucs_org[n].size(); ++i){
				for(int j=0; j<strucs_org[n][i].nAtoms(); ++j){
					strucs_org[n][i].index(j)=j;
				}
			}
		}
		
		//======== set the types ========
		if(WORLD.rank()==0) std::cout<<"setting the types\n";
		for(int n=0; n<nData; ++n){
			for(int i=0; i<strucs_org[n].size(); ++i){
				for(int j=0; j<strucs_org[n][i].nAtoms(); ++j){
					strucs_org[n][i].type(j)=nnpte.nnp().index(strucs_org[n][i].name(j));
				}
			}
		}
		
		//======== set the charges ========
		if(atomT.charge){
			if(WORLD.rank()==0) std::cout<<"setting charges\n";
			for(int n=0; n<nData; ++n){
				for(int i=0; i<strucs_org[n].size(); ++i){
					for(int j=0; j<strucs_org[n][i].nAtoms(); ++j){
						strucs_org[n][i].charge(j)=nnpte.nnp().nnh(strucs_org[n][i].type(j)).type().charge().val();
					}
				}
			}
		}
		
		//======== set the electronegativities ========
		if(atomT.chi){
			if(WORLD.rank()==0) std::cout<<"setting electronegativities\n";
			for(int n=0; n<nData; ++n){
				for(int i=0; i<strucs_org[n].size(); ++i){
					for(int j=0; j<strucs_org[n][i].nAtoms(); ++j){
						strucs_org[n][i].chi(j)=nnpte.nnp().nnh(strucs_org[n][i].type(j)).type().chi().val();
					}
				}
			}
		}
		
		//======== set the idempotentials ========
		if(atomT.eta){
			if(WORLD.rank()==0) std::cout<<"setting idempotentials\n";
			for(int n=0; n<nData; ++n){
				for(int i=0; i<strucs_org[n].size(); ++i){
					for(int j=0; j<strucs_org[n][i].nAtoms(); ++j){
						strucs_org[n][i].eta(j)=nnpte.nnp().nnh(strucs_org[n][i].type(j)).type().eta().val();
					}
				}
			}
		}
		
		//======== add new structures ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"adding structures\n";
		std::shared_ptr<rng::dist::Base> rdistp;
		switch(rdist){
			case rng::dist::Name::UNIFORM:{
				rdistp.reset(new rng::dist::Uniform(0.0,rdelta));
			}break;
			case rng::dist::Name::LAPLACE:{
				rdistp.reset(new rng::dist::Laplace(0.0,rdelta));
			}break;
			case rng::dist::Name::NORMAL:{
				rdistp.reset(new rng::dist::Normal(0.0,rdelta));
			}break;
			case rng::dist::Name::LOGISTIC:{
				rdistp.reset(new rng::dist::Logistic(0.0,rdelta));
			}break;
			case rng::dist::Name::SECH:{
				rdistp.reset(new rng::dist::Sech(0.0,rdelta));
			}break;
			case rng::dist::Name::COSINE:{
				rdistp.reset(new rng::dist::Cosine(0.0,rdelta));
			}break;
			default:{
				throw std::invalid_argument("Invalid position distribution.");
			}break;
		}
		std::vector<int> fzero(nData,0);
		for(int n=0; n<nData; ++n){
			std::uniform_real_distribution<double> dista(-1.0,1.0);
			if(strucs_org[n].size()>0){
				int nstrucstot=0;
				//find the number of structures without force information
				for(int i=0; i<strucs_org[n].size(); ++i){
					const int nAtoms=strucs_org[n][i].nAtoms();
					//compute total force
					double favg=0;
					for(int j=0; j<nAtoms; ++j){
						favg+=strucs_org[n][i].force(j).norm();
					}
					favg/=nAtoms;
					if(favg<1.0e-12) fzero[n]++;
				}
				//resize the total number of structures
				nstrucstot=strucs_org[n].size()+(strucs_org[n].size()-fzero[n])*nadd;
				strucs_tot[n].resize(nstrucstot);
				//add new structures
				int count=0;
				for(int i=0; i<strucs_org[n].size(); ++i){
					const int nAtoms=strucs_org[n][i].nAtoms();
					//original
					strucs_tot[n][count]=strucs_org[n][i];
					++count;
					//compute average/total force
					double favg=0;
					for(int j=0; j<nAtoms; ++j){
						favg+=strucs_org[n][i].force(j).norm();
					}
					favg/=nAtoms;
					//printf("file[n][%i] = %s pe = %.12f\n",i,files[n][i].c_str(),strucs_org[n][i].pe());
					//additional structures
					if(favg>1.0e-12){
						for(int m=0; m<nadd; ++m){
							strucs_tot[n][count]=strucs_org[n][i];
							for(int j=0; j<nAtoms; ++j){
								const Eigen::Vector3d& force=strucs_tot[n][count].force(j);
								//compute norm and angles
								double vr=0;
								switch(perturb){
									case Perturb::CONSTANT:{
										vr=rdistp->rand(rngen);
									} break;
									case Perturb::COVALENT:{
										const int type=nnpte.nnp().index(strucs_tot[n][count].name(j));
										const double rcov=nnpte.nnp().nnh(type).type().rcov().val();
										vr=rdistp->rand(rngen)*rcov;
									} break;
									default:{
										throw std::invalid_argument("Invalid perturbation.");
									} break;
								}
								//make perturbation vector
								const double vctheta=dista(rngen);
								const double vstheta=sqrt(1.0-vctheta*vctheta);
								const double vphi=dista(rngen)*math::constant::PI;
								Eigen::Vector3d dr=Eigen::Vector3d::Zero();
								dr[0]=vr*vstheta*cos(vphi);
								dr[1]=vr*vstheta*sin(vphi);
								dr[2]=vr*vctheta;
								//perturb structure
								strucs_tot[n][count].posn(j).noalias()+=dr;
								strucs_tot[n][count].pe()-=dr.dot(force);
							}
							//printf("pe - new [%i] = %.12f\n",count,strucs_tot[n][count].pe());
							//clear forces - no longer needed
							strucs_tot[n][count].force().clear();
							++count;
						}
					}
				}
			}
		}
		MPI_Allreduce(MPI_IN_PLACE,fzero.data(),nData,MPI_INT,MPI_SUM,WORLD.mpic());
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0){
			std::cout<<"zero-force structures - train : "<<fzero[0]<<"\n";
			std::cout<<"zero-force structures - val   : "<<fzero[1]<<"\n";
			std::cout<<"zero-force structures - test  : "<<fzero[2]<<"\n";
		}
		
		//************************************************************************************
		// EXTERNAL POTENTIALS
		//************************************************************************************
		
		//======== compute coulomb energies ========
		if(compute.coul){
			if(WORLD.rank()==0) std::cout<<"computing coulomb energies\n";
			for(int n=0; n<nData; ++n){
				for(int i=0; i<strucs_org[n].size(); i++){
					NeighborList nlist(strucs_org[n][i],pot_coul.rc());
					const double ecoul=pot_coul.energy(strucs_org[n][i],nlist);
					strucs_org[n][i].ecoul()=ecoul;
					strucs_org[n][i].pe()-=ecoul;
				}
			}
			for(int n=0; n<nData; ++n){
				std::vector<double> ecoul(strucs_tot[n].size(),std::numeric_limits<double>::max());
				for(int i=0; i<strucs_tot[n].size(); i++){
					NeighborList nlist(strucs_tot[n][i],pot_coul.rc());
					const double ecoul=pot_coul.energy(strucs_tot[n][i],nlist);
					strucs_tot[n][i].ecoul()=ecoul;
					strucs_tot[n][i].pe()-=ecoul;
				}
			}
		}
		
		//======== compute vdw energies ========
		if(compute.vdwl){
			if(WORLD.rank()==0) std::cout<<"computing vdw_l energies\n";
			Reduce<1> ralpha,rerrer,rerrek,rNK;
			std::vector<Reduce<1> > rnk(nData);
			//compute parameters
			pot_vdw_l.resize(nnpte.nnp().ntypes());
			pot_vdw_l.ksl().rc()=pot_vdw_l.rc();
			pot_vdw_l.ksl().prec()=pot_vdw_l.prec();
			for(int i=0; i<nnpte.nnp().ntypes(); ++i){
				const double ri=nnpte.nnp().nnh(i).type().rvdw().val();
				const double ci=nnpte.nnp().nnh(i).type().c6().val();
				pot_vdw_l.rvdw()(i,i)=ri;
				pot_vdw_l.c6()(i,i)=ci;
			}
			pot_vdw_l.init();
			if(WORLD.rank()==0){
				for(int i=0; i<nnpte.nnp().ntypes(); ++i){
					const std::string ni=nnpte.nnp().nnh(i).type().name();
					for(int j=0; j<nnpte.nnp().ntypes(); ++j){
						const std::string nj=nnpte.nnp().nnh(j).type().name();
						std::cout<<"c6("<<ni<<","<<nj<<") = "<<pot_vdw_l.c6()(i,j)<<"\n";
						std::cout<<"rvdw("<<ni<<","<<nj<<") = "<<pot_vdw_l.rvdw()(i,j)<<"\n";
					}
				}
			}
			//compute energy - org
			for(int n=0; n<nData; ++n){
				for(int i=0; i<strucs_org[n].size(); i++){
					NeighborList nlist(strucs_org[n][i],pot_vdw_l.rc());
					const double evdw=pot_vdw_l.energy(strucs_org[n][i],nlist);
					strucs_org[n][i].evdw()=evdw;
					strucs_org[n][i].pe()-=evdw;
					ralpha.push(pot_vdw_l.ksl().alpha());
					rerrer.push(pot_vdw_l.ksl().errEr());
					rerrek.push(pot_vdw_l.ksl().errEk());
					rnk[0].push(pot_vdw_l.ksl().nk()[0]*1.0);
					rnk[1].push(pot_vdw_l.ksl().nk()[1]*1.0);
					rnk[2].push(pot_vdw_l.ksl().nk()[2]*1.0);
					rNK.push(pot_vdw_l.ksl().nk().prod());
				}
			}
			//compute energy - tot
			for(int n=0; n<nData; ++n){
				std::vector<double> evdw(strucs_tot[n].size(),std::numeric_limits<double>::max());
				for(int i=0; i<strucs_tot[n].size(); i++){
					NeighborList nlist(strucs_tot[n][i],pot_vdw_l.rc());
					const double evdw=pot_vdw_l.energy(strucs_tot[n][i],nlist);
					strucs_tot[n][i].evdw()=evdw;
					strucs_tot[n][i].pe()-=evdw;
				}
			}
			//print
			if(WORLD.rank()==0){
				std::cout<<"alpha = "<<ralpha.avg()<<" "<<ralpha.min()<<" "<<ralpha.max()<<" "<<ralpha.dev()<<"\n";
				std::cout<<"errEr = "<<rerrer.avg()<<" "<<rerrer.min()<<" "<<rerrer.max()<<" "<<rerrer.dev()<<"\n";
				std::cout<<"errEk = "<<rerrek.avg()<<" "<<rerrek.min()<<" "<<rerrek.max()<<" "<<rerrek.dev()<<"\n";
				std::cout<<"nk[0] = "<<rnk[0].avg()<<" "<<rnk[0].min()<<" "<<rnk[0].max()<<" "<<rnk[0].dev()<<"\n";
				std::cout<<"nk[1] = "<<rnk[1].avg()<<" "<<rnk[1].min()<<" "<<rnk[1].max()<<" "<<rnk[1].dev()<<"\n";
				std::cout<<"nk[2] = "<<rnk[2].avg()<<" "<<rnk[2].min()<<" "<<rnk[2].max()<<" "<<rnk[2].dev()<<"\n";
				std::cout<<"nk    = "<<rNK.avg()<<" "<<rNK.min()<<" "<<rNK.max()<<" "<<rNK.dev()<<"\n";
			}
		}
		
		//======== compute vdw energies ========
		if(compute.vdws){
			if(WORLD.rank()==0) std::cout<<"computing vdw_s energies\n";
			//compute parameters
			pot_vdw_s.resize(nnpte.nnp().ntypes());
			for(int i=0; i<nnpte.nnp().ntypes(); ++i){
				const double ri=nnpte.nnp().nnh(i).type().rvdw().val();
				const double ci=nnpte.nnp().nnh(i).type().c6().val();
				pot_vdw_s.rvdw()(i,i)=ri;
				pot_vdw_s.c6()(i,i)=ci;
			}
			pot_vdw_s.init();
			if(WORLD.rank()==0){
				for(int i=0; i<nnpte.nnp().ntypes(); ++i){
					const std::string ni=nnpte.nnp().nnh(i).type().name();
					for(int j=0; j<nnpte.nnp().ntypes(); ++j){
						const std::string nj=nnpte.nnp().nnh(j).type().name();
						std::cout<<"c6("<<ni<<","<<nj<<") = "<<pot_vdw_s.c6()(i,j)<<"\n";
						std::cout<<"rvdw("<<ni<<","<<nj<<") = "<<pot_vdw_s.rvdw()(i,j)<<"\n";
					}
				}
			}
			//compute energy - org
			for(int n=0; n<nData; ++n){
				for(int i=0; i<strucs_org[n].size(); i++){
					NeighborList nlist(strucs_org[n][i],pot_vdw_s.rc());
					const double evdw=pot_vdw_s.energy(strucs_org[n][i],nlist);
					strucs_org[n][i].evdw()=evdw;
					strucs_org[n][i].pe()-=evdw;
				}
			}
			//compute energy - tot
			for(int n=0; n<nData; ++n){
				std::vector<double> evdw(strucs_tot[n].size(),std::numeric_limits<double>::max());
				for(int i=0; i<strucs_tot[n].size(); i++){
					NeighborList nlist(strucs_tot[n][i],pot_vdw_s.rc());
					const double evdw=pot_vdw_s.energy(strucs_tot[n][i],nlist);
					strucs_tot[n][i].evdw()=evdw;
					strucs_tot[n][i].pe()-=evdw;
				}
			}
		}
		
		//======== compute repulsive energies ========
		if(compute.rep){
			if(WORLD.rank()==0) std::cout<<"computing repulsive energies\n";
			//compute parameters
			pot_rep.resize(nnpte.nnp().ntypes());
			for(int i=0; i<nnpte.nnp().ntypes(); ++i){
				const double ri=nnpte.nnp().nnh(i).type().rcov().val();
				const double zi=nnpte.nnp().nnh(i).type().z().val();
				pot_rep.r()[i]=ri;
				pot_rep.z()[i]=zi;
				pot_rep.f()[i]=1;
			}
			pot_rep.init();
			if(WORLD.rank()==0){
				for(int i=0; i<nnpte.nnp().ntypes(); ++i){
					const std::string ni=nnpte.nnp().nnh(i).type().name();
					std::cout<<"r("<<ni<<") = "<<pot_rep.r()(i)<<"\n";
					std::cout<<"z("<<ni<<") = "<<pot_rep.z()(i)<<"\n";
				}
			}
			//compute energy
			for(int n=0; n<nData; ++n){
				for(int i=0; i<strucs_org[n].size(); i++){
					NeighborList nlist(strucs_org[n][i],pot_rep.rc());
					const double erep=pot_rep.energy(strucs_org[n][i],nlist);
					strucs_org[n][i].erep()=erep;
					strucs_org[n][i].pe()-=erep;
				}
			}
			for(int n=0; n<nData; ++n){
				std::vector<double> erep(strucs_tot[n].size(),std::numeric_limits<double>::max());
				for(int i=0; i<strucs_tot[n].size(); i++){
					NeighborList nlist(strucs_tot[n][i],pot_rep.rc());
					const double erep=pot_rep.energy(strucs_tot[n][i],nlist);
					strucs_tot[n][i].erep()=erep;
					strucs_tot[n][i].pe()-=erep;
				}
			}
		}
		
		//************************************************************************************
		// SET INPUTS
		//************************************************************************************
		
		//======== initialize the symmetry functions ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"initializing symmetry functions\n";
		for(int n=0; n<nData; ++n){
			for(int i=0; i<strucs_org[n].size(); ++i){
				for(int j=0; j<strucs_org[n][i].nAtoms(); ++j){
					strucs_org[n][i].symm(j).resize(nnpte.nnp().nnh(strucs_org[n][i].type(j)).nInput());
				}
			}
			for(int i=0; i<strucs_tot[n].size(); ++i){
				for(int j=0; j<strucs_tot[n][i].nAtoms(); ++j){
					strucs_tot[n][i].symm(j).resize(nnpte.nnp().nnh(strucs_tot[n][i].type(j)).nInput());
				}
			}
		}
		
		//======== compute the symmetry functions ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing the inputs (symmetry functions)\n";
		for(int n=0; n<nData; ++n){
			clock.begin();
			if(strucs_org[n].size()>0){
				for(int i=0; i<strucs_org[n].size(); i++){
					if(NNPTEFR_PRINT_STATUS>0) std::cout<<"structure-train["<<i<<"]\n";
					NeighborList nlist(strucs_org[n][i],nnpte.nnp().rc());
					NNP::symm(nnpte.nnp(),strucs_org[n][i],nlist);
				}
			}
			clock.end();
			time_symm[n]=clock.duration();
		}
		for(int n=0; n<nData; ++n){
			clock.begin();
			if(strucs_tot[n].size()>0){
				for(int i=0; i<strucs_tot[n].size(); i++){
					if(NNPTEFR_PRINT_STATUS>0) std::cout<<"structure-train["<<i<<"]\n";
					NeighborList nlist(strucs_tot[n][i],nnpte.nnp().rc());
					NNP::symm(nnpte.nnp(),strucs_tot[n][i],nlist);
				}
			}
			clock.end();
			time_symm[n]=clock.duration();
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== clear excess memory ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"clearing excess memory\n";
		for(int n=0; n<nData; ++n){
			if(strucs_tot[n].size()>0){
				for(int i=0; i<strucs_tot[n].size(); ++i){
					strucs_tot[n][i].posn().clear();
					strucs_tot[n][i].name().clear();
					strucs_tot[n][i].charge().clear();
					strucs_tot[n][i].eta().clear();
					strucs_tot[n][i].chi().clear();
				}
			}
		}
		
		//======== print the memory ========
		{
			//compute memory (gb)
			double meml[nData]={0,0,0};
			for(int n=0; n<nData; ++n) for(int i=0; i<strucs_org[n].size(); ++i) meml[n]+=serialize::nbytes(strucs_org[n][i])/1.0e9;
			//allocate arrays
			std::vector<std::vector<double> > mem(nData,std::vector<double>(WORLD.size(),0.0));
			//gather memory
			for(int n=0; n<nData; ++n) MPI_Gather(&meml[n],1,MPI_DOUBLE,mem[n].data(),1,MPI_DOUBLE,0,WORLD.mpic());
			//compute total
			std::vector<double> memt(nData,0.0);
			for(int n=0; n<nData; ++n) for(int i=0; i<WORLD.size(); ++i) memt[n]+=mem[n][i];
			//print
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MEMORY - ORG",strbuf)<<"\n";
				std::cout<<"memory unit - GB\n";
				std::cout<<"mem - train - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<mem[0][i]<<" "; std::cout<<"\n";
				std::cout<<"mem - val   - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<mem[1][i]<<" "; std::cout<<"\n";
				std::cout<<"mem - test  - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<mem[2][i]<<" "; std::cout<<"\n";
				std::cout<<"mem - train - tot = "<<memt[0]<<"\n";
				std::cout<<"mem - val   - tot = "<<memt[1]<<"\n";
				std::cout<<"mem - test  - tot = "<<memt[2]<<"\n";
				std::cout<<print::buf(strbuf)<<"\n";
			}
		}
		{
			//compute memory (gb)
			double meml[nData]={0,0,0};
			for(int n=0; n<nData; ++n) for(int i=0; i<strucs_tot[n].size(); ++i) meml[n]+=serialize::nbytes(strucs_tot[n][i])/1.0e9;
			//allocate arrays
			std::vector<std::vector<double> > mem(nData,std::vector<double>(WORLD.size(),0));
			//gather memory
			for(int n=0; n<nData; ++n) MPI_Gather(&meml[n],1,MPI_DOUBLE,mem[n].data(),1,MPI_DOUBLE,0,WORLD.mpic());
			//compute total
			std::vector<double> memt(nData,0.0);
			for(int n=0; n<nData; ++n) for(int i=0; i<WORLD.size(); ++i) memt[n]+=mem[n][i];
			//print
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MEMORY - TOT",strbuf)<<"\n";
				std::cout<<"memory unit - GB\n";
				std::cout<<"mem - train - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<mem[0][i]<<" "; std::cout<<"\n";
				std::cout<<"mem - val   - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<mem[1][i]<<" "; std::cout<<"\n";
				std::cout<<"mem - test  - loc = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<mem[2][i]<<" "; std::cout<<"\n";
				std::cout<<"mem - train - tot = "<<memt[0]<<"\n";
				std::cout<<"mem - val   - tot = "<<memt[1]<<"\n";
				std::cout<<"mem - test  - tot = "<<memt[2]<<"\n";
				std::cout<<print::buf(strbuf)<<"\n";
			}
		}
		
		//************************************************************************************
		// TRAINING
		//************************************************************************************
		
		//======== subtract ground-state energies ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"subtracting ground-state energies\n";
		for(int n=0; n<nData; ++n){
			for(int i=0; i<strucs_org[n].size(); ++i){
				for(int j=0; j<strucs_org[n][i].nAtoms(); ++j){
					strucs_org[n][i].pe()-=nnpte.nnp().nnh(strucs_org[n][i].type(j)).type().energy().val();
				}
			}
			for(int i=0; i<strucs_tot[n].size(); ++i){
				for(int j=0; j<strucs_tot[n][i].nAtoms(); ++j){
					strucs_tot[n][i].pe()-=nnpte.nnp().nnh(strucs_tot[n][i].type(j)).type().energy().val();
				}
			}
		}
		
		//======== train the nn potential ========
		if(mode==Mode::TRAIN){
			if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"training the nn potential\n";
			nnpte.train(dist_struc[nData].size(),strucs_tot[0],strucs_tot[1]);
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== add ground-state energies ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"adding ground-state energies\n";
		for(int n=0; n<nData; ++n){
			for(int i=0; i<strucs_org[n].size(); ++i){
				for(int j=0; j<strucs_org[n][i].nAtoms(); ++j){
					strucs_org[n][i].pe()+=nnpte.nnp().nnh(strucs_org[n][i].type(j)).type().energy().val();
				}
			}
			for(int i=0; i<strucs_tot[n].size(); ++i){
				for(int j=0; j<strucs_tot[n][i].nAtoms(); ++j){
					strucs_tot[n][i].pe()+=nnpte.nnp().nnh(strucs_tot[n][i].type(j)).type().energy().val();
				}
			}
		}
		
		//************************************************************************************
		// EVALUATION
		//************************************************************************************
		
		//======== statistical data - energies/forces/errors ========
		std::vector<double> kendall(nData,0);
		std::vector<Reduce<1> > r1_energy(nData);
		std::vector<Reduce<2> > r2_energy(nData);
		std::vector<Reduce<1> > r1_force(nData);
		std::vector<std::vector<Reduce<2> > > r2_force(nData,std::vector<Reduce<2> >(3));
		
		//======== compute the final energies ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final energies\n";
		for(int n=0; n<nData; ++n){
			if(dist_struc[n].size()>0){
				std::vector<double> energy_n(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_n_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<double> energy_r_t(nstrucs[n],std::numeric_limits<double>::max());
				std::vector<int> natoms(nstrucs[n],0); std::vector<int> natoms_t(nstrucs[n],0);
				//compute energies
				clock.begin();
				for(int i=0; i<dist_struc[n].size(); ++i){
					if(NNPTEFR_PRINT_STATUS>0) std::cout<<"structure["<<WORLD.rank()<<"]["<<i<<"]\i";
					energy_r[dist_struc[n].index(i)]=strucs_org[n][i].pe();
					energy_n[dist_struc[n].index(i)]=NNP::energy(nnpte.nnp(),strucs_org[n][i]);
					natoms[dist_struc[n].index(i)]=strucs_org[n][i].nAtoms();
				}
				clock.end();
				time_energy[n]=clock.duration();
				if(compute.zero){
					for(int i=0; i<dist_struc[n].size(); ++i){
						for(int j=0; j<strucs_org[n][i].nAtoms(); ++j){
							energy_r[dist_struc[n].index(i)]-=nnpte.nnp().nnh(strucs_org[n][i].type(j)).type().energy().val();
							energy_n[dist_struc[n].index(i)]-=nnpte.nnp().nnh(strucs_org[n][i].type(j)).type().energy().val();
						}
					}
				}
				MPI_Reduce(energy_r.data(),energy_r_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
				MPI_Reduce(energy_n.data(),energy_n_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
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
		if(write.coul){
			if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final coulomb energies\n";
			for(int n=0; n<nData; ++n){
				if(dist_struc[n].size()>0){
					std::vector<double> energy_r(nstrucs[n],std::numeric_limits<double>::max());
					std::vector<double> energy_r_t(nstrucs[n],std::numeric_limits<double>::max());
					std::vector<int> natoms(nstrucs[n],0); std::vector<int> natoms_t(nstrucs[n],0);
					//compute energies
					clock.begin();
					for(int i=0; i<dist_struc[n].size(); ++i){
						if(NNPTEFR_PRINT_STATUS>0) std::cout<<"structure-train["<<WORLD.rank()<<"]["<<i<<"]\n";
						energy_r[dist_struc[n].index(i)]=strucs_org[n][i].ecoul();
						natoms[dist_struc[n].index(i)]=strucs_org[n][i].nAtoms();
					}
					clock.end();
					MPI_Reduce(energy_r.data(),energy_r_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
					MPI_Reduce(natoms.data(),natoms_t.data(),nstrucs[n],MPI_INT,MPI_MAX,0,WORLD.mpic());
					//normalize
					if(compute.norm){
						for(int i=0; i<nstrucs[n]; ++i) energy_r_t[i]/=natoms_t[i];
					}
					//write energies
					if(WORLD.rank()==0){
						std::string file;
						switch(n){
							case 0: file="nnp_ecoul_train.dat"; break;
							case 1: file="nnp_ecoul_val.dat"; break;
							case 2: file="nnp_ecoul_test.dat"; break;
							default: file="ERROR.dat"; break;
						}
						FILE* writer=fopen(file.c_str(),"w");
						if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
						else{
							std::vector<std::pair<int,double> > energy_r_pair(nstrucs[n]);
							for(int i=0; i<nstrucs[n]; ++i){
								energy_r_pair[i].first=indices[n][i];
								energy_r_pair[i].second=energy_r_t[i];
							}
							std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
							fprintf(writer,"#STRUCTURE ENERGY_COUL\n");
							for(int i=0; i<nstrucs[n]; ++i){
								fprintf(writer,"%s %f\n",files[n][i].c_str(),energy_r_pair[i].second);
							}
							fclose(writer); writer=NULL;
						}
					}
				}
			}
		}
		
		//======== compute the final vdw energies ========
		if(write.vdw){
			if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final vdw energies\n";
			for(int n=0; n<nData; ++n){
				if(dist_struc[n].size()>0){
					std::vector<double> energy_r(nstrucs[n],std::numeric_limits<double>::max());
					std::vector<double> energy_r_t(nstrucs[n],std::numeric_limits<double>::max());
					std::vector<int> natoms(nstrucs[n],0); std::vector<int> natoms_t(nstrucs[n],0);
					//compute energies
					clock.begin();
					for(int i=0; i<dist_struc[n].size(); ++i){
						if(NNPTEFR_PRINT_STATUS>0) std::cout<<"structure-train["<<WORLD.rank()<<"]["<<i<<"]\n";
						energy_r[dist_struc[n].index(i)]=strucs_org[n][i].evdw();
						natoms[dist_struc[n].index(i)]=strucs_org[n][i].nAtoms();
					}
					clock.end();
					MPI_Reduce(energy_r.data(),energy_r_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
					MPI_Reduce(natoms.data(),natoms_t.data(),nstrucs[n],MPI_INT,MPI_MAX,0,WORLD.mpic());
					//normalize
					if(compute.norm){
						for(int i=0; i<nstrucs[n]; ++i) energy_r_t[i]/=natoms_t[i];
					}
					//write energies
					if(WORLD.rank()==0){
						std::string file;
						switch(n){
							case 0: file="nnp_evdw_train.dat"; break;
							case 1: file="nnp_evdw_val.dat"; break;
							case 2: file="nnp_evdw_test.dat"; break;
							default: file="ERROR.dat"; break;
						}
						FILE* writer=fopen(file.c_str(),"w");
						if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
						else{
							std::vector<std::pair<int,double> > energy_r_pair(nstrucs[n]);
							for(int i=0; i<nstrucs[n]; ++i){
								energy_r_pair[i].first=indices[n][i];
								energy_r_pair[i].second=energy_r_t[i];
							}
							std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
							fprintf(writer,"#STRUCTURE ENERGY_VDW\n");
							for(int i=0; i<nstrucs[n]; ++i){
								fprintf(writer,"%s %f\n",files[n][i].c_str(),energy_r_pair[i].second);
							}
							fclose(writer); writer=NULL;
						}
					}
				}
			}
		}
		
		//======== compute the final repulsive energies ========
		if(write.rep){
			if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing final repulsive energies\n";
			for(int n=0; n<nData; ++n){
				if(dist_struc[n].size()>0){
					std::vector<double> energy_r(nstrucs[n],std::numeric_limits<double>::max());
					std::vector<double> energy_r_t(nstrucs[n],std::numeric_limits<double>::max());
					std::vector<int> natoms(nstrucs[n],0); std::vector<int> natoms_t(nstrucs[n],0);
					//compute energies
					for(int i=0; i<dist_struc[n].size(); ++i){
						energy_r[dist_struc[n].index(i)]=strucs_org[n][i].erep();
						natoms[dist_struc[n].index(i)]=strucs_org[n][i].nAtoms();
					}
					MPI_Reduce(energy_r.data(),energy_r_t.data(),nstrucs[n],MPI_DOUBLE,MPI_MIN,0,WORLD.mpic());
					MPI_Reduce(natoms.data(),natoms_t.data(),nstrucs[n],MPI_INT,MPI_MAX,0,WORLD.mpic());
					//normalize
					if(compute.norm){
						for(int i=0; i<nstrucs[n]; ++i) energy_r_t[i]/=natoms_t[i];
					}
					//write energies
					if(WORLD.rank()==0){
						std::string file;
						switch(n){
							case 0: file="nnp_erep_train.dat"; break;
							case 1: file="nnp_erep_val.dat"; break;
							case 2: file="nnp_erep_test.dat"; break;
							default: file="ERROR.dat"; break;
						}
						FILE* writer=fopen(file.c_str(),"w");
						if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
						else{
							std::vector<std::pair<int,double> > energy_r_pair(nstrucs[n]);
							for(int i=0; i<nstrucs[n]; ++i){
								energy_r_pair[i].first=indices[n][i];
								energy_r_pair[i].second=energy_r_t[i];
							}
							std::sort(energy_r_pair.begin(),energy_r_pair.end(),compare_pair);
							fprintf(writer,"#STRUCTURE ENERGY_PAULI\n");
							for(int i=0; i<nstrucs[n]; ++i){
								fprintf(writer,"%s %f\n",files[n][i].c_str(),energy_r_pair[i].second);
							}
							fclose(writer); writer=NULL;
						}
					}
				}
			}
		}
		
		//======== compute the final forces ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0 && compute.force) std::cout<<"computing final forces\n";
		if(compute.force && write.force){
			for(int n=0; n<nData; ++n){
				if(dist_struc[n].size()>0){
					//compute forces
					clock.begin();
					for(int i=0; i<dist_struc[n].size(); ++i){
						if(NNPTEFR_PRINT_STATUS>0) std::cout<<"structure["<<n<<"]["<<i<<"]\n";
						Structure& struc=strucs_org[n][i];
						//compute exact forces
						std::vector<Eigen::Vector3d> f_r(struc.nAtoms());
						for(int j=0; j<struc.nAtoms(); ++j) f_r[j]=struc.force(j);
						//compute nn forces
						NeighborList nlist(struc,nnpte.nnp().rc());
						NNP::force(nnpte.nnp(),struc,nlist);
						std::vector<Eigen::Vector3d> f_n(struc.nAtoms());
						for(int j=0; j<struc.nAtoms(); ++j) f_n[j]=struc.force(j);
						//compute statistics
						for(int j=0; j<struc.nAtoms(); ++j){
							r1_force[n].push((f_r[j]-f_n[j]).norm());
							r2_force[n][0].push(f_r[j][0],f_n[j][0]);
							r2_force[n][1].push(f_r[j][1],f_n[j][1]);
							r2_force[n][2].push(f_r[j][2],f_n[j][2]);
						}
					}
					clock.end();
					time_force[n]=clock.duration();
					//accumulate statistics
					std::vector<Reduce<1> > r1fv(WORLD.size());
					thread::gather(r1_force[n],r1fv,WORLD.mpic());
					if(WORLD.rank()==0) for(int i=1; i<WORLD.size(); ++i) r1_force[n]+=r1fv[i];
					for(int i=0; i<3; ++i){
						std::vector<Reduce<2> > r2fv(WORLD.size());
						thread::gather(r2_force[n][i],r2fv,WORLD.mpic());
						if(WORLD.rank()==0) for(int j=1; j<WORLD.size(); ++j) r2_force[n][i]+=r2fv[j];
					}
				}
			}
		}
		
		//======== write the inputs ========
		if(write.input){
			for(int nn=0; nn<nData; ++nn){
				if(dist_struc[nn].size()>0){
					std::string file;
					switch(nn){
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
								for(int n=0; n<dist_struc[nn].size(); ++n){
									for(int i=0; i<strucs_org[nn][n].nAtoms(); ++i){
										fprintf(writer,"%s%i ",strucs_org[nn][n].name(i).c_str(),i);
										for(int j=0; j<strucs_org[nn][n].symm(i).size(); ++j){
											fprintf(writer,"%f ",strucs_org[nn][n].symm(i)[j]);
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
		}
		
		//======== principal component analysis ========
		if(compute.pca){
			if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"computing PCA\n";
			//==== compute PCA ====
			const int nobs=nnpte.params().size();
			const int nparams=nnpte.obj().dim();
			PCA pca(nobs,nparams);
			for(int i=0; i<nobs; ++i){
				pca.X().row(i)=nnpte.params()[i];
			}
			pca.compute();
			
			//==== store first two pc vectors ====
			if(WORLD.rank()==0) std::cout<<"computing pc vectors\n";
			const double w1=pca.w()[pca.ci(0)];
			const double w2=pca.w()[pca.ci(1)];
			const Eigen::VectorXd p1=pca.W().row(pca.ci(0));
			const Eigen::VectorXd p2=pca.W().row(pca.ci(1));
			if(WORLD.rank()==0) std::cout<<"PCA = \n"<<w1<<"\n"<<w2<<"\n";
			if(WORLD.rank()==0) std::cout<<"skew = "<<w1/w2<<"\n";
			
			//==== compute moore-penrose pseudo-inverse ====
			if(WORLD.rank()==0) std::cout<<"computing pseudo-inverse\n";
			Eigen::MatrixXd pmat(nparams,2);
			pmat.col(0)=p1;
			pmat.col(1)=p2;
			Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> decomp(pmat);
			const Eigen::MatrixXd pinverse=decomp.pseudoInverse();
			
			//==== create temporary nnp object ====
			NNP nnp=nnpte.nnp();
			
			//==== compute loss pathway ====
			if(WORLD.rank()==0) std::cout<<"computing loss pathway\n";
			const Eigen::VectorXd pcenter=nnpte.params().back();
			std::vector<Eigen::VectorXd> pvectors(nnpte.nnp().ntypes());
			for(int n=0; n<nnpte.nnp().ntypes(); ++n){
				pvectors[n].resize(nnpte.nnp().nnh(n).nn().size());
			}
			std::vector<Eigen::Vector3d> lp(nobs);
			for(int i=0; i<nobs; ++i){
				const Eigen::MatrixXd vec=pinverse*(nnpte.params()[i]-pcenter);
				//split into nnp parameter vectors
				int pcount=0;
				for(int n=0; n<nnp.ntypes(); ++n){
					std::memcpy(pvectors[n].data(),nnpte.params()[i].data()+pcount,pvectors[n].size()*sizeof(double));
					pcount+=pvectors[n].size();
					nnp.nnh(n).nn()<<pvectors[n];
				}
				//compute loss
				double error=0;
				for(int n=0; n<strucs_org[0].size(); ++n){
					double norm=0;
					const double nAtoms=strucs_org[0][n].nAtoms();
					switch(nnpte.norm()){
						case Norm::IDENTITY: norm=1.0; break;
						case Norm::LINEAR: norm=1.0/nAtoms; break;
						case Norm::SQRT: norm=1.0/sqrt(nAtoms); break;
						case Norm::CBRT: norm=1.0/cbrt(nAtoms); break;
						case Norm::LOG: norm=1.0/(log(nAtoms)+1.0); break;
						default: throw std::invalid_argument("Invalid normalization method."); break;
					}
					const double pe=NNP::energy(nnp,strucs_org[0][n]);
					const double dE=(pe-strucs_org[0][n].pe());
					const double dU=dE*norm;
					switch(nnpte.iter().loss()){
						case opt::Loss::MSE:{
							error+=0.5*dU*dU;
						} break;
						case opt::Loss::MAE:{
							error+=std::fabs(dU);
						} break;
						case opt::Loss::HUBER:{
							const double arg=dU*nnpte.deltai();
							error+=nnpte.delta2()*(sqrt(1.0+(arg*arg))-1.0);
						} break;
						case opt::Loss::ASINH:{
							const double arg=dU*nnpte.deltai();
							const double sqrtf=sqrt(1.0+arg*arg);
							const double logf=log(arg+sqrtf);
							error+=nnpte.delta2()*(1.0-sqrtf+arg*logf);
						} break;
						default: break;
					}
				}
				lp[i]<<vec(0,0),vec(1,0),error;
			}
			//reduce
			{
				double* data_loc=new double[nobs];
				double* data_tot=new double[nobs];
				for(int i=0; i<nobs; ++i) data_loc[i]=lp[i][2];
				for(int i=0; i<nobs; ++i) data_tot[i]=0.0;
				MPI_Reduce(data_loc,data_tot,nobs,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
				for(int i=0; i<nobs; ++i) lp[i][2]=data_tot[i]/nstrucs[0];
				delete[] data_loc;
				delete[] data_tot;
			}
			
			//==== compute loss surface ====
			if(WORLD.rank()==0) std::cout<<"computing loss surface\n";
			//find max diff
			double pdmax=0;
			for(int i=0; i<nobs-1; ++i){
				const double diff=(nnpte.params()[i]-pcenter).norm();
				if(diff>pdmax) pdmax=diff;
			}
			//find max lp diff
			double lpmax=0;
			for(int i=0; i<nobs-1; ++i){
				const Eigen::Vector3d lpi=lp[i];
				const Eigen::Vector3d lpb=lp.back();
				const double diff=std::sqrt(
					(lpi[0]-lpb[0])*(lpi[0]-lpb[0])+
					(lpi[1]-lpb[1])*(lpi[1]-lpb[1])
				);
				if(diff>lpmax) lpmax=diff;
			}
			if(WORLD.rank()==0){
				std::cout<<"pdmax = "<<pdmax<<"\n";
				std::cout<<"lpmax = "<<lpmax<<"\n";
				std::cout<<"pdbe  = "<<(nnpte.params().front()-nnpte.params().back()).norm()<<"\n";
			}
			const double plen=lpmax*1.5;
			//compute rg
			Eigen::VectorXd rm=nnpte.params().front();
			for(int i=1; i<nobs; ++i) rm.noalias()+=nnpte.params()[i];
			double rg=0;
			for(int i=0; i<nobs; ++i) rg+=(nnpte.params()[i]-rm).squaredNorm();
			rg=std::sqrt(rg/nobs);
			if(WORLD.rank()==0) std::cout<<"rg = "<<rg<<"\n";
			//generate grid
			const int npoints=(2*ngrid+1)*(2*ngrid+1);
			if(WORLD.rank()==0) std::cout<<"np = "<<npoints<<"\n";
			const double dp=plen/(1.0*ngrid);
			if(WORLD.rank()==0) std::cout<<"dp = "<<dp<<"\n";
			std::vector<Eigen::Vector3d> ll(npoints);
			int count=0;
			for(int i=-ngrid; i<=ngrid; ++i){
				for(int j=-ngrid; j<=ngrid; ++j){
					//generate parameter point
					const Eigen::VectorXd params=pcenter+i*dp*p1+j*dp*p2;
					//split into nnp parameter vectors
					int pcount=0;
					for(int n=0; n<nnp.ntypes(); ++n){
						std::memcpy(pvectors[n].data(),params.data()+pcount,pvectors[n].size()*sizeof(double));
						pcount+=pvectors[n].size();
						nnp.nnh(n).nn()<<pvectors[n];
					}
					//compute loss
					double error=0;
					for(int n=0; n<strucs_org[0].size(); ++n){
						double norm=0;
						const double nAtoms=strucs_org[0][n].nAtoms();
						switch(nnpte.norm()){
							case Norm::IDENTITY: norm=1.0; break;
							case Norm::LINEAR: norm=1.0/nAtoms; break;
							case Norm::SQRT: norm=1.0/sqrt(nAtoms); break;
							case Norm::CBRT: norm=1.0/cbrt(nAtoms); break;
							case Norm::LOG: norm=1.0/(log(nAtoms)+1.0); break;
							default: throw std::invalid_argument("Invalid normalization method."); break;
						}
						const double pe=NNP::energy(nnp,strucs_org[0][n]);
						const double dE=(pe-strucs_org[0][n].pe());
						const double dU=dE*norm;
						switch(nnpte.iter().loss()){
							case opt::Loss::MSE:{
								error+=0.5*dU*dU;
							} break;
							case opt::Loss::MAE:{
								error+=std::fabs(dU);
							} break;
							case opt::Loss::HUBER:{
								const double arg=dU*nnpte.deltai();
								error+=nnpte.delta2()*(sqrt(1.0+(arg*arg))-1.0);
							} break;
							case opt::Loss::ASINH:{
								const double arg=dU*nnpte.deltai();
								const double sqrtf=sqrt(1.0+arg*arg);
								const double logf=log(arg+sqrtf);
								error+=nnpte.delta2()*(1.0-sqrtf+arg*logf);
							} break;
							default: break;
						}
					}
					ll[count++]<<i*dp,j*dp,error;
				}
			}
			//reduce
			{
				double* data_loc=new double[npoints];
				double* data_tot=new double[npoints];
				for(int i=0; i<npoints; ++i) data_loc[i]=ll[i][2];
				for(int i=0; i<npoints; ++i) data_tot[i]=0.0;
				MPI_Reduce(data_loc,data_tot,npoints,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
				for(int i=0; i<npoints; ++i) ll[i][2]=data_tot[i]/nstrucs[0];
				delete[] data_loc;
				delete[] data_tot;
			}
			
			//==== write loss function ====
			if(WORLD.rank()==0) std::cout<<"writing loss function\n";
			if(WORLD.rank()==0){
				std::cout<<"writing loss function\n";
				FILE* writer=fopen("loss.dat","w");
				if(writer==NULL) throw std::runtime_error("Could not open loss file.\n");
				for(int i=0; i<ll.size(); ++i){
					fprintf(writer,"%f %f %e\n",ll[i][0],ll[i][1],ll[i][2]);
				}
			}
			
			//==== write loss path ====
			if(WORLD.rank()==0) std::cout<<"writing loss path\n";
			if(WORLD.rank()==0){
				std::cout<<"writing loss path\n";
				FILE* writer=fopen("loss_path.dat","w");
				if(writer==NULL) throw std::runtime_error("Could not open loss file.\n");
				for(int i=0; i<lp.size(); ++i){
					fprintf(writer,"%f %f %e\n",lp[i][0],lp[i][1],lp[i][2]);
				}
			}
			
			//==== write pca scores ===
			if(WORLD.rank()==0) std::cout<<"writing pca scores\n";
			if(WORLD.rank()==0){
				std::cout<<"writing pca scores\n";
				const double trace=pca.S().trace();
				FILE* writer=fopen("loss_pca.dat","w");
				if(writer==NULL) throw std::runtime_error("Could not open loss file.\n");
				const int nvars=pca.nvars();
				for(int i=0; i<nvars; ++i){
					fprintf(writer,"%e\n",pca.w()[i]/trace);
				}
			}
		}

		//======== stop the wall clock ========
		if(WORLD.rank()==0){
			clock_wall.end();
			time_wall=clock_wall.duration();
		}
		
		//************************************************************************************
		// OUTPUT
		//************************************************************************************
		
		//======== print the timing info ========
		for(int n=0; n<nData; ++n){
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
			if(strucs_org[0].size()>0){
				std::cout<<"time - symm   - train = "<<time_symm[0]<<"\n";
				std::cout<<"time - energy - train = "<<time_energy[0]<<"\n";
				std::cout<<"time - force  - train = "<<time_force[0]<<"\n";
			}
			if(strucs_org[1].size()>0){
				std::cout<<"time - symm   - val   = "<<time_symm[1]<<"\n";
				std::cout<<"time - energy - val   = "<<time_energy[1]<<"\n";
				std::cout<<"time - force  - val   = "<<time_force[1]<<"\n";
			}
			if(strucs_org[2].size()>0){
				std::cout<<"time - symm   - test  = "<<time_symm[2]<<"\n";
				std::cout<<"time - energy - test  = "<<time_energy[2]<<"\n";
				std::cout<<"time - force  - test  = "<<time_force[2]<<"\n";
			}
			std::cout<<"time - wall           = "<<time_wall<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print the error statistics ========
		if(WORLD.rank()==0){
			for(int n=0; n<nData; ++n){
				if(nstrucs[n]>0){
					std::cout<<print::buf(strbuf)<<"\n";
					if(n==0) std::cout<<print::title("ERROR - STATISTICS - TRAINING",strbuf)<<"\n";
					else if(n==1) std::cout<<print::title("ERROR - STATISTICS - VALIDATION",strbuf)<<"\n";
					else if(n==2) std::cout<<print::title("ERROR - STATISTICS - TESTING",strbuf)<<"\n";
					std::cout<<"\tERROR - AVG - "<<dnames[n]<<" - ENERGY/ATOM = "<<r1_energy[n].avg()<<"\n";
					std::cout<<"\tERROR - DEV - "<<dnames[n]<<" - ENERGY/ATOM = "<<r1_energy[n].dev()<<"\n";
					std::cout<<"\tERROR - MAX - "<<dnames[n]<<" - ENERGY/ATOM = "<<r1_energy[n].max()<<"\n";
					std::cout<<"\tERROR - M   - "<<dnames[n]<<" - ENERGY/ATOM = "<<fabs(r2_energy[n].m()-1.0)<<"\n";
					std::cout<<"\tERROR - R2  - "<<dnames[n]<<" - ENERGY/ATOM = "<<fabs(r2_energy[n].r2()-1.0)<<"\n";
					std::cout<<"\tERROR - KEN - "<<dnames[n]<<" - ENERGY/ATOM = "<<fabs(kendall[n]-1.0)<<"\n";
					if(compute.force){
					//std::cout<<"FORCE:\n";
					std::cout<<"\tERROR - AVG - FORCE - "<<dnames[n]<<" = "<<r1_force[n].avg()<<"\n";
					std::cout<<"\tERROR - DEV - FORCE - "<<dnames[n]<<" = "<<r1_force[n].dev()<<"\n";
					std::cout<<"\tERROR - MAX - FORCE - "<<dnames[n]<<" = "<<r1_force[n].max()<<"\n";
					Eigen::Vector3d m; m<<r2_force[n][0].m(),r2_force[n][1].m(),r2_force[n][2].m();
					Eigen::Vector3d r2; r2<<r2_force[n][0].r2(),r2_force[n][1].r2(),r2_force[n][2].r2();
					std::cout<<"\tERROR - M   - FORCE - "<<dnames[n]<<" = "<<(m-Eigen::Vector3d::Constant(1)).norm()<<"\n";
					std::cout<<"\tERROR - R2  - FORCE - "<<dnames[n]<<" = "<<(r2-Eigen::Vector3d::Constant(1)).norm()<<"\n";
					}
					std::cout<<print::buf(strbuf)<<"\n";
				}
			}
		}
		
		//======== write the nn's ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing the nn's\n";
		if(WORLD.rank()==0){
			NNP::write(nnpte.file_ann_.c_str(),nnpte.nnp());
		}
		//======== write restart file ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"writing restart file\n";
		if(WORLD.rank()==0){
			nnpte.write_restart(nnpte.file_restart_.c_str());
		}
		
		//======== finalize mpi ========
		if(NNPTEFR_PRINT_STATUS>-1 && WORLD.rank()==0) std::cout<<"finalizing mpi\n";
		std::cout<<std::flush;
		MPI_Barrier(WORLD.mpic());
		MPI_Finalize();
	}catch(std::exception& e){
		std::cout<<"ERROR in nnpte::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//======== free local variables ========
	delete[] paramfile;
	delete[] input;
	delete[] strbuf;
	
	return 0;
}
