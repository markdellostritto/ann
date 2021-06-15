// c++ libraries
#include <iostream>
// c libraries
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
// ann - eigen
#include "eigen.hpp"
// ann - string
#include "string.hpp"
// ann - print
#include "print.hpp"
// ann - nn - train
#include "nn_train_ekf.hpp"
// ann - list
#include "list.hpp"
// ann - time
#include "time.hpp"

//***********************************************************************
// NN Optimization
//***********************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const NNOpt& nnopt){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - OPT",str)<<"\n";
	out<<"pre-cond  = "<<nnopt.preCond_<<"\n";
	out<<"post-cond = "<<nnopt.postCond_<<"\n";
	out<<"restart   = "<<nnopt.restart_<<"\n";
	out<<"batch     = "<<nnopt.batch_<<"\n";
	out<<"alpha     = "<<nnopt.alpha_<<"\n";
	out<<"gamma     = "<<nnopt.gamma_<<"\n";
	out<<"power     = "<<nnopt.power_<<"\n";
	out<<"max_iter  = "<<nnopt.max_<<"\n";
	out<<"n_print   = "<<nnopt.nPrint_<<"\n";
	out<<"n_write   = "<<nnopt.nWrite_<<"\n";
	out<<"period    = "<<nnopt.period_<<"\n";
	out<<"step      = "<<nnopt.step_<<"\n";
	out<<"count     = "<<nnopt.count_<<"\n";
	out<<"decay     = "<<nnopt.decay_<<"\n";
	out<<print::title("NN - OPT",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

/**
* set defaults
*/
void NNOpt::defaults(){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::defaults():\n";
	//conditioning
	preCond_=false;
	postCond_=false;
	//optimization
	batch_.clear();
	nPrint_=0;
	nWrite_=0;
	step_=0;
	count_=0;
	max_=0;
	err_train_=0;
	err_val_=0;
	alpha_=0.0;
	gamma_=0.0;
	power_=0.0;
	period_=0;
	decay_=Opt::DECAY::CONST;
	//random
	cg2_.init(std::time(NULL));
	//input/output
	restart_=false;
	file_error_="nn_train_error.dat";
}

/**
* clear data
*/
void NNOpt::clear(){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::clear():\n";
	//neural network
	nn_.reset();
	//optimization
	batch_.clear();
	step_=0;
	count_=0;
	err_train_=0;
	err_val_=0;
	//conditioning
	preCond_=true;
	postCond_=true;
	//data
	inT_.reset();
	outT_.reset();
	inV_.reset();
	outV_.reset();
	//file i/o
	restart_=false;
}

void NNOpt::update_step(int step){
	switch(decay_){
		case Opt::DECAY::CONST: break;
		case Opt::DECAY::EXP:  gamma_*=exp(-alpha_); break;
		case Opt::DECAY::SQRT: gamma_*=sqrt((1.0+alpha_*step)/(1.0+alpha_*(step+1))); break;
		case Opt::DECAY::INV:  gamma_*=(1.0+alpha_*step)/(1.0+alpha_*(step+1)); break;
		case Opt::DECAY::POW:  gamma_*=pow((1.0+alpha_*step)/(1.0+alpha_*(step+1)),power_); break;
		case Opt::DECAY::STEP: if(step>0 && step%period_==0) gamma_*=alpha_;
		break;
		default: break;
	}
}

/**
* train a neural network given a set of training and validation data
* @param nn - pointer to neural network which will be optimized
* @param nbatchl - the size of the local batch (number of elements in the batch handled by a single processor)
*/
void NNOpt::train(std::shared_ptr<NeuralNet::ANN>& nn, int nbatchl){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,nbatchl)):\n";
	
	//==== set MPI variables ====
	int nprocs=1,rank=0;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	//==== initialize the randomizer ====
	std::srand(std::time(NULL));
	
	//==== set the network ====
	nn_=nn;
	
	//==== check the data ====
	std::cout<<"checking data\n";
	if(inT_==NULL) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): no training inputs provided.");
	if(outT_==NULL) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): no training outputs provided.");
	if(inV_==NULL) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): no training inputs provided.");
	if(outV_==NULL) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): no training outputs provided.");
	if(inT_->size()!=outT_->size()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): Invalid input/output - training: must be same size.");
	if(inV_->size()!=outV_->size()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): Invalid input/output - validation: must be same size.");
	for(int n=0; n<inT_->size(); ++n){
		if((*inT_)[n].size()!=nn_->nIn()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): Invalid inputs - training: incompatible with network size.");
		if((*outT_)[n].size()!=nn_->nOut()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): Invalid outputs - training: incompatible with network size.");
	}
	for(int n=0; n<inV_->size(); ++n){
		if((*inV_)[n].size()!=nn_->nIn()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): Invalid inputs - validation: incompatible with network size.");
		if((*outV_)[n].size()!=nn_->nOut()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<NeuralNet::ANN>&,int): Invalid outputs - validation: incompatible with network size.");
	}
	const int nTrainL_=inT_->size();//local number of training samples
	const int nValL_  =inV_->size();//local number of validation samples
	MPI_Allreduce(&nTrainL_,&nTrain_,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Allreduce(&nValL_,&nVal_,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	
	//==== resize the batch and indices ====
	std::cout<<"resizing batch\n";
	batch_.resize(nbatchl,nTrainL_);
	std::cout<<batch_<<"\n";
	
	//==== resize the ekf ====
	std::cout<<"resizing ekf\n";
	ekf_.resize(nn_->size(),nn_->nOut());
	std::cout<<ekf_<<"\n";
	
	//==== compute input statistics ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"computing input statistics\n";
	Eigen::VectorXd avg_in_loc=Eigen::VectorXd::Zero(nn_->nIn());
	Eigen::VectorXd avg_in_tot=Eigen::VectorXd::Zero(nn_->nIn());
	Eigen::VectorXd dev_in_loc=Eigen::VectorXd::Zero(nn_->nIn());
	Eigen::VectorXd dev_in_tot=Eigen::VectorXd::Zero(nn_->nIn());
	//compute average
	for(int n=0; n<nTrainL_; ++n){
		avg_in_loc.noalias()+=(*inT_)[n];
	}
	MPI_Allreduce(avg_in_loc.data(),avg_in_tot.data(),nn_->nIn(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	avg_in_tot/=nTrain_;
	//compute deviation
	for(int n=0; n<nTrainL_; ++n){
		dev_in_loc.noalias()+=((*inT_)[n]-avg_in_tot).cwiseProduct((*inT_)[n]-avg_in_tot);
	}
	MPI_Allreduce(dev_in_loc.data(),dev_in_tot.data(),nn_->nIn(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	for(int i=0; i<nn_->nIn(); ++i) dev_in_tot[i]=std::sqrt(dev_in_tot[i]/(nTrain_-1.0));
	//print
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
		std::cout<<"avg - in = "<<avg_in_tot.transpose()<<"\n";
		std::cout<<"dev - in = "<<dev_in_tot.transpose()<<"\n";
	}
	
	//==== compute output statistics ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"computing output statistics\n";
	Eigen::VectorXd avg_out_loc=Eigen::VectorXd::Zero(nn_->nOut());
	Eigen::VectorXd avg_out_tot=Eigen::VectorXd::Zero(nn_->nOut());
	Eigen::VectorXd dev_out_loc=Eigen::VectorXd::Zero(nn_->nOut());
	Eigen::VectorXd dev_out_tot=Eigen::VectorXd::Zero(nn_->nOut());
	//compute average
	for(int n=0; n<nTrainL_; ++n){
		avg_out_loc.noalias()+=(*outT_)[n];
	}
	MPI_Allreduce(avg_out_loc.data(),avg_out_tot.data(),nn_->nIn(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	//compute deviation
	for(int n=0; n<nTrainL_; ++n){
		dev_out_loc.noalias()+=((*outT_)[n]-avg_out_tot).cwiseProduct((*outT_)[n]-avg_out_tot);
	}
	MPI_Allreduce(dev_out_loc.data(),dev_out_tot.data(),nn_->nOut(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	for(int i=0; i<nn_->nOut(); ++i) dev_out_tot[i]=std::sqrt(dev_out_tot[i]/(nTrain_-1.0));
	//print
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
		std::cout<<"avg - out = "<<avg_out_tot.transpose()<<"\n";
		std::cout<<"dev - out = "<<dev_out_tot.transpose()<<"\n";
	}
	
	//==== precondition the input ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"pre-conditioning input\n";
	//set bias and weight
	if(preCond_){
		for(int i=0; i<nn_->nIn(); ++i) nn_->inb()[i]=-1.0*avg_in_tot[i];
		for(int i=0; i<nn_->nIn(); ++i) nn_->inw()[i]=1.0/dev_in_tot[i];
	} else {
		for(int i=0; i<nn_->nIn(); ++i) nn_->inb()[i]=0.0;
		for(int i=0; i<nn_->nIn(); ++i) nn_->inw()[i]=1.0;
	}
	//print
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
		std::cout<<"bias-in  = "<<nn_->inb().transpose()<<"\n";
		std::cout<<"scale-in = "<<nn_->inw().transpose()<<"\n";
	}
	
	//==== postcondition the output ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"post-conditioning output\n";
	if(postCond_){
		for(int i=0; i<nn_->nOut(); ++i) nn_->outb()[i]=avg_out_tot[i];
		for(int i=0; i<nn_->nOut(); ++i) nn_->outw()[i]=dev_out_tot[i];
	} else {
		for(int i=0; i<nn_->nOut(); ++i) nn_->outb()[i]=0.0;
		for(int i=0; i<nn_->nOut(); ++i) nn_->outw()[i]=1.0;
	}
	//print
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
		std::cout<<"bias-out  = "<<nn_->outb().transpose()<<"\n";
		std::cout<<"scale-out = "<<nn_->outw().transpose()<<"\n";
	}
	
	//==== set the initial values ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"setting the initial values\n";
	//set values
	*nn_>>ekf_.p();
	const int hdim=nn_->size()*nn_->nOut();
	hLocal_.resize(nn_->size(),nn_->nOut());
	hGlobal_.resize(nn_->size(),nn_->nOut());
	eLocal_.resize(nn_->nOut());
	eGlobal_.resize(nn_->nOut());
	
	//==== allocate status vectors ====
	std::vector<int> step;
	std::vector<double> gamma,err_t,err_v;
	if(rank==0){
		step.resize(max_/nPrint_);
		gamma.resize(max_/nPrint_);
		err_v.resize(max_/nPrint_);
		err_t.resize(max_/nPrint_);
	}
	
	//==== execute the optimization ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"executing the optimization\n";
	//bcast parameters
	MPI_Bcast(ekf_.p().data(),ekf_.p().size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
	bool fbreak=false;
	const double nbatchi_=1.0/batch_.size();
	const double nVali_=1.0/nVal_;
	dOutDP_.resize(*nn_);
	Clock clock;
	clock.begin();
	for(int iter=0; iter<max_; ++iter){
		double err_train_sum_=0,err_val_sum_=0;
		//set the parameters of the network
		*nn_<<ekf_.p();
		//compute the error and gradient
		error();
		//accumulate error
		MPI_Reduce(&err_train_,&err_train_sum_,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&err_val_,&err_val_sum_,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		//accumulate the error vector
		eGlobal_.setZero();
		MPI_Reduce(eLocal_.data(),eGlobal_.data(),nn_->nOut(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		//accumulate gradient
		hGlobal_.setZero();
		MPI_Reduce(hLocal_.data(),hGlobal_.data(),hdim,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		//compute step
		if(rank==0){
			//normalize error and gradient
			err_train_=err_train_sum_*nbatchi_;
			err_val_=err_val_sum_*nVali_;
			eGlobal_*=nbatchi_;
			hGlobal_*=nbatchi_;
			//print error
			if(step_%nPrint_==0){
				const int t=iter/nPrint_;
				step[t]=step_;
				gamma[t]=gamma_;
				err_t[t]=std::sqrt(2.0*err_train_);
				err_v[t]=std::sqrt(2.0*err_val_);
				printf("opt %8i gamma %12.10f err_t %12.10f err_v %12.10f\n",step[t],gamma[t],err_t[t],err_v[t]);
			}
			//set ekf derivative matrix
			ekf_.H()=hGlobal_;
			//set ekf error vector
			ekf_.E()=eGlobal_;
			//compute the new position
			ekf_.step(gamma_);
			//compute new step
			update_step(step_);
		}
		//bcast parameters
		MPI_Bcast(ekf_.p().data(),ekf_.p().size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
		//bcast break condition
		MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		if(fbreak) break;
		//increment step
		++step_;
		++count_;
	}
	clock.end();
	MPI_Barrier(MPI_COMM_WORLD);
	
	//==== write the error ====
	if(rank==0){
		FILE* writer_error_=NULL;
		if(!restart_){
			writer_error_=fopen(file_error_.c_str(),"w");
			fprintf(writer_error_,"#STEP GAMMA ERROR_RMS_TRAIN ERROR_RMS_VAL\n");
		} else writer_error_=fopen(file_error_.c_str(),"a");
		if(writer_error_==NULL) throw std::runtime_error("I/O Error: Could not open error file.");
		for(int t=0; t<step.size(); ++t){
			fprintf(writer_error_,"%6i %12.10f %12.10f %12.10f\n",step[t],gamma[t],err_t[t],err_v[t]);
		}
		fclose(writer_error_);
		writer_error_=NULL;
	}
	
	if(NN_TRAIN_PRINT_STATUS>-1 && rank==0){
		char* str=new char[print::len_buf];
		std::cout<<print::buf(str)<<"\n";
		std::cout<<print::title("OPT - SUMMARY",str)<<"\n";
		std::cout<<"n-steps = "<<step_<<"\n";
		//std::cout<<"opt-val = "<<data_.val()<<"\n";
		std::cout<<"time    = "<<clock.duration()<<"\n";
		std::cout<<print::title("OPT - SUMMARY",str)<<"\n";
		std::cout<<print::buf(str)<<"\n";
		delete[] str;
	}
}

/**
* compute the error associated with the neural network with respect to the training data
* @return the error associated with the training data
*/
double NNOpt::error(){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::error():\n";
	//reset the gradient
	hLocal_.setZero();
	eLocal_.setZero();
	
	//randomize the batch
	for(int i=0; i<batch_.size(); ++i) batch_[i]=batch_.data((batch_.count()++)%batch_.capacity());
	list::sort::insertion(batch_.elements(),batch_.size());
	if(batch_.count()>=batch_.capacity()){
		list::shuffle(batch_.data(),batch_.capacity(),cg2_);
		batch_.count()=0;
	}
	
	//compute the training error
	err_train_=0;
	for(int n=0; n<batch_.size(); ++n){
		const int m=batch_[n];
		//set the inputs
		for(int i=0; i<nn_->nIn(); ++i) nn_->in()[i]=(*inT_)[m][i];
		//execute the network
		nn_->execute();
		//compute the gradient
		dOutDP_.grad(*nn_);
		//compute the error
		err_train_+=0.5*((*outT_)[m]-nn_->out()).squaredNorm();
		//add the error
		eLocal_.noalias()+=(*outT_)[m]-nn_->out();
		//add the gradient
		for(int i=0; i<nn_->nOut(); ++i){
			int c_=0;
			for(int l=0; l<nn_->nlayer(); ++l){
				for(int j=0; j<nn_->bias(l).size(); ++j){
					hLocal_(c_++,i)+=1*dOutDP_.dodb()[i][l](j);
				}
			}
			for(int l=0; l<nn_->nlayer(); ++l){
				for(int j=0; j<nn_->edge(l).cols(); ++j){
					for(int k=0; k<nn_->edge(l).rows(); ++k){
						hLocal_(c_++,i)+=1*dOutDP_.dodw()[i][l](k,j);
					}
				}
			}
		}
	}
	
	//compute the validation error
	err_val_=0;
	if(step_%nPrint_==0 || step_%nWrite_==0){
		for(int i=0; i<inV_->size(); ++i){
			//set the inputs
			for(int j=0; j<nn_->nIn(); ++j) nn_->in()[j]=(*inV_)[i][j];
			//execute the network
			nn_->execute();
			//compute the error
			err_val_+=0.5*((*outV_)[i]-nn_->out()).squaredNorm();
		}
	}
	
	//return the error
	return err_train_;
}

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const NNOpt& obj){
		if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const NNOpt&):\n";
		int size=0;
		size+=sizeof(bool);//preCond_
		size+=sizeof(bool);//postCond_
		size+=nbytes(obj.batch());
		size+=sizeof(int);//nPrint_
		size+=sizeof(int);//nWrite_
		size+=sizeof(int);//step_
		size+=sizeof(int);//count_
		size+=sizeof(int);//max_
		size+=nbytes(obj.ekf());
		size+=sizeof(bool);//restart_
		size+=nbytes(obj.file_error());//file_error_
		size+=nbytes(obj.file_restart());//file_restart_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNOpt& obj, char* arr){
		if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"serialize::pack(const NNOpt&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.preCond(),sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.postCond(),sizeof(bool)); pos+=sizeof(bool);
		pos+=pack(obj.batch(),arr+pos);
		std::memcpy(arr+pos,&obj.nPrint(),sizeof(int)); pos+=sizeof(int);//nPrint_
		std::memcpy(arr+pos,&obj.nWrite(),sizeof(int)); pos+=sizeof(int);//nWrite_
		std::memcpy(arr+pos,&obj.step(),sizeof(int)); pos+=sizeof(int);//step_
		std::memcpy(arr+pos,&obj.count(),sizeof(int)); pos+=sizeof(int);//count_
		std::memcpy(arr+pos,&obj.max(),sizeof(int)); pos+=sizeof(int);//max_
		pos+=pack(obj.ekf(),arr+pos);
		std::memcpy(arr+pos,&obj.restart(),sizeof(bool)); pos+=sizeof(bool);
		pos+=pack(obj.file_error(),arr+pos);
		pos+=pack(obj.file_restart(),arr+pos);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNOpt& obj, const char* arr){
		if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"serialize::unpack(NNPot&,const char*):\n";
		int pos=0;
		std::memcpy(&obj.preCond(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.postCond(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		pos+=unpack(obj.batch(),arr+pos);
		std::memcpy(&obj.nPrint(),arr+pos,sizeof(int)); pos+=sizeof(int);//nPrint_
		std::memcpy(&obj.nWrite(),arr+pos,sizeof(int)); pos+=sizeof(int);//nWrite_
		std::memcpy(&obj.step(),arr+pos,sizeof(int)); pos+=sizeof(int);//step_
		std::memcpy(&obj.count(),arr+pos,sizeof(int)); pos+=sizeof(int);//count_
		std::memcpy(&obj.max(),arr+pos,sizeof(int)); pos+=sizeof(int);//max_
		pos+=unpack(obj.ekf(),arr+pos);
		std::memcpy(&obj.restart(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		pos+=unpack(obj.file_error(),arr+pos);
		pos+=unpack(obj.file_restart(),arr+pos);
		return pos;
	}
	
}