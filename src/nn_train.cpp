// c++ libraries
#include <iostream>
// c libraries
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
// ann - string
#include "string.hpp"
// ann - print
#include "print.hpp"
// ann - nn - train
#include "nn_train.hpp"

namespace NN{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const NNOpt& nnopt){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - OPT",str)<<"\n";
	out<<"PRE-COND  = "<<nnopt.preCond_<<"\n";
	out<<"POST-COND = "<<nnopt.postCond_<<"\n";
	out<<"N-BATCH   = "<<nnopt.nbatch_<<"\n";
	out<<"P-BATCH   = "<<nnopt.pbatch_<<"\n";
	out<<print::title("NN - OPT",str)<<"\n";
	out<<print::buf(str)<<"\n";
}

//==== member functions ====

/**
* set defaults
*/
void NNOpt::defaults(){
	//batch
	cbatch_=0;
	nbatch_=0;
	pbatch_=1;
	//conditioning
	preCond_=false;
	postCond_=false;
	//optimization
	data_.clear();
	err_train_=0;
	err_val_=0;
	//input/output
	restart_=false;
	file_error_="nn_train_error.dat";
	//file i/o
	writer_error_=NULL;
}

/**
* clear data
*/
void NNOpt::clear(){
	//batch
	cbatch_=0;
	batch_.clear();
	indices_.clear();
	//conditioning
	preCond_=true;
	postCond_=true;
	//data
	inT_=NULL;
	outT_=NULL;
	inV_=NULL;
	outV_=NULL;
	//optimization
	data_.clear();
	err_train_=0;
	err_val_=0;
	//neural network
	nn_=NULL;
	//input/output
	restart_=false;
	//file i/o
	if(writer_error_!=NULL){
		fclose(writer_error_);
		writer_error_=NULL;
	}
}

/**
* train a neural network given a set of training and validation data
* @param nn - pointer to neural network which will be optimized
*/
void NNOpt::train(std::shared_ptr<Network>& nn){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::train(Network&,Opt&,int):\n";
	
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
	if(inT_==NULL) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): no training inputs provided.");
	if(outT_==NULL) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): no training outputs provided.");
	if(inV_==NULL) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): no training inputs provided.");
	if(outV_==NULL) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): no training outputs provided.");
	if(inT_->size()!=outT_->size()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): Invalid input/output - training: must be same size.");
	if(inV_->size()!=outV_->size()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): Invalid input/output - validation: must be same size.");
	for(int n=0; n<inT_->size(); ++n){
		if((*inT_)[n].size()!=nn_->nIn()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): Invalid inputs - training: incompatible with network size.");
		if((*outT_)[n].size()!=nn_->nOut()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): Invalid outputs - training: incompatible with network size.");
	}
	for(int n=0; n<inV_->size(); ++n){
		if((*inV_)[n].size()!=nn_->nIn()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): Invalid inputs - validation: incompatible with network size.");
		if((*outV_)[n].size()!=nn_->nOut()) throw std::invalid_argument("NNOpt::train(std::shared_ptr<Network>&): Invalid outputs - validation: incompatible with network size.");
	}
	const int nTrainL_=inT_->size();
	const int nValL_  =inV_->size();
	MPI_Reduce(&nTrainL_,&nTrain_,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(&nValL_,&nVal_,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	
	//==== resize the batch and indices ====
	std::cout<<"resizing batch\n";
	batch_.resize(nbatch_);
	for(int i=nbatch_-1; i>=0; --i) batch_[i]=i;
	indices_.resize(inT_->size());
	for(int i=indices_.size()-1; i>=0; --i) indices_[i]=i;
	
	//==== precondition the input ====
	if(preCond_){
		if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"pre-conditioning input\n";
		Eigen::VectorXd avg=Eigen::VectorXd::Zero(nn_->nIn());
		Eigen::VectorXd avg_t=Eigen::VectorXd::Zero(nn_->nIn());
		Eigen::VectorXd dev=Eigen::VectorXd::Zero(nn_->nIn());
		Eigen::VectorXd dev_t=Eigen::VectorXd::Zero(nn_->nIn());
		double N=0,N_t=0;
		for(int n=0; n<inT_->size(); ++n){
			avg.noalias()+=(*inT_)[n]; ++N;
		}
		MPI_Reduce(&N,&N_t,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(avg.data(),avg_t.data(),nn_->nIn(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		avg_t/=N_t;
		avg=avg_t;
		N=N_t;
		MPI_Bcast(avg.data(),nn_->nIn(),MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&N,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		for(int n=0; n<inT_->size(); ++n){
			dev.noalias()+=((*inT_)[n]-avg).cwiseProduct((*inT_)[n]-avg);
		}
		MPI_Reduce(dev.data(),dev_t.data(),nn_->nIn(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		dev=dev_t;
		MPI_Bcast(dev.data(),nn_->nIn(),MPI_DOUBLE,0,MPI_COMM_WORLD);
		for(int i=nn_->nIn()-1; i>=0; --i) dev[i]=std::sqrt(dev[i]/(N-1.0));
		for(int i=nn_->nIn()-1; i>=0; --i) nn_->inb()[i]=-1.0*avg[i];
		for(int i=nn_->nIn()-1; i>=0; --i) nn_->inw()[i]=1.0/dev[i];
		if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
			std::cout<<"scaling-input:\n";
			std::cout<<"avg      = "<<avg.transpose()<<"\n";
			std::cout<<"dev      = "<<dev.transpose()<<"\n";
			std::cout<<"bias-in  = "<<nn_->inb().transpose()<<"\n";
			std::cout<<"scale-in = "<<nn_->inw().transpose()<<"\n";
		}
	} else {
		if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"no pre-conditioning of input\n";
		for(int i=nn_->nIn()-1; i>=0; --i) nn_->inb()[i]=0.0;
		for(int i=nn_->nIn()-1; i>=0; --i) nn_->inw()[i]=1.0;
	}
	//==== precondition the output ====
	if(postCond_){
		if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"pre-conditioning output\n";
		Eigen::VectorXd avg=Eigen::VectorXd::Zero(nn_->nOut());
		Eigen::VectorXd avg_t=Eigen::VectorXd::Zero(nn_->nOut());
		Eigen::VectorXd dev=Eigen::VectorXd::Zero(nn_->nOut());
		Eigen::VectorXd dev_t=Eigen::VectorXd::Zero(nn_->nOut());
		double N=0,N_t=0;
		for(int n=0; n<outT_->size(); ++n){
			avg.noalias()+=(*outT_)[n]; ++N;
		}
		MPI_Reduce(&N,&N_t,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(avg.data(),avg_t.data(),nn_->nOut(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		avg_t/=N_t;
		avg=avg_t;
		N=N_t;
		MPI_Bcast(avg.data(),nn_->nOut(),MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&N,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		for(int n=0; n<outT_->size(); ++n){
			dev.noalias()+=((*outT_)[n]-avg).cwiseProduct((*outT_)[n]-avg);
		}
		MPI_Reduce(dev.data(),dev_t.data(),nn_->nOut(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		dev=dev_t;
		MPI_Bcast(dev.data(),nn_->nOut(),MPI_DOUBLE,0,MPI_COMM_WORLD);
		for(int i=nn_->nOut()-1; i>=0; --i) dev[i]=std::sqrt(dev[i]/(N-1.0));
		for(int i=nn_->nOut()-1; i>=0; --i) nn_->outb()[i]=avg[i];
		for(int i=nn_->nOut()-1; i>=0; --i) nn_->outw()[i]=dev[i];
		if(NN_TRAIN_PRINT_STATUS>0 && rank==0){
			std::cout<<"scaling-output:\n";
			std::cout<<"avg       = "<<avg.transpose()<<"\n";
			std::cout<<"dev       = "<<dev.transpose()<<"\n";
			std::cout<<"bias-out  = "<<nn_->outb().transpose()<<"\n";
			std::cout<<"scale-out = "<<nn_->outw().transpose()<<"\n";
		}
	} else {
		if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"no pre-conditioning of output\n";
		for(int i=nn_->nOut()-1; i>=0; --i) nn_->outb()[i]=0.0;
		for(int i=nn_->nOut()-1; i>=0; --i) nn_->outw()[i]=1.0;
	}
	
	//==== initalize the optimizer ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"initializing the optimizer\n";
	data_.init(nn_->size());
	model_->init(nn_->size());
	
	//==== set the initial values ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"setting the initial values\n";
	//set values
	*nn_>>data_.p();
	*nn_>>data_.pOld();
	data_.g().setZero();
	data_.gOld().setZero();
	
	//==== open the error file ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"opening the ouput file\n";
	
	//allocate status vectors
	std::vector<int> step;
	std::vector<double> gamma,err_t,err_v;
	if(rank==0){
		step.resize(data_.max()/data_.nPrint());
		gamma.resize(data_.max()/data_.nPrint());
		err_v.resize(data_.max()/data_.nPrint());
		err_t.resize(data_.max()/data_.nPrint());
	}
	
	//==== execute the optimization ====
	if(NN_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"executing the optimization\n";
	//bcast parameters
	MPI_Bcast(data_.p().data(),data_.dim(),MPI_DOUBLE,0,MPI_COMM_WORLD);
	Eigen::VectorXd gSum_(data_.dim());
	bool fbreak=false;
	const double nbatchi_=1.0/nbatch_;
	const double nVali_=1.0/nVal_;
	const clock_t start=std::clock();
	for(int iter=0; iter<data_.max(); ++iter){
		double err_train_sum_=0,err_val_sum_=0;
		//set the parameters of the network
		*nn_<<data_.p();
		//compute the error and gradient
		error();
		//accumulate error
		MPI_Reduce(&err_train_,&err_train_sum_,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&err_val_,&err_val_sum_,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		//accumulate gradient
		MPI_Reduce(data_.g().data(),gSum_.data(),data_.dim(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		//compute step
		if(rank==0){
			//normalize error and gradient
			err_train_=err_train_sum_*nbatchi_;
			err_val_=err_val_sum_*nVali_;
			data_.g().noalias()=gSum_*nbatchi_;
			//compute regularization error and gradient
			//data_.g().noalias()+=nn_->grad_lambda(gTemp)/inT_->size();
			//print error
			if(data_.step()%data_.nPrint()==0){
				const int t=iter/data_.nPrint();
				step[t]=data_.step();
				gamma[t]=model_->gamma();
				err_t[t]=std::sqrt(2.0*err_train_);
				err_v[t]=std::sqrt(2.0*err_val_);
				printf("opt %8i gamma %12.10f err_t %12.10f err_v %12.10f\n",step[t],gamma[t],err_t[t],err_v[t]);
			}
			//compute the new position
			data_.val()=err_train_;
			model_->step(data_);
			//compute the difference
			data_.dv()=std::fabs(data_.val()-data_.valOld());
			data_.dp()=(data_.p()-data_.pOld()).norm();
			//set the new "old" values
			data_.pOld()=data_.p();//set "old" p value
			data_.gOld()=data_.g();//set "old" g value
			data_.valOld()=data_.val();//set "old" value
			//check the break condition
			switch(data_.optVal()){
				case Opt::VAL::FTOL_ABS: fbreak=(data_.val()<data_.tol()); break;
				case Opt::VAL::FTOL_REL: fbreak=(data_.dv()<data_.tol()); break;
				case Opt::VAL::XTOL_REL: fbreak=(data_.dp()<data_.tol()); break;
			}
		}
		//bcast parameters
		MPI_Bcast(data_.p().data(),data_.p().size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
		//bcast break condition
		MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		if(fbreak) break;
		//increment step
		++data_.step();
	}
	const clock_t stop=std::clock();
	const double time=((double)(stop-start))/CLOCKS_PER_SEC;
	MPI_Barrier(MPI_COMM_WORLD);
	
	//==== write the error ====
	if(rank==0){
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
		std::cout<<"n-steps = "<<data_.step()<<"\n";
		std::cout<<"opt-val = "<<data_.val()<<"\n";
		std::cout<<"time    = "<<time<<"\n";
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
	Eigen::VectorXd& g=data_.g();
	g.setZero();
	Eigen::VectorXd gTemp=g;
	//randomize the batch
	if(batch_.size()<inT_->size()){
		for(int i=0; i<batch_.size(); ++i) batch_[i]=indices_[(cbatch_++)%indices_.size()];
		std::sort(batch_.begin(),batch_.end());
		if(cbatch_>=indices_.size()){
			std::random_shuffle(indices_.begin(),indices_.end());
			cbatch_=0;
		}
	}
	//compute the training error
	err_train_=0;
	for(int i=0; i<batch_.size(); ++i){
		const int ii=batch_[i];
		//set the inputs
		for(int j=nn_->nIn()-1; j>=0; --j) nn_->in()[j]=(*inT_)[ii][j];
		//execute the network
		nn_->execute();
		//compute the error
		err_train_+=nn_->error((*outT_)[ii],gTemp);
		//add the gradient
		g.noalias()+=gTemp;
	}
	//compute the validation error
	err_val_=0;
	if(data_.step()%data_.nPrint()==0 || data_.step()%data_.nWrite()==0){
		for(int i=0; i<inV_->size(); ++i){
			//set the inputs
			for(int j=nn_->nIn()-1; j>=0; --j) nn_->in()[j]=(*inV_)[i][j];
			//execute the network
			nn_->execute();
			//compute the error
			err_val_+=nn_->error((*outV_)[i]);
		}
	}
	//return the error
	return err_train_;
}

}

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const NN::NNOpt& obj){
		int size=0;
		size+=sizeof(int);//nbatch_
		size+=sizeof(int);//cbatch_
		size+=sizeof(double);//pbatch_
		size+=sizeof(bool);//preCond_
		size+=sizeof(bool);//postCond_
		size+=nbytes(obj.data());
		switch(obj.data().algo()){
			case Opt::ALGO::SGD:      size+=nbytes(static_cast<const Opt::SGD&>(*obj.model())); break;
			case Opt::ALGO::SDM:      size+=nbytes(static_cast<const Opt::SDM&>(*obj.model())); break;
			case Opt::ALGO::NAG:      size+=nbytes(static_cast<const Opt::NAG&>(*obj.model())); break;
			case Opt::ALGO::ADAGRAD:  size+=nbytes(static_cast<const Opt::ADAGRAD&>(*obj.model())); break;
			case Opt::ALGO::ADADELTA: size+=nbytes(static_cast<const Opt::ADADELTA&>(*obj.model())); break;
			case Opt::ALGO::RMSPROP:  size+=nbytes(static_cast<const Opt::RMSPROP&>(*obj.model())); break;
			case Opt::ALGO::ADAM:     size+=nbytes(static_cast<const Opt::ADAM&>(*obj.model())); break;
			case Opt::ALGO::NADAM:    size+=nbytes(static_cast<const Opt::NADAM&>(*obj.model())); break;
			case Opt::ALGO::BFGS:     size+=nbytes(static_cast<const Opt::BFGS&>(*obj.model())); break;
			case Opt::ALGO::RPROP:    size+=nbytes(static_cast<const Opt::RPROP&>(*obj.model())); break;
			default: throw std::runtime_error("nbytes(const NNPotOpt&): Invalid optimization method."); break;
		}
		size+=sizeof(bool);//restart_
		size+=nbytes(obj.file_error());//file_error_
		size+=nbytes(obj.file_restart());//file_restart_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NN::NNOpt& obj, char* arr){
		int pos=0;
		std::memcpy(arr+pos,&obj.nbatch(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.cbatch(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.pbatch(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.preCond(),sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.postCond(),sizeof(bool)); pos+=sizeof(bool);
		pos+=pack(obj.data(),arr+pos);
		switch(obj.data().algo()){
			case Opt::ALGO::SGD:      pos+=pack(static_cast<const Opt::SGD&>(*obj.model()),arr+pos); break;
			case Opt::ALGO::SDM:      pos+=pack(static_cast<const Opt::SDM&>(*obj.model()),arr+pos); break;
			case Opt::ALGO::NAG:      pos+=pack(static_cast<const Opt::NAG&>(*obj.model()),arr+pos); break;
			case Opt::ALGO::ADAGRAD:  pos+=pack(static_cast<const Opt::ADAGRAD&>(*obj.model()),arr+pos); break;
			case Opt::ALGO::ADADELTA: pos+=pack(static_cast<const Opt::ADADELTA&>(*obj.model()),arr+pos); break;
			case Opt::ALGO::RMSPROP:  pos+=pack(static_cast<const Opt::RMSPROP&>(*obj.model()),arr+pos); break;
			case Opt::ALGO::ADAM:     pos+=pack(static_cast<const Opt::ADAM&>(*obj.model()),arr+pos); break;
			case Opt::ALGO::NADAM:    pos+=pack(static_cast<const Opt::NADAM&>(*obj.model()),arr+pos); break;
			case Opt::ALGO::BFGS:     pos+=pack(static_cast<const Opt::BFGS&>(*obj.model()),arr+pos); break;
			case Opt::ALGO::RPROP:    pos+=pack(static_cast<const Opt::RPROP&>(*obj.model()),arr+pos); break;
			default: throw std::runtime_error("pack(const NNPotOpt&,char*): Invalid optimization method."); break;
		}
		std::memcpy(arr+pos,&obj.restart(),sizeof(bool)); pos+=sizeof(bool);
		pos+=pack(obj.file_error(),arr+pos);
		pos+=pack(obj.file_restart(),arr+pos);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NN::NNOpt& obj, const char* arr){
		int pos=0;
		std::memcpy(&obj.nbatch(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.cbatch(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.pbatch(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.preCond(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.postCond(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		pos+=unpack(obj.data(),arr+pos);
		switch(obj.data().algo()){
			case Opt::ALGO::SGD:
				obj.model().reset(new Opt::SGD());
				pos+=unpack(static_cast<Opt::SGD&>(*obj.model()),arr+pos);
			break;
			case Opt::ALGO::SDM:
				obj.model().reset(new Opt::SDM());
				pos+=unpack(static_cast<Opt::SDM&>(*obj.model()),arr+pos);
			break;
			case Opt::ALGO::NAG:
				obj.model().reset(new Opt::NAG());
				pos+=unpack(static_cast<Opt::NAG&>(*obj.model()),arr+pos);
			break;
			case Opt::ALGO::ADAGRAD:
				obj.model().reset(new Opt::ADAGRAD());
				pos+=unpack(static_cast<Opt::ADAGRAD&>(*obj.model()),arr+pos);
			break;
			case Opt::ALGO::ADADELTA:
				obj.model().reset(new Opt::ADADELTA());
				pos+=unpack(static_cast<Opt::ADADELTA&>(*obj.model()),arr+pos);
			break;
			case Opt::ALGO::RMSPROP:
				obj.model().reset(new Opt::RMSPROP());
				pos+=unpack(static_cast<Opt::RMSPROP&>(*obj.model()),arr+pos);
			break;
			case Opt::ALGO::ADAM:
				obj.model().reset(new Opt::ADAM());
				pos+=unpack(static_cast<Opt::ADAM&>(*obj.model()),arr+pos);
			break;
			case Opt::ALGO::NADAM:
				obj.model().reset(new Opt::NADAM());
				pos+=unpack(static_cast<Opt::NADAM&>(*obj.model()),arr+pos);
			break;
			case Opt::ALGO::BFGS:
				obj.model().reset(new Opt::BFGS());
				pos+=unpack(static_cast<Opt::BFGS&>(*obj.model()),arr+pos);
			break;
			case Opt::ALGO::RPROP:
				obj.model().reset(new Opt::RPROP());
				pos+=unpack(static_cast<Opt::RPROP&>(*obj.model()),arr+pos);
			break;
			default:
				throw std::runtime_error("unpack(NNPotOpt&,const char*): Invalid optimization method.");
			break;
		}
		std::memcpy(&obj.restart(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		pos+=unpack(obj.file_error(),arr+pos);
		pos+=unpack(obj.file_restart(),arr+pos);
		return pos;
	}
	
}