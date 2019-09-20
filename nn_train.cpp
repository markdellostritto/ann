#include "nn_train.hpp"

namespace NN{

void NNOpt::defaults(){
	//conditioning
	preCond_=false;
	postCond_=false;
	//data
	inputsT_=NULL;
	outputsT_=NULL;
	inputsV_=NULL;
	outputsV_=NULL;
	//optimization
	data_.clear();
	err_train_=0;
	err_val_=0;
	err_lambda_=0;
	//neural network
	nn_=NULL;
	//input/output
	restart_=false;
	nPrint_=0;
	nWrite_=0;
	file_error_="nn_train_error.dat";
	//file i/o
	writer_error_=NULL;
}

void NNOpt::clear(){
	//batch
	batch_.clear();
	indices_.clear();
	//conditioning
	preCond_=true;
	postCond_=true;
	//data
	inputsT_=NULL;
	outputsT_=NULL;
	inputsV_=NULL;
	outputsV_=NULL;
	//optimization
	data_.clear();
	err_train_=0;
	err_val_=0;
	err_lambda_=0;
	//neural network
	nn_=NULL;
	//input/output
	restart_=false;
	nPrint_=0;
	nWrite_=0;
	//file i/o
	if(writer_error_!=NULL){
		fclose(writer_error_);
		writer_error_=NULL;
	}
}
	
//***********************************************************************
// NN Optimization
//***********************************************************************

void NNOpt::train(Network& nn, int nbatch){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::train(Network&,Opt&,int):\n";
	Eigen::VectorXd gTemp;
	clock_t start,stop;
	double time=0;
	
	//==== initialize the randomizer ====
	std::srand(std::time(NULL));
	
	//==== set the network ====
	nn_=&nn;
	
	//==== check the data ====
	if(inputsT_==NULL) throw std::invalid_argument("ERROR: no training inputs provided.");
	if(outputsT_==NULL) throw std::invalid_argument("ERROR: no training outputs provided.");
	if(inputsV_==NULL) throw std::invalid_argument("ERROR: no training inputs provided.");
	if(outputsV_==NULL) throw std::invalid_argument("ERROR: no training outputs provided.");
	if(inputsT_->size()!=outputsT_->size()) throw std::invalid_argument("Invalid input/output - training: must be same size.");
	if(inputsV_->size()!=outputsV_->size()) throw std::invalid_argument("Invalid input/output - validation: must be same size.");
	for(unsigned int n=0; n<inputsT_->size(); ++n){
		if((*inputsT_)[n].size()!=nn_->nInput()) throw std::invalid_argument("Invalid inputs - training: incompatible with network size.");
		if((*outputsT_)[n].size()!=nn_->nOutput()) throw std::invalid_argument("Invalid outputs - training: incompatible with network size.");
	}
	for(unsigned int n=0; n<inputsV_->size(); ++n){
		if((*inputsV_)[n].size()!=nn_->nInput()) throw std::invalid_argument("Invalid inputs - training: incompatible with network size.");
		if((*outputsV_)[n].size()!=nn_->nOutput()) throw std::invalid_argument("Invalid outputs - training: incompatible with network size.");
	}
	
	//==== check the batch and indices ====
	if(nbatch<0) nbatch=inputsT_->size();
	else if(nbatch==0) throw std::invalid_argument("Invalid batch size.");
	if(nbatch>inputsT_->size()) throw std::invalid_argument("Invalid batch size.");
	batch_.resize(nbatch);
	for(int i=nbatch-1; i>=0; --i) batch_[i]=i;
	indices_.resize(inputsT_->size());
	for(int i=indices_.size()-1; i>=0; --i) indices_[i]=i;
	
	//==== precondition the input ====
	if(preCond_){
		if(NN_TRAIN_PRINT_STATUS>0) std::cout<<"pre-conditioning input\n";
		Eigen::VectorXd avg=Eigen::VectorXd::Zero(nn_->nInput());
		Eigen::VectorXd stddev=Eigen::VectorXd::Zero(nn_->nInput());
		double N=0;
		for(unsigned int n=0; n<inputsT_->size(); ++n){
			avg.noalias()+=(*inputsT_)[n]; ++N;
		}
		avg/=N;
		for(unsigned int n=0; n<inputsT_->size(); ++n){
			stddev.noalias()+=((*inputsT_)[n]-avg).cwiseProduct((*inputsT_)[n]-avg);
		}
		for(unsigned int i=0; i<nn_->nInput(); ++i) stddev[i]=std::sqrt(stddev[i]/(N-1.0));
		for(unsigned int i=0; i<nn_->nInput(); ++i) nn_->preScale(i)=-1.0*avg[i];
		for(unsigned int i=0; i<nn_->nInput(); ++i) nn_->preScale(i)=1.0/stddev[i];
		if(NN_TRAIN_PRINT_STATUS>0){
			std::cout<<"scaling-input:\n";
			std::cout<<"avg      = "<<avg.transpose()<<"\n";
			std::cout<<"stddev   = "<<stddev.transpose()<<"\n";
			std::cout<<"bias-in  = "<<nn_->preBias().transpose()<<"\n";
			std::cout<<"scale-in = "<<nn_->preScale().transpose()<<"\n";
		}
	} else {
		if(NN_TRAIN_PRINT_STATUS>0) std::cout<<"no pre-conditioning of input\n";
		for(unsigned int i=0; i<nn_->nInput(); ++i) nn_->preBias(i)=0.0;
		for(unsigned int i=0; i<nn_->nInput(); ++i) nn_->preScale(i)=1.0;
	}
	//==== precondition the output ====
	if(postCond_){
		if(NN_TRAIN_PRINT_STATUS>0) std::cout<<"pre-conditioning output\n";
		Eigen::VectorXd avg=Eigen::VectorXd::Zero(nn_->nOutput());
		Eigen::VectorXd stddev=Eigen::VectorXd::Zero(nn_->nOutput());
		double N=0;
		for(unsigned int n=0; n<outputsT_->size(); ++n){
			avg.noalias()+=(*outputsT_)[n]; ++N;
		}
		avg/=N;
		for(unsigned int n=0; n<outputsT_->size(); ++n){
			stddev.noalias()+=((*outputsT_)[n]-avg).cwiseProduct((*outputsT_)[n]-avg);
		}
		for(unsigned int i=0; i<nn_->nOutput(); ++i) stddev[i]=std::sqrt(stddev[i]/(N-1.0));
		for(unsigned int i=0; i<nn_->nOutput(); ++i) nn_->postBias(i)=avg[i];
		for(unsigned int i=0; i<nn_->nOutput(); ++i) nn_->postScale(i)=stddev[i];
		if(NN_TRAIN_PRINT_STATUS>0){
			std::cout<<"scaling-output:\n";
			std::cout<<"avg       = "<<avg.transpose()<<"\n";
			std::cout<<"stddev    = "<<stddev.transpose()<<"\n";
			std::cout<<"bias-out  = "<<nn_->postBias().transpose()<<"\n";
			std::cout<<"scale-out = "<<nn_->postScale().transpose()<<"\n";
		}
	} else {
		if(NN_TRAIN_PRINT_STATUS>0) std::cout<<"no pre-conditioning of output\n";
		for(unsigned int i=0; i<nn_->nOutput(); ++i) nn_->postBias(i)=0.0;
		for(unsigned int i=0; i<nn_->nOutput(); ++i) nn_->postScale(i)=1.0;
	}
	
	//==== initalize the optimizer ====
	if(NN_TRAIN_PRINT_STATUS>0) std::cout<<"initializing the optimizer\n";
	data_.init(nn_->size());
	model_->init(nn_->size());
	
	//==== set the initial values ====
	if(NN_TRAIN_PRINT_STATUS>0) std::cout<<"setting the initial values\n";
	//set values
	*nn_>>data_.p(); *nn_>>data_.pOld();
	*nn_>>data_.g(); *nn_>>data_.gOld();
	*nn_>>gTemp;
	data_.g().setZero();
	data_.gOld().setZero();
	
	//==== open the error file ====
	if(NN_TRAIN_PRINT_STATUS>0) std::cout<<"opening the ouput file\n";
	std::cout<<"file_error_ = "<<file_error_<<"\n";
	if(!restart_){
		writer_error_=fopen(file_error_.c_str(),"w");
		fprintf(writer_error_,"#STEP ERROR_RMS_TRAIN ERROR_RMS_VAL\n");
	} else writer_error_=fopen(file_error_.c_str(),"a");
	if(writer_error_==NULL) throw std::runtime_error("I/O Error: Could not open error file.");
	
	//==== execute the optimization ====
	if(NN_TRAIN_PRINT_STATUS>0) std::cout<<"executing the optimization\n";
	start=std::clock();
	for(unsigned int n=0; n<data_.max(); ++n){
		//set the parameters of the network
		*nn_<<data_.p();
		//compute the error and gradient
		error(data_.p(),data_.g());
		//compute regularization error and gradient
		err_lambda_+=nn_->error_lambda()/inputsT_->size();
		data_.g().noalias()+=nn_->grad_lambda(gTemp)/inputsT_->size();
		//print error
		if(data_.step()%data_.nPrint()==0) printf("opt %8i err_t %12.10f err_v %12.10f\n",
			data_.step(),std::sqrt(2.0*err_train_),std::sqrt(2.0*err_val_));
		//write error
		if(data_.step()%data_.nWrite()==0) fprintf(writer_error_,"%6i %12.10f %12.10f\n",
			data_.step(),std::sqrt(2.0*err_train_),std::sqrt(2.0*err_val_));
		//compute the new position
		data_.val()=err_train_+err_lambda_;
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
			case Opt::VAL::FTOL_REL: if(data_.dv()<data_.tol()) break;
			case Opt::VAL::XTOL_REL: if(data_.dp()<data_.tol()) break;
			case Opt::VAL::FTOL_ABS: if(data_.val()<data_.tol()) break;
		}
		//increment step
		++data_.step();
	}
	stop=std::clock();
	time=((double)(stop-start))/CLOCKS_PER_SEC;
	
	//==== close the error file ====
	fclose(writer_error_);
	writer_error_=NULL;
	
	if(NN_TRAIN_PRINT_STATUS>-1){
		std::cout<<"**************************************************\n";
		std::cout<<"******************* OPT - SUMM *******************\n";
		std::cout<<"n-steps = "<<data_.step()<<"\n";
		std::cout<<"opt-val = "<<data_.val()<<"\n";
		std::cout<<"time    = "<<time<<"\n";
		std::cout<<"******************* OPT - SUMM *******************\n";
		std::cout<<"**************************************************\n";
	}
}

double NNOpt::error(const Eigen::VectorXd& x, Eigen::VectorXd& g){
	if(NN_TRAIN_PRINT_FUNC>0) std::cout<<"NNOpt::error(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	//reset the gradient
	g.setZero();
	Eigen::VectorXd gTemp=g;
	//randomize the batch
	if(batch_.size()<inputsT_->size()){
		std::random_shuffle(indices_.begin(),indices_.end());
		for(unsigned int i=0; i<batch_.size(); ++i) batch_[i]=indices_[i];
	}
	//compute the training error
	err_train_=0;
	for(unsigned int i=0; i<batch_.size(); ++i){
		//set the inputs
		for(unsigned int j=0; j<nn_->nInput(); ++j) nn_->input(j)=(*inputsT_)[batch_[i]][j];
		//execute the network
		nn_->execute();
		//compute the error
		err_train_+=nn_->error((*outputsT_)[batch_[i]],gTemp);
		//add the gradient
		g.noalias()+=gTemp;
	}
	err_train_/=batch_.size();
	//compute the validation error
	err_val_=0;
	for(unsigned int i=0; i<inputsV_->size(); ++i){
		//set the inputs
		for(unsigned int j=0; j<nn_->nInput(); ++j) nn_->input(j)=(*inputsV_)[i][j];
		//execute the network
		nn_->execute();
		//compute the error
		err_val_+=nn_->error((*outputsV_)[batch_[i]]);
	}
	err_val_/=inputsV_->size();
	//normalize the gradient
	g/=batch_.size();
	//return the error
	return err_train_;
}

}