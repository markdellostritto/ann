#include "nn_train.hpp"

namespace NN{
	
//***********************************************************************
// NN Optimization
//***********************************************************************

void NNOpt::train(Network& nn, VecList& inputs, VecList& outputs, Opt& opt){
	if(NN_PRINT_FUNC) std::cout<<"NNOpt::train(Network&,VecList&,VecList&,Opt&):\n";
	Eigen::VectorXd p;//parameters to be optimized
	
	/* initialize the randomizer */
	std::srand(std::time(NULL));
	
	/* set the network*/
	this->nn=&nn;
	/* set the inputs */
	inputsT=inputs;
	outputsT=outputs;
	
	/* check the variables */
	if(NN_PRINT_STATUS>0) std::cout<<"Checking the variables...\n";
	if(inputs.size()!=outputs.size()) throw std::invalid_argument("Invalid input/output: must be same size.");
	for(unsigned int n=0; n<inputs.size(); ++n){
		if(inputs[n].size()!=nn.nInput()) throw std::invalid_argument("Invalid inputs: incompatible with network size.");
		if(outputs[n].size()!=nn.nOutput()) throw std::invalid_argument("Invalid outputs: incompatible with network size.");
	}
	
	/* precondition the input */
	if(nn.preCond()){
		if(NN_PRINT_STATUS>0) std::cout<<"Pre-conditioning input...\n";
		Eigen::VectorXd avg=Eigen::VectorXd::Zero(nn.nInput());
		Eigen::VectorXd stddev=Eigen::VectorXd::Zero(nn.nInput());
		double N=0;
		for(unsigned int n=0; n<inputs.size(); ++n){
			avg.noalias()+=inputs[n]; ++N;
		}
		avg/=N;
		for(unsigned int n=0; n<inputs.size(); ++n){
			stddev.noalias()+=(inputs[n]-avg).cwiseProduct(inputs[n]-avg);
		}
		for(unsigned int i=0; i<nn.nInput(); ++i) stddev[i]=std::sqrt(stddev[i]/(N-1.0));
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preScale(i)=-1.0*avg[i];
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preScale(i)=1.0/stddev[i];
		if(NN_PRINT_STATUS>0){
			std::cout<<"scaling-input:\n";
			std::cout<<"avg = "<<avg.transpose()<<"\n";
			std::cout<<"stddev = "<<stddev.transpose()<<"\n";
			std::cout<<"bias-in = "<<nn.preBias().transpose()<<"\n";
			std::cout<<"scale-in = "<<nn.preScale().transpose()<<"\n";
		}
	} else {
		if(NN_PRINT_STATUS>0) std::cout<<"No pre-conditioning of input.\n";
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preBias(i)=0.0;
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preScale(i)=1.0;
	}
	/* precondition the output */
	if(nn.postCond()){
		if(NN_PRINT_STATUS>0) std::cout<<"Pre-conditioning output...\n";
		Eigen::VectorXd avg=Eigen::VectorXd::Zero(nn.nOutput());
		Eigen::VectorXd stddev=Eigen::VectorXd::Zero(nn.nOutput());
		double N=0;
		for(unsigned int n=0; n<outputs.size(); ++n){
			avg.noalias()+=outputs[n]; ++N;
		}
		avg/=N;
		for(unsigned int n=0; n<outputs.size(); ++n){
			stddev.noalias()+=(outputs[n]-avg).cwiseProduct(outputs[n]-avg);
		}
		for(unsigned int i=0; i<nn.nOutput(); ++i) stddev[i]=std::sqrt(stddev[i]/(N-1.0));
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postBias(i)=avg[i];
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postScale(i)=stddev[i];
		if(NN_PRINT_STATUS>0){
			std::cout<<"scaling-output:\n";
			std::cout<<"avg = "<<avg.transpose()<<"\n";
			std::cout<<"stddev = "<<stddev.transpose()<<"\n";
			std::cout<<"bias-out = "<<nn.postBias().transpose()<<"\n";
			std::cout<<"scale-out = "<<nn.postScale().transpose()<<"\n";
		}
	} else {
		if(NN_PRINT_STATUS>0) std::cout<<"No pre-conditioning of output.\n";
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postBias(i)=0.0;
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postScale(i)=1.0;
	}
	
	/* set the initial values */
	if(NN_PRINT_STATUS>0) std::cout<<"Setting the initial values...\n";
	//set values
	nn>>p;
	//resize gradient
	if(NN_PRINT_STATUS>0) std::cout<<"Resizing the gradient...\n";
	grads.resize(inputs.size());
	for(unsigned int i=0; i<grads.size(); ++i){
		grads[i]=Eigen::VectorXd::Zero(nn.size());
	}
	
	/* set the objective function */
	if(NN_PRINT_STATUS>0) std::cout<<"Setting the objective function...\n";
	std::function<double(NNOpt&, const Eigen::VectorXd&, Eigen::VectorXd&)> func = &NNOpt::error;
	
	/* execute the optimization */
	if(NN_PRINT_STATUS>0) std::cout<<"Executing the optimization...\n";
	opt.opt<NNOpt>(func,*this,p);
	
	if(NN_PRINT_STATUS>-1){
		std::cout<<"n-steps = "<<opt.nStep()<<"\n";
		std::cout<<"opt-val = "<<opt.val()<<"\n";
		if(NN_PRINT_STATUS>0){std::cout<<"p = "; for(int i=0; i<p.size(); ++i) std::cout<<p[i]<<" "; std::cout<<"\n";}
	}
}

void NNOpt::train(Network& nn, VecList& inputs, VecList& outputs, Opt& opt, unsigned int batchSize){
	if(NN_PRINT_FUNC) std::cout<<"NNOpt::train(Network&,VecList&,VecList&,Opt&,unsigned int):\n";
	Eigen::VectorXd p;//parameters to be optimized
	
	/* initialize the randomizer */
	std::srand(std::time(NULL));
	
	/* check the batch size */
	if(batchSize==0) throw std::invalid_argument("Invalid batch size.");
	if(batchSize>inputs.size()) throw std::invalid_argument("Invalid batch size.");
	batch.resize(batchSize,0);
	
	/* set the network*/
	this->nn=&nn;
	/* set the inputs */
	inputsT=inputs;
	outputsT=outputs;
	/* set the indices */
	indices.resize(inputs.size());
	for(unsigned int i=0; i<indices.size(); ++i) indices[i]=i;
	
	/* check the variables */
	if(NN_PRINT_STATUS>0) std::cout<<"Checking the variables...\n";
	if(inputs.size()!=outputs.size()) throw std::invalid_argument("Invalid input/output: must be same size.");
	for(unsigned int n=0; n<inputs.size(); ++n){
		if(inputs[n].size()!=nn.nInput()) throw std::invalid_argument("Invalid inputs: incompatible with network size.");
		if(outputs[n].size()!=nn.nOutput()) throw std::invalid_argument("Invalid outputs: incompatible with network size.");
	}
	
	/* precondition the input */
	if(nn.preCond()){
		if(NN_PRINT_STATUS>0) std::cout<<"Pre-conditioning input...\n";
		Eigen::VectorXd avg=Eigen::VectorXd::Zero(nn.nInput());
		Eigen::VectorXd stddev=Eigen::VectorXd::Zero(nn.nInput());
		double N=0;
		for(unsigned int n=0; n<inputs.size(); ++n){
			avg.noalias()+=inputs[n]; ++N;
		}
		avg/=N;
		for(unsigned int n=0; n<inputs.size(); ++n){
			stddev.noalias()+=(inputs[n]-avg).cwiseProduct(inputs[n]-avg);
		}
		for(unsigned int i=0; i<nn.nInput(); ++i) stddev[i]=std::sqrt(stddev[i]/(N-1.0));
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preBias(i)=-1*avg[i];
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preScale(i)=1.0/stddev[i];
		if(NN_PRINT_STATUS>0){
			std::cout<<"scaling-input:\n";
			std::cout<<"avg = "<<avg.transpose()<<"\n";
			std::cout<<"stddev = "<<stddev.transpose()<<"\n";
			std::cout<<"bias-in = "<<nn.preBias().transpose()<<"\n";
			std::cout<<"scale-in = "<<nn.preScale().transpose()<<"\n";
		}
	} else {
		if(NN_PRINT_STATUS>0) std::cout<<"No pre-conditioning of input.\n";
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preBias(i)=0.0;
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.preScale(i)=1.0;
	}
	/* precondition the output */
	if(nn.postCond()){
		if(NN_PRINT_STATUS>0) std::cout<<"Pre-conditioning output...\n";
		Eigen::VectorXd avg=Eigen::VectorXd::Zero(nn.nOutput());
		Eigen::VectorXd stddev=Eigen::VectorXd::Zero(nn.nOutput());
		double N=0;
		for(unsigned int n=0; n<outputs.size(); ++n){
			avg.noalias()+=outputs[n]; ++N;
		}
		avg/=N;
		for(unsigned int n=0; n<outputs.size(); ++n){
			stddev.noalias()+=(outputs[n]-avg).cwiseProduct(outputs[n]-avg);
		}
		for(unsigned int i=0; i<nn.nOutput(); ++i) stddev[i]=std::sqrt(stddev[i]/(N-1.0));
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postBias(i)=avg[i];
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postScale(i)=stddev[i];
		if(NN_PRINT_STATUS>0){
			std::cout<<"scaling-output:\n";
			std::cout<<"avg = "<<avg.transpose()<<"\n";
			std::cout<<"stddev = "<<stddev.transpose()<<"\n";
			std::cout<<"bias-out = "<<nn.postBias().transpose()<<"\n";
			std::cout<<"scale-out = "<<nn.postScale().transpose()<<"\n";
		}
	} else {
		if(NN_PRINT_STATUS>0) std::cout<<"No pre-conditioning of output.\n";
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postBias(i)=0.0;
		for(unsigned int i=0; i<nn.nOutput(); ++i) nn.postScale(i)=1.0;
	}
	
	/* set the initial values */
	if(NN_PRINT_STATUS>0) std::cout<<"Setting the initial values...\n";
	//set values
	nn>>p;
	//resize gradient
	if(NN_PRINT_STATUS>0) std::cout<<"Resizing the gradient...\n";
	grads.resize(inputs.size());
	for(unsigned int i=0; i<grads.size(); ++i){
		grads[i]=Eigen::VectorXd::Zero(nn.size());
	}
	
	/* set the objective function */
	if(NN_PRINT_STATUS>0) std::cout<<"Setting the objective function...\n";
	std::function<double(NNOpt&, const Eigen::VectorXd&, Eigen::VectorXd&)> func = &NNOpt::error_batch;
	
	/* execute the optimization */
	if(NN_PRINT_STATUS>0) std::cout<<"Executing the optimization...\n";
	opt.opt<NNOpt>(func,*this,p);
	
	if(NN_PRINT_STATUS>-1){
		std::cout<<"n-steps = "<<opt.nStep()<<"\n";
		std::cout<<"opt-val = "<<opt.val()<<"\n";
		if(NN_PRINT_STATUS>0){std::cout<<"p = "; for(int i=0; i<p.size(); ++i) std::cout<<p[i]<<" "; std::cout<<"\n";}
	}
}

double NNOpt::error(const Eigen::VectorXd& x, Eigen::VectorXd& grad){
	if(NN_PRINT_FUNC) std::cout<<"NNOpt::error(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	//local function variables
	double err=0;
	//reset the grad
	grad.setZero();
	//load the weights and biases into the network
	*nn<<x;
	Eigen::VectorXd dcda=Eigen::VectorXd::Zero(nn->output().size());
	Eigen::VectorXd gradLocal=Eigen::VectorXd::Zero(nn->size());
	//loop over all inputs
	for(unsigned int i=0; i<inputsT.size(); ++i){
		//set the inputs
		nn->input_=inputsT[i];
		//execute the network
		nn->execute();
		//compute the error
		err+=nn->error(outputsT[i],grads[i]);
		//add the gradient
		grad.noalias()+=grads[i];
	}
	//normalize the error and gradient
	err/=inputsT.size();
	grad/=inputsT.size();
	//return the error
	return err;
}

double NNOpt::error_batch(const Eigen::VectorXd& x, Eigen::VectorXd& grad){
	if(NN_PRINT_FUNC) std::cout<<"NNOpt::error_batch(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	//local function variables
	double err=0;
	//reset the grad
	grad.setZero();
	//load the weights and biases into the network
	*nn<<x;
	//randomize the batch
	std::random_shuffle(indices.begin(),indices.end());
	for(unsigned int i=0; i<batch.size(); ++i) batch[i]=indices[i];
	//loop over all inputs in the batch
	for(unsigned int i=0; i<batch.size(); ++i){
		//set the inputs
		nn->input_=inputsT[batch[i]];
		//execute the network
		nn->execute();
		//compute the error
		err+=nn->error(outputsT[batch[i]],grads[batch[i]]);
		//add the gradient
		grad.noalias()+=grads[batch[i]];
	}
	//normalize the error and gradient
	err/=batch.size();
	grad/=batch.size();
	//return the error
	return err;
}

double NNOpt::error_batch_val(const Eigen::VectorXd& x, Eigen::VectorXd& grad){
	if(NN_PRINT_FUNC) std::cout<<"NNOpt::error_batch_val(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	//local function variables
	double err=0;
	//reset the grad
	grad.setZero();
	//load the weights and biases into the network
	*nn<<x;
	//randomize the batch
	std::random_shuffle(indices.begin(),indices.end());
	for(unsigned int i=0; i<batch.size(); ++i) batch[i]=indices[i];
	//loop over all inputs in the batch
	for(unsigned int i=0; i<batch.size(); ++i){
		//set the inputs
		nn->input_=inputsT[batch[i]];
		//execute the network
		nn->execute();
		//compute the error
		err+=nn->error(outputsT[batch[i]],grads[batch[i]]);
		//add the gradient
		grad.noalias()+=grads[batch[i]];
	}
	//normalize the error and gradient
	err/=batch.size();
	grad/=batch.size();
	//calculate the error from the validation data
	err=0;
	for(unsigned int i=0; i<inputsV.size(); ++i){
		//set the inputs
		nn->input_=inputsV[i];
		//execute the network
		nn->execute();
		//compute the error
		err+=nn->error(outputsV[i]);
	}
	//normalize the error
	err/=inputsV.size();
	//return the error
	return err;
}

}