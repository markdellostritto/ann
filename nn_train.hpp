#ifndef NN_TRAIN_HPP
#define NN_TRAIN_HPP

//c++ libraries
#include <iostream>
//c libraries
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <memory>
//Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>
//nn
#include "nn.hpp"
//string
#include "string.hpp"
//opt
#include "optimize.hpp"

#ifndef NN_TRAIN_PRINT_FUNC
#define NN_TRAIN_PRINT_FUNC 0
#endif

#ifndef NN_TRAIN_PRINT_STATUS
#define NN_TRAIN_PRINT_STATUS 0
#endif

#ifndef NN_TRAIN_PRINT_DATA
#define NN_TRAIN_PRINT_DATA 0
#endif

namespace NN{
	
//***********************************************************************
// NN Optimization
//***********************************************************************

class NNOpt{
private:
	//batch
	std::vector<unsigned int> batch_;//random batch
	std::vector<unsigned int> indices_;//indices of the batch
	//conditioning
	bool preCond_;  //whether to pre-condition  the inputs
	bool postCond_; //whether to post-condition the inputs
	//data
	VecList* inputsT_;  //data - inputs - training
	VecList* outputsT_; //data - inputs - training
	VecList* inputsV_;  //data - inputs - validation
	VecList* outputsV_; //data - inputs - validation
	//optimization
	Opt::Data data_;
	std::shared_ptr<Opt::Model> model_;
	double err_train_; //error - training
	double err_val_;   //error - validation
	double err_lambda_;//error - regularization
	//neural networks
	Network* nn_;//neural network
	//input/output
	bool restart_;//whether restarting
	unsigned int nPrint_;//print status every n steps
	unsigned int nWrite_;//write potential every n steps
	//file i/o
	FILE* writer_error_;//file pointer which writes the error
	std::string file_error_;//file storing the error
	std::string file_restart_;//restart file
public:
	//constructors/destructors
	NNOpt(){defaults();}
	~NNOpt(){}
	
	//access
	//conditioning
	bool& preCond(){return preCond_;}
	const bool& preCond()const{return preCond_;}
	bool& postCond(){return postCond_;}
	const bool& postCond()const{return postCond_;}
	//optimization
	Opt::Data& data(){return data_;}
	const Opt::Data& data()const{return data_;}
	std::shared_ptr<Opt::Model>& model(){return model_;}
	const std::shared_ptr<Opt::Model>& model()const{return model_;}
	double err_train()const{return err_train_;}
	double err_val()const{return err_val_;}
	double err_lambda()const{return err_lambda_;}
	//input/output
	bool& restart(){return restart_;}
	const bool& restart()const{return restart_;}
	unsigned int& nPrint(){return nPrint_;}
	const unsigned int& nPrint()const{return nPrint_;}
	unsigned int& nWrite(){return nWrite_;}
	const unsigned int& nWrite()const{return nWrite_;}
	//file i/o
	std::string& file_error(){return file_error_;}
	const std::string& file_error()const{return file_error_;}
	std::string& file_restart(){return file_restart_;}
	const std::string& file_restart()const{return file_restart_;}
	//data
	VecList*& inputsT(){return inputsT_;}
	VecList*& outputsT(){return outputsT_;}
	VecList*& inputsV(){return inputsV_;}
	VecList*& outputsV(){return outputsV_;}
	Network*& nn(){return nn_;}
	
	//training
	void train(Network& nn, int batchSize=-1);
	//error
	double error(const Eigen::VectorXd& x, Eigen::VectorXd& grad);
	//member functions
	void defaults();
	void clear();
};

}

#endif