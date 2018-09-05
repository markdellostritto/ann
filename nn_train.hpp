#ifndef NN_TRAIN_HPP
#define NN_TRAIN_HPP

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "nn.hpp"
#include "string.hpp"
#include "optimize.hpp"

namespace NN{
	
//***********************************************************************
// NN Optimization
//***********************************************************************

class NNOpt{
public:
	//data
	std::vector<unsigned int> batch;//random batch
	std::vector<unsigned int> indices;//indices of the batch
	VecList inputsT,outputsT;//training
	VecList inputsV,outputsV;//validation
	VecList grads;//gradients
	Network* nn;//neural network
	//constructors/destructors
	NNOpt(){};
	~NNOpt(){};
	void train(Network& nn, VecList& inputs, VecList& outputs, Opt& opt);
	void train(Network& nn, VecList& inputs, VecList& outputs, Opt& opt, unsigned int batchSize);
	double error(const Eigen::VectorXd& x, Eigen::VectorXd& grad);
	double error_batch(const Eigen::VectorXd& x, Eigen::VectorXd& grad);
	double error_batch_val(const Eigen::VectorXd& x, Eigen::VectorXd& grad);
};

}

#endif