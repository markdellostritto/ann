// c libraries
#include <cstdlib>
// c++ libraries
#include <iostream>
#include <exception>
#include <chrono>
#include <limits>
// local libraries - structure
#include "cell.hpp"
#include "structure.hpp"
#include "vasp.hpp"
// local libraries - nn
#include "nn.hpp"
#include "nn_pot.hpp"
#include "string.hpp"
#include "statistics.hpp"

#ifndef NN_POT_TRAIN_DEBUG
#define NN_POT_TRAIN_DEBUG 1
#endif

class NNPotOpt{
public:
	//simulation data
		std::vector<Structure<AtomT> >* strucTrain_;//training configurations (one timestep only)
		std::vector<Structure<AtomT> >* strucVal_;//validation configurations (one timestep only)
		std::vector<Structure<AtomT> >* strucTest_;//training configurations (one timestep only)
	//elements
		std::vector<unsigned int> nAtoms_;//number of atoms of each element
		VecList gElement_;//gradients of the parameters - subset for each element (nElements)
		VecList pElement_;//parameters to be optimized - subset for each element (nElements)
		std::vector<VecList> gElementT_;//gradients - distributed among each thread
	//batch
		std::vector<unsigned int> batch_;//small batch of inputs (simulations)
		std::vector<unsigned int> indices_;//indices of all the inputs (simulations)
	//nn
		unsigned int nParams_;//total number of parameters
		std::vector<NNPot> nnpot_;//neural network potentials (nThreads)
		VecList preBias_,preScale_;//one for each specie (nn)
		bool preCond_;
		bool postCond_;
	//input/output
		unsigned int nPrint_;//print status every n steps
		unsigned int nWrite_;//write potential every n steps
	//optimization
		unsigned int opt_count_;//optimization count
		unsigned int memory_;//memory for monitoring changes in the error
		double gl_,progress_,pq_;//generalization loss, progress, and progress quotient
		double tol_val_;//tolerance of the validation set
		unsigned int nBatch_;
		double pBatch_;
	//error
		double error_val_min_;//minimum error for the validation set
		double error_train_min_;//minum error for the training set
		std::vector<double> error_train_vec_;//vector storing the error for the training set
		std::vector<double> error_val_vec_;//vector storing the error for the validation set
		std::vector<double> error_train_thread_;//vector storing the error for the training set
		std::vector<double> error_val_thread_;//vector storing the error for the validation set
	//parallel
		unsigned int nThreads_;
		std::vector<unsigned int> tdTrain_,tdVal_,tdTest_;//thread distributions
		std::vector<unsigned int> toTrain_,toVal_,toTest_;//thread offsets
	//constructors/destructors
		NNPotOpt():nnpot_(NULL),strucTrain_(NULL),strucVal_(NULL),strucTest_(NULL){defaults();};
		~NNPotOpt(){};
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPotOpt& nnPotOpt);
	//member functions
		void defaults();
		void clear();
		void train(NNPot& nnpot, Opt& opt, unsigned int batchSize);
		double error(const Eigen::VectorXd& x, Eigen::VectorXd& grad);
};
