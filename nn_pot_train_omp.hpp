// c libraries
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cfloat>
// c++ libraries
#include <iostream>
#include <exception>
#include <memory>
// local libraries - stats
#include "accumulator.hpp"
// local libraries - structure
#include "cell.hpp"
#include "structure.hpp"
#include "vasp.hpp"
#include "qe.hpp"
#include "ame.hpp"
// local libraries - nn
#include "nn.hpp"
#include "nn_pot.hpp"
#include "string.hpp"
// local libraries - units
#include "units.hpp"

#ifndef NN_POT_TRAIN_DEBUG
#define NN_POT_TRAIN_DEBUG 1
#endif

class NNPotOpt{
public:
	//simulation data
		std::vector<Structure>* strucTrain_;//training configurations (one timestep only)
		std::vector<Structure>* strucVal_;//validation configurations (one timestep only)
		std::vector<Structure>* strucTest_;//training configurations (one timestep only)
	//elements
		unsigned int nElements_;
		std::vector<unsigned int> nAtoms_;//number of atoms of each element
		VecList pElement_;//parameters to be optimized - subset for each element (nElements)
		VecList gElement_;//gradients of the parameters - subset for each element (nElements)
		std::vector<VecList> gElementT_;//gradients - distributed among each thread
		std::vector<VecList> gTemp_;//gradients - distributed among each thread
	//batch
		unsigned int nBatch_;//number of structures in batch
		double pBatch_;//percentage of structures in batch (overrides nBatch_)
		std::vector<unsigned int> batch_;//small batch of inputs (simulations)
		std::vector<unsigned int> indices_;//indices of all the inputs (simulations)
	//nn
		unsigned int nParams_;//total number of parameters
		NNPot nnpot_;
		std::vector<NNPot> nnpotv_;//neural network potential vector (nThreads)
		VecList preBias_,preScale_;//one for each specie (nn)
		bool preCond_;
	//input/output
		std::string restart_file_;//restart file
		bool restart_;//restart
		unsigned int nPrint_;//print status every n steps
		unsigned int nWrite_;//write potential every n steps
	//optimization
		OPT_METHOD::type algo_;
		std::shared_ptr<Opt> opt_;//optimization object
		Eigen::VectorXd p;//parameters to be optimized
		Eigen::VectorXd g;//gradient of parameters to be optimized
	//error
		double error_val_min_;//minimum error for the validation set
		double error_train_min_;//minum error for the training set
		std::vector<double> error_train_thread_;//vector storing the error for the training set
		std::vector<double> error_val_thread_;//vector storing the error for the validation set
	//parallel
		unsigned int nThreads_;
	//memory
		bool memsave;//whether to save on memory
	//file i/o
		FILE* writer_error;
		std::string file_error;
	//constructors/destructors
		NNPotOpt():strucTrain_(NULL),strucVal_(NULL),strucTest_(NULL){defaults();};
		~NNPotOpt(){};
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPotOpt& nnPotOpt);
	//member functions
		void defaults();
		void clear();
		void train(unsigned int batchSize);
		double error(const Eigen::VectorXd& x, Eigen::VectorXd& grad);
		void write_restart(const char* file);
		void read_restart(const char* file);
};

struct Stats{
	double min,max;
	double avg,avgp;
	double stddev;
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************

	template <> unsigned int nbytes(const NNPotOpt& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const NNPotOpt& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(NNPotOpt& obj, const char* arr);
	
}
