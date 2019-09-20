// mpi
#include <mpi.h>
// c libraries
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cfloat>
// c++ libraries
#include <iostream>
#include <exception>
#include <memory>
#include <algorithm>
// statistics
#include "accumulator.hpp"
// structure
#include "cell.hpp"
#include "structure.hpp"
// file i/o
#include "vasp.hpp"
#include "qe.hpp"
#include "ame.hpp"
#include "string.hpp"
// nn
#include "nn.hpp"
#include "nn_pot.hpp"
// optimization
#include "optimize.hpp"
// units
#include "units.hpp"
// charge
#include "ewald3D.hpp"
// mpi - utility
#include "mpi_util.hpp"
#include "parallel.hpp"
// compiler
#include "compiler.hpp"

#define EIGEN_NO_DEBUG

#ifndef NN_POT_TRAIN_PRINT_FUNC
#define NN_POT_TRAIN_PRINT_FUNC 0
#endif

#ifndef NN_POT_TRAIN_PRINT_STATUS
#define NN_POT_TRAIN_PRINT_STATUS 0
#endif

#ifndef NN_POT_TRAIN_PRINT_DATA
#define NN_POT_TRAIN_PRINT_DATA 0
#endif

bool compare_pair(const std::pair<unsigned int,double>& p1, const std::pair<unsigned int,double>& p2);

class NNPotOpt{
public:
	//simulation data
		std::vector<Structure>* strucTrain_; //structures - training
		std::vector<Structure>* strucVal_;   //structures - validation
		std::vector<Structure>* strucTest_;  //structures - testing
		unsigned int nTrain_; //number - structures - training
		unsigned int nVal_;   //number - structures - validation
		unsigned int nTest_;  //number - structures - testing
	//elements
		std::vector<Atom> atoms_;//unique atomic species
		unsigned int nElements_;//number of unique atomic species
		std::vector<unsigned int> nAtoms_;//number of atoms of each element
		VecList pElement_; //parameters - subset for each element (nElements)
		VecList gElement_; //gradients - subset for each element (nElements)
		VecList gTemp_;    //gradients - temp vector
		VecList gTempSum_; //gradients - temp vector for sums
	//batch
		unsigned int nBatch_;//number of structures in batch
		double pBatch_;//percentage of structures in batch (overrides nBatch_)
		std::vector<unsigned int> batch_;//small batch of inputs (simulations)
		std::vector<unsigned int> indices_;//indices of all the inputs (simulations)
	//nn
		unsigned int nParams_;//total number of parameters
		NNPot nnpot_;//the neural network potential
		VecList preBias_,preScale_;//one for each specie (nn)
		bool preCond_;//whether to pre-condition the inputs
		bool charge_;//whether the atoms are charged
	//input/output
		std::string restart_file_;//restart file
		bool restart_;//restart
		std::vector<std::string> atoms_basis_radial_;//radial basis - atom name
		std::vector<std::pair<std::string,std::string> > atoms_basis_angular_;//angular basis - atom name
		std::vector<std::string> files_basis_radial_;//radial basis file
		std::vector<std::string> files_basis_angular_;//angular basis file
		bool calcForce_;//whether to compute the force
	//optimization
		Opt::Data data_;//optimization - data
		std::shared_ptr<Opt::Model> model_;//optimization - data
		Eigen::VectorXd identity_;//identity vector
	//error
		double error_train_;  //error - training
		double error_val_;    //error - validation
		double error_lambda_; //error - regularization
	//file i/o
		FILE* writer_error_;//file pointer which writes the error
		std::string file_error_;//file storing the error
	//constructors/destructors
		NNPotOpt();
		~NNPotOpt(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPotOpt& nnPotOpt);
	//member functions
		void defaults();//default variable values
		void clear();//reset object
		void train(unsigned int batchSize);//train potential
		double error(const Eigen::VectorXd& x);//compute error for a potential
		void write_restart(const char* file);//write restart file
		void read_restart(const char* file);//read restart file
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
