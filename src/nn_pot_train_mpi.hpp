// c++ libraries
#include <memory>
// ann - structure
#include "structure_fwd.hpp"
// nn
#include "nn.hpp"
#include "nn_pot.hpp"
// optimization
#include "optimize.hpp"

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

//************************************************************
// Mode
//************************************************************

struct MODE{
	enum type{
		TRAIN,
		TEST,
		SYMM,
		UNKNOWN
	};
	static type read(const char* str);
	static const char* name(const MODE::type& mode);
};
std::ostream& operator<<(std::ostream& out, const MODE::type& mode);

//************************************************************
// NNPotOpt - Neural Network Potential - Optimization
//************************************************************

class NNPotOpt{
public:
	//simulation data
		std::vector<Structure>* strucTrain_; //structures - training
		std::vector<Structure>* strucVal_;   //structures - validation
		std::vector<Structure>* strucTest_;  //structures - testing
		int nTrain_; //number - structures - training
		int nVal_;   //number - structures - validation
		int nTest_;  //number - structures - testing
	//elements
		std::vector<Atom> atoms_;//unique atomic species
		int nElements_;//number of unique atomic species
		std::vector<Eigen::VectorXd> pElement_; //parameters - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gElement_; //gradients - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gLocal_;   //gradients - local to each processor
		std::vector<Eigen::VectorXd> gTotal_;   //gradients - total for each structure
	//batch
		int cBatch_;//batch count
		int nBatch_;//number of structures in batch
		std::vector<int> batch_;//small batch of inputs (simulations)
		std::vector<int> indices_;//indices of all the inputs (simulations)
		std::mt19937 rngen_;//random number generator
	//nn
		int nParams_;//total number of parameters
		bool preCond_;//whether to pre-condition the inputs
		bool charge_;//whether the atoms are charged
		NNPot nnpot_;//the neural network potential
		std::vector<NeuralNet::Cost> cost_;//gradient of the cost function
		NeuralNet::LossN::type loss_;//loss function
		NeuralNet::ANNInit init_;//neural network initialization parameters
		NeuralNet::TransferN::type tfType_;//transfer function
		std::vector<std::vector<int> > nh_;//hidden layer configuration
		double huberw_;//huber loss width
	//input/output
		std::string file_error_;//file storing the error
		std::string file_ann_;//ann file
		std::string file_restart_;//restart file
		std::vector<std::string> files_basis_;//files storing the basis
		bool restart_;//restart
		bool calcForce_;//whether to compute the force
		bool calcSymm_;//whether to compute the symmetry functions
		bool norm_;//normalize the energies
	//optimization
		Opt::Data data_;//optimization - data
		std::shared_ptr<Opt::Model> model_;//optimization - model
		Eigen::VectorXd identity_;//identity vector
	//error
		double error_train_; //error - training
		double error_val_;   //error - validation
	//constructors/destructors
		NNPotOpt();
		~NNPotOpt(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPotOpt& nnPotOpt);
	//member functions
		void defaults();//default variable values
		void clear();//reset object
		void train(int batchSize);//train potential
		double error(const Eigen::VectorXd& x);//compute error for a potential
		void write_restart(const char* file);//write restart file
		void read_restart(const char* file);//read restart file
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const NNPotOpt& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNPotOpt& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNPotOpt& obj, const char* arr);
	
}
