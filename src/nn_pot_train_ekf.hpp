// c++ libraries
#include <memory>
// ann - structure
#include "structure_fwd.hpp"
// nn
#include "nn.hpp"
#include "nn_pot.hpp"
// optimization
#include "optimize.hpp"
#include "ekf.hpp"

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
		std::vector<Eigen::MatrixXd> hElement_; //gradients - subset for each element (nElements)
		std::vector<Eigen::MatrixXd> hLocal_;   //gradients - local to each processor
		std::vector<Eigen::MatrixXd> hTotal_;   //gradients - total for each structure
		Eigen::MatrixXd hGlobal_;
		Eigen::VectorXd eGlobal_;
		Eigen::VectorXd eLocal_;
	//batch
		int cBatch_;//batch count
		int nBatch_;//number of structures in batch
		std::vector<int> batch_;//small batch of inputs (simulations)
		std::vector<int> indices_;//indices of all the inputs (simulations)
	//nn
		int nParams_;//total number of parameters
		bool preCond_;//whether to pre-condition the inputs
		bool charge_;//whether the atoms are charged
		NNPot nnpot_;//the neural network potential
		std::vector<NeuralNet::DOutDP> dOutDP_;//gradient of the cost function
		NeuralNet::LossN::type loss_;//loss function
		NeuralNet::ANNInit init_;//neural network initialization parameters
		NeuralNet::TransferN::type tfType_;//transfer function
		std::vector<std::vector<int> > nh_;//hidden layer configuration
	//input/output
		std::string file_error_;//file storing the error
		std::string file_ann_;//ann file
		std::string file_restart_;//restart file
		std::vector<std::string> files_basis_;//files storing the basis
		bool restart_;//restart
		bool calcForce_;//whether to compute the force
		bool calcSymm_;//whether to compute the symmetry functions
		bool writeSymm_;//whether to compute the symmetry functions
		bool norm_;//normalize the energies
	//optimization
		parallel::Dist dist_batch_;
		int max_;//max number of iterations
		int nPrint_;//print this many steps
		int nWrite_;//write this many steps
		int step_;//the current step for decay
		int count_;//the total number of steps taken
		int period_;//the period of decay
		double tol_;//the tolerance for optimization
		double gamma_;//the step size
		double alpha_;//the step decay constant
		double power_;//the power for decay
		double val_,valOld_;//optimization value
		double dv_;//change in optimization value
		Opt::DECAY::type decayg_;//decay schedule - gamma
		Opt::DECAY::type decayn_;//decay schedule - noise
		Opt::VAL::type optVal_;
		EKF ekf_;//extended Kalman filter
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
		void update_step(int step);
		void update_noise(int step);
		void train(parallel::Dist dist_batch);//train potential
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
