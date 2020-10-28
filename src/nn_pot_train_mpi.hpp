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
		VecList pElement_; //parameters - subset for each element (nElements)
		VecList gElement_; //gradients - subset for each element (nElements)
		VecList gTemp_;    //gradients - temp vector
		VecList gTempSum_; //gradients - temp vector for sums
	//batch
		int cBatch_;//batch count
		int nBatch_;//number of structures in batch
		double pBatch_;//percentage of structures in batch (overrides nBatch_)
		std::vector<int> batch_;//small batch of inputs (simulations)
		std::vector<int> indices_;//indices of all the inputs (simulations)
	//nn
		int seed_;//random number seed
		int nParams_;//total number of parameters
		NNPot nnpot_;//the neural network potential
		VecList inb_,inw_;//input bias, weight for each specie
		bool preCond_;//whether to pre-condition the inputs
		bool charge_;//whether the atoms are charged
		double idev_;//initial deviation
		NN::InitN::type initType_;//initialization method
		NN::TransferN::type tfType_;//transfer function
		std::vector<std::vector<int> > nh_;//hidden layer configuration
	//input/output
		std::string restart_file_;//restart file
		std::vector<std::string> files_basis_;//
		bool restart_;//restart
		bool calcForce_;//whether to compute the force
		bool calcSymm_;//whether to compute the symmetry functions
		bool writeSymm_;//whether to compute the symmetry functions
	//optimization
		Opt::Data data_;//optimization - data
		std::shared_ptr<Opt::Model> model_;//optimization - data
		Eigen::VectorXd identity_;//identity vector
	//error
		double error_train_; //error - training
		double error_val_;   //error - validation
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
