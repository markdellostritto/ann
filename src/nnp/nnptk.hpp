// c++ libraries
#include <memory>
// structure
#include "struc/structure_fwd.hpp"
// ml
#include "ml/nn.hpp"
#include "ml/batch.hpp"
// nnp
#include "nnp/nnp.hpp"
// optimization
#include "opt/objective.hpp"
#include "opt/decay.hpp"

#define EIGEN_NO_DEBUG

#ifndef NNPTK_PRINT_FUNC
#define NNPTK_PRINT_FUNC 0
#endif

#ifndef NNPTK_PRINT_STATUS
#define NNPTK_PRINT_STATUS 0
#endif

#ifndef NNPTK_PRINT_DATA
#define NNPTK_PRINT_DATA 0
#endif

//************************************************************
// Mode
//************************************************************

class Mode{
public:
	enum Type{
		TRAIN,
		TEST,
		SYMM,
		UNKNOWN
	};
	//constructor
	Mode():t_(Type::UNKNOWN){}
	Mode(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Mode read(const char* str);
	static const char* name(const Mode& mode);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Mode& mode);

//************************************************************
// NNPTK - Neural Network Potential - Optimization
//************************************************************

class NNPTK{
public:
	//nnp
		int nTypes_;//number of unique atomic species
		std::vector<Eigen::VectorXd> grad_;     //gradients - local to each processor
		NNP nnp_;//the neural network potential
	//input/output
		std::string file_params_;   //file - stores parameters
		std::string file_error_;   //file - stores error
		std::string file_ann_;     //file - stores ann
		std::string file_restart_; //file - stores restart info
	//flags
		bool restart_; //flag - whether to restart
		bool preCond_; //flag - whether to pre-condition the inputs
		bool wparams_; //flag - whether to write the parameters to file
	//optimization
		std::mt19937 rngen_;//random number generator
		Batch batch_; //batch
		opt::Objective obj_;//objective
		std::shared_ptr<opt::decay::Base> decay_;//step decay
		std::vector<NN::DODP> dODP_;
	//kalman
		double noise_;
		std::vector<int> natoms_;
		std::vector<Eigen::VectorXd> w_;//nnp parameters
		std::vector<Eigen::VectorXd> dy_;//errors
		std::vector<Eigen::MatrixXd> AI_;//scaling matrix - inverse
		std::vector<Eigen::MatrixXd> A_;//scaling matrix
		std::vector<Eigen::MatrixXd> K_;//gain matrix
		std::vector<Eigen::MatrixXd> H_;//nnp gradients
		std::vector<Eigen::MatrixXd> P_;//covariance matrix
		std::vector<Eigen::MatrixXd> Q_;//noise matrix
	//error
		double error_[4];
	//constructors/destructors
		NNPTK();
		~NNPTK(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPTK& nnPotOpt);
public:
	//==== utility ====
	void defaults();//default variable values
	void clear();//reset object
	
	//==== reading/writing restart ====
	void write_restart(const char* file);//write restart file
	void read_restart(const char* file);//read restart file
	
	//==== error ====
	void train(int batchSize, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//train potential
	double error(const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//compute error for a potential
	
	//==== static functions ====
	static void read(const char* file, NNPTK& nnptk);
	static void read(FILE* reader, NNPTK& nnptk);
	
	//==== access ====
	//elements
		int nTypes()const{return nTypes_;}
	//files
		std::string& file_params(){return file_params_;}
		const std::string& file_params()const{return file_params_;}
		std::string& file_error(){return file_error_;}
		const std::string& file_error()const{return file_error_;}
		std::string& file_ann(){return file_ann_;}
		const std::string& file_ann()const{return file_ann_;}
		std::string& file_restart(){return file_restart_;}
		const std::string& file_restart()const{return file_restart_;}
	//flags
		bool& restart(){return restart_;}
		const bool& restart()const{return restart_;}
		bool& preCond(){return preCond_;}
		const bool& preCond()const{return preCond_;}
		bool& wparams(){return wparams_;}
		const bool& wparams()const{return wparams_;}
	//nnp
		NNP& nnp(){return nnp_;}
		const NNP& nnp()const{return nnp_;}
	//optimization
		std::mt19937& rngen(){return rngen_;}
		const std::mt19937& rngen()const{return rngen_;}
		const Batch& batch()const{return batch_;}
		opt::Objective& obj(){return obj_;}
		const opt::Objective& obj()const{return obj_;}
		std::shared_ptr<opt::decay::Base>& decay(){return decay_;}
		const std::shared_ptr<opt::decay::Base>& decay()const{return decay_;}
	//kalman
		double& noise(){return noise_;}
		const double& noise()const{return noise_;}
	
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNPTK& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNPTK& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNPTK& obj, const char* arr);
	
}
