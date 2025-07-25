// c++ libraries
#include <memory>
// structure
#include "struc/structure_fwd.hpp"
// ml
#include "ml/nn.hpp"
#include "ml/batch.hpp"
// nnp
#include "nnp/nnp.hpp"
// torch
#include "torch/qeq.hpp"
// optimization
#include "opt/objective.hpp"
#include "opt/algo.hpp"
#include "opt/decay.hpp"

#define EIGEN_NO_DEBUG

#ifndef NNPTEQ_PRINT_FUNC
#define NNPTEQ_PRINT_FUNC 0
#endif

#ifndef NNPTEQ_PRINT_STATUS
#define NNPTEQ_PRINT_STATUS 0
#endif

#ifndef NNPTEQ_PRINT_DATA
#define NNPTEQ_PRINT_DATA 0
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
// NNPTEQ - Neural Network Potential - Optimization
//************************************************************

class NNPTEQ{
public:
	//nnp
		NNP nnp_;//the neural network potential
		int nTypes_;//number of unique atomic species
		std::vector<Eigen::VectorXd> pElement_; //parameters - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gElement_; //gradients - subset for each element (nElements)
		std::vector<Eigen::VectorXd> grad_;     //gradients - local to each processor
		std::vector<NN::Cost> cost_;//gradient of the cost functions
	//input/output
		std::string file_params_;   //file - stores parameters
		std::string file_error_;   //file - stores error
		std::string file_ann_;     //file - stores ann
		std::string file_restart_; //file - stores restart info
	//flags
		bool restart_; //flag - whether to restart
		bool preCond_; //flag - whether to pre-condition the inputs
		bool wparams_; //flag - whether to write the parameters to file
	//charge
		QEQ qeq_;
		std::vector<Eigen::VectorXd> bt_;//solution vector - training
		std::vector<Eigen::VectorXd> xt_;//variable vector - training
		std::vector<Eigen::MatrixXd> AIt_;//inverse of qeq matrix - training
		std::vector<Eigen::VectorXd> bv_;//solution vector - training
		std::vector<Eigen::VectorXd> xv_;//variable vector - training
		std::vector<Eigen::MatrixXd> AIv_;//inverse of qeq matrix - training
	//optimization
		Batch batch_; //batch
		opt::Objective obj_;//objective
		std::shared_ptr<opt::algo::Base> algo_; //optimization algorithm
		std::shared_ptr<opt::decay::Base> decay_;//step decay
		std::vector<thread::Dist> dist_atomt;
		std::vector<thread::Dist> dist_atomv;
		std::mt19937 rngen_;//random number generator
		double delta_;
		double deltai_;
		double delta2_;
	//error
		double error_[4];
	//constructors/destructors
		NNPTEQ();
		~NNPTEQ(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPTEQ& nnPotOpt);
public:
	//==== utility ====
	void defaults();//default variable values
	void clear();//reset object
	
	//==== reading/writing restart ====
	void write_restart(const char* file);//write restart file
	void read_restart(const char* file);//read restart file
	
	//==== error ====
	void train(int batchSize, std::vector<Structure>& struc_train, std::vector<Structure>& struc_val);//train potential
	void error(const Eigen::VectorXd& x, std::vector<Structure>& struc_train, std::vector<Structure>& struc_val);//compute error for a potential
	
	//==== static functions ====
	static void read(const char* file, NNPTEQ& nnpte);
	static void read(FILE* reader, NNPTEQ& nnpte);
	
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
	//charge
		QEQ& qeq(){return qeq_;}
		const QEQ& qeq()const{return qeq_;}
	//nnp
		NNP& nnp(){return nnp_;}
		const NNP& nnp()const{return nnp_;}
	//optimization
		std::mt19937& rngen(){return rngen_;}
		const std::mt19937& rngen()const{return rngen_;}
		const Batch& batch()const{return batch_;}
		opt::Objective& obj(){return obj_;}
		const opt::Objective& obj()const{return obj_;}
		std::shared_ptr<opt::algo::Base>& algo(){return algo_;}
		const std::shared_ptr<opt::algo::Base>& algo()const{return algo_;}
		std::shared_ptr<opt::decay::Base>& decay(){return decay_;}
		const std::shared_ptr<opt::decay::Base>& decay()const{return decay_;}
		double& delta(){return delta_;}
		const double& delta()const{return delta_;}
		double& deltai(){return deltai_;}
		const double& deltai()const{return deltai_;}
		double& delta2(){return delta2_;}
		const double& delta2()const{return delta2_;}
	
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNPTEQ& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNPTEQ& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNPTEQ& obj, const char* arr);
	
}
