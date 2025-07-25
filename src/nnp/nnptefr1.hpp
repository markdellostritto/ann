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
#include "opt/algo.hpp"
#include "opt/decay.hpp"

#define EIGEN_NO_DEBUG

#ifndef NNPTEFR_PRINT_FUNC
#define NNPTEFR_PRINT_FUNC 0
#endif

#ifndef NNPTEFR_PRINT_STATUS
#define NNPTEFR_PRINT_STATUS 0
#endif

#ifndef NNPTEFR_PRINT_DATA
#define NNPTEFR_PRINT_DATA 0
#endif

//************************************************************
//  Perturb
//************************************************************

class Perturb{
public:
	enum Type{
		CONSTANT,
		COVALENT,
		NONE
	};
	//constructor
	Perturb():t_(Type::NONE){}
	Perturb(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Perturb read(const char* str);
	static const char* name(const Perturb& perturb);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Perturb& perturb);

//************************************************************
//  PreScale
//************************************************************

class PreScale{
public:
	enum Type{
		NONE,
		DEV,
		MINMAX,
		MAX,
		NONE
	};
	//constructor
	PreScale():t_(Type::NONE){}
	PreScale(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static PreScale read(const char* str);
	static const char* name(const PreScale& preScale);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const PreScale& preScale);

//************************************************************
//  PreBias
//************************************************************

class PreBias{
public:
	enum Type{
		NONE,
		MEAN,
		HAR,
		MID,
		MIN,
		NONE
	};
	//constructor
	PreBias():t_(Type::NONE){}
	PreBias(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static PreBias read(const char* str);
	static const char* name(const PreBias& preBias);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const PreBias& preBias);

//************************************************************
//  Norm
//************************************************************

class Norm{
public:
	enum Type{
		NONE,
		LINEAR,
		SQRT,
		CBRT,
		LOG,
		NONE
	};
	//constructor
	Norm():t_(Type::NONE){}
	Norm(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Norm read(const char* str);
	static const char* name(const Norm& norm);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Norm& norm);

//************************************************************
// Mode
//************************************************************

class Mode{
public:
	enum Type{
		TRAIN,
		TEST,
		SYMM,
		NONE
	};
	//constructor
	Mode():t_(Type::NONE){}
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
// Regularization
//************************************************************

class Regularization{
public:
	enum Type{
		NONE,
		LASSO,
		RIDGE,
		HUBER,
		ASINH,
		NONE
	};
	//constructor
	Regularization():t_(Type::NONE){}
	Regularization(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Regularization read(const char* str);
	static const char* name(const Regularization& reg);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Regularization& reg);

//************************************************************
// NNPTEFR - Neural Network Potential - Optimization
//************************************************************

/**
* Neural Network Potential - Training - Energy (NNPTEFR)
*/
class NNPTEFR{
public:
	//random
		int seed_;
		std::mt19937 rngen_;//random number generator
	//nnp
		int nTypes_;//number of unique atomic species
		std::vector<Eigen::VectorXd> pElement_; //parameters - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gElement_; //gradients - subset for each element (nElements)
		std::vector<Eigen::VectorXd> grad_;     //gradients - subset for each element (nElements)
		NNP nnp_;//the neural network potential
		std::vector<NN::Cost> cost_;//gradient of the cost functions
		std::vector<NN::DODP> dodp_;//gradient of the cost functions
	//input/output
		std::string file_params_;   //file - stores parameters
		std::string file_error_;   //file - stores error
		std::string file_ann_;     //file - stores ann
		std::string file_restart_; //file - stores restart info
	//flags
		bool restart_; //flag - whether to restart
		bool reset_;   //flag - whether to reset optimization
		bool wparams_; //flag - whether to write the parameters to file
	//batch
		std::vector<double> normt_;
		std::vector<double> normv_;
		std::vector<thread::Dist> dist_atomt;
		std::vector<thread::Dist> dist_atomv;
		Batch batcht_; //batch - training
		Batch batchv_; //batch - validation
	//optimization
		opt::Objective obj_;//optimization - objective
		opt::Iterator iter_;//optimization - iterator
		opt::Decay decay_;//step decay
		std::shared_ptr<opt::algo::Base> algo_; //optimization algorithm
		Norm norm_;
		PreScale prescale_;
		PreBias prebias_;
		Regularization reg_;
		double inscale_,inbias_;
		double delta_,deltai_,delta2_;//loss width
		double beta_,betai_;//exponential averageing
		double rmse_t_a_,rmse_v_a_;//rmse - time averaged
		double eta_;//weight decay
	//error
		double error_[4];
	//constructors/destructors
		NNPTEFR();
		~NNPTEFR(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPTEFR& nnPotOpt);
public:
	//==== utility ====
	void defaults();//default variable values
	void clear();//reset object
	
	//==== reading/writing restart ====
	void write_restart(const char* file);//write restart file
	void read_restart(const char* file);//read restart file
	
	//==== error ====
	void train(int batchSize, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//train potential
	void error_cost(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//compute error for a potential
	void error_cost_o1(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//compute error for a potential
	void error_dodp(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//compute error for a potential
	
	//==== static functions ====
	static void read(const char* file, NNPTEFR& nnpte);
	static void read(FILE* reader, NNPTEFR& nnpte);
	
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
		bool& reset(){return reset_;}
		const bool& reset()const{return reset_;}
		bool& wparams(){return wparams_;}
		const bool& wparams()const{return wparams_;}
	//nnp
		NNP& nnp(){return nnp_;}
		const NNP& nnp()const{return nnp_;}
	//optimization
		int& seed(){return seed_;}
		const int& seed()const{return seed_;}
		std::mt19937& rngen(){return rngen_;}
		const std::mt19937& rngen()const{return rngen_;}
		const Batch& batcht()const{return batcht_;}
		const Batch& batchv()const{return batchv_;}
		opt::Objective& obj(){return obj_;}
		const opt::Objective& obj()const{return obj_;}
		opt::Iterator& iter(){return iter_;}
		const opt::Iterator& iter()const{return iter_;}
		opt::Decay& decay(){return decay_;}
		const opt::Decay& decay()const{return decay_;}
		std::shared_ptr<opt::algo::Base>& algo(){return algo_;}
		const std::shared_ptr<opt::algo::Base>& algo()const{return algo_;}
		Norm& norm(){return norm_;}
		const Norm& norm()const{return norm_;}
		Regularization& reg(){return reg_;}
		const Regularization& reg()const{return reg_;}
		PreScale& prescale(){return prescale_;}
		const PreScale& prescale()const{return prescale_;}
		PreBias& prebias(){return prebias_;}
		const PreBias& prebias()const{return prebias_;}
		double& inscale(){return inscale_;}
		const double& inscale()const{return inscale_;}
		double& inbias(){return inbias_;}
		const double& inbias()const{return inbias_;}
		double& beta(){return beta_;}
		const double& beta()const{return beta_;}
		double& betai(){return betai_;}
		const double& betai()const{return betai_;}
		double& eta(){return eta_;}
		const double& eta()const{return eta_;}
		double& delta(){return delta_;}
		const double& delta()const{return delta_;}
		double& deltai(){return deltai_;}
		const double& deltai()const{return deltai_;}
		double& delta2(){return delta2_;}
		const double& delta2()const{return delta2_;}
	//error
		double* error(){return error_;}
		const double* error()const{return error_;}
		
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNPTEFR& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNPTEFR& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNPTEFR& obj, const char* arr);
	
}
