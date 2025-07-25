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

#ifndef NNPTE_PRINT_FUNC
#define NNPTE_PRINT_FUNC 0
#endif

#ifndef NNPTE_PRINT_STATUS
#define NNPTE_PRINT_STATUS 0
#endif

#ifndef NNPTE_PRINT_DATA
#define NNPTE_PRINT_DATA 0
#endif

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
		UNKNOWN
	};
	//constructor
	PreScale():t_(Type::UNKNOWN){}
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
		MID,
		MIN,
		UNKNOWN
	};
	//constructor
	PreBias():t_(Type::UNKNOWN){}
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
// NNPTE - Neural Network Potential - Optimization
//************************************************************

/**
* Neural Network Potential - Training - Energy (NNPTE)
*/
class NNPTE{
public:
	//random
		std::mt19937 rngen_;//random number generator
	//nnp
		int nTypes_;//number of unique atomic species
		std::vector<Eigen::VectorXd> pElement_; //parameters - subset for each element (nElements)
		std::vector<Eigen::VectorXd> gElement_; //gradients - subset for each element (nElements)
		std::vector<Eigen::VectorXd> grad_; //gradients - subset for each element (nElements)
		NNP nnp_;//the neural network potential
		std::vector<NN::Cost> cost_;//gradient of the cost functions
	//input/output
		std::string file_params_;   //file - stores parameters
		std::string file_error_;   //file - stores error
		std::string file_ann_;     //file - stores ann
		std::string file_restart_; //file - stores restart info
	//flags
		bool restart_; //flag - whether to restart
		bool reset_;   //flag - whether to reset
		bool wparams_; //flag - whether to write the parameters to file
	//optimization
		std::vector<thread::Dist> dist_atomt;
		std::vector<thread::Dist> dist_atomv;
		Batch batch_; //batch
		opt::Objective obj_;//objective
		opt::Iterator iter_;//optimization - iterator
		opt::Decay decay_;//step decay
		std::shared_ptr<opt::algo::Base> algo_; //optimization algorithm
		PreScale prescale_;
		PreBias prebias_;
		double inscale_,inbias_;
		double delta_,deltai_,delta2_;
	//quantum adam
		static const double beta1_,beta2_;
		int nqstep_;
		double eps_;//zero
		double rho_;//mass term
		double beta1i_,beta2i_;
		Eigen::VectorXd mgrad_;//EMA of gradient
		Eigen::VectorXd mgrad2_;//EMA of gradient squared
		Eigen::VectorXd mgradq_;//EMA of quantum gradient
		Eigen::VectorXd mgradq2_;//EMA of quantum gradient squared
		Eigen::VectorXd gradq_;
		Eigen::VectorXd wm1_;
		Eigen::VectorXd wp1_;
		std::vector<int> roots_bead_;
	//error
		double error_[4];
	//constructors/destructors
		NNPTE();
		~NNPTE(){}
	//operators
		friend std::ostream& operator<<(std::ostream& out, const NNPTE& nnPotOpt);
public:
	//==== utility ====
	void defaults();//default variable values
	void clear();//reset object
	
	//==== reading/writing restart ====
	void write_restart(const char* file);//write restart file
	void read_restart(const char* file);//read restart file
	
	//==== error ====
	void train(int batchSize, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//train potential
	void error(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//compute error for a potential
	void error2(const Eigen::VectorXd& x, const std::vector<Structure>& struc_train, const std::vector<Structure>& struc_val);//compute error for a potential
	
	//==== static functions ====
	static void read(const char* file, NNPTE& nnpte);
	static void read(FILE* reader, NNPTE& nnpte);
	
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
		std::mt19937& rngen(){return rngen_;}
		const std::mt19937& rngen()const{return rngen_;}
		const Batch& batch()const{return batch_;}
		opt::Objective& obj(){return obj_;}
		const opt::Objective& obj()const{return obj_;}
		opt::Decay& decay(){return decay_;}
		const opt::Decay& decay()const{return decay_;}
		opt::Iterator& iter(){return iter_;}
		const opt::Iterator& iter()const{return iter_;}
		double& delta(){return delta_;}
		const double& delta()const{return delta_;}
		double& deltai(){return deltai_;}
		const double& deltai()const{return deltai_;}
		double& delta2(){return delta2_;}
		const double& delta2()const{return delta2_;}
		PreScale& prescale(){return prescale_;}
		const PreScale& prescale()const{return prescale_;}
		PreBias& prebias(){return prebias_;}
		const PreBias& prebias()const{return prebias_;}
		double& inscale(){return inscale_;}
		const double& inscale()const{return inscale_;}
		double& inbias(){return inbias_;}
		const double& inbias()const{return inbias_;}
	//quantum adam
		int& nqstep(){return nqstep_;}
		const int& nqstep()const{return nqstep_;}
		double& eps(){return eps_;}
		const double& eps()const{return eps_;}
		double& rho(){return rho_;}
		const double& rho()const{return rho_;}
		std::vector<int>& roots_bead(){return roots_bead_;}
		const std::vector<int>& roots_bead()const{return roots_bead_;}
	//error
		double* error(){return error_;}
		const double* error()const{return error_;}
		
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NNPTE& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNPTE& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNPTE& obj, const char* arr);
	
}
