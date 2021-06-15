#ifndef NN_TRAIN_HPP
#define NN_TRAIN_HPP

// mpi
#include <mpi.h>
// c++ libraries
#include <iosfwd>
// c libraries
#include <memory>
// ann - typedefs
#include "typedef.hpp"
// ann - nn
#include "nn.hpp"
// ann - batch
#include "batch.hpp"
// ann - random
#include "random.hpp"
// ann - ekf
#include "ekf.hpp"
#include "optimize.hpp"

#ifndef NN_TRAIN_PRINT_FUNC
#define NN_TRAIN_PRINT_FUNC 0
#endif

#ifndef NN_TRAIN_PRINT_STATUS
#define NN_TRAIN_PRINT_STATUS 0
#endif

#ifndef NN_TRAIN_PRINT_DATA
#define NN_TRAIN_PRINT_DATA 0
#endif

//***********************************************************************
// NN Optimization
//***********************************************************************

class NNOpt{
private:
	//neural networks
	NeuralNet::DOutDP dOutDP_;//gradient of output w.r.t. parameters
	std::shared_ptr<NeuralNet::ANN> nn_;//neural network
	//optimization
	Batch batch_;//batch
	EKF ekf_;//extended Kalman filter
	Opt::DECAY::type decay_;//decay schedule
	double alpha_;//step decay constant
	double gamma_;//gradient step size
	double power_;
	int max_;
	int nPrint_;
	int nWrite_;
	int step_;
	int count_;
	int period_;
	double err_train_; //error - training
	double err_val_;   //error - validation
	Eigen::MatrixXd hLocal_;
	Eigen::MatrixXd hGlobal_;
	Eigen::VectorXd eLocal_;
	Eigen::VectorXd eGlobal_;
	//conditioning
	bool preCond_;  //whether to pre-condition  the inputs
	bool postCond_; //whether to post-condition the inputs
	//data
	int nTrain_,nVal_;
	std::shared_ptr<std::vector<VecXd> > inT_;  //data - inputs - training
	std::shared_ptr<std::vector<VecXd> > outT_; //data - inputs - training
	std::shared_ptr<std::vector<VecXd> > inV_;  //data - inputs - validation
	std::shared_ptr<std::vector<VecXd> > outV_; //data - inputs - validation
	//random
	rng::gen::CG2 cg2_;
	//file i/o
	bool restart_;//whether restarting
	std::string file_error_;//file storing the error
	std::string file_restart_;//restart file
public:
	//==== constructors/destructors ====
	NNOpt(){defaults();}
	~NNOpt(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NNOpt& nnopt);
	
	//==== access ====
	//conditioning
	bool& preCond(){return preCond_;}
	const bool& preCond()const{return preCond_;}
	bool& postCond(){return postCond_;}
	const bool& postCond()const{return postCond_;}
	//optimization
	double err_train()const{return err_train_;}
	double err_val()const{return err_val_;}
	Batch& batch(){return batch_;}
	const Batch& batch()const{return batch_;}
	double& alpha(){return alpha_;}
	const double& alpha()const{return alpha_;}
	double& gamma(){return gamma_;}
	const double& gamma()const{return gamma_;}
	double& power(){return power_;}
	const double& power()const{return power_;}
	int& max(){return max_;}
	const int& max()const{return max_;}
	int& nPrint(){return nPrint_;}
	const int& nPrint()const{return nPrint_;}
	int& nWrite(){return nWrite_;}
	const int& nWrite()const{return nWrite_;}
	int& step(){return step_;}
	const int& step()const{return step_;}
	int& count(){return count_;}
	const int& count()const{return count_;}
	int& period(){return period_;}
	const int& period()const{return period_;}
	EKF& ekf(){return ekf_;}
	const EKF& ekf()const{return ekf_;}
	//random
	rng::gen::CG2& cg2(){return cg2_;}
	const rng::gen::CG2& cg2()const{return cg2_;}
	//file i/o
	bool& restart(){return restart_;}
	const bool& restart()const{return restart_;}
	std::string& file_error(){return file_error_;}
	const std::string& file_error()const{return file_error_;}
	std::string& file_restart(){return file_restart_;}
	const std::string& file_restart()const{return file_restart_;}
	Opt::DECAY::type& decay(){return decay_;}
	const Opt::DECAY::type& decay()const{return decay_;}
	//data
	std::shared_ptr<std::vector<VecXd> >& inT(){return inT_;}
	std::shared_ptr<std::vector<VecXd> >& outT(){return outT_;}
	std::shared_ptr<std::vector<VecXd> >& inV(){return inV_;}
	std::shared_ptr<std::vector<VecXd> >& outV(){return outV_;}
	const std::shared_ptr<std::vector<VecXd> >& inT()const{return inT_;}
	const std::shared_ptr<std::vector<VecXd> >& outT()const{return outT_;}
	const std::shared_ptr<std::vector<VecXd> >& inV()const{return inV_;}
	const std::shared_ptr<std::vector<VecXd> >& outV()const{return outV_;}
	//nn
	std::shared_ptr<NeuralNet::ANN>& nn(){return nn_;}
	const std::shared_ptr<NeuralNet::ANN>& nn()const{return nn_;}
	
	//==== member functions ====
	//training
	void train(std::shared_ptr<NeuralNet::ANN>& nn, int nbatchl);
	void update_step(int step);
	//error
	double error();
	//misc
	void defaults();
	void clear();
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const NNOpt& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NNOpt& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NNOpt& obj, const char* arr);
	
}

#endif