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
// ann - opt
#include "optimize.hpp"

#ifndef NN_TRAIN_PRINT_FUNC
#define NN_TRAIN_PRINT_FUNC 0
#endif

#ifndef NN_TRAIN_PRINT_STATUS
#define NN_TRAIN_PRINT_STATUS 1
#endif

#ifndef NN_TRAIN_PRINT_DATA
#define NN_TRAIN_PRINT_DATA 0
#endif

namespace NN{
	
//***********************************************************************
// NN Optimization
//***********************************************************************

class NNOpt{
private:
	//batch
	int nbatch_,cbatch_;
	double pbatch_;
	std::vector<int> batch_;//random batch
	std::vector<int> indices_;//indices of the batch
	//conditioning
	bool preCond_;  //whether to pre-condition  the inputs
	bool postCond_; //whether to post-condition the inputs
	//data
	int nTrain_,nVal_;
	std::shared_ptr<VecList> inT_;  //data - inputs - training
	std::shared_ptr<VecList> outT_; //data - inputs - training
	std::shared_ptr<VecList> inV_;  //data - inputs - validation
	std::shared_ptr<VecList> outV_; //data - inputs - validation
	//optimization
	Opt::Data data_;
	std::shared_ptr<Opt::Model> model_;
	double err_train_; //error - training
	double err_val_;   //error - validation
	//neural networks
	std::shared_ptr<Network> nn_;//neural network
	//input/output
	bool restart_;//whether restarting
	//file i/o
	FILE* writer_error_;//file pointer which writes the error
	std::string file_error_;//file storing the error
	std::string file_restart_;//restart file
public:
	//==== constructors/destructors ====
	NNOpt(){defaults();}
	~NNOpt(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NNOpt& nnopt);
	
	//==== access ====
	//batch
	int& nbatch(){return nbatch_;}
	const int& nbatch()const{return nbatch_;}
	int& cbatch(){return cbatch_;}
	const int& cbatch()const{return cbatch_;}
	double& pbatch(){return pbatch_;}
	const double& pbatch()const{return pbatch_;}
	//conditioning
	bool& preCond(){return preCond_;}
	const bool& preCond()const{return preCond_;}
	bool& postCond(){return postCond_;}
	const bool& postCond()const{return postCond_;}
	//optimization
	Opt::Data& data(){return data_;}
	const Opt::Data& data()const{return data_;}
	std::shared_ptr<Opt::Model>& model(){return model_;}
	const std::shared_ptr<Opt::Model>& model()const{return model_;}
	double err_train()const{return err_train_;}
	double err_val()const{return err_val_;}
	//input/output
	bool& restart(){return restart_;}
	const bool& restart()const{return restart_;}
	//file i/o
	std::string& file_error(){return file_error_;}
	const std::string& file_error()const{return file_error_;}
	std::string& file_restart(){return file_restart_;}
	const std::string& file_restart()const{return file_restart_;}
	//data
	std::shared_ptr<VecList>& inT(){return inT_;}
	std::shared_ptr<VecList>& outT(){return outT_;}
	std::shared_ptr<VecList>& inV(){return inV_;}
	std::shared_ptr<VecList>& outV(){return outV_;}
	const std::shared_ptr<VecList>& inT()const{return inT_;}
	const std::shared_ptr<VecList>& outT()const{return outT_;}
	const std::shared_ptr<VecList>& inV()const{return inV_;}
	const std::shared_ptr<VecList>& outV()const{return outV_;}
	//nn
	std::shared_ptr<Network>& nn(){return nn_;}
	const std::shared_ptr<Network>& nn()const{return nn_;}
	
	//==== member functions ====
	//training
	void train(std::shared_ptr<Network>& nn);
	//error
	double error();
	//misc
	void defaults();
	void clear();
};

}

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const NN::NNOpt& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NN::NNOpt& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NN::NNOpt& obj, const char* arr);
	
}

#endif