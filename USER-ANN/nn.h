#ifndef ANN_NN_HPP
#define ANN_NN_HPP

//no bounds checking in Eigen
#define EIGEN_NO_DEBUG

// c libraries
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
// c++ libraries
#include <iostream>
// ann library - eigen utilities
#include <Eigen/Dense>
#include <Eigen/StdVector>
// ann library - math 
#include "ann_math_special.h"
// ann library - string
#include "ann_string.h"
// ann library - serialization
#include "serialize.h"

namespace NN{
	
//***********************************************************************
// COMPILER DIRECTIVES
//***********************************************************************

#ifndef NN_PRINT_FUNC
#define NN_PRINT_FUNC 0
#endif

#ifndef NN_PRINT_STATUS
#define NN_PRINT_STATUS 0
#endif

#ifndef NN_PRINT_DATA
#define NN_PRINT_DATA 0
#endif

//***********************************************************************
// TYPEDEFS
//***********************************************************************

typedef std::vector<Eigen::VectorXd,Eigen::aligned_allocator<Eigen::VectorXd> > VecList;
typedef std::vector<Eigen::MatrixXd,Eigen::aligned_allocator<Eigen::MatrixXd> > MatList;

//***********************************************************************
// FORWARD DECLARATIONS
//***********************************************************************

class NNOpt;

//***********************************************************************
// TRANSFER FUNCTIONS - NAMES
//***********************************************************************

struct TransferN{
	enum type{
		UNKNOWN=-1,
		TANH=0,
		SIGMOID=1,
		LINEAR=2
	};
	static type load(const char* str);
};
std::ostream& operator<<(std::ostream& out, const TransferN::type& tf);

//***********************************************************************
// TRANSFER FUNCTIONS - FUNCTION WRAPPERS
//***********************************************************************

struct TransferF{
	static inline double f_tanh(double x){return std::tanh(x);}
	static inline double f_sigmoid(double x){return special::sigmoid(x);}
	static inline double f_lin(double x){return x;}
};

struct TransferFD{
	static inline double f_tanh(double x){x=std::cosh(x); return 1.0/(x*x);}
	static inline double f_sigmoid(double x){return 1.0/((1.0+std::exp(-x))*(1.0+std::exp(x)));}
	static inline double f_lin(double x){return 1;}
};

struct TransferFFD{
	static inline void f_tanh(double x, double& v, double& d){v=std::tanh(x);d=1.0-v*v;}
	static inline void f_sigmoid(double x, double& v, double& d){x=std::exp(-x);v=1.0/(1.0+x);d=1.0/((1.0+x)*(1.0+1.0/x));}
	static inline void f_lin(double x, double& v, double& d){v=x;d=1;}
};

//***********************************************************************
// NETWORK CLASS
//***********************************************************************

class Network{
private:
	//typedefs
		typedef double (*FuncP)(double);
		typedef void (*FFDP)(double,double&,double&);
	//network dimensions
		unsigned int nlayer_;//number of layers of the network (hidden + output)
	//initialize
		double bInit_;
		double wInit_;
	//node weights and biases
		Eigen::VectorXd input_;//input layer
		Eigen::VectorXd sinput_;//scaled input
		Eigen::VectorXd output_;//output layer
		VecList node_;//nodes - not including input layer (nlayer_)
		VecList bias_;//bias - same indexing as nodes_ (nlayer_)
		MatList edge_;//edges - input for indexed layer (nodes_[n-1] -> nodes_[n]) => (nodes_[n-1] x nodes_[n])
		Eigen::VectorXd preScale_,postScale_;//scaling layers for input/output
		Eigen::VectorXd preBias_,postBias_;//biasing layers for input/output
	//gradients
		Eigen::VectorXd grad_;//input layer
		VecList dndz_;//node derivative - not including input layer
		VecList delta_;//derivative of cost function w.r.t. node inputs
		MatList dOut_;//derivative of the output nodes w.r.t. to all other nodes
	//transfer functions
		TransferN::type tfType_;//transfer function type
		//std::vector<std::function<double(double)> > tf_;//transfer function - input for indexed layer (nlayer_)
		//std::vector<std::function<double(double)> > tfd_;//transfer derivative - input for indexed layer (nlayer_)
		std::vector<FuncP> tf_;//transfer function - input for indexed layer (nlayer_)
		std::vector<FuncP> tfd_;//transfer derivative - input for indexed layer (nlayer_)
	//regularization
		double lambda_;//regularization weight
public:
	//======== friend declarations ========
	friend class NNOpt;
	
	//======== constructors/destructors ========
	Network(){defaults();};
	~Network(){};
	
	//======== operators ========
	friend std::ostream& operator<<(std::ostream& out, const Network& n);
	friend Eigen::VectorXd& operator>>(const Network& nn, Eigen::VectorXd& v);
	friend Network& operator<<(Network& nn, const Eigen::VectorXd& v);
	
	//======== access ========
	//network dimensions
		unsigned int nlayer()const{return nlayer_;};
		unsigned int nhidden()const{return nlayer_-1;};
		unsigned int nNodes(unsigned int l)const{return node_[l].size();};
	//initialization
		double& bInit(){return bInit_;};
		const double& bInit()const{return bInit_;};
		double& wInit(){return wInit_;};
		const double& wInit()const{return wInit_;};
	//nodes
		double& input(unsigned int n){return input_[n];};
		const double& input(unsigned int n)const{return input_[n];};
		const Eigen::VectorXd& input()const{return input_;};
		double& sinput(unsigned int n){return sinput_[n];};
		const double& sinput(unsigned int n)const{return sinput_[n];};
		double& hidden(unsigned int l, unsigned int n){return node_[l][n];};
		const double& hidden(unsigned int l, unsigned int n)const{return node_[l][n];};
		double& layer(unsigned int l, unsigned int n){return node_[l][n];};
		const double& layer(unsigned int l, unsigned int n)const{return node_[l][n];};
		double& output(unsigned int n){return output_[n];};
		const double& output(unsigned int n)const{return output_[n];};
		const Eigen::VectorXd& output()const{return output_;};
	//scaling
		double& preScale(unsigned int n){return preScale_[n];};
		const double& preScale(unsigned int n)const{return preScale_[n];};
		const Eigen::VectorXd& preScale()const{return preScale_;};
		double& postScale(unsigned int n){return postScale_[n];};
		const double& postScale(unsigned int n)const{return postScale_[n];};
		const Eigen::VectorXd& postScale()const{return postScale_;};
		double& preBias(unsigned int n){return preBias_[n];};
		const double& preBias(unsigned int n)const{return preBias_[n];};
		const Eigen::VectorXd& preBias()const{return preBias_;};
		double& postBias(unsigned int n){return postBias_[n];};
		const double& postBias(unsigned int n)const{return postBias_[n];};
		const Eigen::VectorXd& postBias()const{return postBias_;};
	//bias
		double& bias(unsigned int l, unsigned int n){return bias_[l][n];};
		const double& bias(unsigned int l, unsigned int n)const{return bias_[l][n];};
		const Eigen::VectorXd& bias(unsigned int l)const{return bias_[l];};
	//edges
		double& edge(unsigned int l, unsigned int n, unsigned int m){return edge_[l](n,m);};
		const double& edge(unsigned int l, unsigned int n, unsigned int m)const{return edge_[l](n,m);};
		const Eigen::MatrixXd& edge(unsigned int l)const{return edge_[l];};
	//size
		unsigned int nInput()const{return input_.size();};
		unsigned int nHidden(unsigned int l)const{return node_[l].size();};
		unsigned int nlayer(unsigned int l)const{return node_[l].size();};
		unsigned int nOutput()const{return node_.back().size();};
	//gradients
		double& grad(unsigned int n){return grad_[n];};
		const double& grad(unsigned int n)const{return grad_[n];};
		double& dndz(unsigned int l, unsigned int n){return dndz_[l][n];};
		const double& dndz(unsigned int l, unsigned int n)const{return dndz_[l][n];};
		double& delta(unsigned int l, unsigned int n){return delta_[l][n];};
		const double& delta(unsigned int l, unsigned int n)const{return delta_[l][n];};
		Eigen::MatrixXd& dOut(unsigned int i){return dOut_[i];};
		const Eigen::MatrixXd& dOut(unsigned int i)const{return dOut_[i];};
		void grad_out();
	//transfer functions
		TransferN::type& tfType(){return tfType_;};
		const TransferN::type& tfType()const{return tfType_;};
		//std::function<double(double)>& tf(unsigned int l){return tf_[l];};
		//const std::function<double(double)>& tf(unsigned int l)const{return tf_[l];};
		//std::function<double(double)>& tfd(unsigned int l){return tfd_[l];};
		//const std::function<double(double)>& tfd(unsigned int l)const{return tfd_[l];};
		FuncP tf(unsigned int l){return tf_[l];};
		const FuncP tf(unsigned int l)const{return tf_[l];};
		FuncP tfd(unsigned int l){return tfd_[l];};
		const FuncP tfd(unsigned int l)const{return tfd_[l];};
		FFDP tfd(unsigned int l){return tffd_[l];};
		const FFDP tfd(unsigned int l)const{return tffd_[l];};
	//regularization
		double& lambda(){return lambda_;};
		const double& lambda()const{return lambda_;};
		
	//======== member functions ========
	//clearing/initialization
		void defaults();
		void clear();
	//error
		double error(const Eigen::VectorXd& output);
		double error(const Eigen::VectorXd& output, Eigen::VectorXd& grad);
		Eigen::VectorXd& dcda(const Eigen::VectorXd& output, Eigen::VectorXd& grad);
	//info
		unsigned int size()const;
	//resizing
		void resize(unsigned int nInput, unsigned int nOutput);
		void resize(unsigned int nInput, const std::vector<unsigned int>& nNodes, unsigned int nOutput);
		void resize(unsigned int nInput, const std::vector<unsigned int>& nNodes);
		Eigen::VectorXd& grad(const Eigen::VectorXd& dcda, Eigen::VectorXd& grad);
	//execution
		const Eigen::VectorXd& execute();
		const Eigen::VectorXd& execute(const Eigen::VectorXd& input);
		
	//======== static functions ========
	static void write(FILE* writer, const Network& nn);
	static void write(const char*, const Network& nn);
	static void read(FILE* writer, Network& nn);
	static void read(const char*, Network& nn);
};

}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const NN::Network& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const NN::Network& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(NN::Network& obj, const char* arr);
	
}

#endif
