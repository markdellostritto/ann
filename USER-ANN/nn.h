#ifndef ANN_NN_HPP
#define ANN_NN_HPP

#define EIGEN_NO_DEBUG

// c++ libraries
#include <iosfwd>
// eigen
#include <Eigen/Dense>
// ann - serialization
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

typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatX;

//***********************************************************************
// INITIALIZATION METHOD
//***********************************************************************

struct InitN{
	enum type{
		UNKNOWN=0,
		RAND=1,
		XAVIER=2,
		HE=3,
		MEAN=4
	};
	static type read(const char* str);
	static const char* name(const InitN::type& tf);
};
std::ostream& operator<<(std::ostream& out, const InitN::type& tf);

//***********************************************************************
// TRANSFER FUNCTIONS - NAMES
//***********************************************************************

struct TransferN{
	enum type{
		UNKNOWN=0,
		LINEAR=1,
		SIGMOID=2,
		TANH=3,
		ISRU=4,
		ARCTAN=5,
		SOFTSIGN=6,
		RELU=7,
		SOFTPLUS=8,
		ELU=9,
		GELU=10,
		SOFTPLUS2=11
	};
	static type read(const char* str);
	static const char* name(const TransferN::type& tf);
};
std::ostream& operator<<(std::ostream& out, const TransferN::type& tf);

//***********************************************************************
// TRANSFER FUNCTIONS
//***********************************************************************

struct TransferFFDV{
	static void f_lin(VecX& f, VecX& d);
	static void f_sigmoid(VecX& f, VecX& d);
	static void f_tanh(VecX& f, VecX& d);
	static void f_isru(VecX& f, VecX& d);
	static void f_arctan(VecX& f, VecX& d);
	static void f_softsign(VecX& f, VecX& d);
	static void f_relu(VecX& f, VecX& d);
	static void f_softplus(VecX& f, VecX& d);
	static void f_softplus2(VecX& f, VecX& d);
	static void f_elu(VecX& f, VecX& d);
	static void f_gelu(VecX& f, VecX& d);
};

//***********************************************************************
// NETWORK CLASS
//***********************************************************************

/*
DEFINITIONS:
	ensemble - total set of all data (e.g. training "ensemble")
	element - single datum from ensemble
	c - "c" donotes the cost function, e.g. the gradient of the cost function w.r.t. the value of a node is dc/da
	z - "z" is the input to each node, e.g. the gradient of a node w.r.t. to its input is da/dz
	a - "a" is the value of a node, e.g. the gradient of a node w.r.t. to its input is da/dz
	o - "o" is the output of the network (i.e. out_), e.g. the gradient of the output w.r.t. the input is do/di
	i - "i" is the input of the network (i.e. in_), e.g. the gradient of the output w.r.t. the input is do/di
PRIVATE:
	VecX in_ - raw input data for a single element of the ensemble (e.g. training set)
	VecX inw_ - weight used to scale the input data
	VecX inb_ - bias used to shift the input data
	VecX out_ - raw output data given a single input element
	VecX outw_ - weight used to scale the output data
	VecX outb_ - bias used to shift the output data
	int nlayer_ - 
		total number of hidden layers
		best thought of as the number of "connections" between layers
		thus, if we have just the input and output, we have one "layer" - 
			one set of weights,biases connecting input/output
		if we have two layers, we have the input, output, and one "hidden" layer - 
			two sets of weights,biases connecting input/layer0/output
		if we have three layers, we have the input, output, and two "hidden" layers - 
			three sets of weights,biases connecting input/layer0/layer1/output
		et cetera
	std::vector<VecX> node_ - 
		all nodes, including the input, output, and hidden layers
		the raw input and output (in_,out_) are separate from "node_"
		this is because the raw input/output may be shifted/scaled before being used
		thus, while in_/out_ are the "raw" input/output,
		the front/back of "node_" can be thought of the "scaled" input/output
		note that scaling is not necessary, but made optional with the use of in_/out_
		has a size of "nlayer_+1", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<VecX> bias_ - 
		the bias of each layer, best thought of as the bias "between" layers n,n+1
		bias_[n] must have the size node_[n+1] - we add this bias when going from node_[n] to node_[n+1]
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<MatX> edge_ -
		the weights of each layer, best though of as transforming from layers n to n+1
		edge_[n] must have the size (node_[n+1],node_[n]) - matrix multiplying (node_[n]) to get (node_[n+1])
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<VecX> dadz_ - 
		the gradient of the value of a node (a) w.r.t. the input of the node (z) - da/dz
		practically, the gradient of the transfer function of each layer
		best thought of as the gradient associated with function transferring "between" layers n,n+1
		thus, dadz_[n] must have the size node_[n+1]
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	dcdz_ - 
		the gradient of the cost function (c) w.r.t. the node inputs (z) - dc/dz
	dcdo_ -
		the gradient of the cost function (c) w.r.t. the output (o) - dc/do
	doda_ -
		the derivative of the output (o) w.r.t. the value of all nodes (a)
		has a size of "nlayer_+1" as we need to compute the gradient w.r.t. all nodes
		thus, doda_[n] must of the size node_[n]
		this includes the hidden layers as well as the input/ouput layers
		note these are the scaled inputs/outputs
	dodi_ -
		the derivative of the output w.r.t. the raw input
		this is the first element of doda_ multiplied by the input scaling
	tfType_ -
		the type of the transfer function
		note the transfer function for the last layer is always linear
	tffdv_ - 
		(Transfer Function, Function Derivative, Vector)
		the transfer function for each layer, operates on entire vector at once
		computes both function and derivative simultaneously
*/
class Network{
private:
	//typedefs
		typedef void (*FFDVP)(VecX&,VecX&);
	//initialization
		double bInit_;//initial value - bias
		double wInit_;//initial value - weight
		double idev_;//initialization deviation
		InitN::type initType_;//initialization scheme
		int seed_;//random seed
	//layers
		int nlayer_;//number of layers (weights,biases)
	//input/output
		VecX in_;//input layer
		VecX out_;//output layer
		VecX inw_,inb_;//input weight, bias
		VecX outw_,outb_;//output weight, bias
	//node weights and biases
		std::vector<VecX> node_;//nodes (nlayer_+1)
		std::vector<VecX> bias_;//bias (nlayer_)
		std::vector<MatX> edge_;//edges (nlayer_)
	//gradients - nodes
		std::vector<VecX> dadz_;//node derivative - not including input layer (nlayer_)
	//gradients - cost function
		std::vector<VecX> dcdz_;//derivative of cost function w.r.t. node inputs (nlayer_)
		VecX dcdo_;//gradient of cost w.r.t. output (out_.size())
	//gradients - output
		std::vector<MatX> doda_;//derivative of out_ w.r.t. to the value "a" of all nodes (nlayer_+1)
		MatX dodi_;//derivative of out_ w.r.t. to in_ (out_.size() x in_.size())
	//transfer functions
		TransferN::type tfType_;//transfer function type
		std::vector<FFDVP> tffdv_;//transfer function - input for indexed layer (nlayer_)
public:
	//==== constructors/destructors ====
	Network(){defaults();}
	~Network(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Network& n);
	friend FILE* operator<<(FILE* out, const Network& n);
	friend VecX& operator>>(const Network& nn, VecX& v);
	friend Network& operator<<(Network& nn, const VecX& v);
	
	//==== access ====
	//network dimensions
		int nlayer()const{return nlayer_;}
	//initialization
		double& bInit(){return bInit_;}
		const double& bInit()const{return bInit_;}
		double& wInit(){return wInit_;}
		const double& wInit()const{return wInit_;}
		double& idev(){return idev_;}
		const double& idev()const{return idev_;}
		InitN::type& initType(){return initType_;}
		const InitN::type& initType()const{return initType_;}
		int& seed(){return seed_;}
		const int& seed()const{return seed_;}
	//nodes
		VecX& in(){return in_;}
		const VecX& in()const{return in_;}
		VecX& out(){return out_;}
		const VecX& out()const{return out_;}
		VecX& node(int n){return node_[n];}
		const VecX& node(int n)const{return node_[n];}
		int nNodes(int n)const{return node_[n].size();}
	//scaling
		VecX& inw(){return inw_;}
		const VecX& inw()const{return inw_;}
		VecX& inb(){return inb_;}
		const VecX& inb()const{return inb_;}
		VecX& outw(){return outw_;}
		const VecX& outw()const{return outw_;}
		VecX& outb(){return outb_;}
		const VecX& outb()const{return outb_;}
	//bias
		VecX& bias(int l){return bias_[l];}
		const VecX& bias(int l)const{return bias_[l];}
	//edge
		MatX& edge(int l){return edge_[l];}
		const MatX& edge(int l)const{return edge_[l];}
	//size
		int nIn()const{return in_.size();}
		int nOut()const{return out_.size();}
	//gradients - nodes
		VecX& dadz(int n){return dadz_[n];}
		const VecX& dadz(int n)const{return dadz_[n];}
	//gradients - cost function
		VecX& dcdo(){return dcdo_;}
		const VecX& dcdo()const{return dcdo_;}
		VecX& dcdz(int n){return dcdz_[n];}
		const VecX& dcdz(int n)const{return dcdz_[n];}
	//gradients - output
		MatX& doda(int n){return doda_[n];}
		const MatX& doda(int n)const{return doda_[n];}
		MatX dodi(){return dodi_;}
		const MatX dodi()const{return dodi_;}
	//transfer functions
		TransferN::type& tfType(){return tfType_;}
		const TransferN::type& tfType()const{return tfType_;}
		FFDVP tffdv(int l){return tffdv_[l];}
		const FFDVP tffdv(int l)const{return tffdv_[l];}
		
	//==== member functions ====
	//clearing/initialization
		void defaults();
		void clear();
	//error
		double error(const VecX& output)const;
		double error(const VecX& output, VecX& grad);
		double error_lambda()const;
		VecX& dcda(const VecX& output, VecX& grad)const;
	//info
		int size()const;
		int nBias()const;
		int nWeight()const;
	//resizing
		void resize(int nInput, int nOutput);
		void resize(int nInput, const std::vector<int>& nNodes, int nOutput);
		void resize(const std::vector<int>& nNodes);
		void reset();
		VecX& grad(const VecX& dcda, VecX& grad);
		VecX& grad_lambda(VecX& grad)const;
	//execution
		const VecX& execute();
		const VecX& execute(const VecX& in){in_.noalias()=in;return execute();}
		void grad_out();
		
	//==== static functions ====
	static void write(FILE* writer, const Network& nn);
	static void write(const char*, const Network& nn);
	static void read(FILE* writer, Network& nn);
	static void read(const char*, Network& nn);
};

bool operator==(const Network& n1, const Network& n2);
inline bool operator!=(const Network& n1, const Network& n2){return !(n1==n2);}

}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NN::Network& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NN::Network& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NN::Network& obj, const char* arr);
	
}

#endif