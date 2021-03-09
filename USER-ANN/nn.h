#ifndef ANN_NN_HPP
#define ANN_NN_HPP

#define EIGEN_NO_DEBUG

// c++ libraries
#include <iosfwd>
// eigen
#include <Eigen/Dense>
// ann - random
#include "random_ann.h"
// ann - serialization
#include "serialize.h"

namespace NeuralNet{
	
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

typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecXd;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatXd;

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
	static void f_lin(VecXd& f, VecXd& d);
	static void f_sigmoid(VecXd& f, VecXd& d);
	static void f_tanh(VecXd& f, VecXd& d);
	static void f_isru(VecXd& f, VecXd& d);
	static void f_arctan(VecXd& f, VecXd& d);
	static void f_softsign(VecXd& f, VecXd& d);
	static void f_relu(VecXd& f, VecXd& d);
	static void f_softplus(VecXd& f, VecXd& d);
	static void f_softplus2(VecXd& f, VecXd& d);
	static void f_elu(VecXd& f, VecXd& d);
	static void f_gelu(VecXd& f, VecXd& d);
};

//***********************************************************************
// ANNInit
//***********************************************************************

class ANNInit{
private:
	double bInit_;//initial value - bias
	double wInit_;//initial value - weight
	double sigma_;//distribution size parameter
	RNG::DistN::type distT_;//distribution type
	InitN::type initType_;//initialization scheme
	int seed_;//random seed	
public:
	//==== constructors/destructors ====
	ANNInit(){defaults();}
	~ANNInit(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ANNInit& init);
	
	//==== access ====
	double& bInit(){return bInit_;}
	const double& bInit()const{return bInit_;}
	double& wInit(){return wInit_;}
	const double& wInit()const{return wInit_;}
	double& sigma(){return sigma_;}
	const double& sigma()const{return sigma_;}
	RNG::DistN::type& distT(){return distT_;}
	const RNG::DistN::type& distT()const{return distT_;}
	InitN::type& initType(){return initType_;}
	const InitN::type& initType()const{return initType_;}
	int& seed(){return seed_;}
	const int& seed()const{return seed_;}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
};

//***********************************************************************
// ANN
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
	VecXd in_ - raw input data for a single element of the ensemble (e.g. training set)
	VecXd inw_ - weight used to scale the input data
	VecXd inb_ - bias used to shift the input data
	VecXd out_ - raw output data given a single input element
	VecXd outw_ - weight used to scale the output data
	VecXd outb_ - bias used to shift the output data
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
	std::vector<VecXd> node_ - 
		all nodes, including the input, output, and hidden layers
		the raw input and output (in_,out_) are separate from "node_"
		this is because the raw input/output may be shifted/scaled before being used
		thus, while in_/out_ are the "raw" input/output,
		the front/back of "node_" can be thought of the "scaled" input/output
		note that scaling is not necessary, but made optional with the use of in_/out_
		has a size of "nlayer_+1", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<VecXd> bias_ - 
		the bias of each layer, best thought of as the bias "between" layers n,n+1
		bias_[n] must have the size node_[n+1] - we add this bias when going from node_[n] to node_[n+1]
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<MatXd> edge_ -
		the weights of each layer, best though of as transforming from layers n to n+1
		edge_[n] must have the size (node_[n+1],node_[n]) - matrix multiplying (node_[n]) to get (node_[n+1])
		has a size of "nlayer_", as there are "nlayer_" connections between "nlayer_+1" nodes
	std::vector<VecXd> dadz_ - 
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
class ANN{
private:
	//typedefs
		typedef void (*FFDVP)(VecXd&,VecXd&);
	//layers
		int nlayer_;//number of layers (weights,biases)
	//input/output
		VecXd in_;//input layer
		VecXd out_;//output layer
		VecXd inw_,inb_;//input weight, bias
		VecXd outw_,outb_;//output weight, bias
	//node weights and biases
		std::vector<VecXd> node_;//nodes (nlayer_+1)
		std::vector<VecXd> bias_;//bias (nlayer_)
		std::vector<MatXd> edge_;//edges (nlayer_)
	//gradients - nodes
		std::vector<VecXd> dadz_;//node derivative - not including input layer (nlayer_)
	//transfer functions
		TransferN::type tfType_;//transfer function type
		std::vector<FFDVP> tffdv_;//transfer function - input for indexed layer (nlayer_)
public:
	//==== constructors/destructors ====
	ANN(){defaults();}
	~ANN(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ANN& n);
	friend FILE* operator<<(FILE* out, const ANN& n);
	friend VecXd& operator>>(const ANN& nn, VecXd& v);
	friend ANN& operator<<(ANN& nn, const VecXd& v);
	
	//==== access ====
	//network dimensions
		int nlayer()const{return nlayer_;}
	//nodes
		VecXd& in(){return in_;}
		const VecXd& in()const{return in_;}
		VecXd& out(){return out_;}
		const VecXd& out()const{return out_;}
		VecXd& node(int n){return node_[n];}
		const VecXd& node(int n)const{return node_[n];}
		int nNodes(int n)const{return node_[n].size();}
	//scaling
		VecXd& inw(){return inw_;}
		const VecXd& inw()const{return inw_;}
		VecXd& inb(){return inb_;}
		const VecXd& inb()const{return inb_;}
		VecXd& outw(){return outw_;}
		const VecXd& outw()const{return outw_;}
		VecXd& outb(){return outb_;}
		const VecXd& outb()const{return outb_;}
	//bias
		VecXd& bias(int l){return bias_[l];}
		const VecXd& bias(int l)const{return bias_[l];}
	//edge
		MatXd& edge(int l){return edge_[l];}
		const MatXd& edge(int l)const{return edge_[l];}
	//size
		int nIn()const{return in_.size();}
		int nOut()const{return out_.size();}
	//gradients - nodes
		VecXd& dadz(int n){return dadz_[n];}
		const VecXd& dadz(int n)const{return dadz_[n];}
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
		double error(const VecXd& output)const;
		double error_lambda()const;
		VecXd& dcdo(const VecXd& output, VecXd& grad)const;
		VecXd& grad_lambda(VecXd& grad)const;
	//info
		int size()const;
		int nBias()const;
		int nWeight()const;
	//resizing
		void resize(const ANNInit& init, int nInput, int nOutput);
		void resize(const ANNInit& init, int nInput, const std::vector<int>& nNodes, int nOutput);
		void resize(const ANNInit& init, const std::vector<int>& nNodes);
	//execution
		const VecXd& execute();
		const VecXd& execute(const VecXd& in){in_=in;return execute();}
		
	//==== static functions ====
	static void write(FILE* writer, const ANN& nn);
	static void write(const char*, const ANN& nn);
	static void read(FILE* writer, ANN& nn);
	static void read(const char*, ANN& nn);
};

bool operator==(const ANN& n1, const ANN& n2);
inline bool operator!=(const ANN& n1, const ANN& n2){return !(n1==n2);}

//***********************************************************************
// Cost
//***********************************************************************

/*
dcdz_ - 
	the gradient of the cost function (c) w.r.t. the node inputs (z) - dc/dz
dcdo_ -
	the gradient of the cost function (c) w.r.t. the output (o) - dc/do
*/
class Cost{
private:
	VecXd dcdo_;//gradient of cost w.r.t. output (out_.size()) 
	std::vector<VecXd> dcdz_;//derivative of cost function w.r.t. node inputs (nlayer_)
	VecXd grad_;//gradient of the cost function with respect to each parameter (bias + weight)
public:
	//==== constructors/destructors ====
	Cost(){}
	Cost(const ANN& nn){resize(nn);}
	~Cost(){}
	
	//==== access ====
	VecXd& dcdo(){return dcdo_;}
	const VecXd& dcdo()const{return dcdo_;}
	std::vector<VecXd>& dcdz(){return dcdz_;}
	const std::vector<VecXd>& dcdz()const{return dcdz_;}
	VecXd& grad(){return grad_;}
	const VecXd& grad()const{return grad_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	double error(const ANN& nn, const VecXd& out);
	const VecXd& grad(const ANN& nn, const VecXd& dcdo);
};

//***********************************************************************
// DOutDVal
//***********************************************************************

/*
doda_ -
	the derivative of the output (o) w.r.t. the value of all nodes (a)
	has a size of "nlayer_+1" as we need to compute the gradient w.r.t. all nodes
	thus, doda_[n] must of the size node_[n]
	this includes the hidden layers as well as the input/ouput layers
	note these are the scaled inputs/outputs
dodi_ -
	the derivative of the output w.r.t. the raw input
	this is the first element of doda_ multiplied by the input scaling
*/
class DOutDVal{
private:
	MatXd dodi_;//derivative of out_ w.r.t. to in_ (out_.size() x in_.size())
	std::vector<MatXd> doda_;//derivative of out_ w.r.t. to the value "a" of all nodes (nlayer_+1)
public:
	//==== constructors/destructors ====
	DOutDVal(){}
	DOutDVal(const ANN& nn){resize(nn);}
	~DOutDVal(){}
	
	//==== access ====
	MatXd& dodi(){return dodi_;}
	const MatXd& dodi()const{return dodi_;}
	MatXd& doda(int n){return doda_[n];}
	const MatXd& doda(int n)const{return doda_[n];}
	std::vector<MatXd>& doda(){return doda_;}
	const std::vector<MatXd>& doda()const{return doda_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// DOutDP
//***********************************************************************

class DOutDP{
private:
	std::vector<MatXd> dodz_;//derivative of output w.r.t. node inputs (nlayer_)
	std::vector<std::vector<MatXd> > dodw_;//derivative of output w.r.t. weights
	VecXd grad_;//gradient of the cost function with respect to each parameter (bias + weight)
public:
	//==== constructors/destructors ====
	DOutDP(){}
	DOutDP(const ANN& nn){resize(nn);}
	~DOutDP(){}
	
	//==== access ====
	MatXd& dodz(int n){return dodz_[n];}
	const MatXd& dodz(int n)const{return dodz_[n];}
	std::vector<MatXd>& dodz(){return dodz_;}
	const std::vector<MatXd>& dodz()const{return dodz_;}
	MatXd& dodb(int n){return dodz_[n];}
	const MatXd& dodb(int n)const{return dodz_[n];}
	std::vector<MatXd>& dodb(){return dodz_;}
	const std::vector<MatXd>& dodb()const{return dodz_;}
	std::vector<std::vector<MatXd> >& dodw(){return dodw_;}
	const std::vector<std::vector<MatXd> >& dodw()const{return dodw_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// D2OutDPDVal
//***********************************************************************

class D2OutDPDVal{
private:
	ANN nnc_;
	DOutDVal dOutDVal_;
	std::vector<MatXd> d2odpda_;
	MatXd pt1_,pt2_;
public:
	//==== constructors/destructors ====
	D2OutDPDVal(){}
	D2OutDPDVal(const ANN& nn){resize(nn);}
	~D2OutDPDVal(){}
	
	//==== access ====
	std::vector<MatXd>& d2odpda(){return d2odpda_;}
	const std::vector<MatXd>& d2odpda()const{return d2odpda_;}
	MatXd& d2odpda(int i){return d2odpda_[i];}
	const MatXd& d2odpda(int i)const{return d2odpda_[i];}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NeuralNet::ANNInit& obj);
	template <> int nbytes(const NeuralNet::ANN& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NeuralNet::ANNInit& obj, char* arr);
	template <> int pack(const NeuralNet::ANN& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NeuralNet::ANNInit& obj, const char* arr);
	template <> int unpack(NeuralNet::ANN& obj, const char* arr);
	
}

#endif