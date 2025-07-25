#pragma once
#ifndef NN_HPP
#define NN_HPP

#define EIGEN_NO_DEBUG
//#define EIGEN_USE_MKL_ALL

// c++ libraries
#include <iosfwd>
// eigen
#include <Eigen/Dense>
// math
#include "math/random.hpp"
// mem
#include "mem/serialize.hpp"

typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecXd;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatXd;

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
// FORWARD DECLARATIONS
//***********************************************************************

class ANN;
class ANNP;

//***********************************************************************
// INITIALIZATION METHOD
//***********************************************************************

/**
* Initialization scheme for the neural network weights.
*/
class Init{
public:
	//enum
	enum Type{
		UNKNOWN,
		RAND,
		LECUN,
		HE,
		XAVIER
	};
	//constructor
	Init():t_(Type::UNKNOWN){}
	Init(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Init read(const char* str);
	static const char* name(const Init& init);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Init& init);

//***********************************************************************
// Neuron
//***********************************************************************

class Neuron{
public:
	//type
	enum Type{
		UNKNOWN,
		//linear
		LINEAR,
		//sigmoidal
		SIGMOID,
		TANH,
		ISRU,
		ARCTAN,
		RELU,
		ELU,
		TANHRE,
		SQRE,
		//gated-switch
		SWISH,
		GELU,
		MISH,
		PFLU,
		SERF,
		LOGISH,
		//switch
		SOFTPLUS,
		SQPLUS,
		ATISH,
		//test
		TEST
	};
	//constructor
	Neuron():t_(Type::UNKNOWN){}
	Neuron(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Neuron read(const char* str);
	static const char* name(const Neuron& neuron);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Neuron& neuron);

//***********************************************************************
// AF_FP - Activation Function - Forward Pass
//***********************************************************************

class AFFP{
public:
	//linear
	static void af_lin(double c, const VecXd& z, VecXd& a);
	//sigmoidal
	static void af_sigmoid(double c, const VecXd& z, VecXd& a);
	static void af_tanh(double c, const VecXd& z, VecXd& a);
	static void af_isru(double c, const VecXd& z, VecXd& a);
	static void af_arctan(double c, const VecXd& z, VecXd& a);
	static void af_relu(double c, const VecXd& z, VecXd& a);
	static void af_elu(double c, const VecXd& z, VecXd& a);
	static void af_tanhre(double c, const VecXd& z, VecXd& a);
	static void af_sqre(double c, const VecXd& z, VecXd& a);
	//gated-switch
	static void af_swish(double c, const VecXd& z, VecXd& a);
	static void af_gelu(double c, const VecXd& z, VecXd& a);
	static void af_mish(double c, const VecXd& z, VecXd& a);
	static void af_pflu(double c, const VecXd& z, VecXd& a);
	static void af_serf(double c, const VecXd& z, VecXd& a);
	static void af_logish(double c, const VecXd& z, VecXd& a);
	//switch
	static void af_softplus(double c, const VecXd& z, VecXd& a);
	static void af_sqplus(double c, const VecXd& z, VecXd& a);
	static void af_atish(double c, const VecXd& z, VecXd& a);
	//test
	static void af_test(double c, const VecXd& z, VecXd& a);
};

//***********************************************************************
// AF_FPBP - Activation Function - Forward Pass + Backward Pass
//***********************************************************************

class AFFPBP{
public:
	//linear
	static void af_lin(double c, const VecXd& z, VecXd& a, VecXd& d);
	//sigmoidal
	static void af_sigmoid(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_tanh(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_isru(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_arctan(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_relu(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_elu(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_tanhre(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_sqre(double c, const VecXd& z, VecXd& a, VecXd& d);
	//gated-switch
	static void af_swish(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_gelu(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_mish(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_pflu(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_serf(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_logish(double c, const VecXd& z, VecXd& a, VecXd& d);
	//switch
	static void af_softplus(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_sqplus(double c, const VecXd& z, VecXd& a, VecXd& d);
	static void af_atish(double c, const VecXd& z, VecXd& a, VecXd& d);
	//test
	static void af_test(double c, const VecXd& z, VecXd& a, VecXd& d);
};

//***********************************************************************
// AF_FPBP2 - Activation Function - Forward Pass + Backward Pass + 2nd derivative
//***********************************************************************

class AFFPBP2{
public:
	//linear
	static void af_lin(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	//sigmoidal
	static void af_sigmoid(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_tanh(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_isru(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_arctan(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_relu(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_elu(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_tanhre(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_sqre(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	//gated-switch
	static void af_swish(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_gelu(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_mish(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_pflu(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_serf(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_logish(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	//switch
	static void af_softplus(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_sqplus(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	static void af_atish(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
	//test
	static void af_test(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2);
};

//***********************************************************************
// ANN
//***********************************************************************

/**
* Class defining the function and parameters of an artificial neural network.
* The neural network is defined by several layers of nodes, each represented by a vector.
* Each set of nodes in a given layer is fully connected to the nodes in adjacent layers
* by a set of edges with associated weights, represented as a matrix.
* Each node has an associated input, bias, and value.  Given a set of inputs to the network,
* the input for each node is determined by the product of the weight matrix by the values
* of the previous layer added to the bias.  The value of each node is then determined by
* the application of a "transfer" or "activation" function on the node input.
*
* Notations (used throughout this compilation unit):
* c - "c" donotes the cost function, e.g. the gradient of the cost function w.r.t. the value of a node is dc/da
* z - "z" is the input to each node, e.g. the gradient of a node w.r.t. to its input is da/dz
* a - "a" is the value of a node, e.g. the gradient of a node w.r.t. to its input is da/dz
* o - "o" is the output of the network, e.g. the gradient of the output w.r.t. the input is do/di
* i - "i" is the input of the network, e.g. the gradient of the output w.r.t. the input is do/di
*/
class ANN{
private:
	//typedefs
		typedef void (*fAFFP)(double,const VecXd&,VecXd&);
		typedef void (*fAFFPBP)(double,const VecXd&,VecXd&,VecXd&);
		typedef void (*fAFFPBP2)(double,const VecXd&,VecXd&,VecXd&,VecXd&);
	//layers
		int nlayer_;//number of layers (weights,biases)
		double c_;//sharpness parameter
	//transfer functions
		Neuron neuron_;//transfer function type
		std::vector<fAFFP> affp_;//transfer functions
		std::vector<fAFFPBP> affpbp_;//transfer functions
		std::vector<fAFFPBP2> affpbp2_;//transfer functions
	//input/output
		VecXd inp_;//input - raw
		VecXd ins_;//input - scaled and shifted
		VecXd out_;//output
		VecXd inpw_,inpb_;//input weight, bias
		VecXd outw_,outb_;//output weight, bias
	//gradients - nodes
		std::vector<VecXd> dadz_;//transfer function first derivative
		std::vector<VecXd> d2adz2_;//transfer function second derivative
	//node weights and biases
		std::vector<VecXd> a_;//node values
		std::vector<VecXd> z_;//node inputs
		std::vector<VecXd> b_;//biases
		std::vector<MatXd> w_;//weights
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
		double& c(){return c_;}
		const double& c()const{return c_;}
	//input/output
		VecXd& inp(){return inp_;}
		const VecXd& inp()const{return inp_;}
		VecXd& ins(){return ins_;}
		const VecXd& ins()const{return ins_;}
		VecXd& out(){return out_;}
		const VecXd& out()const{return out_;}
	//scaling
		VecXd& inpw(){return inpw_;}
		const VecXd& inpw()const{return inpw_;}
		VecXd& inpb(){return inpb_;}
		const VecXd& inpb()const{return inpb_;}
		VecXd& outw(){return outw_;}
		const VecXd& outw()const{return outw_;}
		VecXd& outb(){return outb_;}
		const VecXd& outb()const{return outb_;}
	//nodes
		VecXd& a(int l){return a_[l];}
		const VecXd& a(int l)const{return a_[l];}
		const std::vector<VecXd>& a()const{return a_;}
		VecXd& z(int l){return z_[l];}
		const VecXd& z(int l)const{return z_[l];}
		const std::vector<VecXd>& z()const{return z_;}
	//bias
		VecXd& b(int l){return b_[l];}
		const VecXd& b(int l)const{return b_[l];}
		const std::vector<VecXd>& b()const{return b_;}
	//edge
		MatXd& w(int l){return w_[l];}
		const MatXd& w(int l)const{return w_[l];}
		const std::vector<MatXd>& w()const{return w_;}
	//size
		int nInp()const{return inp_.size();}
		int nOut()const{return out_.size();}
		int nNodes(int n)const{return a_[n].size();}
	//gradients - nodes
		VecXd& dadz(int n){return dadz_[n];}
		const VecXd& dadz(int n)const{return dadz_[n];}
		const std::vector<VecXd>& dadz()const{return dadz_;}
		VecXd& d2adz2(int n){return d2adz2_[n];}
		const VecXd& d2adz2(int n)const{return d2adz2_[n];}
		const std::vector<VecXd>& d2adz2()const{return d2adz2_;}
	//transfer functions
		Neuron& neuron(){return neuron_;}
		const Neuron& neuron()const{return neuron_;}
		fAFFP affp(int l){return affp_[l];}
		const fAFFP affp(int l)const{return affp_[l];}
		fAFFPBP affpbp(int l){return affpbp_[l];}
		const fAFFPBP affpbp(int l)const{return affpbp_[l];}
		fAFFPBP2 affpbp2(int l){return affpbp2_[l];}
		const fAFFPBP2 affpbp2(int l)const{return affpbp2_[l];}
		
	//==== member functions ====
	//clearing/initialization
		void defaults();
		void clear();
	//info
		int size()const;
		int nBias()const;
		int nWeight()const;
	//resizing
		void resize(const ANNP& init, int nInput, int nOutput);
		void resize(const ANNP& init, int nInput, const std::vector<int>& nNodes, int nOutput);
		void resize(const ANNP& init, const std::vector<int>& nNodes);
		void resize(const ANNP& init, int nInput, const std::vector<int>& nNodes);
		void reset(const ANNP& init);
	//error
		double error_lambda()const;
		VecXd& grad_lambda(VecXd& grad)const;
	//execution
		const VecXd& fp();
		const VecXd& fpbp();
		const VecXd& fpbp2();
		const VecXd& fp(const VecXd& inp){inp_=inp;return fp();}
		const VecXd& fpbp(const VecXd& inp){inp_=inp;return fpbp();}
		const VecXd& fpbp2(const VecXd& inp){inp_=inp;return fpbp2();}
		
	//==== static functions ====
	static void write(FILE* writer, const ANN& nn);
	static void write(const char*, const ANN& nn);
	static void read(FILE* writer, ANN& nn);
	static void read(const char*, ANN& nn);
};

//***********************************************************************
// ANNP
//***********************************************************************

/**
* Class which stores Artificial Neural Network Parameters (ANNP).
* These include a set of loosely related parameters which are important
* for initializing a neural network but do not necessarily define the network 
* and thus do not need to be stored with it.
*/
class ANNP{
private:
	int seed_;//random seed	
	rng::dist::Name dist_b_;//distribution type - bias
	rng::dist::Name dist_w_;//distribution type - weight
	Init init_;//initialization scheme
	Neuron neuron_;//nueron type
	double sigma_b_;//distribution size parameter - bias
	double sigma_w_;//distribution size parameter - weight
	double c_;//sharpness parameter
public:
	//==== constructors/destructors ====
	ANNP(){defaults();}
	~ANNP(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ANNP& init);
	
	//==== access ====
	int& seed(){return seed_;}
	const int& seed()const{return seed_;}
	rng::dist::Name& dist_b(){return dist_b_;}
	const rng::dist::Name& dist_b()const{return dist_b_;}
	rng::dist::Name& dist_w(){return dist_w_;}
	const rng::dist::Name& dist_w()const{return dist_w_;}
	Init& init(){return init_;}
	const Init& init()const{return init_;}
	Neuron& neuron(){return neuron_;}
	const Neuron& neuron()const{return neuron_;}
	double& sigma_b(){return sigma_b_;}
	const double& sigma_b()const{return sigma_b_;}
	double& sigma_w(){return sigma_w_;}
	const double& sigma_w()const{return sigma_w_;}
	double& c(){return c_;}
	const double& c()const{return c_;}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
	
	//==== static functions ====
	static void read(const char*, ANNP& annp);
	static void read(FILE* writer, ANNP& annp);
};

//***********************************************************************
// Cost
//***********************************************************************

/**
* Class for computing the derivative of a given cost function (c)
* with respect to the weights and biases of the ntework.
*/
class Cost{
private:
	std::vector<VecXd> dcdz_;//derivative of cost function (c) w.r.t. node inputs (z) (dc/dz)
	VecXd grad_;//gradient of the cost function with respect to each parameter (bias + weight)
public:
	//==== constructors/destructors ====
	Cost(){}
	Cost(const ANN& nn){resize(nn);}
	~Cost(){}
	
	//==== access ====
	std::vector<VecXd>& dcdz(){return dcdz_;}
	const std::vector<VecXd>& dcdz()const{return dcdz_;}
	VecXd& grad(){return grad_;}
	const VecXd& grad()const{return grad_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	const VecXd& grad(const ANN& nn, const VecXd& dcdo);
};

//***********************************************************************
// DODZ
//***********************************************************************

/**
* Class for computing the derivative of network output (o) with respect
* to the node inputs (z)
*/
class DODZ{
private:
	MatXd dodi_;//gradient of the output (o) w.r.t. the network inputs (i) (do/di)
	std::vector<MatXd> dodz_;//gradient of the output (o) w.r.t. node inputs (z) (do/dz)
public:
	//==== constructors/destructors ====
	DODZ(){}
	DODZ(const ANN& nn){resize(nn);}
	~DODZ(){}
	
	//==== access ====
	MatXd& dodi(){return dodi_;}
	const MatXd& dodi()const{return dodi_;}
	MatXd& dodz(int n){return dodz_[n];}
	const MatXd& dodz(int n)const{return dodz_[n];}
	std::vector<MatXd>& dodz(){return dodz_;}
	const std::vector<MatXd>& dodz()const{return dodz_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// DODP
//***********************************************************************

/**
* Class for computing the derivative of network output (o) with respect
* to the parameters (p) of the network (i.e. weights and biases)
*/
class DODP{
private:
	std::vector<MatXd> dodz_;//derivative of output w.r.t. node inputs (nlayer_)
	std::vector<std::vector<VecXd> > dodb_;//derivative of output w.r.t. biases
	std::vector<std::vector<MatXd> > dodw_;//derivative of output w.r.t. weights
	MatXd dodp_;//(nOut,nBias+nWeight)
public:
	//==== constructors/destructors ====
	DODP(){}
	DODP(const ANN& nn){resize(nn);}
	~DODP(){}
	
	//==== access ====
	//dodz
	std::vector<MatXd>& dodz(){return dodz_;}
	const std::vector<MatXd>& dodz()const{return dodz_;}
	//dodb
	std::vector<std::vector<VecXd> >& dodb(){return dodb_;}
	const std::vector<std::vector<VecXd> >& dodb()const{return dodb_;}
	//dodw
	std::vector<std::vector<MatXd> >& dodw(){return dodw_;}
	const std::vector<std::vector<MatXd> >& dodw()const{return dodw_;}
	//dodp
	MatXd& dodp(){return dodp_;}
	const MatXd& dodp()const{return dodp_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// DZDI
//***********************************************************************

/**
* Class for computing the derivative of the node inputs (z) with respect
* to the inputs to the network (i).
*/
class DZDI{
private:
	std::vector<MatXd> dzdi_;//gradient of (z) w.r.t. input (i) (dz/di)
public:
	//==== constructors/destructors ====
	DZDI(){}
	DZDI(const ANN& nn){resize(nn);}
	~DZDI(){}
	
	//==== access ====
	const std::vector<MatXd>& dzdi(){return dzdi_;}
	MatXd& dzdi(int n){return dzdi_[n];}
	const MatXd& dzdi(int n)const{return dzdi_[n];}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// D2ODZDI
//***********************************************************************

/**
* Class for computing the second derivative of the outputs (o) of a 
* nueral network with respect to the node inputs (z) and the network
* inputs (i), thus d2o/dzdi.
*/
class D2ODZDI{
private:
	DODZ dOdZ_;//gradient of the output (o) w.r.t. node inputs (z) (do/dz)
	DZDI dZdI_;//gradient of the node inputs (z) w.r.t the network input (i) (dz/di)
	std::vector<std::vector<MatXd> > d2odzdi_;//gradient of (o) w.r.t (z) and (i) (d2o/dzdi)
	std::vector<std::vector<MatXd> > d2odbdi_;//gradient of (o) w.r.t (b) and (i) (d2o/dbdi)
	std::vector<std::vector<std::vector<MatXd> > > d2odwdi_;//gradient of (o) w.r.t (w) and (i) (d2o/dwdi)
	std::vector<MatXd> d2odpdi_;//gradient of (o) w.r.t (p) and (i) (d2o/dpdi)
public:
	//==== constructors/destructors ====
	D2ODZDI(){}
	D2ODZDI(const ANN& nn){resize(nn);}
	~D2ODZDI(){}
	
	//==== access ====
	std::vector<std::vector<MatXd> >& d2odzdi(){return d2odzdi_;}
	std::vector<std::vector<MatXd> >& d2odbdi(){return d2odbdi_;}
	std::vector<std::vector<std::vector<MatXd> > >& d2odwdi(){return d2odwdi_;}
	std::vector<MatXd>& d2odpdi(){return d2odpdi_;}
	
	//==== member functions ====
	void clear();
	void resize(const ANN& nn);
	void grad(const ANN& nn);
};

//***********************************************************************
// D2ODZDIN
//***********************************************************************

/**
* Class for computing the second derivative of the outputs (o) of a 
* nueral network with respect to the node inputs (z) and the network
* inputs (i), thus d2o/dzdi.
* The gradient is computed in a brute force manner,
* i.e. numerical differention.
* Note: this is very slow and used only for testing purposes.
*/
class D2ODZDIN{
private:
	ANN nnc_;//neural network copy
	DODZ dOdZ_;//gradient of the output (o) w.r.t. node inputs (z) (do/dz)
	std::vector<MatXd> d2odpdi_;//gradient of (o) w.r.t (p) and (i) (d2o/dpdi)
	MatXd pt1_,pt2_;
public:
	//==== constructors/destructors ====
	D2ODZDIN(){}
	D2ODZDIN(const ANN& nn){resize(nn);}
	~D2ODZDIN(){}
	
	//==== access ====
	std::vector<MatXd>& d2odpdi(){return d2odpdi_;}
	const std::vector<MatXd>& d2odpdi()const{return d2odpdi_;}
	MatXd& d2odpdi(int i){return d2odpdi_[i];}
	const MatXd& d2odpdi(int i)const{return d2odpdi_[i];}
	
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
	
	template <> int nbytes(const NN::ANNP& obj);
	template <> int nbytes(const NN::ANN& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NN::ANNP& obj, char* arr);
	template <> int pack(const NN::ANN& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NN::ANNP& obj, const char* arr);
	template <> int unpack(NN::ANN& obj, const char* arr);
	
}

#endif