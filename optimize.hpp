#pragma once
#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

// c++ libraries
#include <iosfwd>
// eigen libraries
#include <Eigen/Dense>
//serialization
#include "serialize.hpp"

namespace Opt{
	
#ifndef OPT_PRINT_FUNC
#define OPT_PRINT_FUNC 0
#endif

#ifndef OPT_PRINT_DATA
#define OPT_PRINT_DATA 0
#endif

//***************************************************
// optimization method
//***************************************************

struct ALGO{
	enum type {
		UNKNOWN=0,
		SGD=1,
		SDM=2,
		NAG=3,
		ADAGRAD=4,
		ADADELTA=5,
		RMSPROP=6,
		ADAM=7,
		NADAM=8,
		BFGS=9,
		RPROP=10
	};
	static type read(const char* str);
};
std::ostream& operator<<(std::ostream& out, const ALGO::type& type);

//***************************************************
// optimization value
//***************************************************

struct VAL{
	enum type{
		UNKNOWN=0,
		FTOL_ABS=1,
		FTOL_REL=2,
		XTOL_ABS=3,
		XTOL_REL=4
	};
	static type read(const char* str);
};
std::ostream& operator<<(std::ostream& out, const VAL::type& type);

//***************************************************
// decay method
//***************************************************

struct DECAY{
	enum type{
		UNKNOWN=0,
		CONST=1,
		EXP=2,
		SQRT=3,
		INV=4,
		POW=5
	};
	static type read(const char* str);
};
std::ostream& operator<<(std::ostream& out, const VAL::type& type);

//***************************************************
// Data
//***************************************************

class Data{
private:
	//count
		unsigned int nPrint_;
		unsigned int nWrite_;
		unsigned int step_;
	//stopping
		unsigned int max_;
		double tol_;
	//status
		double val_,valOld_;
		double dv_,dp_;
	//algorithm
		ALGO::type algo_;//optimization algorithm
		VAL::type optVal_;//the type of value determining the end condition
	//parameters
		unsigned int dim_;
		Eigen::VectorXd p_,pOld_;
		Eigen::VectorXd g_,gOld_;
public:
	//==== constructors/destructors ====
	Data(){defaults();}
	Data(unsigned int dim){defaults();init(dim);}
	~Data(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Data& data);
	
	//==== access ====
	//status
		double& val(){return val_;}
		const double& val()const{return val_;}
		double& valOld(){return valOld_;}
		const double& valOld()const{return valOld_;}
		double& dv(){return dv_;}
		const double& dv()const{return dv_;}
		double& dp(){return dp_;}
		const double& dp()const{return dp_;}
	//count
		unsigned int& nPrint(){return nPrint_;}
		const unsigned int& nPrint()const{return nPrint_;}
		unsigned int& nWrite(){return nWrite_;}
		const unsigned int& nWrite()const{return nWrite_;}
		unsigned int& step(){return step_;}
		const unsigned int& step()const{return step_;}
	//stopping
		double& tol(){return tol_;}
		const double& tol()const{return tol_;}
		unsigned int& max(){return max_;}
		const unsigned int& max()const{return max_;}
	//parameters
		unsigned int& dim(){return dim_;}
		const unsigned int& dim()const{return dim_;}
		Eigen::VectorXd& p(){return p_;}
		const Eigen::VectorXd& p()const{return p_;}
		Eigen::VectorXd& pOld(){return pOld_;}
		const Eigen::VectorXd& pOld()const{return pOld_;}
		Eigen::VectorXd& g(){return g_;}
		const Eigen::VectorXd& g()const{return g_;}
		Eigen::VectorXd& gOld(){return gOld_;}
		const Eigen::VectorXd& gOld()const{return gOld_;}
	//algorithm
		ALGO::type& algo(){return algo_;}
		const ALGO::type& algo()const{return algo_;}
		VAL::type& optVal(){return optVal_;}
		const VAL::type& optVal()const{return optVal_;}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
	void init(unsigned int dim);
};

//***************************************************
// Model
//***************************************************

class Model{
protected:
	unsigned int dim_;//dimension of the problem
	ALGO::type algo_;//optimization algorithm
	DECAY::type decay_;//decay schedule
	double alpha_;//step decay constant
	double pow_;//step decay power
	double gamma_;//gradient step size
	double gamma0_;
public:
	//==== constructors/destructors ====
	Model(){defaults();}
	virtual ~Model(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Model& model);
	
	//==== access ====
	unsigned int& dim(){return dim_;}
	const unsigned int& dim()const{return dim_;}
	ALGO::type& algo(){return algo_;}
	const ALGO::type& algo()const{return algo_;}
	DECAY::type& decay(){return decay_;}
	const DECAY::type& decay()const{return decay_;}
	double& gamma(){return gamma_;}
	const double& gamma()const{return gamma_;}
	double& gamma0(){return gamma0_;}
	const double& gamma0()const{return gamma0_;}
	double& alpha(){return alpha_;}
	const double& alpha()const{return alpha_;}
	double& pow(){return pow_;}
	const double& pow()const{return pow_;}
	
	//==== member functions ====
	void defaults();
	void clear();
	void update_step(unsigned int step);
	
	//==== virtual functions ====
	virtual void step(Data& d)=0;
	virtual void init(unsigned int dim);
};

//steepest-desccent
class SGD: public Model{
public:
	//constructors/destructors
	SGD(){defaults();}
	SGD(unsigned int dim){init(dim);}
	~SGD(){}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const SGD& sgd);
};

//steepest-descent + momentum
class SDM: public Model{
private:
	double eta_;//mixing term
	Eigen::VectorXd dx_;//change in parameters
public:
	//constructors/destructors
	SDM(){defaults();}
	SDM(unsigned int dim){init(dim);}
	~SDM(){}
	//access
	double& eta(){return eta_;}
	const double& eta()const{return eta_;}
	Eigen::VectorXd& dx(){return dx_;}
	const Eigen::VectorXd& dx()const{return dx_;}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const SDM& sdm);
};

//nesterov accelerated gradient
class NAG: public Model{
private:
	double eta_;//mixing term
	Eigen::VectorXd dx_;
public:
	//constructors/destructors
	NAG(){defaults();}
	NAG(unsigned int dim){init(dim);}
	~NAG(){}
	//access
	double& eta(){return eta_;}
	const double& eta()const{return eta_;}
	Eigen::VectorXd& dx(){return dx_;}
	const Eigen::VectorXd& dx()const{return dx_;}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const NAG& nag);
};

//adagrad
class ADAGRAD: public Model{
private:
	static const double eps_;//small term to prevent divergence
	Eigen::VectorXd mgrad2_;//avg of square of gradient
public:
	//constructors/destructors
	ADAGRAD(){defaults();}
	ADAGRAD(unsigned int dim){init(dim);}
	~ADAGRAD(){}
	//access
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const ADAGRAD& adagrad);
};

//adadelta
class ADADELTA: public Model{
private:
	static const double eps_;//small term to prevent divergence
	double eta_;//mixing fraction
	Eigen::VectorXd mgrad2_;//avg of square of gradient
	Eigen::VectorXd mdx2_;//avg of square of dx
	Eigen::VectorXd dx_;//change in x
public:
	//constructors/destructors
	ADADELTA(){defaults();}
	ADADELTA(unsigned int dim){init(dim);}
	~ADADELTA(){}
	//access
	double& eta(){return eta_;}
	const double& eta()const{return eta_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	Eigen::VectorXd& mdx2(){return mdx2_;}
	const Eigen::VectorXd& mdx2()const{return mdx2_;}
	Eigen::VectorXd& dx(){return dx_;}
	const Eigen::VectorXd& dx()const{return dx_;}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const ADADELTA& adadelta);
};

//rmsprop
class RMSPROP: public Model{
private:
	static const double eps_;//small term to prevent divergence
	Eigen::VectorXd mgrad2_;//avg of square of gradient
public:
	//constructors/destructors
	RMSPROP(){defaults();}
	RMSPROP(unsigned int dim){init(dim);}
	~RMSPROP(){}
	//access
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const RMSPROP& rmsprop);
};

//adam
class ADAM: public Model{
private:
	static const double eps_;//small term to prevent divergence
	static const double beta1_;
	static const double beta2_;
	double beta1i_;//power w.r.t i
	double beta2i_;//power w.r.t i
	Eigen::VectorXd mgrad_;//avg of gradient
	Eigen::VectorXd mgrad2_;//avg of square of gradient
public:
	//constructors/destructors
	ADAM(){defaults();}
	ADAM(unsigned int dim){init(dim);}
	~ADAM(){}
	//access
	double& beta1i(){return beta1i_;}
	const double& beta1i()const{return beta1i_;}
	double& beta2i(){return beta2i_;}
	const double& beta2i()const{return beta2i_;}
	Eigen::VectorXd& mgrad(){return mgrad_;}
	const Eigen::VectorXd& mgrad()const{return mgrad_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const ADAM& adam);
};

//nadam
class NADAM: public Model{
private:
	static const double eps_;//small term to prevent divergence
	static const double beta1_;
	static const double beta2_;
	double beta1i_;//power w.r.t i
	double beta2i_;//power w.r.t i
	Eigen::VectorXd mgrad_;//avg of gradient
	Eigen::VectorXd mgrad2_;//avg of square of gradient
public:
	//constructors/destructors
	NADAM(){defaults();}
	NADAM(unsigned int dim){init(dim);}
	~NADAM(){}
	//access
	double& beta1i(){return beta1i_;}
	const double& beta1i()const{return beta1i_;}
	double& beta2i(){return beta2i_;}
	const double& beta2i()const{return beta2i_;}
	Eigen::VectorXd& mgrad(){return mgrad_;}
	const Eigen::VectorXd& mgrad()const{return mgrad_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const NADAM& nadam);
};

//bfgs
class BFGS: public Model{
private:
	Eigen::MatrixXd B_,BOld_;
	Eigen::VectorXd s_,y_;
public:
	//constructors/destructors
	BFGS(){defaults();}
	BFGS(unsigned int dim){init(dim);}
	~BFGS(){}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const BFGS& bfgs);
};

//rprop
class RPROP: public Model{
private:
	static const double etaP;
	static const double etaM;
	static const double deltaMax;
	static const double deltaMin;
	Eigen::VectorXd delta_;
	Eigen::VectorXd dx_;
public:
	//constructors/destructors
	RPROP(){defaults();}
	RPROP(unsigned int dim){init(dim);}
	~RPROP(){}
	//access
	Eigen::VectorXd& delta(){return delta_;}
	const Eigen::VectorXd& delta()const{return delta_;}
	Eigen::VectorXd& dx(){return dx_;}
	const Eigen::VectorXd& dx()const{return dx_;}
	//member functions
	void step(Data& d);
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const RPROP& rprop);
};
/*
	Christian Igel and Michael Hüsken. 
		Improving the Rprop Learning Algorithm. 
		Second International Symposium on Neural Computation 
		(NC 2000), pp. 115-121, ICSC Academic Press, 2000
	Christian Igel and Michael Hüsken. 
		Empirical Evaluation of the Improved Rprop Learning Algorithm. 
		Neurocomputing 50:105-123, 2003
*/

//read from file

Model& read(Model& model, const char* file);
Data& read(Data& data, const char* file);
SGD& read(SGD& sdg, const char* file);
SDM& read(SDM& sdm, const char* file);
NAG& read(NAG& nag, const char* file);
ADAGRAD& read(ADAGRAD& adagrad, const char* file);
ADADELTA& read(ADADELTA& adadelta, const char* file);
RMSPROP& read(RMSPROP& rmsprop, const char* file);
ADAM& read(ADAM& adam, const char* file);
NADAM& read(NADAM& nadam, const char* file);
BFGS& read(BFGS& bfgs, const char* file);
RPROP& read(RPROP& rprop, const char* file);

//read from file pointer

Model& read(Model& model, FILE* reader);
Data& read(Data& data, FILE* reader);
SGD& read(SGD& sdg, FILE* reader);
SDM& read(SDM& sdm, FILE* reader);
NAG& read(NAG& nag, FILE* reader);
ADAGRAD& read(ADAGRAD& adagrad, FILE* reader);
ADADELTA& read(ADADELTA& adadelta, FILE* reader);
RMSPROP& read(RMSPROP& rmsprop, FILE* reader);
ADAM& read(ADAM& adam, FILE* reader);
NADAM& read(NADAM& nadam, FILE* reader);
BFGS& read(BFGS& bfgs, FILE* reader);
RPROP& read(RPROP& rprop, FILE* reader);

//opterators - comparison

bool operator==(const SGD& obj1, const SGD& obj2);
bool operator==(const SDM& obj1, const SDM& obj2);
bool operator==(const NAG& obj1, const NAG& obj2);
bool operator==(const ADAGRAD& obj1, const ADAGRAD& obj2);
bool operator==(const ADADELTA& obj1, const ADADELTA& obj2);
bool operator==(const RMSPROP& obj1, const RMSPROP& obj2);
bool operator==(const ADAM& obj1, const ADAM& obj2);
bool operator==(const NADAM& obj1, const NADAM& obj2);
bool operator==(const BFGS& obj1, const BFGS& obj2);
bool operator==(const RPROP& obj1, const RPROP& obj2);

}

namespace serialize{

	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const Opt::Data& obj);
	template <> unsigned int nbytes(const Opt::Model& obj);
	template <> unsigned int nbytes(const Opt::SGD& obj);
	template <> unsigned int nbytes(const Opt::SDM& obj);
	template <> unsigned int nbytes(const Opt::NAG& obj);
	template <> unsigned int nbytes(const Opt::ADAGRAD& obj);
	template <> unsigned int nbytes(const Opt::ADADELTA& obj);
	template <> unsigned int nbytes(const Opt::RMSPROP& obj);
	template <> unsigned int nbytes(const Opt::ADAM& obj);
	template <> unsigned int nbytes(const Opt::NADAM& obj);
	template <> unsigned int nbytes(const Opt::BFGS& obj);
	template <> unsigned int nbytes(const Opt::RPROP& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> unsigned int pack(const Opt::Data& obj, char* arr);
	template <> unsigned int pack(const Opt::Model& obj, char* arr);
	template <> unsigned int pack(const Opt::SGD& obj, char* arr);
	template <> unsigned int pack(const Opt::SDM& obj, char* arr);
	template <> unsigned int pack(const Opt::NAG& obj, char* arr);
	template <> unsigned int pack(const Opt::ADAGRAD& obj, char* arr);
	template <> unsigned int pack(const Opt::ADADELTA& obj, char* arr);
	template <> unsigned int pack(const Opt::RMSPROP& obj, char* arr);
	template <> unsigned int pack(const Opt::ADAM& obj, char* arr);
	template <> unsigned int pack(const Opt::NADAM& obj, char* arr);
	template <> unsigned int pack(const Opt::BFGS& obj, char* arr);
	template <> unsigned int pack(const Opt::RPROP& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> unsigned int unpack(Opt::Data& obj, const char* arr);
	template <> unsigned int unpack(Opt::Model& obj, const char* arr);
	template <> unsigned int unpack(Opt::SGD& obj, const char* arr);
	template <> unsigned int unpack(Opt::SDM& obj, const char* arr);
	template <> unsigned int unpack(Opt::NAG& obj, const char* arr);
	template <> unsigned int unpack(Opt::ADAGRAD& obj, const char* arr);
	template <> unsigned int unpack(Opt::ADADELTA& obj, const char* arr);
	template <> unsigned int unpack(Opt::RMSPROP& obj, const char* arr);
	template <> unsigned int unpack(Opt::ADAM& obj, const char* arr);
	template <> unsigned int unpack(Opt::NADAM& obj, const char* arr);
	template <> unsigned int unpack(Opt::BFGS& obj, const char* arr);
	template <> unsigned int unpack(Opt::RPROP& obj, const char* arr);
	
}

#endif