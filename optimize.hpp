#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

// c libraries
#include <cstdlib>
#include <cmath>
// c++ libraries
#include <iostream>
#include <string>
#include <limits>
#include <functional>
// eigen libraries
#include <Eigen/Dense>
#include "eigen.hpp"
// local libraries - math
#include "math_const.hpp"
#include "math_cmp.hpp"
#include "math_special.hpp"
//serialization
#include "serialize.hpp"

#ifndef PRINT_OPT_FUNC
#define PRINT_OPT_FUNC 0
#endif

#ifndef PRINT_OPT_DATA
#define PRINT_OPT_DATA 0
#endif

//***************************************************
// optimization method
//***************************************************

struct OPT_METHOD{
	enum type {
		SGD,
		SDM,
		NAG,
		ADAGRAD,
		ADADELTA,
		RMSPROP,
		ADAM,
		BFGS,
		LM,
		RPROP,
		UNKNOWN
	};
	static type load(const char* str);
};

std::ostream& operator<<(std::ostream& out, const OPT_METHOD::type& type);

//***************************************************
// optimization value
//***************************************************

struct OPT_VAL{
	enum type{
		XTOL_REL,
		XTOL_ABS,
		FTOL_REL,
		FTOL_ABS,
		UNKNOWN
	};
	static type load(const char* str);
};

std::ostream& operator<<(std::ostream& out, const OPT_VAL::type& type);

//***************************************************
// Opt
//***************************************************

class Opt{
public:
	typedef const std::function<double (const Eigen::VectorXd& x, Eigen::VectorXd& grad)> Func;
	template <class T> using FuncT=std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>;
protected:
	/*optimization*/
	//status
		unsigned int nPrint_;//frequency of printing
		unsigned int nStep_;//the number of steps
		unsigned int nEval_;//the number of evaluations
	//stopping
		double tol_;//optimization tolerance
		unsigned int maxIter_;//the max number of iterations
	//parameters
		unsigned int dim_;//dimension of the problem
		double val_,valOld_;//value of the objective function
		Eigen::VectorXd x_,xOld_;//parameters
		Eigen::VectorXd grad_,gradOld_;//gradient
	//algorithm
		OPT_METHOD::type algo_;//optimization algorithm
		OPT_VAL::type optVal_;//the type of value determining the end condition
	//line searching
		double precln_;//precision for line searches
		unsigned int maxln_;//max for line searches
		Eigen::VectorXd a_,b_,c_,d_;//for line search
public:
	//constructors/destructors
	Opt(){defaults();};
	~Opt(){};
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const Opt& opt);
	
	/*access*/
	//status
		unsigned int& nPrint(){return nPrint_;};
		const unsigned int& nPrint()const{return nPrint_;};
		unsigned int& nStep(){return nStep_;};
		const unsigned int& nStep()const{return nStep_;};
		unsigned int& nEval(){return nEval_;};
		const unsigned int& nEval()const{return nEval_;};
		double& val(){return val_;};
		const double& val()const{return val_;};
		double& valOld(){return valOld_;};
		const double& valOld()const{return valOld_;};
	//stopping
		double& tol(){return tol_;};
		const double& tol()const{return tol_;};
		unsigned int& maxIter(){return maxIter_;};
		const unsigned int& maxIter()const{return maxIter_;};
	//parameters
		unsigned int& dim(){return dim_;};
		const unsigned int& dim()const{return dim_;};
		Eigen::VectorXd& x(){return x_;};
		const Eigen::VectorXd& x()const{return x_;};
		Eigen::VectorXd& xOld(){return xOld_;};
		const Eigen::VectorXd& xOld()const{return xOld_;};
		Eigen::VectorXd& grad(){return grad_;};
		const Eigen::VectorXd& grad()const{return grad_;};
		Eigen::VectorXd& gradOld(){return gradOld_;};
		const Eigen::VectorXd& gradOld()const{return gradOld_;};
	//algorithm
		OPT_METHOD::type& algo(){return algo_;};
		const OPT_METHOD::type& algo()const{return algo_;};
		OPT_VAL::type& optVal(){return optVal_;};
		const OPT_VAL::type& optVal()const{return optVal_;};
	//linear optimization
		double& precln(){return precln_;};
		const double& precln()const{return precln_;};
		unsigned int& maxln(){return maxln_;};
		const unsigned int& maxln()const{return maxln_;};
	
	//member functions
	void defaults();
	void clear(){defaults();};
	void resize(unsigned int n);
	
	//optimization functions
	double opt_ln(Func& func);
	double opt_ln(Func& func, Eigen::VectorXd& x0, Eigen::VectorXd& x1);
	template <class T> double opt_ln(const FuncT<T>& func, T& obj);
	double opts(const Func& func, Eigen::VectorXd& x0);
	double opts(const Func& func, Eigen::VectorXd& x0, Eigen::VectorXd& grad, double& val);
	template <class T> double opts(const FuncT<T>& func, T& obj, Eigen::VectorXd& x0);
	template <class T> double opts(const FuncT<T>& func, T& obj, Eigen::VectorXd& x0, Eigen::VectorXd& grad, double& val);
	void opts_impl(const Func& func);
	template <class T> void opts_impl(const FuncT<T>& func, T& obj);
	
	//virtual functions
	virtual void step(){}; 
	virtual void init(unsigned int dim){};
};

template <class T>
double Opt::opt_ln(const FuncT<T>& func, T& obj){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::opt_ln<T>(const FuncT<T>&,T&):\n";
	a_.noalias()=x_; b_.noalias()=xOld_;
	double va=func(obj,a_,grad_);
	double vb=func(obj,b_,grad_);
	c_.noalias()=b_-(b_-a_)*1.0/num_const::PHI;
	d_.noalias()=a_+(b_-a_)*1.0/num_const::PHI;
	double vc,vd;
	unsigned int count=0;
	while((c_-d_).norm()>precln_ && count<maxln_){
		//calculate the new function values
		vc=func(obj,c_,grad_);
		vd=func(obj,d_,grad_);
		//check the values to find min
		if(vc<vd) {b_.noalias()=d_; vb=vd;}
		else {a_.noalias()=c_; va=vc;}
		//recompute c/d to prevent loss of precision
		c_.noalias()=b_-(b_-a_)*1.0/num_const::PHI;
		d_.noalias()=a_+(b_-a_)*1.0/num_const::PHI;
		++count;
	}
	nEval_+=count*2;
	if(count==maxln_) std::cout<<"WARNING: Could not resolve line optimization.\n";
	if(vc<vd) x_.noalias()=c_;
	else x_.noalias()=d_;
	if(PRINT_OPT_DATA>1){
		std::cout<<"count_ln = "<<count<<"\n";
		std::cout<<"val_ln = "<<((vc<vd)?vc:vd)<<"\n";
	}
	return (vc<vd)?vc:vd;
}

template <class T>
double Opt::opts(const FuncT<T>& func, T& obj, Eigen::VectorXd& x0){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::opts<T>(const FuncT<T>&,T&,Eigen::VectorXd&):\n";
	//initialization/resizing
	resize(x0.size());
	x_=x0; 
	xOld_=x0;
	init(dim_);
	opts_impl(func,obj);
	//finalization
	x0=x_;
	//return value
	return val_;
}

template <class T>
double Opt::opts(const FuncT<T>& func, T& obj, Eigen::VectorXd& x0, Eigen::VectorXd& grad, double& val){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::opts(const FuncT<T>&,T&,Eigen::VectorXd&,Eigen::VectorXd&,double&):\n";
	//initialization/resizing
	resize(x0.size());
	if(grad.size()==0) grad=Eigen::VectorXd::Zero(dim_);
	else if(x0.size()!=grad.size()) throw std::runtime_error("Invalid initial gradient.");
	x_=x0; 
	xOld_=x0;
	grad_=grad;
	gradOld_=grad;
	init(dim_);
	//optimization
	opts_impl(func,obj);
	//finalization
	x0=x_;
	grad=grad_;
	val=val_;
	//return value
	return val_;
}

template <class T>
void Opt::opts_impl(const FuncT<T>& func, T& obj){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::opts_impl(const FuncT<T>&,T&,const OptAlgo*):\n";
	double dx_=0;
	double dv_=0;
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//calculate the new position
		step();
		//set the new "old" values
		xOld_=x_;
		gradOld_=grad_;
		valOld_=val_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		//check the break condition
		if(optVal_==OPT_VAL::FTOL_REL && dv_<tol_) break;
		else if(optVal_==OPT_VAL::XTOL_REL && dx_<tol_) break;
		else if(optVal_==OPT_VAL::FTOL_ABS && val_<tol_) break;
		//print the status
		if(PRINT_OPT_FUNC>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(nPrint_>0){if(i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";}
		//update the counts
		++nStep_;
	}
}

//steepest-desccent
class SGD: public Opt{
private:
	unsigned int period_;
	unsigned int decay_;
	double gamma_;
public:
	//constructors/destructors
	SGD(){defaults();};
	SGD(const Opt& opt):Opt(opt){defaults();};
	~SGD(){};
	//access
	unsigned int& period(){return period_;};
	const unsigned int& period()const{return period_;};
	unsigned int& decay(){return decay_;};
	const unsigned int& decay()const{return decay_;};
	double& gamma(){return gamma_;};
	const double& gamma()const{return gamma_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim){};
	//operators
	friend std::ostream& operator<<(std::ostream& out, const SGD& sgd);
};

//steepest-descent + momentum
class SDM: public Opt{
private:
	double gamma_;//gradient step size
	double eta_;//mixing term
public:
	//constructors/destructors
	SDM(){defaults();};
	SDM(const Opt& opt):Opt(opt){defaults();};
	~SDM(){};
	//access
	double& gamma(){return gamma_;};
	const double& gamma()const{return gamma_;};
	double& eta(){return eta_;};
	const double& eta()const{return eta_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim){};
	//operators
	friend std::ostream& operator<<(std::ostream& out, const SDM& sdm);
};

//nesterov accelerated gradient
class NAG: public Opt{
private:
	double gamma_;//gradient step size
	double eta_;//mixing term
public:
	//constructors/destructors
	NAG(){defaults();};
	NAG(const Opt& opt):Opt(opt){defaults();};
	~NAG(){};
	//access
	double& gamma(){return gamma_;};
	const double& gamma()const{return gamma_;};
	double& eta(){return eta_;};
	const double& eta()const{return eta_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim){};
	//operators
	friend std::ostream& operator<<(std::ostream& out, const NAG& nag);
};

//adagrad
class ADAGRAD: public Opt{
private:
	static const double eps_;//small term to prevent divergence
	double gamma_;//gradient step size
	Eigen::VectorXd mgrad2_;//avg of square of gradient
public:
	//constructors/destructors
	ADAGRAD(){defaults();};
	ADAGRAD(const Opt& opt):Opt(opt){defaults();};
	~ADAGRAD(){};
	//access
	double& gamma(){return gamma_;};
	const double& gamma()const{return gamma_;};
	Eigen::VectorXd& mgrad2(){return mgrad2_;};
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const ADAGRAD& adagrad);
};

class ADADELTA: public Opt{
private:
	static const double eps_;//small term to prevent divergence
	double gamma_;//gradient step size
	double eta_;//mixing fraction
	Eigen::VectorXd mgrad2_;//avg of square of gradient
	Eigen::VectorXd mdx2_;//avg of square of dx
	Eigen::VectorXd dxv_;//change in x
public:
	//constructors/destructors
	ADADELTA(){defaults();};
	ADADELTA(const Opt& opt):Opt(opt){defaults();};
	~ADADELTA(){};
	//access
	double& gamma(){return gamma_;};
	const double& gamma()const{return gamma_;};
	double& eta(){return eta_;};
	const double& eta()const{return eta_;};
	Eigen::VectorXd& mgrad2(){return mgrad2_;};
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;};
	Eigen::VectorXd& dxv(){return dxv_;};
	const Eigen::VectorXd& dxv()const{return dxv_;};
	Eigen::VectorXd& mdx2(){return mdx2_;};
	const Eigen::VectorXd& mdx2()const{return mdx2_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const ADADELTA& adadelta);
};

class RMSPROP: public Opt{
private:
	static const double eps_;//small term to prevent divergence
	double gamma_;//gradient step size
	Eigen::VectorXd mgrad2_;//avg of square of gradient
public:
	//constructors/destructors
	RMSPROP(){defaults();};
	~RMSPROP(){};
	//access
	double& gamma(){return gamma_;};
	const double& gamma()const{return gamma_;};
	Eigen::VectorXd& mgrad2(){return mgrad2_;};
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const RMSPROP& rmsprop);
};

class ADAM: public Opt{
private:
	static const double eps_;//small term to prevent divergence
	static const double beta1;
	static const double beta2;
	double beta1i_;//power w.r.t i
	double beta2i_;//power w.r.t i
	double gamma_;//gradient step size
	Eigen::VectorXd mgrad_;//avg of gradient
	Eigen::VectorXd mgrad2_;//avg of square of gradient
public:
	//constructors/destructors
	ADAM(){defaults();};
	ADAM(const Opt& opt):Opt(opt){defaults();};
	~ADAM(){};
	//access
	double& gamma(){return gamma_;};
	const double& gamma()const{return gamma_;};
	double& beta1i(){return beta1i_;};
	const double& beta1i()const{return beta1i_;};
	double& beta2i(){return beta2i_;};
	const double& beta2i()const{return beta2i_;};
	Eigen::VectorXd& mgrad(){return mgrad_;};
	const Eigen::VectorXd& mgrad()const{return mgrad_;};
	Eigen::VectorXd& mgrad2(){return mgrad2_;};
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const ADAM& adam);
};

class BFGS: public Opt{
private:
	double gamma_;//gradient step size
	Eigen::MatrixXd B_,BOld_;
	Eigen::VectorXd s_,y_;
	unsigned int period_;
	unsigned int decay_;
public:
	//constructors/destructors
	BFGS(){defaults();};
	BFGS(const Opt& opt):Opt(opt){defaults();};
	~BFGS(){};
	//access
	unsigned int& period(){return period_;};
	const unsigned int& period()const{return period_;};
	unsigned int& decay(){return decay_;};
	const unsigned int& decay()const{return decay_;};
	double& gamma(){return gamma_;};
	const double& gamma()const{return gamma_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const BFGS& bfgs);
};

class LM: public Opt{
private:
	unsigned int period_;
	unsigned int decay_;
	double gamma_;//gradient step size
	double lambda_;
	double damp_;
	double min_,max_;
	Eigen::MatrixXd H_,D_;//hessian
public:
	//constructors/destructors
	LM(){defaults();};
	LM(const Opt& opt):Opt(opt){defaults();};
	~LM(){};
	//access
	unsigned int& period(){return period_;};
	const unsigned int& period()const{return period_;};
	unsigned int& decay(){return decay_;};
	const unsigned int& decay()const{return decay_;};
	double& gamma(){return gamma_;};
	const double& gamma()const{return gamma_;};
	double& damp(){return damp_;};
	const double& damp()const{return damp_;};
	double& lambda(){return lambda_;};
	const double& lambda()const{return lambda_;};
	double& min(){return min_;};
	const double& min()const{return min_;};
	double& max(){return max_;};
	const double& max()const{return max_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const LM& lm);
};

class RPROP: public Opt{
private:
	static const double etaP;
	static const double etaM;
	static const double deltaMax;
	static const double deltaMin;
	unsigned int period_;
	unsigned int decay_;
	Eigen::VectorXd delta_;
public:
	//constructors/destructors
	RPROP(){defaults();};
	RPROP(const Opt& opt):Opt(opt){defaults();};
	~RPROP(){};
	//access
	unsigned int& period(){return period_;};
	const unsigned int& period()const{return period_;};
	unsigned int& decay(){return decay_;};
	const unsigned int& decay()const{return decay_;};
	Eigen::VectorXd& delta(){return delta_;};
	const Eigen::VectorXd& delta()const{return delta_;};
	//member functions
	void step();
	void defaults();
	void init(unsigned int dim);
	//operators
	friend std::ostream& operator<<(std::ostream& out, const RPROP& rprop);
};

Opt& read(Opt& opt, const char* file);
SGD& read(SGD& sdg, const char* file);
SDM& read(SDM& sdm, const char* file);
NAG& read(NAG& nag, const char* file);
ADAGRAD& read(ADAGRAD& adagrad, const char* file);
ADADELTA& read(ADADELTA& adadelta, const char* file);
RMSPROP& read(RMSPROP& rmsprop, const char* file);
ADAM& read(ADAM& adam, const char* file);
BFGS& read(BFGS& bfgs, const char* file);
LM& read(LM& lm, const char* file);
RPROP& read(RPROP& rprop, const char* file);

Opt& read(Opt& opt, FILE* reader);
SGD& read(SGD& sdg, FILE* reader);
SDM& read(SDM& sdm, FILE* reader);
NAG& read(NAG& nag, FILE* reader);
ADAGRAD& read(ADAGRAD& adagrad, FILE* reader);
ADADELTA& read(ADADELTA& adadelta, FILE* reader);
RMSPROP& read(RMSPROP& rmsprop, FILE* reader);
ADAM& read(ADAM& adam, FILE* reader);
BFGS& read(BFGS& bfgs, FILE* reader);
LM& read(LM& lm, FILE* reader);
RPROP& read(RPROP& rprop, FILE* reader);

namespace serialize{

	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const Opt& obj);
	template <> unsigned int nbytes(const SGD& obj);
	template <> unsigned int nbytes(const SDM& obj);
	template <> unsigned int nbytes(const NAG& obj);
	template <> unsigned int nbytes(const ADAGRAD& obj);
	template <> unsigned int nbytes(const ADADELTA& obj);
	template <> unsigned int nbytes(const RMSPROP& obj);
	template <> unsigned int nbytes(const ADAM& obj);
	template <> unsigned int nbytes(const BFGS& obj);
	template <> unsigned int nbytes(const LM& obj);
	template <> unsigned int nbytes(const RPROP& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const Opt& obj, char* arr);
	template <> void pack(const SGD& obj, char* arr);
	template <> void pack(const SDM& obj, char* arr);
	template <> void pack(const NAG& obj, char* arr);
	template <> void pack(const ADAGRAD& obj, char* arr);
	template <> void pack(const ADADELTA& obj, char* arr);
	template <> void pack(const RMSPROP& obj, char* arr);
	template <> void pack(const ADAM& obj, char* arr);
	template <> void pack(const BFGS& obj, char* arr);
	template <> void pack(const LM& obj, char* arr);
	template <> void pack(const RPROP& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(Opt& obj, const char* arr);
	template <> void unpack(SGD& obj, const char* arr);
	template <> void unpack(SDM& obj, const char* arr);
	template <> void unpack(NAG& obj, const char* arr);
	template <> void unpack(ADAGRAD& obj, const char* arr);
	template <> void unpack(ADADELTA& obj, const char* arr);
	template <> void unpack(RMSPROP& obj, const char* arr);
	template <> void unpack(ADAM& obj, const char* arr);
	template <> void unpack(BFGS& obj, const char* arr);
	template <> void unpack(LM& obj, const char* arr);
	template <> void unpack(RPROP& obj, const char* arr);
	
}

#endif