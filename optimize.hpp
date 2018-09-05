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
// local libraries - math
#include "math_const.hpp"
#include "math_cmp.hpp"
#include "math_function.hpp"

#ifndef DEBUG_OPT
#define DEBUG_OPT 0
#endif

//***************************************************
// optimization method
//***************************************************

struct OPT_METHOD{
	enum type {
		SD,
		SDG,
		BFGS,
		BFGSG,
		SDM,
		NAG,
		ADAGRAD,
		ADADELTA,
		RMSPROP,
		ADAM,
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
private:
	/*optimization*/
	//status
		unsigned int nPrint_;//frequency of printing
		unsigned int nStep_;//the number of steps
		unsigned int nEval_;//the number of evaluations
		int status_;//status of the optimization
	//stopping
		double tol_;//optimization tolerance
		unsigned int maxIter_;//the max number of iterations
		double dx_,dv_;
	//memory
		unsigned int mem_;//memory of previous iterations
	//parameters
		unsigned int dim_;//dimension of the problem
		double val_,valOld_;//value of the objective function
		std::vector<double> valPrev_;//previous values of the objective function
		Eigen::VectorXd x_,xOld_;//parameters
		Eigen::VectorXd grad_;//gradient
		std::vector<Eigen::VectorXd> gradPrev_;//previous values of the gradient
		Eigen::VectorXd lower_,upper_;//bounds
	//algorithm
		OPT_METHOD::type algo_;//optimization algorithm
		OPT_VAL::type optVal_;	
		double gamma_;//descent parameter
		double eta_;//friction parameter
		double eps_;//small number, prevents divide by zero
		unsigned int period_;//period of oscilation of gamma
		unsigned int decay_;//decay constant of gamma
	//line searching
		double precln_;//precision for line searches
		unsigned int maxln_;//max for line searches
		Eigen::VectorXd a_,b_,c_,d_;//for line search
public:
	//constructors/destructors
	Opt(){defaults();};
	Opt(const Opt& opt);
	~Opt(){};
	
	//operators
	Opt& operator=(const Opt& opt);
	friend std::ostream& operator<<(std::ostream& out, const Opt& opt);
	
	/*access*/
	//status
		unsigned int& nPrint(){return nPrint_;};
		const unsigned int& nPrint()const{return nPrint_;};
		const unsigned int& nStep()const{return nStep_;};
		const unsigned int& nEval()const{return nEval_;};
		const int& status()const{return status_;};
		const double& val()const{return val_;};
	//stopping
		double& tol(){return tol_;};
		const double& tol()const{return tol_;};
		unsigned int& maxIter(){return maxIter_;};
		const unsigned int& maxIter()const{return maxIter_;};
	//memory
		unsigned int& mem(){return mem_;};
		const unsigned int& mem()const{return mem_;};
	//parameters
		unsigned int& dim(){return dim_;};
		const unsigned int& dim()const{return dim_;};
		Eigen::VectorXd& x(){return x_;};
		const Eigen::VectorXd& x()const{return x_;};
		Eigen::VectorXd& lower(){return lower_;};
		const Eigen::VectorXd& lower()const{return lower_;};
		Eigen::VectorXd& upper(){return upper_;};
		const Eigen::VectorXd& upper()const{return upper_;};
	//algorithm
		OPT_METHOD::type& algo(){return algo_;};
		const OPT_METHOD::type& algo()const{return algo_;};
		OPT_VAL::type& optVal(){return optVal_;};
		const OPT_VAL::type& optVal()const{return optVal_;};
		double& gamma(){return gamma_;};
		const double& gamma()const{return gamma_;};
		double& eta(){return eta_;};
		const double& eta()const{return eta_;};
		double& eps(){return eps_;};
		const double& eps()const{return eps_;};
		unsigned int& period(){return period_;};
		const unsigned int& period()const{return period_;};
		unsigned int& decay(){return decay_;};
		const unsigned int& decay()const{return decay_;};
	//linear optimization
		double& precln(){return precln_;};
		const double& precln()const{return precln_;};
		unsigned int& maxln(){return maxln_;};
		const unsigned int& maxln()const{return maxln_;};
	
	//member functions
	void defaults();
	void clear(){defaults();};
	void resize(unsigned int n);
	double opt(const Func& func, Eigen::VectorXd& x0);
	double opt_sd(Func& func);
	double opt_sdg(Func& func);
	double opt_bfgs(Func& func);
	double opt_bfgsg(Func& func);
	double opt_sdm(Func& func);
	double opt_nag(Func& func);
	double opt_adagrad(Func& func);
	double opt_adadelta(Func& func);
	double opt_rmsprop(Func& func);
	double opt_adam(Func& func);
	double opt_rprop(Func& func);
	
	//optimization functions
	double opt_ln(Func& func);
	double opt_ln(Func& func, Eigen::VectorXd& x0, Eigen::VectorXd& x1);
	template <class T> double opt(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj, Eigen::VectorXd& x0);
	template <class T> double opt_ln(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_sd(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_sdg(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_sdm(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_nag(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_adagrad(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_adadelta(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_rmsprop(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_adam(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_bfgs(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_bfgsg(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
	template <class T> double opt_rprop(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj);
};

template <class T>
double Opt::opt(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj, Eigen::VectorXd& x0){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt<T>(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>&,T&,Eigen::VectorXd&):\n";
	nStep_=0; nEval_=0;
	val_=0; valOld_=0;
	//if(x0.size()!=dim_) throw std::invalid_argument("Invalid array dimension.");
	resize(x0.size());
	x_.noalias()=x0; xOld_.noalias()=x0;
	grad_.resize(dim_);
	if(algo_==OPT_METHOD::SD) opt_sd<T>(func,obj);
	else if(algo_==OPT_METHOD::SDG) opt_sdg<T>(func,obj);
	else if(algo_==OPT_METHOD::SDM) opt_sdm<T>(func,obj);
	else if(algo_==OPT_METHOD::NAG) opt_nag<T>(func,obj);
	else if(algo_==OPT_METHOD::ADAGRAD) opt_adagrad<T>(func,obj);
	else if(algo_==OPT_METHOD::ADADELTA) opt_adadelta<T>(func,obj);
	else if(algo_==OPT_METHOD::RMSPROP) opt_rmsprop<T>(func,obj);
	else if(algo_==OPT_METHOD::ADAM) opt_adam<T>(func,obj);
	else if(algo_==OPT_METHOD::BFGS) opt_bfgs<T>(func,obj);
	else if(algo_==OPT_METHOD::BFGSG) opt_bfgsg<T>(func,obj);
	else if(algo_==OPT_METHOD::RPROP) opt_rprop<T>(func,obj);
	else throw std::runtime_error("Invalid optimization algorithm.");
	x0=x_;
}

template <class T>
double Opt::opt_ln(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_ln<T>(const std::function<double (T&, const Eigen::VectorXd&,Eigen::VectorXd& grad)>&,T&):\n";
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
	if(DEBUG_OPT>1){
		std::cout<<"count_ln = "<<count<<"\n";
		std::cout<<"val_ln = "<<((vc<vd)?vc:vd)<<"\n";
	}
	return (vc<vd)?vc:vd;
}

template <class T>
double Opt::opt_sd(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_sd<T>(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//calculate the new position
		x_.noalias()=xOld_-grad_;
		//calculate the new position using a line search
		val_=opt_ln(func,obj);
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		if(optVal_==OPT_VAL::FTOL_REL && dv_<tol_) break;
		else if(optVal_==OPT_VAL::XTOL_REL && dx_<tol_) break;
		else if(optVal_==OPT_VAL::FTOL_ABS && val_<tol_) break;
		//set the new "old" values
		xOld_.noalias()=x_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
	return val_;
}

template <class T>
double Opt::opt_sdg(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_sdg<T>(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//calculate the new position
		double gam=gamma_;
		if(period_>0) gam=0.5*gam*(std::cos(num_const::PI*function::mod(((double)i)/period_,1.0))+1.0);
		if(decay_>0) gam=std::exp(-1.0*((double)i)/decay_)*gam;
		grad_*=gam;
		x_.noalias()=xOld_-grad_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		//check the break condition
		if(optVal_==OPT_VAL::FTOL_REL && dv_<tol_) break;
		else if(optVal_==OPT_VAL::XTOL_REL && dx_<tol_) break;
		else if(optVal_==OPT_VAL::FTOL_ABS && val_<tol_) break;
		//set the new "old" values
		xOld_.noalias()=x_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" gam "<<gam<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" gam "<<gam<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
	return val_;
}

template <class T>
double Opt::opt_sdm(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_sdm<T>(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(grad_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//scale the gradient
		grad_*=gamma_;
		grad_.noalias()+=eta_*gradOld_;
		//calculate the new position
		x_.noalias()=xOld_-grad_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		if(optVal_==OPT_VAL::FTOL_REL && dv_<tol_) break;
		else if(optVal_==OPT_VAL::XTOL_REL && dx_<tol_) break;
		else if(optVal_==OPT_VAL::FTOL_ABS && val_<tol_) break;
		//set the new "old" values
		xOld_=x_;
		gradOld_=grad_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
	return val_;
}

template <class T>
double Opt::opt_nag(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_nag<T>(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		x_.noalias()-=eta_*gradOld_;
		val_=func(obj,x_,grad_); ++nEval_;
		//scale the gradient
		grad_*=gamma_;
		grad_.noalias()+=eta_*gradOld_;
		//calculate the new position
		x_.noalias()=xOld_-grad_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		if(dv_<tol_) break;
		//set the new "old" values
		xOld_=x_;
		gradOld_=grad_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" eta "<<eta_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" eta "<<eta_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
	return val_;
}

template <class T>
double Opt::opt_adagrad(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_adagrad<T>(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	Eigen::VectorXd gradAvg_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//add to the running average of the square of the gradients
		gradAvg_.noalias()+=grad_.cwiseProduct(grad_);
		//scale the gradient
		for(unsigned int n=0; n<grad_.size(); ++n) grad_[n]*=gamma_/std::sqrt(gradAvg_[n]+eps_);
		//calculate the new position
		x_.noalias()=xOld_-grad_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		if(dv_<tol_) break;
		//update the "old" values
		xOld_=x_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
	return val_;
}

template <class T>
double Opt::opt_adadelta(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_adagrad<T>(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	Eigen::VectorXd gradAvg_=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd dxAvg_=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd dxv_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//add to the running average of the square of the gradients
		gradAvg_*=eta_;
		gradAvg_.noalias()+=(1.0-eta_)*grad_.cwiseProduct(grad_);
		//add to the running average of the square of the deltas
		dxAvg_*=eta_;
		dxAvg_.noalias()+=(1.0-eta_)*dxv_.cwiseProduct(dxv_);
		//scale the gradient
		for(unsigned int n=0; n<grad_.size(); ++n) grad_[n]*=std::sqrt(dxAvg_[n]+eps_)/std::sqrt(gradAvg_[n]+eps_);
		//calculate the new position
		x_.noalias()=xOld_-grad_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dxv_.noalias()=x_-xOld_;
		dx_=dxv_.norm();
		if(dv_<tol_) break;
		//update the "old" values
		xOld_=x_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
	return val_;
}

template <class T>
double Opt::opt_rmsprop(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_rmsprop<T>(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	Eigen::VectorXd gradAvg_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//add to the running average of the square of the gradients
		gradAvg_*=0.9;
		gradAvg_.noalias()+=0.1*grad_.cwiseProduct(grad_);
		//scale the gradient
		for(unsigned int n=0; n<grad_.size(); ++n) grad_[n]*=gamma_/std::sqrt(gradAvg_[n]+eps_);
		//calculate the new position
		x_.noalias()=xOld_-grad_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		if(dv_<tol_) break;
		//update the "old" values
		xOld_=x_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
	return val_;
}

template <class T>
double Opt::opt_adam(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_adam<T>(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	Eigen::VectorXd gradAvg_=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd grad2Avg_=Eigen::VectorXd::Zero(x_.size());
	const double beta1=0.9;
	const double beta2=0.999;
	double beta1i=beta1;//power w.r.t i
	double beta2i=beta2;//power w.r.t i
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//add to the running average of the gradients
		gradAvg_*=beta1;
		gradAvg_.noalias()+=(1.0-beta1)*grad_;
		//add to the running average of the square of the gradients
		grad2Avg_*=beta2;
		grad2Avg_.noalias()+=(1.0-beta2)*grad_.cwiseProduct(grad_);
		//calculate the update
		for(unsigned int n=0; n<grad_.size(); ++n) grad_[n]=gamma_*gradAvg_[n]/(1.0-beta1i)/(std::sqrt(grad2Avg_[n]/(1.0-beta2i))+eps_);
		//calculate the new position
		x_.noalias()=xOld_-grad_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		if(dv_<tol_) break;
		//update the "old" values
		xOld_=x_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" gam "<<gamma_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
		//update the powers of betas
		beta1i*=beta1;
		beta2i*=beta2;
	}
	return val_;
}

template <class T>
double Opt::opt_bfgs(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_bfgs(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	Eigen::MatrixXd B=Eigen::MatrixXd::Identity(x_.size(),x_.size());
	Eigen::MatrixXd BOld=Eigen::MatrixXd::Identity(x_.size(),x_.size());
	Eigen::VectorXd s=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd y=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//obtain direction
		x_=xOld_;
		x_.noalias()-=B.llt().solve(grad_);
		//calculate the new position using a line search
		val_=opt_ln(func,obj);
		//find the s vector
		s.noalias()=x_-xOld_;
		//set the y vector
		y.noalias()=grad_-gradOld_;
		//set the new B matrix
		B=BOld;
		B.noalias()+=y*y.transpose()/y.dot(s);
		B.noalias()-=BOld*(s*s.transpose())*BOld.transpose()/(s.dot(BOld*s));
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		//set the new "old" values
		xOld_=x_;
		valOld_=val_;
		gradOld_=grad_;
		BOld=B;
		if(optVal_==OPT_VAL::FTOL_REL && dv_<tol_) break;
		else if(optVal_==OPT_VAL::XTOL_REL && dx_<tol_) break;
		else if(optVal_==OPT_VAL::FTOL_ABS && val_<tol_) break;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
}

template <class T>
double Opt::opt_bfgsg(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_bfgs(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	Eigen::MatrixXd B=Eigen::MatrixXd::Identity(x_.size(),x_.size());
	Eigen::MatrixXd BOld=Eigen::MatrixXd::Identity(x_.size(),x_.size());
	Eigen::VectorXd s=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd y=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//obtain direction
		x_=xOld_;
		x_.noalias()-=B.llt().solve(grad_);
		//calculate the new position
		double gam=gamma_;
		if(period_>0) gam=0.5*gam*(std::cos(num_const::PI*function::mod(((double)i)/period_,1.0))+1.0);
		if(decay_>0) gam=std::exp(-1.0*((double)i)/decay_)*gam;
		x_.noalias()=xOld_-gam*grad_;
		//calculate the new value
		val_=func(obj,x_,grad_); ++nEval_;
		//find the s vector
		s.noalias()=x_-xOld_;
		//set the y vector
		y.noalias()=gam*(grad_-gradOld_);
		//set the new B matrix
		B=BOld;
		B.noalias()+=y*y.transpose()/y.dot(s);
		B.noalias()-=(BOld*s)*(BOld*s).transpose()/(s.dot(BOld*s));
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		//set the new "old" values
		xOld_=x_;
		valOld_=val_;
		gradOld_=grad_;
		BOld=B;
		if(optVal_==OPT_VAL::FTOL_REL && dv_<tol_) break;
		else if(optVal_==OPT_VAL::XTOL_REL && dx_<tol_) break;
		else if(optVal_==OPT_VAL::FTOL_ABS && val_<tol_) break;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" gam "<<gam<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" gam "<<gam<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
}

template <class T>
double Opt::opt_rprop(const std::function<double (T&, const Eigen::VectorXd& x, Eigen::VectorXd& grad)>& func, T& obj){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_bfgs(const std::function<double (T&,const Eigen::VectorXd&,Eigen::VectorXd&)>&,T&):\n";
	//initialize the optimization
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(dim_);
	Eigen::VectorXd delta_=Eigen::VectorXd::Constant(dim_,0.1);
	double etaP=1.2,etaM=0.5;
	double deltaMax=50,deltaMin=1e-14;
	//execute the optimization
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(obj,x_,grad_); ++nEval_;
		//calculate new position and delta
		double gam=deltaMin;
		if(period_>0) gam=0.5*gam*(std::cos(num_const::PI*function::mod(((double)i)/period_,1.0))+1.0);
		if(decay_>0) gam=std::exp(-1.0*((double)i)/decay_)*gam;
		for(unsigned int n=0; n<dim_; ++n){
			double s=grad_[n]*gradOld_[n];
			if(s>0){
				delta_[n]=cmp::min(delta_[n]*etaP,deltaMax);
				x_[n]-=function::sign(grad_[n])*delta_[n];
			}else if(s<0){
				delta_[n]=cmp::max(delta_[n]*etaM,gam);
				grad_[n]=0.0;
				if(val_>valOld_) x_[n]-=function::sign(grad_[n])*delta_[n];
			} else if(s==0){
				x_[n]-=function::sign(grad_[n])*delta_[n];
			}
		}
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		//check the break condition
		if(optVal_==OPT_VAL::FTOL_REL && dv_<tol_) break;
		else if(optVal_==OPT_VAL::XTOL_REL && dx_<tol_) break;
		else if(optVal_==OPT_VAL::FTOL_ABS && val_<tol_) break;
		//set the new "old" values
		xOld_=x_;
		valOld_=val_;
		gradOld_=grad_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" dmin "<<gam<<" delta "<<delta_.norm()<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" delta "<<delta_.norm()<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the count
		++nStep_;
	}
}

#endif