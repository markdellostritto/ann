#include "optimize.hpp"

//***************************************************
// optimization method
//***************************************************

std::ostream& operator<<(std::ostream& out, const OPT_METHOD::type& type){
	if(type==OPT_METHOD::SD) return out<<"SD";
	else if(type==OPT_METHOD::SDG) return out<<"SDG";
	else if(type==OPT_METHOD::BFGS) return out<<"BFGS";
	else if(type==OPT_METHOD::BFGSG) return out<<"BFGSG";
	else if(type==OPT_METHOD::SDM) return out<<"SDM";
	else if(type==OPT_METHOD::NAG) return out<<"NAG";
	else if(type==OPT_METHOD::ADAGRAD) return out<<"ADAGRAD";
	else if(type==OPT_METHOD::ADADELTA) return out<<"ADADELTA";
	else if(type==OPT_METHOD::RMSPROP) return out<<"RMSPROP";
	else if(type==OPT_METHOD::ADAM) return out<<"ADAM";
	else if(type==OPT_METHOD::RPROP) return out<<"RPROP";
	else return out;
}

OPT_METHOD::type OPT_METHOD::load(const char* str){
	if(std::strcmp(str,"SD")==0) return OPT_METHOD::SD;
	else if(std::strcmp(str,"SDG")==0) return OPT_METHOD::SDG;
	else if(std::strcmp(str,"BFGS")==0) return OPT_METHOD::BFGS;
	else if(std::strcmp(str,"BFGSG")==0) return OPT_METHOD::BFGSG;
	else if(std::strcmp(str,"SDM")==0) return OPT_METHOD::SDM;
	else if(std::strcmp(str,"NAG")==0) return OPT_METHOD::NAG;
	else if(std::strcmp(str,"ADAGRAD")==0) return OPT_METHOD::ADAGRAD;
	else if(std::strcmp(str,"ADADELTA")==0) return OPT_METHOD::ADADELTA;
	else if(std::strcmp(str,"RMSPROP")==0) return OPT_METHOD::RMSPROP;
	else if(std::strcmp(str,"ADAM")==0) return OPT_METHOD::ADAM;
	else if(std::strcmp(str,"RPROP")==0) return OPT_METHOD::RPROP;
	else throw std::invalid_argument("Invalid optimization algorithm");
}

//***************************************************
// optimization value
//***************************************************

std::ostream& operator<<(std::ostream& out, const OPT_VAL::type& type){
	if(type==OPT_VAL::XTOL_REL) return out<<"XTOL_REL";
	else if(type==OPT_VAL::XTOL_ABS) return out<<"XTOL_ABS";
	else if(type==OPT_VAL::FTOL_REL) return out<<"FTOL_REL";
	else if(type==OPT_VAL::FTOL_ABS) return out<<"FTOL_ABS";
	else return out;
}

OPT_VAL::type OPT_VAL::load(const char* str){
	if(std::strcmp(str,"XTOL_REL")==0) return OPT_VAL::XTOL_REL;
	else if(std::strcmp(str,"XTOL_ABS")==0) return OPT_VAL::XTOL_ABS;
	else if(std::strcmp(str,"FTOL_REL")==0) return OPT_VAL::FTOL_REL;
	else if(std::strcmp(str,"FTOL_ABS")==0) return OPT_VAL::FTOL_ABS;
	else throw std::runtime_error("Invalid optimization value.");
}

//*********************************************
//Opt class
//*********************************************

//constructors/destructors

Opt::Opt(const Opt& opt){
	if(DEBUG_OPT>0) std::cout<<"Opt::Opt(const Opt&):\n";
	resize(opt.dim());
	nStep_=0; nEval_=0;
	maxIter_=opt.maxIter();
	tol_=opt.tol();
	algo_=opt.algo();
	gamma_=opt.gamma();
}

//operators

Opt& Opt::operator=(const Opt& opt){
	if(DEBUG_OPT>0) std::cout<<"Opt::operator=(const Opt&):\n";
	resize(opt.dim());
	nStep_=0; nEval_=0;
	maxIter_=opt.maxIter();
	tol_=opt.tol();
	algo_=opt.algo();
	gamma_=opt.gamma();
	return *this;
}

std::ostream& operator<<(std::ostream& out, const Opt& opt){
	out<<"**************************************\n";
	out<<"**************** OPT ****************\n";
	out<<"DIM      = "<<opt.dim_<<"\n";
	out<<"ALGO     = "<<opt.algo_<<"\n";
	out<<"TOL      = "<<opt.tol_<<"\n";
	out<<"MAX_ITER = "<<opt.maxIter_<<"\n";
	out<<"N_PRINT  = "<<opt.nPrint_<<"\n";
	out<<"GAMMA    = "<<opt.gamma_<<"\n";
	out<<"ETA      = "<<opt.eta_<<"\n";
	out<<"PERIOD   = "<<opt.period_<<"\n";
	out<<"DECAY    = "<<opt.decay_<<"\n";
	out<<"OPT_VAL  = "<<opt.optVal_<<"\n";
	out<<"PREC-LN  = "<<opt.precln_<<"\n";
	out<<"MAX-LN   = "<<opt.maxln_<<"\n";
	out<<"**************** OPT ****************\n";
	out<<"**************************************";
	return out;
}

//member functions

void Opt::defaults(){
	if(DEBUG_OPT>0) std::cout<<"Opt::defaults():\n";
	//status
		nPrint_=1000;
		nStep_=0;
		nEval_=0;
	//memory
		mem_=0;
	//parameters
		dim_=0;
		val_=0;
		valOld_=0;
		valPrev_.clear();
		x_.resize(0);
		grad_.resize(0);
		gradPrev_.clear();
		upper_.resize(0);
		lower_.resize(0);
	//stopping
		maxIter_=100;
		tol_=1e-5;
	//algorithm
		algo_=OPT_METHOD::UNKNOWN;
		gamma_=0.1;
		eta_=0.0;
		eps_=1e-8;
		period_=0;
		decay_=0;
	//linear optimization
		precln_=1e-5;
		maxln_=100;
}

void Opt::resize(unsigned int n){
	if(DEBUG_OPT>0) std::cout<<"Opt::resize(unsigned int):\n";
	dim_=n;
	x_=Eigen::VectorXd::Constant(n,std::sqrt(1.0/(1.0*n)));
	xOld_=Eigen::VectorXd::Constant(n,std::sqrt(1.0/(1.0*n)));
}

double Opt::opt(const Func& func, Eigen::VectorXd& x0){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt(const Func&,Eigen::VectorXd&):\n";
	nStep_=0; nEval_=0;
	val_=0; valOld_=0;
	if(x0.size()!=dim_) throw std::invalid_argument("Invalid array dimension.");
	x_.noalias()=x0; xOld_.noalias()=x0;
	grad_.resize(dim_);
	if(algo_==OPT_METHOD::SD) opt_sd(func);
	else if(algo_==OPT_METHOD::SDG) opt_sdg(func);
	else if(algo_==OPT_METHOD::BFGS) opt_bfgs(func);
	else if(algo_==OPT_METHOD::BFGSG) opt_bfgsg(func);
	else if(algo_==OPT_METHOD::SDM) opt_sdm(func);
	else if(algo_==OPT_METHOD::NAG) opt_nag(func);
	else if(algo_==OPT_METHOD::ADAGRAD) opt_adagrad(func);
	else if(algo_==OPT_METHOD::ADADELTA) opt_adadelta(func);
	else if(algo_==OPT_METHOD::RMSPROP) opt_rmsprop(func);
	else if(algo_==OPT_METHOD::ADAM) opt_adam(func);
	else if(algo_==OPT_METHOD::RPROP) opt_rprop(func);
	x0=x_;
}

double Opt::opt_sd(const Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_sd(const Func&):\n";
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
		//calculate the new position
		x_.noalias()=xOld_-grad_;
		//calculate the new position using a line search
		val_=opt_ln(func);
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		if(dv_<tol_) break;
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

double Opt::opt_sdg(const Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_sdg(const Func&):\n";
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
		//calculate the new position
		x_.noalias()=xOld_-gamma_*grad_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		if(dv_<tol_) break;
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

double Opt::opt_bfgs(Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_bfgs(const Func&):\n";
	Eigen::MatrixXd B=Eigen::MatrixXd::Identity(x_.size(),x_.size());
	Eigen::MatrixXd BOld=Eigen::MatrixXd::Identity(x_.size(),x_.size());
	Eigen::VectorXd s=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd y=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
		//obtain direction
		x_=xOld_;
		x_.noalias()-=B.llt().solve(grad_);
		//calculate the new position using a line search
		val_=opt_ln(func);
		//find the s vector
		s.noalias()=x_-xOld_;
		//set the y vector
		y.noalias()=grad_-gradOld_;
		//set the new B matrix
		B=BOld;
		B.noalias()+=y*y.transpose()/y.dot(s);
		B.noalias()-=B*(s*s.transpose())*B/(s.dot(B*s));
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		//set the new "old" values
		xOld_=x_;
		valOld_=val_;
		gradOld_=grad_;
		BOld=B;
		if(dv_<tol_) break;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
}

double Opt::opt_bfgsg(Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_bfgsg(const Func&):\n";
	Eigen::MatrixXd B=Eigen::MatrixXd::Identity(x_.size(),x_.size());
	Eigen::MatrixXd BOld=Eigen::MatrixXd::Identity(x_.size(),x_.size());
	Eigen::VectorXd s=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd y=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
		//obtain direction
		x_.noalias()=xOld_-gamma_*B.llt().solve(grad_);
		//find the s vector
		s.noalias()=x_-xOld_;
		//set the y vector
		y.noalias()=grad_-gradOld_;
		//set the new B matrix
		B.noalias()=BOld+y*y.transpose()/y.dot(s)-B*(s*s.transpose())*B/(s.dot(B*s));
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		//set the new "old" values
		xOld_.noalias()=x_;
		valOld_=val_;
		gradOld_.noalias()=grad_;
		BOld.noalias()=B;
		if(dv_<tol_) break;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
}

double Opt::opt_sdm(const Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_sdm(const Func&):\n";
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
		//scale the gradient
		grad_*=gamma_;
		grad_.noalias()+=eta_*gradOld_;
		//calculate the new position
		x_.noalias()=xOld_-grad_;
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		if(dv_<tol_) break;
		//update the "old" values
		xOld_.noalias()=x_;
		gradOld_.noalias()=grad_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
	return val_;
}

double Opt::opt_nag(const Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_nag(const Func&):\n";
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		x_.noalias()-=eta_*gradOld_;
		val_=func(x_,grad_); ++nEval_;
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
		xOld_.noalias()=x_;
		gradOld_.noalias()=grad_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
	return val_;
}

double Opt::opt_adagrad(const Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_adagrad(const Func&):\n";
	Eigen::VectorXd gradAvg_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
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

double Opt::opt_adadelta(const Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_adadelta(const Func&):\n";
	Eigen::VectorXd gradAvg_=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd dxAvg_=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd dxv_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
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

double Opt::opt_rmsprop(const Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_rmsprop(const Func&):\n";
	Eigen::VectorXd gradAvg_=Eigen::VectorXd::Zero(x_.size());
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
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

double Opt::opt_adam(const Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_adam(const Func&):\n";
	Eigen::VectorXd gradAvg_=Eigen::VectorXd::Zero(x_.size());
	Eigen::VectorXd grad2Avg_=Eigen::VectorXd::Zero(x_.size());
	const double beta1=0.9;
	const double beta2=0.999;
	double beta1i=beta1;//power w.r.t i
	double beta2i=beta2;//power w.r.t i
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
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
		xOld_.noalias()=x_;
		valOld_=val_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
		//update the powers of betas
		beta1i*=beta1;
		beta2i*=beta2;
	}
	return val_;
}

double Opt::opt_rprop(Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_rprop(const Func&):\n";
	//initialize the optimization
	Eigen::VectorXd gradOld_=Eigen::VectorXd::Zero(dim_);
	Eigen::VectorXd delta=Eigen::VectorXd::Constant(dim_,0.1);
	double etaP=1.2,etaM=0.5;
	for(unsigned int i=0; i<maxIter_; ++i){
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
		//calculate new position
		for(unsigned int n=0; n<dim_; ++n){
			if(grad_[n]*gradOld_[n]>0){
				delta[n]*=etaP;
				x_[n]-=function::sign(grad_[n])*delta[n];
			} else if(grad_[n]*gradOld_[n]<0){
				delta[n]*=etaM;
				grad_[n]=0.0;
			} else if(grad_[n]*gradOld_[n]==0){
				x_[n]-=function::sign(grad_[n])*delta[n];
			}
		}
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		//set the new "old" values
		xOld_.noalias()=x_;
		valOld_=val_;
		gradOld_=grad_;
		//print the status
		if(DEBUG_OPT==0 && i%nPrint_==0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(DEBUG_OPT>0) std::cout<<"opt step "<<i<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		//update the counts
		++nStep_;
	}
}

double Opt::opt_ln(Opt::Func& func, Eigen::VectorXd& x0, Eigen::VectorXd& x1){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_ln(Opt::Func& func, Eigen::VectorXd x0, Eigen::VectorXd x1):\n";
	a_.noalias()=x0; b_.noalias()=x1;
	double va=func(a_,grad_);
	double vb=func(b_,grad_);
	c_.noalias()=b_-(b_-a_)*1.0/num_const::PHI;
	d_.noalias()=a_+(b_-a_)*1.0/num_const::PHI;
	double vc,vd;
	unsigned int count=0;
	while((c_-d_).norm()>precln_ && count<maxln_){
		//calculate the new function values
		vc=func(c_,grad_);
		vd=func(d_,grad_);
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
	x0.noalias()=c_; x1.noalias()=d_;
	if(DEBUG_OPT>1){
		std::cout<<"count_ln = "<<count<<"\n";
		std::cout<<"val_ln = "<<((vc<vd)?vc:vd)<<"\n";
	}
	return (vc<vd)?vc:vd;
}

double Opt::opt_ln(Opt::Func& func){
	if(DEBUG_OPT>0) std::cout<<"Opt::opt_ln(Opt::Func& func):\n";
	a_.noalias()=x_; b_.noalias()=xOld_;
	double va=func(a_,grad_);
	double vb=func(b_,grad_);
	c_.noalias()=b_-(b_-a_)*1.0/num_const::PHI;
	d_.noalias()=a_+(b_-a_)*1.0/num_const::PHI;
	double vc,vd;
	unsigned int count=0;
	while((c_-d_).norm()>precln_ && count<maxln_){
		//calculate the new function values
		vc=func(c_,grad_);
		vd=func(d_,grad_);
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
