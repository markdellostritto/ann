#include "optimize.hpp"

//***************************************************
// optimization method
//***************************************************

std::ostream& operator<<(std::ostream& out, const OPT_METHOD::type& type){
	if(type==OPT_METHOD::SGD) return out<<"SGD";
	else if(type==OPT_METHOD::SDM) return out<<"SDM";
	else if(type==OPT_METHOD::NAG) return out<<"NAG";
	else if(type==OPT_METHOD::ADAGRAD) return out<<"ADAGRAD";
	else if(type==OPT_METHOD::ADADELTA) return out<<"ADADELTA";
	else if(type==OPT_METHOD::RMSPROP) return out<<"RMSPROP";
	else if(type==OPT_METHOD::ADAM) return out<<"ADAM";
	else if(type==OPT_METHOD::BFGS) return out<<"BFGS";
	else if(type==OPT_METHOD::LM) return out<<"LM";
	else if(type==OPT_METHOD::RPROP) return out<<"RPROP";
	else return out;
}

OPT_METHOD::type OPT_METHOD::load(const char* str){
	if(std::strcmp(str,"SGD")==0) return OPT_METHOD::SGD;
	else if(std::strcmp(str,"SDM")==0) return OPT_METHOD::SDM;
	else if(std::strcmp(str,"NAG")==0) return OPT_METHOD::NAG;
	else if(std::strcmp(str,"ADAGRAD")==0) return OPT_METHOD::ADAGRAD;
	else if(std::strcmp(str,"ADADELTA")==0) return OPT_METHOD::ADADELTA;
	else if(std::strcmp(str,"RMSPROP")==0) return OPT_METHOD::RMSPROP;
	else if(std::strcmp(str,"ADAM")==0) return OPT_METHOD::ADAM;
	else if(std::strcmp(str,"BFGS")==0) return OPT_METHOD::BFGS;
	else if(std::strcmp(str,"LM")==0) return OPT_METHOD::LM;
	else if(std::strcmp(str,"RPROP")==0) return OPT_METHOD::RPROP;
	else return OPT_METHOD::UNKNOWN;
}

//***************************************************
// optimization value
//***************************************************

std::ostream& operator<<(std::ostream& out, const OPT_VAL::type& type){
	if(type==OPT_VAL::XTOL_REL) out<<"XTOL_REL";
	else if(type==OPT_VAL::XTOL_ABS) out<<"XTOL_ABS";
	else if(type==OPT_VAL::FTOL_REL) out<<"FTOL_REL";
	else if(type==OPT_VAL::FTOL_ABS) out<<"FTOL_ABS";
	else out<<"UNKNOWN";
	return out;
}

OPT_VAL::type OPT_VAL::load(const char* str){
	if(std::strcmp(str,"XTOL_REL")==0) return OPT_VAL::XTOL_REL;
	else if(std::strcmp(str,"XTOL_ABS")==0) return OPT_VAL::XTOL_ABS;
	else if(std::strcmp(str,"FTOL_REL")==0) return OPT_VAL::FTOL_REL;
	else if(std::strcmp(str,"FTOL_ABS")==0) return OPT_VAL::FTOL_ABS;
	else return OPT_VAL::UNKNOWN;
}

//*********************************************
//Opt class
//*********************************************

//operators

std::ostream& operator<<(std::ostream& out, const Opt& opt){
	out<<"**************************************************\n";
	out<<"********************** OPT **********************\n";
	out<<"DIM      = "<<opt.dim_<<"\n";
	out<<"ALGO     = "<<opt.algo_<<"\n";
	out<<"TOL      = "<<opt.tol_<<"\n";
	out<<"MAX_ITER = "<<opt.maxIter_<<"\n";
	out<<"N_PRINT  = "<<opt.nPrint_<<"\n";
	out<<"OPT_VAL  = "<<opt.optVal_<<"\n";
	out<<"PREC-LN  = "<<opt.precln_<<"\n";
	out<<"MAX-LN   = "<<opt.maxln_<<"\n";
	out<<"********************** OPT **********************\n";
	out<<"**************************************************";
	return out;
}

//member functions

void Opt::defaults(){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::defaults():\n";
	//status
		nPrint_=0;
		nStep_=0;
		nEval_=0;
	//parameters
		dim_=0;
		val_=0;
		valOld_=0;
		x_.resize(0);
		xOld_.resize(0);
		grad_.resize(0);
		gradOld_.resize(0);
	//stopping
		maxIter_=100;
		tol_=1e-5;
	//algorithm
		algo_=OPT_METHOD::UNKNOWN;
		optVal_=OPT_VAL::FTOL_ABS;
	//linear optimization
		precln_=1e-5;
		maxln_=100;
}

void Opt::resize(unsigned int n){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::init(unsigned int):\n";
	if(n==0) throw std::invalid_argument("Invalid optimization dimenstion.");
	val_=0;
	valOld_=0;
	dim_=n;
	x_=Eigen::VectorXd::Constant(n,std::sqrt(1.0/(1.0*n)));
	xOld_=Eigen::VectorXd::Constant(n,std::sqrt(1.0/(1.0*n)));
	grad_=Eigen::VectorXd::Constant(n,0.0);
	gradOld_=Eigen::VectorXd::Constant(n,0.0);
}

double Opt::opt_ln(Opt::Func& func, Eigen::VectorXd& x0, Eigen::VectorXd& x1){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::opt_ln(Opt::Func& func, Eigen::VectorXd x0, Eigen::VectorXd x1):\n";
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
	if(PRINT_OPT_DATA>1){
		std::cout<<"count_ln = "<<count<<"\n";
		std::cout<<"val_ln = "<<((vc<vd)?vc:vd)<<"\n";
	}
	return (vc<vd)?vc:vd;
}

double Opt::opt_ln(Opt::Func& func){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::opt_ln(Opt::Func& func):\n";
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
	if(PRINT_OPT_DATA>1){
		std::cout<<"count_ln = "<<count<<"\n";
		std::cout<<"val_ln = "<<((vc<vd)?vc:vd)<<"\n";
	}
	return (vc<vd)?vc:vd;
}

double Opt::opts(const Func& func, Eigen::VectorXd& x0){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::opt(const Func&,Eigen::VectorXd&):\n";
	//initialization/resizing
	resize(x0.size());
	x_=x0; 
	xOld_=x0;
	init(dim_);
	//optimization
	opts_impl(func);
	//finalization
	x0=x_;
	//return the value
	return val_;
}

double Opt::opts(const Func& func, Eigen::VectorXd& x0, Eigen::VectorXd& grad, double& val){
	std::cout<<"Opt::opts(const Func&,Eigen::VectorXd&,Eigen::VectorXd&,double&):\n";
	//initialization/resizing
	resize(x0.size());
	if(grad.size()==0) grad=Eigen::VectorXd::Zero(dim_);
	else if(x0.size()!=grad.size()) throw std::runtime_error("Invalid initial gradient.");
	x_=x0; 
	xOld_=x0;
	grad_=grad;
	gradOld_=grad;
	val_=val;
	valOld_=val;
	init(dim_);
	//optimization
	opts_impl(func);
	//finalization
	x0=x_;
	grad=grad_;
	val=val_;
	//return value
	return val_;
}

void Opt::opts_impl(const Func& func){
	if(PRINT_OPT_FUNC>0) std::cout<<"Opt::opts_impl(const Func&):\n";
	double dv_=0,dx_=0;
	for(unsigned int i=0; i<maxIter_; ++i){
		//set the new "old" values
		if(i>0){
			xOld_=x_;
			gradOld_=grad_;
			valOld_=val_;
		}
		//calculate the value and gradient
		val_=func(x_,grad_); ++nEval_;
		//calculate the new position
		step();
		//calculate the difference
		dv_=std::fabs(val_-valOld_);
		dx_=(x_-xOld_).norm();
		//check the break condition
		if(optVal_==OPT_VAL::FTOL_REL && dv_<tol_) break;
		else if(optVal_==OPT_VAL::XTOL_REL && dx_<tol_) break;
		else if(optVal_==OPT_VAL::FTOL_ABS && val_<tol_) break;
		//print the status
		if(PRINT_OPT_DATA>0) std::cout<<"opt step "<<nStep_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";
		else if(nPrint_>0){if(i%nPrint_==0) std::cout<<"opt step "<<nStep_<<" val "<<val_<<" dv "<<dv_<<" dx "<<dx_<<"\n";}
		//update the counts
		++nStep_;
	}
}

//***************************************************
// OptAlgo
//***************************************************

//steepest-descent

void SGD::defaults(){
	period_=0;
	decay_=0;
	gamma_=0.001;
}

std::ostream& operator<<(std::ostream& out, const SGD& sgd){
	out<<static_cast<const Opt&>(sgd)<<"\n";
	out<<"**************************************************\n";
	out<<"********************* SGD *********************\n";
	out<<"PERIOD = "<<sgd.period_<<"\n";
	out<<"DECAY = "<<sgd.decay_<<"\n";
	out<<"GAMMA = "<<sgd.gamma_<<"\n";
	out<<"********************* SGD *********************\n";
	out<<"**************************************************";
}

void SGD::step(){
	//calculate the new position
	double gam=gamma_;
	if(period_>0) gam=0.5*gam*(std::cos(num_const::PI*special::mod(((double)nStep_)/period_,1.0))+1.0);
	if(decay_>0) gam=std::exp(-1.0*((double)nStep_)/decay_)*gam;
	grad_*=gam;
	x_.noalias()=xOld_-grad_;
}

//steepest-descent + momentum

void SDM::defaults(){
	gamma_=0.001;
	eta_=0.9;
}

std::ostream& operator<<(std::ostream& out, const SDM& sdm){
	out<<static_cast<const Opt&>(sdm)<<"\n";
	out<<"**************************************************\n";
	out<<"********************* SDM *********************\n";
	out<<"GAMMA = "<<sdm.gamma_<<"\n";
	out<<"ETA = "<<sdm.eta_<<"\n";
	out<<"********************* SDM *********************\n";
	out<<"**************************************************";
}

void SDM::step(){
	//scale the gradient
	grad_*=gamma_;
	grad_.noalias()+=eta_*gradOld_;
	//calculate the new position
	x_.noalias()=xOld_-grad_;
}

//nesterov accelerated gradient

void NAG::defaults(){
	gamma_=0.001;
	eta_=0.9;
}

std::ostream& operator<<(std::ostream& out, const NAG& nag){
	out<<static_cast<const Opt&>(nag)<<"\n";
	out<<"**************************************************\n";
	out<<"********************* NAG *********************\n";
	out<<"GAMMA = "<<nag.gamma_<<"\n";
	out<<"ETA = "<<nag.eta_<<"\n";
	out<<"********************* NAG *********************\n";
	out<<"**************************************************";
}

void NAG::step(){
	//scale the gradient
	grad_*=gamma_;
	grad_.noalias()+=eta_*gradOld_;
	//calculate the new position
	x_.noalias()=xOld_-grad_;
	x_.noalias()-=eta_*grad_;
}

//adagrad

const double ADAGRAD::eps_=1e-8;

void ADAGRAD::defaults(){
	gamma_=0.001;
}

std::ostream& operator<<(std::ostream& out, const ADAGRAD& adagrad){
	out<<static_cast<const Opt&>(adagrad)<<"\n";
	out<<"**************************************************\n";
	out<<"********************* ADAGRAD *********************\n";
	out<<"GAMMA = "<<adagrad.gamma_<<"\n";
	out<<"********************* ADAGRAD *********************\n";
	out<<"**************************************************";
}

void ADAGRAD::step(){
	//add to the running average of the square of the gradients
	mgrad2_.noalias()+=grad_.cwiseProduct(grad_);
	//calculate the new position
	for(unsigned int n=0; n<grad_.size(); ++n){
		x_[n]=xOld_[n]-grad_[n]*gamma_/std::sqrt(mgrad2_[n]+eps_);
	}
}

void ADAGRAD::init(unsigned int dim){
	if(mgrad2_.size()!=dim) mgrad2_=Eigen::VectorXd::Zero(dim);
}

//adadelta

const double ADADELTA::eps_=1e-8;

void ADADELTA::defaults(){
	gamma_=0.001;
	eta_=0.9;
}

std::ostream& operator<<(std::ostream& out, const ADADELTA& adadelta){
	out<<static_cast<const Opt&>(adadelta)<<"\n";
	out<<"**************************************************\n";
	out<<"********************* ADADELTA *********************\n";
	out<<"GAMMA = "<<adadelta.gamma_<<"\n";
	out<<"ETA = "<<adadelta.eta_<<"\n";
	out<<"********************* ADADELTA *********************\n";
	out<<"**************************************************";
}

void ADADELTA::step(){
	//add to the running average of the square of the gradients
	mgrad2_*=eta_;
	mgrad2_.noalias()+=(1.0-eta_)*grad_.cwiseProduct(grad_);
	//calculate the new position
	for(unsigned int n=0; n<grad_.size(); ++n){
		x_[n]=xOld_[n]-grad_[n]*std::sqrt(mdx2_[n]+eps_)/std::sqrt(mgrad2_[n]+eps_);
	}
	//calculate the difference
	dxv_.noalias()=x_-xOld_;
	//add to the running average of the square of the deltas
	mdx2_*=eta_;
	mdx2_.noalias()+=(1.0-eta_)*dxv_.cwiseProduct(dxv_);
}

void ADADELTA::init(unsigned int dim){
	if(mgrad2_.size()!=dim) mgrad2_=Eigen::VectorXd::Zero(dim);
	if(mdx2_.size()!=dim) mdx2_=Eigen::VectorXd::Zero(dim);
	if(dxv_.size()!=dim) dxv_=Eigen::VectorXd::Zero(dim);
}

//rmsprop

const double RMSPROP::eps_=1e-8;

void RMSPROP::defaults(){
	gamma_=0.001;
}

std::ostream& operator<<(std::ostream& out, const RMSPROP& rmsprop){
	out<<static_cast<const Opt&>(rmsprop)<<"\n";
	out<<"**************************************************\n";
	out<<"********************* RMSPROP *********************\n";
	out<<"GAMMA = "<<rmsprop.gamma_<<"\n";
	out<<"********************* RMSPROP *********************\n";
	out<<"**************************************************";
}

void RMSPROP::step(){
	//add to the running average of the square of the gradients
	mgrad2_*=0.9;
	mgrad2_.noalias()+=0.1*grad_.cwiseProduct(grad_);
	//calculate the new position
	for(unsigned int n=0; n<grad_.size(); ++n){
		x_[n]=xOld_[n]-grad_[n]*gamma_/std::sqrt(mgrad2_[n]+eps_);
	}
}

void RMSPROP::init(unsigned int dim){
	if(mgrad2_.size()!=dim) mgrad2_=Eigen::VectorXd::Zero(dim);
}

//adam

const double ADAM::eps_=1e-8;
const double ADAM::beta1=0.9;
const double ADAM::beta2=0.999;
	
void ADAM::defaults(){
	if(PRINT_OPT_FUNC>0) std::cout<<"ADAM::defaults():\n";
	gamma_=0.001;
	beta1i_=0;
	beta2i_=0;
}

std::ostream& operator<<(std::ostream& out, const ADAM& adam){
	out<<static_cast<const Opt&>(adam)<<"\n";
	out<<"**************************************************\n";
	out<<"********************* ADAM *********************\n";
	out<<"GAMMA = "<<adam.gamma_<<"\n";
	out<<"********************* ADAM *********************\n";
	out<<"**************************************************";
}

void ADAM::step(){
	if(PRINT_OPT_FUNC>0) std::cout<<"ADAM::step():\n";
	//add to the running average of the gradients
	mgrad_*=beta1;
	mgrad_.noalias()+=(1.0-beta1)*grad_;
	//add to the running average of the square of the gradients
	mgrad2_*=beta2;
	mgrad2_.noalias()+=(1.0-beta2)*grad_.cwiseProduct(grad_);
	//calculate the update
	for(unsigned int n=0; n<grad_.size(); ++n){
		grad_[n]=gamma_*mgrad_[n]/(1.0-beta1i_)/(std::sqrt(mgrad2_[n]/(1.0-beta2i_))+eps_);
	}
	//calculate the new position
	x_.noalias()=xOld_-grad_;
	//update the powers of betas
	beta1i_*=beta1;
	beta2i_*=beta2;
}

void ADAM::init(unsigned int dim){
	if(PRINT_OPT_FUNC>0) std::cout<<"ADAM::init(unsigned int):\n";
	if(mgrad_.size()!=dim) mgrad_=Eigen::VectorXd::Zero(dim);
	if(mgrad2_.size()!=dim) mgrad2_=Eigen::VectorXd::Zero(dim);
	if(beta1i_==0) beta1i_=beta1;//power w.r.t i
	if(beta2i_==0) beta2i_=beta2;//power w.r.t i
}

//BFGS

void BFGS::defaults(){
	period_=0;
	decay_=0;
	gamma_=0.001;
}

std::ostream& operator<<(std::ostream& out, const BFGS& bfgs){
	out<<static_cast<const Opt&>(bfgs)<<"\n";
	out<<"**************************************************\n";
	out<<"********************* BFGS *********************\n";
	out<<"PERIOD = "<<bfgs.period_<<"\n";
	out<<"DECAY = "<<bfgs.decay_<<"\n";
	out<<"GAMMA = "<<bfgs.gamma_<<"\n";
	out<<"********************* BFGS *********************\n";
	out<<"**************************************************";
}

void BFGS::step(){
	//obtain direction
	s_.noalias()=B_.llt().solve(grad_);
	//set the s vector
	double gam=gamma_;
	if(period_>0) gam=0.5*gam*(std::cos(num_const::PI*special::mod(((double)nStep_)/period_,1.0))+1.0);
	if(decay_>0) gam=std::exp(-1.0*((double)nStep_)/decay_)*gam;
	s_*=-gam/s_.norm();
	//calculate new position
	x_.noalias()=xOld_+s_;
	//set the y vector
	y_.noalias()=gam*(grad_-gradOld_);
	//set the new B matrix
	B_.noalias()+=y_*y_.transpose()/y_.dot(s_);
	y_.noalias()=BOld_*s_;
	B_.noalias()-=y_*y_.transpose()/(s_.dot(y_));
	//set the old B matrix
	BOld_=B_;
}

void BFGS::init(unsigned int dim){
	if(B_.rows()!=dim) B_=Eigen::MatrixXd::Identity(dim,dim);
	if(BOld_.rows()!=dim) BOld_=Eigen::MatrixXd::Identity(dim,dim);
	if(s_.size()!=dim) s_=Eigen::VectorXd::Zero(dim);
	if(y_.size()!=dim) y_=Eigen::VectorXd::Zero(dim);
}

//LM

void LM::defaults(){
	period_=0;
	decay_=0;
	gamma_=0.001;
	damp_=1e-4;
	min_=1e-6;
	max_=1000;
	lambda_=1.0;
}

std::ostream& operator<<(std::ostream& out, const LM& lm){
	out<<static_cast<const Opt&>(lm)<<"\n";
	out<<"**************************************************\n";
	out<<"*********************** LM ***********************\n";
	out<<"PERIOD = "<<lm.period_<<"\n";
	out<<"DECAY  = "<<lm.decay_<<"\n";
	out<<"GAMMA  = "<<lm.gamma_<<"\n";
	out<<"DAMP   = "<<lm.damp_<<"\n";
	out<<"LAMBDA = "<<lm.lambda_<<"\n";
	out<<"MIN    = "<<lm.min_<<"\n";
	out<<"MAX    = "<<lm.max_<<"\n";
	out<<"*********************** LM ***********************\n";
	out<<"**************************************************";
}

void LM::step(){
	//check the value, update step
	if(val_<valOld_) lambda_*=1.0/(1.0+damp_);
	else lambda_*=(1.0+damp_);
	if(lambda_<min_) lambda_=min_;
	else if(lambda_>max_) lambda_=max_;
	//compute the Hessian
	H_=grad_*grad_.transpose();
	D_=H_.diagonal().asDiagonal();
	//calculate new position
	x_.noalias()-=gamma_*(H_+D_*lambda_).llt().solve(grad_)*val_;
}

void LM::init(unsigned int dim){
	if(H_.rows()!=dim) H_=Eigen::MatrixXd::Identity(dim,dim);
	if(D_.rows()!=dim) D_=Eigen::MatrixXd::Identity(dim,dim);
}

//RPROP

const double RPROP::etaP=1.2;
const double RPROP::etaM=0.5;
const double RPROP::deltaMax=50.0;
const double RPROP::deltaMin=1e-14;

void RPROP::defaults(){
	period_=0;
	decay_=0;
}

std::ostream& operator<<(std::ostream& out, const RPROP& rprop){
	out<<static_cast<const Opt&>(rprop)<<"\n";
	out<<"**************************************************\n";
	out<<"********************* RPROP *********************\n";
	out<<"PERIOD   = "<<rprop.period_<<"\n";
	out<<"DECAY    = "<<rprop.decay_<<"\n";
	out<<"********************* RPROP *********************\n";
	out<<"**************************************************";
}

void RPROP::step(){
	//calculate new position and delta
	double gam=deltaMin;
	if(period_>0) gam=0.5*gam*(std::cos(num_const::PI*special::mod(((double)nStep_)/period_,1.0))+1.0);
	if(decay_>0) gam=std::exp(-1.0*((double)nStep_)/decay_)*gam;
	for(unsigned int n=0; n<dim_; ++n){
		double s=grad_[n]*gradOld_[n];
		if(s>0){
			delta_[n]=cmp::min(delta_[n]*etaP,deltaMax);
			x_[n]-=special::sign(grad_[n])*delta_[n];
		}else if(s<0){
			delta_[n]=cmp::max(delta_[n]*etaM,gam);
			grad_[n]=0.0;
			if(val_>valOld_) x_[n]-=special::sign(grad_[n])*delta_[n];
		} else if(s==0){
			x_[n]-=special::sign(grad_[n])*delta_[n];
		}
	}
}

void RPROP::init(unsigned int dim){
	if(delta_.size()!=dim) delta_=Eigen::VectorXd::Constant(dim,0.1);
}

//reading - file

Opt& read(Opt& opt, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(Opt&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(opt,reader);
	fclose(reader); reader=NULL;
	return opt;
} 

SGD& read(SGD& sgd, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(SGD&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(sgd),reader);
	read(sgd,reader);
	fclose(reader); reader=NULL;
	return sgd;
}

SDM& read(SDM& sdm, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(SDM&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(sdm),reader);
	read(sdm,reader);
	fclose(reader); reader=NULL;
	return sdm;
}

NAG& read(NAG& nag, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(NAG&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(nag),reader);
	read(nag,reader);
	fclose(reader); reader=NULL;
	return nag;
}

ADAGRAD& read(ADAGRAD& adagrad, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(ADAGRAD&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(adagrad),reader);
	read(adagrad,reader);
	fclose(reader); reader=NULL;
	return adagrad;
}

ADADELTA& read(ADADELTA& adadelta, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(ADADELTA&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(adadelta),reader);
	read(adadelta,reader);
	fclose(reader); reader=NULL;
	return adadelta;
}

RMSPROP& read(RMSPROP& rmsprop, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(RMSPROP&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(rmsprop),reader);
	read(rmsprop,reader);
	fclose(reader); reader=NULL;
	return rmsprop;
}

ADAM& read(ADAM& adam, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(ADAM&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(adam),reader);
	read(adam,reader);
	fclose(reader); reader=NULL;
	return adam;
}

BFGS& read(BFGS& bfgs, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(BFGS&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(bfgs),reader);
	read(bfgs,reader);
	fclose(reader); reader=NULL;
	return bfgs;
}

LM& read(LM& lm, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(LM&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(lm),reader);
	read(lm,reader);
	fclose(reader); reader=NULL;
	return lm;
}

RPROP& read(RPROP& rprop, const char* file){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(RPROP&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Opt&>(rprop),reader);
	read(rprop,reader);
	fclose(reader); reader=NULL;
	return rprop;
}

//reading - file pointer

Opt& read(Opt& opt, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(Opt&,FILE*):\n";
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="ALGO"){
			opt.algo()=OPT_METHOD::load(string::to_upper(strlist.at(1)).c_str());
		} else if(strlist.at(0)=="OPT_VAL"){
			opt.optVal()=OPT_VAL::load(string::to_upper(strlist.at(1)).c_str());
		} else if(strlist.at(0)=="TOL"){
			opt.tol()=std::atof(strlist.at(1).c_str());
		} else if(strlist.at(0)=="MAX_ITER"){
			opt.maxIter()=std::atof(strlist.at(1).c_str());
		} else if(strlist.at(0)=="N_PRINT"){
			opt.nPrint()=std::atoi(strlist.at(1).c_str());
		} else if(strlist.at(0)=="PREC_LN"){
			opt.precln()=std::atoi(strlist.at(1).c_str());
		} else if(strlist.at(0)=="MAX_LN"){
			opt.maxln()=std::atoi(strlist.at(1).c_str());
		}
	}
	delete[] input;
	return opt;
}

SGD& read(SGD& sgd, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(SGD&,FILE*):\n";
	read(static_cast<Opt&>(sgd),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="DECAY"){
			sgd.decay()=std::atoi((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="PERIOD"){
			sgd.period()=std::atoi((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="GAMMA"){
			sgd.gamma()=std::atof((strlist.at(1)).c_str());
		} 
	}
	delete[] input;
	return sgd;
}

SDM& read(SDM& sdm, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(SDM&,FILE*):\n";
	read(static_cast<Opt&>(sdm),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="GAMMA"){
			sdm.gamma()=std::atof((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="ETA"){
			sdm.eta()=std::atof((strlist.at(1)).c_str());
		} 
	}
	delete[] input;
	return sdm;
}

NAG& read(NAG& nag, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(NAG&,FILE*):\n";
	read(static_cast<Opt&>(nag),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="GAMMA"){
			nag.gamma()=std::atof((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="ETA"){
			nag.eta()=std::atof((strlist.at(1)).c_str());
		} 
	}
	delete[] input;
	return nag;
}

ADAGRAD& read(ADAGRAD& adagrad, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(ADAGRAD&,FILE*):\n";
	read(static_cast<Opt&>(adagrad),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="GAMMA"){
			adagrad.gamma()=std::atof((strlist.at(1)).c_str());
		}
	}
	delete[] input;
	return adagrad;
}

ADADELTA& read(ADADELTA& adadelta, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(ADADELTA&,FILE*):\n";
	read(static_cast<Opt&>(adadelta),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="GAMMA"){
			adadelta.gamma()=std::atof((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="ETA"){
			adadelta.eta()=std::atof((strlist.at(1)).c_str());
		}
	}
	delete[] input;
	return adadelta;
}

RMSPROP& read(RMSPROP& rmsprop, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(RMSPROP&,FILE*):\n";
	read(static_cast<Opt&>(rmsprop),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="GAMMA"){
			rmsprop.gamma()=std::atof((strlist.at(1)).c_str());
		}
	}
	delete[] input;
	return rmsprop;
}

ADAM& read(ADAM& adam, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(ADAM&,FILE*):\n";
	read(static_cast<Opt&>(adam),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="GAMMA"){
			adam.gamma()=std::atof((strlist.at(1)).c_str());
		}
	}
	delete[] input;
	return adam;
}

BFGS& read(BFGS& bfgs, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(BFGS&,FILE*):\n";
	read(static_cast<Opt&>(bfgs),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="DECAY"){
			bfgs.decay()=std::atoi((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="PERIOD"){
			bfgs.period()=std::atoi((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="GAMMA"){
			bfgs.gamma()=std::atoi((strlist.at(1)).c_str());
		} 
	}
	delete[] input;
	return bfgs;
}

LM& read(LM& lm, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(LM&,FILE*):\n";
	read(static_cast<Opt&>(lm),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="DECAY"){
			lm.decay()=std::atoi((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="PERIOD"){
			lm.period()=std::atoi((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="GAMMA"){
			lm.gamma()=std::atof((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="DAMP"){
			lm.damp()=std::atof((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="MAX"){
			lm.max()=std::atof((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="MIN"){
			lm.min()=std::atof((strlist.at(1)).c_str());
		} 
	}
	delete[] input;
	return lm;
}

RPROP& read(RPROP& rprop, FILE* reader){
	if(PRINT_OPT_FUNC>0) std::cout<<"read(RPROP&,FILE*):\n";
	read(static_cast<Opt&>(rprop),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="DECAY"){
			rprop.decay()=std::atoi((strlist.at(1)).c_str());
		} else if(strlist.at(0)=="PERIOD"){
			rprop.period()=std::atoi((strlist.at(1)).c_str());
		} 
	}
	delete[] input;
	return rprop;
}

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************
	
template <> unsigned int nbytes(const Opt& obj){
	unsigned int size=0;
	//status
		size+=sizeof(unsigned int);//nStep_
		size+=sizeof(unsigned int);//nEval_
	//parameters
		size+=sizeof(unsigned int);//dim_
		size+=sizeof(double);//val_
		size+=sizeof(double);//valOld_
		size+=nbytes(obj.x());//x
		size+=nbytes(obj.xOld());//xOld
		size+=nbytes(obj.grad());//grad
		size+=nbytes(obj.gradOld());//gradOld
	//algorithm
		size+=sizeof(OPT_METHOD::type);//optMethod_
		size+=sizeof(OPT_VAL::type);//optVal_
	//return the size
		return size;
}
template <> unsigned int nbytes(const SGD& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(unsigned int);//period_
	size+=sizeof(unsigned int);//decay_
	size+=sizeof(double);//gamma_
	return size;
}
template <> unsigned int nbytes(const SDM& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(double);//gamma_
	size+=sizeof(double);//eta_
	return size;
}
template <> unsigned int nbytes(const NAG& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(double);//gamma_
	size+=sizeof(double);//eta_
	return size;
}
template <> unsigned int nbytes(const ADAGRAD& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(double);//gamma_
	size+=nbytes(obj.mgrad2());
	return size;
}
template <> unsigned int nbytes(const ADADELTA& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(double);//gamma_
	size+=sizeof(double);//eta_
	size+=nbytes(obj.mgrad2());
	size+=nbytes(obj.mdx2());
	size+=nbytes(obj.dxv());
	return size;
}
template <> unsigned int nbytes(const RMSPROP& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(double);//gamma_
	size+=nbytes(obj.mgrad2());
	return size;
}
template <> unsigned int nbytes(const ADAM& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(double);//gamma_
	size+=nbytes(obj.mgrad());
	size+=nbytes(obj.mgrad2());
	size+=sizeof(double);//beta1i
	size+=sizeof(double);//beta2i
	return size;
}
template <> unsigned int nbytes(const BFGS& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(double);//gamma_
	return size;
}
template <> unsigned int nbytes(const LM& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(unsigned int);//period_
	size+=sizeof(unsigned int);//decay_
	size+=sizeof(double);//gamma_
	size+=sizeof(double);//lambda_
	size+=sizeof(double);//damp_
	size+=sizeof(double);//min_
	size+=sizeof(double);//max_
	return size;
}
template <> unsigned int nbytes(const RPROP& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt&>(obj));
	size+=sizeof(unsigned int);
	size+=sizeof(unsigned int);
	size+=nbytes(obj.delta());
	return size;
}

//**********************************************
// packing
//**********************************************

template <> void pack(const Opt& obj, char* arr){
	unsigned int pos=0;
	//status
		std::memcpy(arr+pos,&obj.nStep(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//nStep_
		std::memcpy(arr+pos,&obj.nEval(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//nEval_
	//paramters
		std::memcpy(arr+pos,&obj.dim(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//dim_
		std::memcpy(arr+pos,&obj.val(),sizeof(double)); pos+=sizeof(double);//val_
		std::memcpy(arr+pos,&obj.valOld(),sizeof(double)); pos+=sizeof(double);//valOld_
		pack(obj.x(),arr+pos); pos+=nbytes(obj.x());//x_
		pack(obj.xOld(),arr+pos); pos+=nbytes(obj.xOld());//xOld_
		pack(obj.grad(),arr+pos); pos+=nbytes(obj.grad());//grad_
		pack(obj.gradOld(),arr+pos); pos+=nbytes(obj.gradOld());//gradOld_
	//algorithm
		std::memcpy(arr+pos,&obj.algo(),sizeof(OPT_METHOD::type)); pos+=sizeof(OPT_METHOD::type);//optMethod_
		std::memcpy(arr+pos,&obj.optVal(),sizeof(OPT_VAL::type)); pos+=sizeof(OPT_VAL::type);//optVal_
}
template <> void pack(const SGD& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.period(),sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(arr+pos,&obj.decay(),sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);
}
template <> void pack(const SDM& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.eta(),sizeof(double)); pos+=sizeof(double);
}
template <> void pack(const NAG& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.eta(),sizeof(double)); pos+=sizeof(double);
}
template <> void pack(const ADAGRAD& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);
	pack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());
}
template <> void pack(const ADADELTA& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.eta(),sizeof(double)); pos+=sizeof(double);
	pack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());
	pack(obj.mdx2(),arr+pos); pos+=nbytes(obj.mdx2());
	pack(obj.dxv(),arr+pos); pos+=nbytes(obj.dxv());
}
template <> void pack(const RMSPROP& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);
	pack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());
}
template <> void pack(const ADAM& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.beta1i(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.beta2i(),sizeof(double)); pos+=sizeof(double);
	pack(obj.mgrad(),arr+pos); pos+=nbytes(obj.mgrad());
	pack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());
}
template <> void pack(const BFGS& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);
}
template <> void pack(const LM& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.period(),sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(arr+pos,&obj.decay(),sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.lambda(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.damp(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.min(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.max(),sizeof(double)); pos+=sizeof(double);
}
template <> void pack(const RPROP& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(arr+pos,&obj.period(),sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(arr+pos,&obj.decay(),sizeof(unsigned int)); pos+=sizeof(unsigned int);
	pack(obj.delta(),arr+pos); pos+=nbytes(obj.delta());
}

//**********************************************
// unpacking
//**********************************************

template <> void unpack(Opt& obj, const char* arr){
	unsigned int pos=0;
	//status
		std::memcpy(&obj.nStep(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//nStep_
		std::memcpy(&obj.nEval(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//nEval_
	//paramters
		std::memcpy(&obj.dim(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//dim_
		std::memcpy(&obj.val(),arr+pos,sizeof(double)); pos+=sizeof(double);//val_
		std::memcpy(&obj.valOld(),arr+pos,sizeof(double)); pos+=sizeof(double);//valOld_
		unpack(obj.x(),arr+pos); pos+=nbytes(obj.x());//x_
		unpack(obj.xOld(),arr+pos); pos+=nbytes(obj.xOld());//xOld_
		unpack(obj.grad(),arr+pos); pos+=nbytes(obj.grad());//grad_
		unpack(obj.gradOld(),arr+pos); pos+=nbytes(obj.gradOld());//gradOld_
	//algorithm
		std::memcpy(&obj.algo(),arr+pos,sizeof(OPT_METHOD::type)); pos+=sizeof(OPT_METHOD::type);//optMethod_
		std::memcpy(&obj.optVal(),arr+pos,sizeof(OPT_VAL::type)); pos+=sizeof(OPT_VAL::type);//optVal_
}
template <> void unpack(SGD& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.period(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(&obj.decay(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);
}
template <> void unpack(SDM& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.eta(),arr+pos,sizeof(double)); pos+=sizeof(double);
}
template <> void unpack(NAG& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.eta(),arr+pos,sizeof(double)); pos+=sizeof(double);
}
template <> void unpack(ADAGRAD& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);
	unpack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());
}
template <> void unpack(ADADELTA& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.eta(),arr+pos,sizeof(double)); pos+=sizeof(double);
	unpack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());
	unpack(obj.mdx2(),arr+pos); pos+=nbytes(obj.mdx2());
	unpack(obj.dxv(),arr+pos); pos+=nbytes(obj.dxv());
}
template <> void unpack(RMSPROP& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);
	unpack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());
}
template <> void unpack(ADAM& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.beta1i(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.beta2i(),arr+pos,sizeof(double)); pos+=sizeof(double);
	unpack(obj.mgrad(),arr+pos); pos+=nbytes(obj.mgrad());
	unpack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());
}
template <> void unpack(BFGS& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);
}
template <> void unpack(LM& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.period(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(&obj.decay(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.lambda(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.damp(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.min(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.max(),arr+pos,sizeof(double)); pos+=sizeof(double);
}
template <> void unpack(RPROP& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt&>(obj));
	std::memcpy(&obj.period(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(&obj.decay(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	unpack(obj.delta(),arr+pos); pos+=nbytes(obj.delta());
}

}