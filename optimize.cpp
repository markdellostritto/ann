// c libraries
#include <cmath>
// c++ libraries
#include <iostream>
#include <string>
// ann - eigen
#include "eigen.hpp"
// ann - math
#include "math_const.hpp"
#include "math_cmp.hpp"
#include "math_special.hpp"
// ann - string
#include "string.hpp"
// ann - print
#include "print.hpp"
// ann - optimize
#include "optimize.hpp"

namespace Opt{

//***************************************************
// optimization method
//***************************************************

std::ostream& operator<<(std::ostream& out, const ALGO::type& type){
	switch(type){
		case ALGO::SGD: out<<"SGD"; break;
		case ALGO::SDM: out<<"SDM"; break;
		case ALGO::NAG: out<<"NAG"; break;
		case ALGO::ADAGRAD: out<<"ADAGRAD"; break;
		case ALGO::ADADELTA: out<<"ADADELTA"; break;
		case ALGO::RMSPROP: out<<"RMSPROP"; break;
		case ALGO::ADAM: out<<"ADAM"; break;
		case ALGO::NADAM: out<<"NADAM"; break;
		case ALGO::BFGS: out<<"BFGS"; break;
		case ALGO::RPROP: out<<"RPROP"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

ALGO::type ALGO::read(const char* str){
	if(std::strcmp(str,"SGD")==0) return ALGO::SGD;
	else if(std::strcmp(str,"SDM")==0) return ALGO::SDM;
	else if(std::strcmp(str,"NAG")==0) return ALGO::NAG;
	else if(std::strcmp(str,"ADAGRAD")==0) return ALGO::ADAGRAD;
	else if(std::strcmp(str,"ADADELTA")==0) return ALGO::ADADELTA;
	else if(std::strcmp(str,"RMSPROP")==0) return ALGO::RMSPROP;
	else if(std::strcmp(str,"ADAM")==0) return ALGO::ADAM;
	else if(std::strcmp(str,"NADAM")==0) return ALGO::NADAM;
	else if(std::strcmp(str,"BFGS")==0) return ALGO::BFGS;
	else if(std::strcmp(str,"RPROP")==0) return ALGO::RPROP;
	else return ALGO::UNKNOWN;
}

//***************************************************
// optimization value
//***************************************************

std::ostream& operator<<(std::ostream& out, const VAL::type& type){
	switch(type){
		case VAL::FTOL_ABS: out<<"FTOL_ABS"; break;
		case VAL::FTOL_REL: out<<"FTOL_REL"; break;
		case VAL::XTOL_ABS: out<<"XTOL_ABS"; break;
		case VAL::XTOL_REL: out<<"XTOL_REL"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

VAL::type VAL::read(const char* str){
	if(std::strcmp(str,"XTOL_REL")==0) return VAL::XTOL_REL;
	else if(std::strcmp(str,"XTOL_ABS")==0) return VAL::XTOL_ABS;
	else if(std::strcmp(str,"FTOL_REL")==0) return VAL::FTOL_REL;
	else if(std::strcmp(str,"FTOL_ABS")==0) return VAL::FTOL_ABS;
	else return VAL::UNKNOWN;
}

//***************************************************
// decay method
//***************************************************

std::ostream& operator<<(std::ostream& out, const DECAY::type& type){
	switch(type){
		case DECAY::CONST: out<<"CONST"; break;
		case DECAY::EXP: out<<"EXP"; break;
		case DECAY::SQRT: out<<"SQRT"; break;
		case DECAY::INV: out<<"INV"; break;
		case DECAY::POW: out<<"POW"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

DECAY::type DECAY::read(const char* str){
	if(std::strcmp(str,"CONST")==0) return DECAY::CONST;
	else if(std::strcmp(str,"EXP")==0) return DECAY::EXP;
	else if(std::strcmp(str,"SQRT")==0) return DECAY::SQRT;
	else if(std::strcmp(str,"INV")==0) return DECAY::INV;
	else if(std::strcmp(str,"POW")==0) return DECAY::POW;
	else return DECAY::UNKNOWN;
}


//*********************************************
//Data class
//*********************************************

std::ostream& operator<<(std::ostream& out, const Data& data){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("DATA",str)<<"\n";
	out<<"VAL     = "<<data.val_<<"\n";
	out<<"N-PRINT = "<<data.nPrint_<<"\n";
	out<<"N-WRITE = "<<data.nWrite_<<"\n";
	out<<"STEP    = "<<data.step_<<"\n";
	out<<"TOL     = "<<data.tol_<<"\n";
	out<<"MAX     = "<<data.max_<<"\n";
	out<<"DIM     = "<<data.dim_<<"\n";
	out<<"ALGO    = "<<data.algo_<<"\n";
	out<<"OPT-VAL = "<<data.optVal_<<"\n";
	out<<print::title("DATA",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void Data::defaults(){
	//status
		val_=0; valOld_=0;
		dv_=0; dp_=0;
	//count
		nPrint_=0;
		nWrite_=0;
		step_=0;
	//stopping
		tol_=0;
		max_=0;
	//parameters
		dim_=0;
	//algorithm
		algo_=ALGO::UNKNOWN;
		optVal_=VAL::UNKNOWN;
}

void Data::init(unsigned int dim){
	if(dim==0) throw std::invalid_argument("Invalid dimension");
	dim_=dim;
	p_=Eigen::VectorXd::Zero(dim_);
	pOld_=Eigen::VectorXd::Zero(dim_);
	g_=Eigen::VectorXd::Zero(dim_);
	gOld_=Eigen::VectorXd::Zero(dim_);
}

//*********************************************
//Model class
//*********************************************

//operators

std::ostream& operator<<(std::ostream& out, const Model& model){
	out<<"DIM    = "<<model.dim_<<"\n";
	out<<"DECAY  = "<<model.decay_<<"\n";
	out<<"GAMMA0 = "<<model.gamma0_<<"\n";
	out<<"GAMMA  = "<<model.gamma_<<"\n";
	out<<"ALPHA  = "<<model.alpha_<<"\n";
	out<<"POW    = "<<model.pow_;
	return out;
}

//member functions

void Model::defaults(){
	if(OPT_PRINT_FUNC>0) std::cout<<"Model::defaults():\n";
	dim_=0;
	algo_=ALGO::UNKNOWN;
	decay_=DECAY::CONST;
	gamma0_=0;
	gamma_=0;
	alpha_=1;
	pow_=1;
}

void Model::init(unsigned int n){
	if(OPT_PRINT_FUNC>0) std::cout<<"Model::init(unsigned int):\n";
	if(n==0) throw std::invalid_argument("Invalid optimization dimenstion.");
	dim_=n;
}

void Model::update_step(unsigned int step){
	switch(decay_){
		case DECAY::CONST: break;
		case DECAY::EXP: gamma_*=alpha_; break;
		case DECAY::SQRT: gamma_=gamma0_/std::sqrt(1.0+step*alpha_); break;
		case DECAY::INV: gamma_=gamma0_/(1.0+step*alpha_); break;
		case DECAY::POW: gamma_=gamma0_/std::pow(1.0+step*alpha_,pow_); break;
		default: break;
	}
}

//***************************************************
// Algorithms
//***************************************************

//steepest-descent

void SGD::defaults(){
	if(OPT_PRINT_FUNC>0) std::cout<<"Model::defaults():\n";
	algo_=ALGO::SGD;
}

std::ostream& operator<<(std::ostream& out, const SGD& sgd){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("SGD",str)<<"\n";
	out<<static_cast<const Model&>(sgd)<<"\n";
	out<<print::title("SGD",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void SGD::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"SGD::step(Data&):\n";
	//update gradient step
	update_step(d.step());
	//compute new position
	d.p().noalias()-=gamma_*d.g();
}

void SGD::init(unsigned int dim){
	if(OPT_PRINT_FUNC>0) std::cout<<"SGD::init(unsigned int):\n";
	Model::init(dim);
}

//steepest-descent + momentum

void SDM::defaults(){
	algo_=ALGO::SDM;
	eta_=0.9;
}

std::ostream& operator<<(std::ostream& out, const SDM& sdm){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("SDM",str)<<"\n";
	out<<static_cast<const Model&>(sdm)<<"\n";
	out<<"ETA   = "<<sdm.eta_<<"\n";
	out<<print::title("SDM",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void SDM::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"SDM::step(Data&):\n";
	//update gradient step
	update_step(d.step());
	//compute step
	dx_*=eta_;
	dx_.noalias()+=gamma_*d.g();
	//compute new position
	d.p().noalias()-=dx_;
}

void SDM::init(unsigned int dim){
	if(OPT_PRINT_FUNC>0) std::cout<<"SDM::init(unsigned int):\n";
	Model::init(dim);
	dx_=Eigen::VectorXd::Zero(dim);
}

//nesterov accelerated gradient

void NAG::defaults(){
	algo_=ALGO::NAG;
	eta_=0.9;
}

std::ostream& operator<<(std::ostream& out, const NAG& nag){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NAG",str)<<"\n";
	out<<static_cast<const Model&>(nag)<<"\n";
	out<<"ETA   = "<<nag.eta_<<"\n";
	out<<print::title("NAG",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;return out;
}

void NAG::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"NAG::step(Data&):\n";
	const Eigen::VectorXd& g=d.g();
	//update gradient step
	update_step(d.step());
	//compute step
	dx_*=eta_;
	if(d.step()==0) dx_.noalias()+=gamma_*g;
	else dx_.noalias()+=gamma_*(eta_+1.0)*g;
	//calculate the new position
	d.p().noalias()-=dx_;
}

void NAG::init(unsigned int dim){
	if(OPT_PRINT_FUNC>0) std::cout<<"NAG::init(unsigned int):\n";
	Model::init(dim);
	dx_=Eigen::VectorXd::Zero(dim);
}

//ADAGRAD

const double ADAGRAD::eps_=1e-8;

void ADAGRAD::defaults(){
	algo_=ALGO::ADAGRAD;
}

std::ostream& operator<<(std::ostream& out, const ADAGRAD& adagrad){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ADAGRAD",str)<<"\n";
	out<<static_cast<const Model&>(adagrad)<<"\n";
	out<<print::title("ADAGRAD",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void ADAGRAD::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"ADAGRAD::step(Data&):\n";
	//local variables
	const Eigen::VectorXd& g=d.g();
	//update gradient step
	update_step(d.step());
	//add to the running average of the square of the gradients
	mgrad2_.noalias()+=g.cwiseProduct(g);
	//calculate the new position
	for(int n=dim_-1; n>=0; --n){
		d.p()[n]-=gamma_*g[n]/std::sqrt(mgrad2_[n]+eps_);
	}
}

void ADAGRAD::init(unsigned int dim){
	if(OPT_PRINT_FUNC>0) std::cout<<"ADAGRAD::init(unsigned int):\n";
	Model::init(dim);
	mgrad2_=Eigen::VectorXd::Zero(dim);
}

//ADADELTA

const double ADADELTA::eps_=1e-8;

void ADADELTA::defaults(){
	algo_=ALGO::ADADELTA;
	eta_=0;
}

std::ostream& operator<<(std::ostream& out, const ADADELTA& adadelta){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ADADELTA",str)<<"\n";
	out<<static_cast<const Model&>(adadelta)<<"\n";
	out<<"ETA   = "<<adadelta.eta_<<"\n";
	out<<print::title("ADADELTA",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void ADADELTA::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"ADADELTA::step(Data&):\n";
	//local variables
	const Eigen::VectorXd& g=d.g();
	//update gradient step
	update_step(d.step());
	//add to the running average of the square of the gradients
	mgrad2_*=eta_;
	mgrad2_.noalias()+=(1.0-eta_)*g.cwiseProduct(g);
	//calculate the new position
	for(int n=dim_-1; n>=0; --n){
		dx_[n]=g[n]*std::sqrt(mdx2_[n]+eps_)/std::sqrt(mgrad2_[n]+eps_);
		d.p()[n]-=dx_[n];
	}
	//add to the running average of the square of the deltas
	mdx2_*=eta_;
	mdx2_.noalias()+=(1.0-eta_)*dx_.cwiseProduct(dx_);
}

void ADADELTA::init(unsigned int dim){
	if(OPT_PRINT_FUNC>0) std::cout<<"ADADELTA::init(unsigned int):\n";
	Model::init(dim);
	eta_=0.9;
	mgrad2_=Eigen::VectorXd::Zero(dim);
	mdx2_=Eigen::VectorXd::Zero(dim);
	dx_=Eigen::VectorXd::Zero(dim);
}

//RMSPROP

const double RMSPROP::eps_=1e-8;

void RMSPROP::defaults(){
	algo_=ALGO::RMSPROP;
}

std::ostream& operator<<(std::ostream& out, const RMSPROP& rmsprop){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("RMSPROP",str)<<"\n";
	out<<static_cast<const Model&>(rmsprop)<<"\n";
	out<<print::title("RMSPROP",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void RMSPROP::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"RMSPROP::step(Data&):\n";
	//local variables
	const Eigen::VectorXd& g=d.g();
	//update gradient step
	update_step(d.step());
	//add to the running average of the square of the gradients
	mgrad2_*=0.9;
	mgrad2_.noalias()+=0.1*g.cwiseProduct(g);
	//calculate the new position
	for(int n=dim_-1; n>=0; --n){
		d.p()[n]-=gamma_*g[n]/std::sqrt(mgrad2_[n]+eps_);
	}
}

void RMSPROP::init(unsigned int dim){
	if(OPT_PRINT_FUNC>0) std::cout<<"RMSPROP::init(unsigned int):\n";
	Model::init(dim);
	mgrad2_=Eigen::VectorXd::Zero(dim);
}

//ADAM

const double ADAM::eps_=1e-8;
const double ADAM::beta1_=0.9;
const double ADAM::beta2_=0.999;
	
void ADAM::defaults(){
	algo_=ALGO::ADAM;
	beta1i_=beta1_;
	beta2i_=beta2_;
}

std::ostream& operator<<(std::ostream& out, const ADAM& adam){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ADAM",str)<<"\n";
	out<<static_cast<const Model&>(adam)<<"\n";
	out<<print::title("ADAM",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void ADAM::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"ADAM::step(Data&):\n";
	//local variables
	const Eigen::VectorXd& g=d.g();
	//update gradient step
	update_step(d.step());
	//add to the running average of the gradients
	mgrad_*=beta1_;
	mgrad_.noalias()+=(1.0-beta1_)*g;
	//add to the running average of the square of the gradients
	mgrad2_*=beta2_;
	mgrad2_.noalias()+=(1.0-beta2_)*g.cwiseProduct(g);
	//calculate the new position
	const double fac=gamma_*std::sqrt(1.0-beta2i_)/(1.0-beta1i_);
	for(int n=dim_-1; n>=0; --n){
		//d.p()[n]-=gamma_*mgrad_[n]/((1.0-beta1i_)*(std::sqrt(mgrad2_[n]/(1.0-beta2i_))+eps_));
		d.p()[n]-=fac*mgrad_[n]/(std::sqrt(mgrad2_[n])+eps_);
	}
	//update the powers of betas
	beta1i_*=beta1_;
	beta2i_*=beta2_;
}

void ADAM::init(unsigned int dim){
	if(OPT_PRINT_FUNC>0) std::cout<<"ADAM::init(unsigned int):\n";
	Model::init(dim);
	mgrad_=Eigen::VectorXd::Zero(dim);
	mgrad2_=Eigen::VectorXd::Zero(dim);
	beta1i_=beta1_;//power w.r.t i
	beta2i_=beta2_;//power w.r.t i
}

//NADAM

const double NADAM::eps_=1e-8;
const double NADAM::beta1_=0.9;
const double NADAM::beta2_=0.999;

void NADAM::defaults(){
	algo_=ALGO::NADAM;
	beta1i_=beta1_;
	beta2i_=beta2_;
}

std::ostream& operator<<(std::ostream& out, const NADAM& nadam){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NADAM",str)<<"\n";
	out<<static_cast<const Model&>(nadam)<<"\n";
	out<<print::title("NADAM",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void NADAM::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"NADAM::step(Data&):\n";
	//local variables
	const Eigen::VectorXd& g=d.g();
	//update gradient step
	update_step(d.step());
	//add to the running average of the gradients
	mgrad_*=beta1_;
	mgrad_.noalias()+=(1.0-beta1_)*g;
	//add to the running average of the square of the gradients
	mgrad2_*=beta2_;
	mgrad2_.noalias()+=(1.0-beta2_)*g.cwiseProduct(g);
	//calculate the new position
	const double fac=gamma_*std::sqrt(1.0-beta2i_)/(1.0-beta1i_);
	for(int n=dim_-1; n>=0; --n){
		//d.p()[n]-=gamma_*(beta1_*mgrad_[n]+(1.0-beta1_)*g[n])/((1.0-beta1i_)*(std::sqrt(mgrad2_[n]/(1.0-beta2i_))+eps_));
		d.p()[n]-=fac*(beta1_*mgrad_[n]+(1.0-beta1_)*g[n])/(std::sqrt(mgrad2_[n])+eps_);
	}
	//update the powers of betas
	beta1i_*=beta1_;
	beta2i_*=beta2_;
}

void NADAM::init(unsigned int dim){
	if(OPT_PRINT_FUNC>0) std::cout<<"NADAM::init(unsigned int):\n";
	Model::init(dim);
	mgrad_=Eigen::VectorXd::Zero(dim);
	mgrad2_=Eigen::VectorXd::Zero(dim);
	beta1i_=beta1_;//power w.r.t i
	beta2i_=beta2_;//power w.r.t i
}

//BFGS

void BFGS::defaults(){
	algo_=ALGO::BFGS;
}

std::ostream& operator<<(std::ostream& out, const BFGS& bfgs){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("BFGS",str)<<"\n";
	out<<static_cast<const Model&>(bfgs)<<"\n";
	out<<print::title("BFGS",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void BFGS::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"BFGS::step(Data&):\n";
	//update gradient step
	update_step(d.step());
	//obtain direction
	s_.noalias()=B_.llt().solve(d.g());
	//set the s vector
	s_*=-gamma_/s_.norm();
	//calculate new position
	d.p().noalias()+=s_;
	//set the y vector
	y_.noalias()=gamma_*(d.g()-d.gOld());
	//set the new B matrix
	B_.noalias()+=y_*y_.transpose()/y_.dot(s_);
	y_.noalias()=BOld_*s_;
	B_.noalias()-=y_*y_.transpose()/(s_.dot(y_));
	//set the old B matrix
	BOld_=B_;
}

void BFGS::init(unsigned int dim){
	if(OPT_PRINT_FUNC>0) std::cout<<"BFGS::init(unsigned int):\n";
	Model::init(dim);
	B_=Eigen::MatrixXd::Identity(dim,dim);
	BOld_=Eigen::MatrixXd::Identity(dim,dim);
	s_=Eigen::VectorXd::Zero(dim);
	y_=Eigen::VectorXd::Zero(dim);
}

//RPROP

const double RPROP::etaP=1.2;
const double RPROP::etaM=0.5;
const double RPROP::deltaMax=50.0;
const double RPROP::deltaMin=1e-14;

void RPROP::defaults(){
	algo_=ALGO::RPROP;
}

std::ostream& operator<<(std::ostream& out, const RPROP& rprop){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("RPROP",str)<<"\n";
	out<<static_cast<const Model&>(rprop)<<"\n";
	out<<print::title("RPROP",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void RPROP::step(Data& d){
	if(OPT_PRINT_FUNC>1) std::cout<<"RPROP::step(Data&):\n";
	//update gradient step
	update_step(d.step());
	//iRProp+
	const bool inc=d.val()>d.valOld();
	for(int n=dim_-1; n>=0; --n){
		const double s=d.g()[n]*d.gOld()[n];
		if(s>0){
			delta_[n]=cmp::min(delta_[n]*etaP,deltaMax);
			dx_[n]=-1.0*special::sign(d.g()[n])*delta_[n];
			d.p()[n]+=dx_[n];
		}else if(s<0){
			delta_[n]=cmp::max(delta_[n]*etaM,deltaMin);
			if(inc) d.p()[n]-=dx_[n];
			d.g()[n]=0.0;
		} else if(s==0){
			dx_[n]=-1.0*special::sign(d.g()[n])*delta_[n];
			d.p()[n]+=dx_[n];
		}
	}
	//iRProp-
	/*for(int n=dim_-1; n>=0; --n){
		const double s=grad[n]*gradOld_[n];
		if(s>0){
			delta_[n]=cmp::min(delta_[n]*etaP,deltaMax);
		}else if(s<0){
			delta_[n]=cmp::max(delta_[n]*etaM,deltaMin);
			grad[n]=0.0;
		}
		x[n]-=special::sign(grad[n])*delta_[n];
	}*/
}

void RPROP::init(unsigned int dim){
	if(OPT_PRINT_FUNC>1) std::cout<<"RPROP::init(unsigned int):\n";
	Model::init(dim);
	delta_=Eigen::VectorXd::Constant(dim,0.1);
	dx_=Eigen::VectorXd::Constant(dim,0.0);
}

//reading - file

Model& read(Model& model, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(Model&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(model,reader);
	fclose(reader); reader=NULL;
	return model;
} 

Data& read(Data& data, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(Data&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(data,reader);
	fclose(reader); reader=NULL;
	return data;
} 

SGD& read(SGD& sgd, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(SGD&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(sgd),reader);
	read(sgd,reader);
	fclose(reader); reader=NULL;
	return sgd;
}

SDM& read(SDM& sdm, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(SDM&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(sdm),reader);
	read(sdm,reader);
	fclose(reader); reader=NULL;
	return sdm;
}

NAG& read(NAG& nag, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(NAG&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(nag),reader);
	read(nag,reader);
	fclose(reader); reader=NULL;
	return nag;
}

ADAGRAD& read(ADAGRAD& adagrad, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(ADAGRAD&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(adagrad),reader);
	read(adagrad,reader);
	fclose(reader); reader=NULL;
	return adagrad;
}

ADADELTA& read(ADADELTA& adadelta, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(ADADELTA&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(adadelta),reader);
	read(adadelta,reader);
	fclose(reader); reader=NULL;
	return adadelta;
}

RMSPROP& read(RMSPROP& rmsprop, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(RMSPROP&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(rmsprop),reader);
	read(rmsprop,reader);
	fclose(reader); reader=NULL;
	return rmsprop;
}

ADAM& read(ADAM& adam, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(ADAM&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(adam),reader);
	read(adam,reader);
	fclose(reader); reader=NULL;
	return adam;
}

NADAM& read(NADAM& nadam, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(NADAM&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(nadam),reader);
	read(nadam,reader);
	fclose(reader); reader=NULL;
	return nadam;
}

BFGS& read(BFGS& bfgs, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(BFGS&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(bfgs),reader);
	read(bfgs,reader);
	fclose(reader); reader=NULL;
	return bfgs;
}

RPROP& read(RPROP& rprop, const char* file){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(RPROP&,const char*):\n";
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open file.");
	read(static_cast<Model&>(rprop),reader);
	read(rprop,reader);
	fclose(reader); reader=NULL;
	return rprop;
}

//reading - file pointer

Data& read(Data& data, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(Data&,FILE*):\n";
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.size()<2) throw std::runtime_error("Parameter tag without corresponding value.");
		if(strlist.at(0)=="ALGO"){
			data.algo()=ALGO::read(string::to_upper(strlist.at(1)).c_str());
		} else if(strlist.at(0)=="OPT_VAL"){
			data.optVal()=VAL::read(string::to_upper(strlist.at(1)).c_str());
		} else if(strlist.at(0)=="TOL"){
			data.tol()=std::atof(strlist.at(1).c_str());
		} else if(strlist.at(0)=="MAX_ITER"){
			data.max()=std::atof(strlist.at(1).c_str());
		} else if(strlist.at(0)=="N_PRINT"){
			data.nPrint()=std::atoi(strlist.at(1).c_str());
		} else if(strlist.at(0)=="N_WRITE"){
			data.nWrite()=std::atoi(strlist.at(1).c_str());
		} 
	}
	delete[] input;
	return data;
}

Model& read(Model& model, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(Model&,FILE*):\n";
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="GAMMA"){
			model.gamma()=std::atof(strlist.at(1).c_str());
			model.gamma0()=model.gamma();
		} else if(strlist.at(0)=="ALPHA"){
			model.alpha()=std::atof(strlist.at(1).c_str());
		} else if(strlist.at(0)=="DECAY"){
			model.decay()=DECAY::read(string::to_upper(strlist.at(1)).c_str());
		} else if(strlist.at(0)=="POW"){
			model.pow()=std::atof(strlist.at(1).c_str());
		}
	}
	delete[] input;
	return model;
}

SGD& read(SGD& sgd, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(SGD&,FILE*):\n";
	read(static_cast<Model&>(sgd),reader);
	return sgd;
}

SDM& read(SDM& sdm, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(SDM&,FILE*):\n";
	read(static_cast<Model&>(sdm),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="ETA"){
			sdm.eta()=std::atof((strlist.at(1)).c_str());
		}
	}
	delete[] input;
	return sdm;
}

NAG& read(NAG& nag, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(NAG&,FILE*):\n";
	read(static_cast<Model&>(nag),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="ETA"){
			nag.eta()=std::atof((strlist.at(1)).c_str());
		} 
	}
	delete[] input;
	return nag;
}

ADAGRAD& read(ADAGRAD& adagrad, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(ADAGRAD&,FILE*):\n";
	read(static_cast<Model&>(adagrad),reader);
	return adagrad;
}

ADADELTA& read(ADADELTA& adadelta, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(ADADELTA&,FILE*):\n";
	read(static_cast<Model&>(adadelta),reader);
	fseek(reader,0,SEEK_SET);
	char* input=new char[string::M];
	std::vector<std::string> strlist;
	while(fgets(input,string::M,reader)!=NULL){
		string::trim_right(input,string::COMMENT);
		if(string::split(input,string::WS,strlist)==0) continue;
		string::to_upper(strlist.at(0));
		if(strlist.at(0)=="ETA"){
			adadelta.eta()=std::atof((strlist.at(1)).c_str());
		}
	}
	delete[] input;
	return adadelta;
}

RMSPROP& read(RMSPROP& rmsprop, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(RMSPROP&,FILE*):\n";
	read(static_cast<Model&>(rmsprop),reader);
	return rmsprop;
}

ADAM& read(ADAM& adam, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(ADAM&,FILE*):\n";
	read(static_cast<Model&>(adam),reader);
	return adam;
}

NADAM& read(NADAM& nadam, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(NADAM&,FILE*):\n";
	read(static_cast<Model&>(nadam),reader);
	return nadam;
}

BFGS& read(BFGS& bfgs, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(BFGS&,FILE*):\n";
	read(static_cast<Model&>(bfgs),reader);
	return bfgs;
}

RPROP& read(RPROP& rprop, FILE* reader){
	if(OPT_PRINT_FUNC>0) std::cout<<"read(RPROP&,FILE*):\n";
	read(static_cast<Model&>(rprop),reader);
	return rprop;
}

//operators - comparison

bool operator==(const SGD& obj1, const SGD& obj2){
	return (
		obj1.gamma()==obj2.gamma()
	);
}
bool operator==(const SDM& obj1, const SDM& obj2){
	return (
		obj1.gamma()==obj2.gamma() &&
		obj1.eta()==obj2.eta()
	);
}
bool operator==(const NAG& obj1, const NAG& obj2){
	return (
		obj1.gamma()==obj2.gamma() &&
		obj1.eta()==obj2.eta()
	);
}
bool operator==(const ADAGRAD& obj1, const ADAGRAD& obj2){
	return (
		obj1.gamma()==obj2.gamma()
	);
}
bool operator==(const ADADELTA& obj1, const ADADELTA& obj2){
	return (
		obj1.gamma()==obj2.gamma() && 
		obj1.eta()==obj2.eta()
	);
}
bool operator==(const RMSPROP& obj1, const RMSPROP& obj2){
	return (
		obj1.gamma()==obj2.gamma()
	);
}
bool operator==(const ADAM& obj1, const ADAM& obj2){
	return (
		obj1.gamma()==obj2.gamma()
	);
}
bool operator==(const NADAM& obj1, const NADAM& obj2){
	return (
		obj1.gamma()==obj2.gamma()
	);
}
bool operator==(const BFGS& obj1, const BFGS& obj2){
	return (
		obj1.gamma()==obj2.gamma()
	);
}
bool operator==(const RPROP& obj1, const RPROP& obj2){
	return true;
}

}

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************
	
template <> unsigned int nbytes(const Opt::Data& obj){
	unsigned int size=0;
	//count
		size+=sizeof(unsigned int);//nPrint_
		size+=sizeof(unsigned int);//nWrite_
		size+=sizeof(unsigned int);//step_
	//stopping
		size+=sizeof(unsigned int);//max_
		size+=sizeof(double);//tol_
	//status
		size+=sizeof(double);//val_;
		size+=sizeof(double);//valOld_;
		size+=sizeof(double);//dv_;
		size+=sizeof(double);//dp_;
	//algorithm
		size+=sizeof(Opt::ALGO::type);//algo_
		size+=sizeof(Opt::VAL::type);//optVal_
	//parameters
		size+=sizeof(unsigned int);//dim_
		size+=nbytes(obj.p());//p_
		size+=nbytes(obj.pOld());//pOld_
		size+=nbytes(obj.g());//g_
		size+=nbytes(obj.gOld());//gOld_
	//return the size
		return size;
}
template <> unsigned int nbytes(const Opt::Model& obj){
	unsigned int size=0;
	size+=sizeof(unsigned int);//dim_
	size+=sizeof(Opt::ALGO::type);//algo_
	size+=sizeof(Opt::DECAY::type);//decay_
	size+=sizeof(double);//gamma0_
	size+=sizeof(double);//gamma_
	size+=sizeof(double);//alpha_
	size+=sizeof(double);//pow_
	return size;
}
template <> unsigned int nbytes(const Opt::SGD& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	return size;
}
template <> unsigned int nbytes(const Opt::SDM& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	size+=sizeof(double);//eta_
	return size;
}
template <> unsigned int nbytes(const Opt::NAG& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	size+=sizeof(double);//eta_
	size+=nbytes(obj.dx());//dx_
	return size;
}
template <> unsigned int nbytes(const Opt::ADAGRAD& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	size+=nbytes(obj.mgrad2());//mgrad2_
	return size;
}
template <> unsigned int nbytes(const Opt::ADADELTA& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	size+=sizeof(double);//eta_
	size+=nbytes(obj.mgrad2());//mgrad2_
	size+=nbytes(obj.mdx2());//mdx2_
	size+=nbytes(obj.dx());//dx_
	return size;
}
template <> unsigned int nbytes(const Opt::RMSPROP& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	size+=nbytes(obj.mgrad2());//mgrad2_
	return size;
}
template <> unsigned int nbytes(const Opt::ADAM& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	size+=sizeof(double);//beta1i
	size+=sizeof(double);//beta2i
	size+=nbytes(obj.mgrad());//mgrad_
	size+=nbytes(obj.mgrad2());//mgrad2_
	return size;
}
template <> unsigned int nbytes(const Opt::NADAM& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	size+=sizeof(double);//beta1i_
	size+=sizeof(double);//beta2i_
	size+=nbytes(obj.mgrad());//mgrad_
	size+=nbytes(obj.mgrad2());//mgrad2_
	return size;
}
template <> unsigned int nbytes(const Opt::BFGS& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	return size;
}
template <> unsigned int nbytes(const Opt::RPROP& obj){
	unsigned int size=0;
	size+=nbytes(static_cast<const Opt::Model&>(obj));
	size+=nbytes(obj.delta());//delta_
	size+=nbytes(obj.dx());//dx_
	return size;
}

//**********************************************
// packing
//**********************************************

template <> unsigned int pack(const Opt::Data& obj, char* arr){
	unsigned int pos=0;
	//count
		std::memcpy(arr+pos,&obj.nPrint(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//nPrint_
		std::memcpy(arr+pos,&obj.nWrite(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//nWrite_
		std::memcpy(arr+pos,&obj.step(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//step_
	//stopping
		std::memcpy(arr+pos,&obj.max(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//max_
		std::memcpy(arr+pos,&obj.tol(),sizeof(double)); pos+=sizeof(double);//tol_
	//status
		std::memcpy(arr+pos,&obj.val(),sizeof(double)); pos+=sizeof(double);//val_
		std::memcpy(arr+pos,&obj.valOld(),sizeof(double)); pos+=sizeof(double);//valOld_
		std::memcpy(arr+pos,&obj.dv(),sizeof(double)); pos+=sizeof(double);//dv_
		std::memcpy(arr+pos,&obj.dp(),sizeof(double)); pos+=sizeof(double);//dp_
	//algorithm
		std::memcpy(arr+pos,&obj.algo(),sizeof(Opt::ALGO::type)); pos+=sizeof(Opt::ALGO::type);//algo_
		std::memcpy(arr+pos,&obj.optVal(),sizeof(Opt::VAL::type)); pos+=sizeof(Opt::VAL::type);//optVal_
	//parameters
		std::memcpy(arr+pos,&obj.dim(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//dim_
		pack(obj.p(),arr+pos); pos+=nbytes(obj.p());//p_
		pack(obj.pOld(),arr+pos); pos+=nbytes(obj.pOld());//pOld_
		pack(obj.g(),arr+pos); pos+=nbytes(obj.g());//g_
		pack(obj.gOld(),arr+pos); pos+=nbytes(obj.gOld());//gOld_
	//return bytes written
	return pos;
}
template <> unsigned int pack(const Opt::Model& obj, char* arr){
	unsigned int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(unsigned int)); pos+=sizeof(unsigned int);//dim_
	std::memcpy(arr+pos,&obj.algo(),sizeof(Opt::ALGO::type)); pos+=sizeof(Opt::ALGO::type);//algo_
	std::memcpy(arr+pos,&obj.decay(),sizeof(Opt::DECAY::type)); pos+=sizeof(Opt::DECAY::type);//decay_
	std::memcpy(arr+pos,&obj.gamma0(),sizeof(double)); pos+=sizeof(double);//gamma0_
	std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);//gamma_
	std::memcpy(arr+pos,&obj.alpha(),sizeof(double)); pos+=sizeof(double);//alpha_
	std::memcpy(arr+pos,&obj.pow(),sizeof(double)); pos+=sizeof(double);//pow_
	return pos;
}
template <> unsigned int pack(const Opt::SGD& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	return pos;
}
template <> unsigned int pack(const Opt::SDM& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(arr+pos,&obj.eta(),sizeof(double)); pos+=sizeof(double);//eta
	return pos;
}
template <> unsigned int pack(const Opt::NAG& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(arr+pos,&obj.eta(),sizeof(double)); pos+=sizeof(double);//eta
	pack(obj.dx(),arr+pos); pos+=nbytes(obj.dx());//dx_
	return pos;
}
template <> unsigned int pack(const Opt::ADAGRAD& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	pack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	return pos;
}
template <> unsigned int pack(const Opt::ADADELTA& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(arr+pos,&obj.eta(),sizeof(double)); pos+=sizeof(double);
	pack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	pack(obj.mdx2(),arr+pos); pos+=nbytes(obj.mdx2());//mdx2_
	pack(obj.dx(),arr+pos); pos+=nbytes(obj.dx());//dx_
	return pos;
}
template <> unsigned int pack(const Opt::RMSPROP& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	pack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	return pos;
}
template <> unsigned int pack(const Opt::ADAM& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(arr+pos,&obj.beta1i(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.beta2i(),sizeof(double)); pos+=sizeof(double);
	pack(obj.mgrad(),arr+pos); pos+=nbytes(obj.mgrad());//mgrad_
	pack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	return pos;
}
template <> unsigned int pack(const Opt::NADAM& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(arr+pos,&obj.beta1i(),sizeof(double)); pos+=sizeof(double);//beta1i_
	std::memcpy(arr+pos,&obj.beta2i(),sizeof(double)); pos+=sizeof(double);//beta2i_
	pack(obj.mgrad(),arr+pos); pos+=nbytes(obj.mgrad());//mgrad_
	pack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	return pos;
}
template <> unsigned int pack(const Opt::BFGS& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	return pos;
}
template <> unsigned int pack(const Opt::RPROP& obj, char* arr){
	unsigned int pos=0;
	pack(static_cast<const Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	pack(obj.delta(),arr+pos); pos+=nbytes(obj.delta());//delta_
	pack(obj.dx(),arr+pos); pos+=nbytes(obj.dx());//dex_
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> unsigned int unpack(Opt::Data& obj, const char* arr){
	unsigned int pos=0;
	//count
		std::memcpy(&obj.nPrint(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//nPrint_
		std::memcpy(&obj.nWrite(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//nWrite_
		std::memcpy(&obj.step(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//step_
	//stopping
		std::memcpy(&obj.max(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//max_
		std::memcpy(&obj.tol(),arr+pos,sizeof(double)); pos+=sizeof(double);//tol_
	//status
		std::memcpy(&obj.val(),arr+pos,sizeof(double)); pos+=sizeof(double);//val_
		std::memcpy(&obj.valOld(),arr+pos,sizeof(double)); pos+=sizeof(double);//valOld_
		std::memcpy(&obj.dv(),arr+pos,sizeof(double)); pos+=sizeof(double);//dv_
		std::memcpy(&obj.dp(),arr+pos,sizeof(double)); pos+=sizeof(double);//dp_
	//algorithm
		std::memcpy(&obj.algo(),arr+pos,sizeof(Opt::ALGO::type)); pos+=sizeof(Opt::ALGO::type);//algo_
		std::memcpy(&obj.optVal(),arr+pos,sizeof(Opt::VAL::type)); pos+=sizeof(Opt::VAL::type);//optVal_
	//parameters
		std::memcpy(&obj.dim(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//dim_
		unpack(obj.p(),arr+pos); pos+=nbytes(obj.p());//p_
		unpack(obj.pOld(),arr+pos); pos+=nbytes(obj.pOld());//pOld_
		unpack(obj.g(),arr+pos); pos+=nbytes(obj.g());//g_
		unpack(obj.gOld(),arr+pos); pos+=nbytes(obj.gOld());//gOld_
	//return bytes read
	return pos;
}
template <> unsigned int unpack(Opt::Model& obj, const char* arr){
	unsigned int pos=0;
	std::memcpy(&obj.dim(),arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);//dim_
	std::memcpy(&obj.algo(),arr+pos,sizeof(Opt::ALGO::type)); pos+=sizeof(Opt::ALGO::type);//algo_
	std::memcpy(&obj.decay(),arr+pos,sizeof(Opt::DECAY::type)); pos+=sizeof(Opt::DECAY::type);//decay_
	std::memcpy(&obj.gamma0(),arr+pos,sizeof(double)); pos+=sizeof(double);//gamma0_
	std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);//gamma_
	std::memcpy(&obj.alpha(),arr+pos,sizeof(double)); pos+=sizeof(double);//alpha_
	std::memcpy(&obj.pow(),arr+pos,sizeof(double)); pos+=sizeof(double);//pow_
	return pos;
}
template <> unsigned int unpack(Opt::SGD& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	return pos;
}
template <> unsigned int unpack(Opt::SDM& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(&obj.eta(),arr+pos,sizeof(double)); pos+=sizeof(double);//eta
	return pos;
}
template <> unsigned int unpack(Opt::NAG& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(&obj.eta(),arr+pos,sizeof(double)); pos+=sizeof(double);//eta
	unpack(obj.dx(),arr+pos); pos+=nbytes(obj.dx());//dx
	return pos;
}
template <> unsigned int unpack(Opt::ADAGRAD& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	unpack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	return pos;
}
template <> unsigned int unpack(Opt::ADADELTA& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(&obj.eta(),arr+pos,sizeof(double)); pos+=sizeof(double);
	unpack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	unpack(obj.mdx2(),arr+pos); pos+=nbytes(obj.mdx2());//mdx2_
	unpack(obj.dx(),arr+pos); pos+=nbytes(obj.dx());//dx_
	return pos;
}
template <> unsigned int unpack(Opt::RMSPROP& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	unpack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	return pos;
}
template <> unsigned int unpack(Opt::ADAM& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(&obj.beta1i(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.beta2i(),arr+pos,sizeof(double)); pos+=sizeof(double);
	unpack(obj.mgrad(),arr+pos); pos+=nbytes(obj.mgrad());//mgrad_
	unpack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	return pos;
}
template <> unsigned int unpack(Opt::NADAM& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	std::memcpy(&obj.beta1i(),arr+pos,sizeof(double)); pos+=sizeof(double);//beta1i_
	std::memcpy(&obj.beta2i(),arr+pos,sizeof(double)); pos+=sizeof(double);//beta2i_
	unpack(obj.mgrad(),arr+pos); pos+=nbytes(obj.mgrad());//mgrad_
	unpack(obj.mgrad2(),arr+pos); pos+=nbytes(obj.mgrad2());//mgrad2_
	return pos;
}
template <> unsigned int unpack(Opt::BFGS& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	return pos;
}
template <> unsigned int unpack(Opt::RPROP& obj, const char* arr){
	unsigned int pos=0;
	unpack(static_cast<Opt::Model&>(obj),arr+pos); pos+=nbytes(static_cast<const Opt::Model&>(obj));
	unpack(obj.delta(),arr+pos); pos+=nbytes(obj.delta());//delta_
	unpack(obj.dx(),arr+pos); pos+=nbytes(obj.dx());//dx_
	return pos;
}

}
