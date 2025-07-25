// c++
#include <iostream>
#include <stdexcept>
// math
#include "math/special.hpp"
// opt
#include "opt/algo.hpp"

namespace opt{
namespace algo{

//***************************************************
// Name
//***************************************************

std::ostream& operator<<(std::ostream& out, const Name& name){
	switch(name){
		case Name::SGD: out<<"SGD"; break;
		case Name::SDM: out<<"SDM"; break;
		case Name::NAG: out<<"NAG"; break;
		case Name::ADAGRAD: out<<"ADAGRAD"; break;
		case Name::ADADELTA: out<<"ADADELTA"; break;
		case Name::RMSPROP: out<<"RMSPROP"; break;
		case Name::ADAM: out<<"ADAM"; break;
		case Name::ADAMW: out<<"ADAMW"; break;
		case Name::ADAB: out<<"ADAB"; break;
		case Name::YOGI: out<<"YOGI"; break;
		case Name::NOGI: out<<"NOGI"; break;
		case Name::NADAM: out<<"NADAM"; break;
		case Name::AMSGRAD: out<<"AMSGRAD"; break;
		case Name::BFGS: out<<"BFGS"; break;
		case Name::RPROP: out<<"RPROP"; break;
		case Name::CG: out<<"CG"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Name::name(const Name& name){
	switch(name){
		case Name::SGD: return "SGD";
		case Name::SDM: return "SDM";
		case Name::NAG: return "NAG";
		case Name::ADAGRAD: return "ADAGRAD";
		case Name::ADADELTA: return "ADADELTA";
		case Name::RMSPROP: return "RMSPROP";
		case Name::ADAM: return "ADAM";
		case Name::ADAMW: return "ADAMW";
		case Name::ADAB: return "ADAB";
		case Name::YOGI: return "YOGI";
		case Name::NOGI: return "NOGI";
		case Name::NADAM: return "NADAM";
		case Name::AMSGRAD: return "AMSGRAD";
		case Name::BFGS: return "BFGS";
		case Name::RPROP: return "RPROP";
		case Name::CG: return "CG";
		default: return "UNKNOWN";
	}
}

Name Name::read(const char* str){
	if(std::strcmp(str,"SGD")==0) return Name::SGD;
	else if(std::strcmp(str,"SDM")==0) return Name::SDM;
	else if(std::strcmp(str,"NAG")==0) return Name::NAG;
	else if(std::strcmp(str,"ADAGRAD")==0) return Name::ADAGRAD;
	else if(std::strcmp(str,"ADADELTA")==0) return Name::ADADELTA;
	else if(std::strcmp(str,"RMSPROP")==0) return Name::RMSPROP;
	else if(std::strcmp(str,"ADAM")==0) return Name::ADAM;
	else if(std::strcmp(str,"ADAMW")==0) return Name::ADAMW;
	else if(std::strcmp(str,"ADAB")==0) return Name::ADAB;
	else if(std::strcmp(str,"YOGI")==0) return Name::YOGI;
	else if(std::strcmp(str,"NOGI")==0) return Name::NOGI;
	else if(std::strcmp(str,"NADAM")==0) return Name::NADAM;
	else if(std::strcmp(str,"AMSGRAD")==0) return Name::AMSGRAD;
	else if(std::strcmp(str,"BFGS")==0) return Name::BFGS;
	else if(std::strcmp(str,"RPROP")==0) return Name::RPROP;
	else if(std::strcmp(str,"CG")==0) return Name::CG;
	else return Name::UNKNOWN;
}

//***************************************************
// Base
//***************************************************

void Base::resize(int dim){
	if(dim<=0) throw std::invalid_argument("opt::algo:Base::resiz(int): invalid dimension.");
	else dim_=dim;
}

//***************************************************
// SGD
//***************************************************

std::ostream& operator<<(std::ostream& out, const SGD& obj){
	return out<<obj.name();
}

void SGD::resize(int dim){
	Base::resize(dim);
}

void SGD::step(Objective& obj){
	//compute new position
	obj.p().noalias()-=obj.gamma()*obj.g();
}

//***************************************************
// SDM
//***************************************************

std::ostream& operator<<(std::ostream& out, const SDM& obj){
	return out<<obj.name()<<" eta "<<obj.eta();
}

void SDM::read(Token& token){
	eta_=std::atof(token.next().c_str());
	if(eta_<0.0 || eta_>1.0) throw std::invalid_argument("opt::algo::SDM::read(Token&): Invalid eta.");
};

void SDM::resize(int dim){
	Base::resize(dim);
	dx_=Eigen::VectorXd::Zero(dim_);
}

void SDM::step(Objective& obj){
	//compute step
	dx_*=eta_;
	dx_.noalias()+=obj.gamma()*obj.g();
	//compute new position
	obj.p().noalias()-=dx_;
}

//***************************************************
// NAG
//***************************************************

std::ostream& operator<<(std::ostream& out, const NAG& obj){
	return out<<obj.name()<<" eta "<<obj.eta();
}

void NAG::read(Token& token){
	eta_=std::atof(token.next().c_str());
	if(eta_<0.0 || eta_>1.0) throw std::invalid_argument("opt::algo::NAG::read(Token&): Invalid eta.");
};

void NAG::resize(int dim){
	Base::resize(dim);
	dx_=Eigen::VectorXd::Zero(dim_);
}

void NAG::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//compute step
	dx_*=eta_;
	//if(obj.step()==0) dx_.noalias()+=obj.gamma()*g;
	//else dx_.noalias()+=obj.gamma()*(eta_+1.0)*g;
	dx_.noalias()+=obj.gamma()*(eta_+1.0)*g;
	//calculate the new position
	obj.p().noalias()-=dx_;
}

//***************************************************
// ADAGRAD
//***************************************************

std::ostream& operator<<(std::ostream& out, const ADAGRAD& obj){
	return out<<obj.name()<<" eps "<<obj.eps();
}

void ADAGRAD::read(Token& token){
	eps_=std::atof(token.next().c_str());
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::NAG::read(Token&): Invalid eps.");
};
	
void ADAGRAD::resize(int dim){
	Base::resize(dim);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
}

void ADAGRAD::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//add to the running average of the square of the gradients
	mgrad2_.noalias()+=g.cwiseProduct(g);
	//calculate the new position
	for(int n=0; n<dim_; ++n){
		obj.p()[n]-=obj.gamma()*g[n]/(sqrt(mgrad2_[n])+eps_);
	}
}

//***************************************************
// ADADELTA
//***************************************************

std::ostream& operator<<(std::ostream& out, const ADADELTA& obj){
	return out<<obj.name()<<" eta "<<obj.eta()<<" eps "<<obj.eps();
}

void ADADELTA::read(Token& token){
	eta_=std::atof(token.next().c_str());
	eps_=std::atof(token.next().c_str());
	if(eta_<0.0 || eta_>1.0) throw std::invalid_argument("opt::algo::ADADELTA::read(Token&): Invalid eta.");
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::ADADELTA::read(Token&): Invalid eps.");
};

void ADADELTA::resize(int dim){
	Base::resize(dim);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
	mdx2_=Eigen::VectorXd::Zero(dim_);
	dx_=Eigen::VectorXd::Zero(dim_);
}

void ADADELTA::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//add to the running average of the square of the gradients
	mgrad2_*=eta_;
	mgrad2_.noalias()+=(1.0-eta_)*g.cwiseProduct(g);
	//calculate the new position
	for(int n=0; n<dim_; ++n){
		dx_[n]=g[n]*sqrt(mdx2_[n])/(sqrt(mgrad2_[n])+eps_);
		obj.p()[n]-=dx_[n];
	}
	//add to the running average of the square of the deltas
	mdx2_*=eta_;
	mdx2_.noalias()+=(1.0-eta_)*dx_.cwiseProduct(dx_);
}

//***************************************************
// RMSPROP
//***************************************************

std::ostream& operator<<(std::ostream& out, const RMSPROP& obj){
	return out<<obj.name()<<" eps "<<obj.eps();
}

void RMSPROP::read(Token& token){
	eps_=std::atof(token.next().c_str());
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::RMSPROP::read(Token&): Invalid eps.");
};

void RMSPROP::resize(int dim){
	Base::resize(dim);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
}

void RMSPROP::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//add to the running average of the square of the gradients
	mgrad2_*=0.9;
	mgrad2_.noalias()+=0.1*g.cwiseProduct(g);
	//calculate the new position
	for(int n=0; n<dim_; ++n){
		obj.p()[n]-=obj.gamma()*g[n]/(sqrt(mgrad2_[n])+eps_);
	}
}

//***************************************************
// ADAM
//***************************************************

const double ADAM::beta1_=0.9;
const double ADAM::beta2_=0.999;

std::ostream& operator<<(std::ostream& out, const ADAM& obj){
	return out<<obj.name()<<" eps "<<obj.eps();
}

void ADAM::read(Token& token){
	eps_=std::atof(token.next().c_str());
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::ADAM::read(Token&): Invalid eps.");
};

void ADAM::resize(int dim){
	Base::resize(dim);
	mgrad_=Eigen::VectorXd::Zero(dim_);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
	beta1i_=beta1_;//power w.r.t i
	beta2i_=beta2_;//power w.r.t i
}

void ADAM::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//compute step
	const double b1ii=1.0/(1.0-beta1i_);
	const double b2ii=1.0/(1.0-beta2i_);
	for(int n=0; n<dim_; ++n){
		//add to the running average of the gradients
		mgrad_[n]*=beta1_; mgrad_[n]+=(1.0-beta1_)*g[n];
		//add to the running average of the square of the gradients
		mgrad2_[n]*=beta2_; mgrad2_[n]+=(1.0-beta2_)*g[n]*g[n];
		//calculate the new position
		obj.p()[n]-=obj.gamma()*b1ii*mgrad_[n]/(sqrt(b2ii*mgrad2_[n])+eps_);
	}
	//update the powers of betas
	beta1i_*=beta1_;
	beta2i_*=beta2_;
}

//***************************************************
// ADAMW
//***************************************************

const double ADAMW::beta1_=0.9;
const double ADAMW::beta2_=0.999;

std::ostream& operator<<(std::ostream& out, const ADAMW& obj){
	return out<<obj.name()<<" w "<<obj.w()<<" eps "<<obj.eps();
}

void ADAMW::read(Token& token){
	w_=std::atof(token.next().c_str());
	eps_=std::atof(token.next().c_str());
	if(w_<0.0 || w_>1.0) throw std::invalid_argument("opt::algo::ADAMW::read(Token&): Invalid w.");
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::ADAMW::read(Token&): Invalid eps.");
};

void ADAMW::resize(int dim){
	Base::resize(dim);
	mgrad_=Eigen::VectorXd::Zero(dim_);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
	beta1i_=beta1_;//power w.r.t i
	beta2i_=beta2_;//power w.r.t i
}

void ADAMW::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//compute step
	const double b1ii=1.0/(1.0-beta1i_);
	const double b2ii=1.0/(1.0-beta2i_);
	for(int n=0; n<dim_; ++n){
		//add to the running average of the gradients
		mgrad_[n]*=beta1_; mgrad_[n]+=(1.0-beta1_)*g[n];
		//add to the running average of the square of the gradients
		mgrad2_[n]*=beta2_; mgrad2_[n]+=(1.0-beta2_)*g[n]*g[n];
		//calculate the new position
		obj.p()[n]-=obj.gamma()*(b1ii*mgrad_[n]/(sqrt(b2ii*mgrad2_[n])+eps_)+w_*obj.p()[n]);
	}
	//update the powers of betas
	beta1i_*=beta1_;
	beta2i_*=beta2_;
}

//***************************************************
// ADAB
//***************************************************

const double ADAB::beta1_=0.9;
const double ADAB::beta2_=0.999;

std::ostream& operator<<(std::ostream& out, const ADAB& obj){
	return out<<obj.name()<<" eps "<<obj.eps();
}

void ADAB::read(Token& token){
	eps_=std::atof(token.next().c_str());
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::ADAB::read(Token&): Invalid eps.");
};

void ADAB::resize(int dim){
	Base::resize(dim);
	mgrad_=Eigen::VectorXd::Zero(dim_);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
	beta1i_=beta1_;//power w.r.t i
	beta2i_=beta2_;//power w.r.t i
}

void ADAB::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//compute step
	const double b1ii=1.0/(1.0-beta1i_);
	const double b2ii=1.0/(1.0-beta2i_);
	for(int n=0; n<dim_; ++n){
		//add to the running average of the gradients
		mgrad_[n]*=beta1_; mgrad_[n]+=(1.0-beta1_)*g[n];
		//add to the running average of the square of the gradients
		mgrad2_[n]*=beta2_; mgrad2_[n]+=(1.0-beta2_)*(g[n]-mgrad_[n])*(g[n]-mgrad_[n])+eps_;
		//calculate the new position
		obj.p()[n]-=obj.gamma()*b1ii*mgrad_[n]/(sqrt(b2ii*mgrad2_[n])+eps_);
	}
	//update the powers of betas
	beta1i_*=beta1_;
	beta2i_*=beta2_;
}

//***************************************************
// YOGI
//***************************************************

const double YOGI::beta1_=0.9;
const double YOGI::beta2_=0.999;

std::ostream& operator<<(std::ostream& out, const YOGI& obj){
	return out<<obj.name()<<" eps "<<obj.eps();
}

void YOGI::read(Token& token){
	eps_=std::atof(token.next().c_str());
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::YOGI::read(Token&): Invalid eps.");
};

void YOGI::resize(int dim){
	Base::resize(dim);
	mgrad_=Eigen::VectorXd::Zero(dim_);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
	beta1i_=beta1_;//power w.r.t i
	beta2i_=beta2_;//power w.r.t i
}

void YOGI::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//compute step
	const double b1ii=1.0/(1.0-beta1i_);
	const double b2ii=1.0/(1.0-beta2i_);
	for(int n=0; n<dim_; ++n){
		//add to the running average of the gradients
		mgrad_[n]*=beta1_; mgrad_[n]+=(1.0-beta1_)*g[n];
		//add to the running average of the square of the gradients
		const double g2=g[n]*g[n];
		mgrad2_[n]-=(1.0-beta2_)*math::special::sgn(mgrad2_[n]-g2)*g2;
		//calculate the new position
		obj.p()[n]-=obj.gamma()*b1ii*mgrad_[n]/(sqrt(b2ii*mgrad2_[n])+eps_);
	}
	//update the powers of betas
	beta1i_*=beta1_;
	beta2i_*=beta2_;
}

//***************************************************
// NOGI
//***************************************************

const double NOGI::beta1_=0.9;
const double NOGI::beta2_=0.999;

std::ostream& operator<<(std::ostream& out, const NOGI& obj){
	return out<<obj.name()<<" eps "<<obj.eps();
}

void NOGI::read(Token& token){
	eps_=std::atof(token.next().c_str());
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::NOGI::read(Token&): Invalid eps.");
};

void NOGI::resize(int dim){
	Base::resize(dim);
	mgrad_=Eigen::VectorXd::Zero(dim_);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
	beta1i_=beta1_;//power w.r.t i
	beta2i_=beta2_;//power w.r.t i
}

void NOGI::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//compute step
	const double b1ii=1.0/(1.0-beta1i_);
	const double b2ii=1.0/(1.0-beta2i_);
	for(int n=0; n<dim_; ++n){
		//add to the running average of the gradients
		mgrad_[n]*=beta1_; mgrad_[n]+=(1.0-beta1_)*g[n];
		//add to the running average of the square of the gradients
		const double g2=g[n]*g[n];
		mgrad2_[n]-=(1.0-beta2_)*math::special::sgn(mgrad2_[n]-g2)*g2;
		//calculate the new position
		obj.p()[n]-=obj.gamma()*b1ii*(beta1_*mgrad_[n]+(1.0-beta1_)*g[n])/(sqrt(b2ii*mgrad2_[n])+eps_);
	}
	//update the powers of betas
	beta1i_*=beta1_;
	beta2i_*=beta2_;
}

//***************************************************
// NADAM
//***************************************************

const double NADAM::beta1_=0.9;
const double NADAM::beta2_=0.999;

std::ostream& operator<<(std::ostream& out, const NADAM& obj){
	return out<<obj.name()<<" eps "<<obj.eps();
}

void NADAM::read(Token& token){
	eps_=std::atof(token.next().c_str());
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::NADAM::read(Token&): Invalid eps.");
	eps2_=eps_*eps_;
};

void NADAM::resize(int dim){
	Base::resize(dim);
	mgrad_=Eigen::VectorXd::Zero(dim_);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
	beta1i_=beta1_;//power w.r.t i
	beta2i_=beta2_;//power w.r.t i
}

void NADAM::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//compute step
	const double b1ii=1.0/(1.0-beta1i_);
	const double b2ii=1.0/(1.0-beta2i_);
	for(int n=0; n<dim_; ++n){
		//add to the running average of the gradients
		mgrad_[n]*=beta1_; mgrad_[n]+=(1.0-beta1_)*g[n];
		//add to the running average of the square of the gradients
		mgrad2_[n]*=beta2_; mgrad2_[n]+=(1.0-beta2_)*g[n]*g[n];
		//calculate the new position
		obj.p()[n]-=obj.gamma()*b1ii*(beta1_*mgrad_[n]+(1.0-beta1_)*g[n])/(sqrt(b2ii*mgrad2_[n])+eps_);
	}
	//update the powers of betas
	beta1i_*=beta1_;
	beta2i_*=beta2_;
}

//***************************************************
// AMSGRAD
//***************************************************

const double AMSGRAD::beta1_=0.9;
const double AMSGRAD::beta2_=0.999;

std::ostream& operator<<(std::ostream& out, const AMSGRAD& obj){
	return out<<obj.name()<<" eps "<<obj.eps();
}

void AMSGRAD::read(Token& token){
	eps_=std::atof(token.next().c_str());
	if(eps_<0.0 || eps_>1.0) throw std::invalid_argument("opt::algo::AMSGRAD::read(Token&): Invalid eps.");
};

void AMSGRAD::resize(int dim){
	Base::resize(dim);
	mgrad_=Eigen::VectorXd::Zero(dim_);
	mgrad2_=Eigen::VectorXd::Zero(dim_);
	mgrad2m_=Eigen::VectorXd::Zero(dim_);
	beta1i_=beta1_;//power w.r.t i
	beta2i_=beta2_;//power w.r.t i
}

void AMSGRAD::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	//compute step
	const double b1ii=1.0/(1.0-beta1i_);
	const double b2ii=1.0/(1.0-beta2i_);
	for(int n=0; n<dim_; ++n){
		//add to the running average of the gradients
		mgrad_[n]*=beta1_; mgrad_[n]+=(1.0-beta1_)*g[n];
		//add to the running average of the square of the gradients
		mgrad2_[n]*=beta2_; mgrad2_[n]+=(1.0-beta2_)*g[n]*g[n];
		//compute the max
		if(mgrad2_[n]>mgrad2m_[n]) mgrad2m_[n]=mgrad2_[n];
		//calculate the new position
		obj.p()[n]-=obj.gamma()*b1ii*mgrad_[n]/(sqrt(b2ii*mgrad2m_[n])+eps_);
	}
	//update the powers of betas
	beta1i_*=beta1_;
	beta2i_*=beta2_;
}

//***************************************************
// BFGS
//***************************************************

std::ostream& operator<<(std::ostream& out, const BFGS& obj){
	return out<<obj.name();
}

void BFGS::step(Objective& obj){
	//obtain direction
	s_.noalias()=B_.llt().solve(obj.g());
	//set the s vector
	s_*=-obj.gamma()/s_.norm();
	//calculate new position
	obj.p().noalias()+=s_;
	//set the y vector
	y_.noalias()=obj.gamma()*(obj.g()-obj.gOld());
	//set the new B matrix
	B_.noalias()+=y_*y_.transpose()/y_.dot(s_);
	y_.noalias()=BOld_*s_;
	B_.noalias()-=y_*y_.transpose()/(s_.dot(y_));
	//set the old B matrix
	BOld_=B_;
}

void BFGS::resize(int dim){
	BFGS::resize(dim);
	B_=Eigen::MatrixXd::Identity(dim_,dim_);
	BOld_=Eigen::MatrixXd::Identity(dim_,dim_);
	s_=Eigen::VectorXd::Zero(dim_);
	y_=Eigen::VectorXd::Zero(dim_);
}

//***************************************************
// RPROP
//***************************************************

const double RPROP::etaP=1.2;
const double RPROP::etaM=0.5;
const double RPROP::deltaMax=50.0;
const double RPROP::deltaMin=1e-14;

std::ostream& operator<<(std::ostream& out, const RPROP& obj){
	return out<<obj.name();
}

void RPROP::resize(int dim){
	Base::resize(dim);
	delta_=Eigen::VectorXd::Constant(dim_,0.1);
	dx_=Eigen::VectorXd::Constant(dim_,0.0);
}

void RPROP::step(Objective& obj){
	//iRProp+
	const bool inc=obj.val()>obj.valOld();
	for(int n=0; n<dim_; ++n){
		const double s=obj.g()[n]*obj.gOld()[n];
		if(s>0){
			delta_[n]=math::special::min(delta_[n]*etaP,deltaMax);
			dx_[n]=-1.0*math::special::sgn(obj.g()[n])*delta_[n];
			obj.p()[n]+=dx_[n];
		}else if(s<0){
			delta_[n]=math::special::max(delta_[n]*etaM,deltaMin);
			if(inc) obj.p()[n]-=dx_[n];
			obj.g()[n]=0.0;
		} else if(s==0){
			dx_[n]=-1.0*math::special::sgn(obj.g()[n])*delta_[n];
			obj.p()[n]+=dx_[n];
		}
	}
}

//***************************************************
// CG
//***************************************************

std::ostream& operator<<(std::ostream& out, const CG& obj){
	return out<<obj.name();
}

void CG::resize(int dim){
	Base::resize(dim);
	cgd_=Eigen::VectorXd::Zero(dim_);
}

void CG::step(Objective& obj){
	const Eigen::VectorXd& g=obj.g();
	const Eigen::VectorXd& gOld=obj.gOld();
	//compute new cg direction
	//const double beta_=(obj.step()==0)?1.0:g.dot(g-gOld)/(gOld.squaredNorm()+eps_);
	const double beta_=g.dot(g-gOld)/(gOld.squaredNorm()+eps_);
	cgd_*=beta_;
	cgd_.noalias()-=g;
	//compute new position
	obj.p().noalias()+=obj.gamma()*cgd_;
}

//***************************************************
// Factory
//***************************************************

std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Base>& obj){
	switch(obj->name()){
		case Name::SGD: out<<static_cast<const SGD&>(*obj); break;
		case Name::SDM: out<<static_cast<const SDM&>(*obj); break;
		case Name::NAG: out<<static_cast<const NAG&>(*obj); break;
		case Name::ADAGRAD: out<<static_cast<const ADAGRAD&>(*obj); break;
		case Name::ADADELTA: out<<static_cast<const ADADELTA&>(*obj); break;
		case Name::RMSPROP: out<<static_cast<const RMSPROP&>(*obj); break;
		case Name::ADAM: out<<static_cast<const ADAM&>(*obj); break;
		case Name::ADAMW: out<<static_cast<const ADAMW&>(*obj); break;
		case Name::ADAB: out<<static_cast<const ADAB&>(*obj); break;
		case Name::YOGI: out<<static_cast<const YOGI&>(*obj); break;
		case Name::NOGI: out<<static_cast<const NOGI&>(*obj); break;
		case Name::NADAM: out<<static_cast<const NADAM&>(*obj); break;
		case Name::AMSGRAD: out<<static_cast<const AMSGRAD&>(*obj); break;
		case Name::BFGS: out<<static_cast<const BFGS&>(*obj); break;
		case Name::RPROP: out<<static_cast<const RPROP&>(*obj); break;
		case Name::CG: out<<static_cast<const CG&>(*obj); break;
		case Name::UNKNOWN: out<<"ALGO UKNOWN\n"; break;
	}
	return out;
}

std::shared_ptr<Base>& make(std::shared_ptr<Base>& obj, Name name){
	switch(name){
		case Name::SGD: obj.reset(new SGD()); break;
		case Name::SDM: obj.reset(new SDM()); break;
		case Name::NAG: obj.reset(new NAG()); break;
		case Name::ADAGRAD: obj.reset(new ADAGRAD()); break;
		case Name::ADADELTA: obj.reset(new ADADELTA()); break;
		case Name::RMSPROP: obj.reset(new RMSPROP()); break;
		case Name::ADAM: obj.reset(new ADAM()); break;
		case Name::ADAMW: obj.reset(new ADAMW()); break;
		case Name::ADAB: obj.reset(new ADAB()); break;
		case Name::YOGI: obj.reset(new YOGI()); break;
		case Name::NOGI: obj.reset(new NOGI()); break;
		case Name::NADAM: obj.reset(new NADAM()); break;
		case Name::AMSGRAD: obj.reset(new AMSGRAD()); break;
		case Name::BFGS: obj.reset(new BFGS()); break;
		case Name::RPROP: obj.reset(new RPROP()); break;
		case Name::CG: obj.reset(new CG()); break;
		case Name::UNKNOWN: throw std::invalid_argument("opt::algo::make(Base*,Name): Invalid algorithm name."); break;
	}
	return obj;
}

std::shared_ptr<Base>& read(std::shared_ptr<Base>& obj, Token& token){
	Name name=Name::read(string::to_upper(token.next()).c_str());
	make(obj,name);
	obj->read(token);
	return obj;
}

}
}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const std::shared_ptr<opt::algo::Base>& obj){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"nbytes(const std::shared_ptr<opt::algo::Base>&)\n";
	int size=0;
	size+=sizeof(int);
	if(obj!=nullptr){
		size+=sizeof(opt::algo::Name);//name
		switch(obj->name()){
			case opt::algo::Name::SGD: size+=nbytes(static_cast<const opt::algo::SGD&>(*obj)); break;
			case opt::algo::Name::SDM: size+=nbytes(static_cast<const opt::algo::SDM&>(*obj)); break;
			case opt::algo::Name::NAG: size+=nbytes(static_cast<const opt::algo::NAG&>(*obj)); break;
			case opt::algo::Name::ADAGRAD: size+=nbytes(static_cast<const opt::algo::ADAGRAD&>(*obj)); break;
			case opt::algo::Name::ADADELTA: size+=nbytes(static_cast<const opt::algo::ADADELTA&>(*obj)); break;
			case opt::algo::Name::RMSPROP: size+=nbytes(static_cast<const opt::algo::RMSPROP&>(*obj)); break;
			case opt::algo::Name::ADAM: size+=nbytes(static_cast<const opt::algo::ADAM&>(*obj)); break;
			case opt::algo::Name::ADAMW: size+=nbytes(static_cast<const opt::algo::ADAMW&>(*obj)); break;
			case opt::algo::Name::ADAB: size+=nbytes(static_cast<const opt::algo::ADAB&>(*obj)); break;
			case opt::algo::Name::YOGI: size+=nbytes(static_cast<const opt::algo::YOGI&>(*obj)); break;
			case opt::algo::Name::NOGI: size+=nbytes(static_cast<const opt::algo::NOGI&>(*obj)); break;
			case opt::algo::Name::NADAM: size+=nbytes(static_cast<const opt::algo::NADAM&>(*obj)); break;
			case opt::algo::Name::AMSGRAD: size+=nbytes(static_cast<const opt::algo::AMSGRAD&>(*obj)); break;
			case opt::algo::Name::BFGS: size+=nbytes(static_cast<const opt::algo::BFGS&>(*obj)); break;
			case opt::algo::Name::RPROP: size+=nbytes(static_cast<const opt::algo::RPROP&>(*obj)); break;
			case opt::algo::Name::CG: size+=nbytes(static_cast<const opt::algo::CG&>(*obj)); break;
			default: throw std::invalid_argument("serialize::nbytes(const std::shared_ptr<opt::algo::Base>&): Invalid algo name"); break;
		}
	}
	return size;
}
template <> int nbytes(const opt::algo::SGD& obj){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"nbytes(const opt::algo::SGD&):\n";
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	return size;
}
template <> int nbytes(const opt::algo::SDM& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eta
	size+=sizeof(double)*obj.dim();//dx
	return size;
}
template <> int nbytes(const opt::algo::NAG& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eta
	size+=sizeof(double)*obj.dim();//dx
	return size;
}
template <> int nbytes(const opt::algo::ADAGRAD& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//mgrad2_
	return size;
}
template <> int nbytes(const opt::algo::ADADELTA& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eta
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//mgrad2_
	size+=sizeof(double)*obj.dim();//mgrad_
	size+=sizeof(double)*obj.dim();//dx_
	return size;
}
template <> int nbytes(const opt::algo::RMSPROP& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double)*obj.dim();//mgrad2_
	return size;
}
template <> int nbytes(const opt::algo::ADAM& obj){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"nbytes(const opt::algo::ADAM&):\n";
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//mgrad_
	size+=sizeof(double)*obj.dim();//mgrad2_
	return size;
}
template <> int nbytes(const opt::algo::ADAMW& obj){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"nbytes(const opt::algo::ADAMW&):\n";
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//w_
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//mgrad_
	size+=sizeof(double)*obj.dim();//mgrad2_
	return size;
}
template <> int nbytes(const opt::algo::ADAB& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//mgrad_
	size+=sizeof(double)*obj.dim();//mgrad2_
	return size;
}
template <> int nbytes(const opt::algo::YOGI& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//mgrad_
	size+=sizeof(double)*obj.dim();//mgrad2_
	return size;
}
template <> int nbytes(const opt::algo::NOGI& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//mgrad_
	size+=sizeof(double)*obj.dim();//mgrad2_
	return size;
}
template <> int nbytes(const opt::algo::NADAM& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//mgrad_
	size+=sizeof(double)*obj.dim();//mgrad2_
	return size;
}
template <> int nbytes(const opt::algo::AMSGRAD& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//mgrad_
	size+=sizeof(double)*obj.dim();//mgrad2_
	size+=sizeof(double)*obj.dim();//mgrad2m_
	return size;
}
template <> int nbytes(const opt::algo::BFGS& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double)*obj.dim();//s_
	size+=sizeof(double)*obj.dim();//y_
	size+=sizeof(double)*obj.dim()*obj.dim();//B_
	size+=sizeof(double)*obj.dim()*obj.dim();//BOld_
	return size;
}
template <> int nbytes(const opt::algo::RPROP& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double)*obj.dim();//delta_
	size+=sizeof(double)*obj.dim();//dx_
	return size;
}
template <> int nbytes(const opt::algo::CG& obj){
	int size=0;
	size+=sizeof(int);//dim
	size+=sizeof(opt::algo::Name);//name
	size+=sizeof(double);//eps_
	size+=sizeof(double)*obj.dim();//cgd_
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const std::shared_ptr<opt::algo::Base>& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const std::shared_ptr<opt::algo::Base>&,char*):\n";
	int pos=0;
	const int null=(obj==nullptr);
	std::memcpy(arr+pos,&null,sizeof(int)); pos+=sizeof(int);
	if(!null){
		opt::algo::Name name=obj->name();
		std::memcpy(arr+pos,&name,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
		switch(obj->name()){
			case opt::algo::Name::SGD: pos+=pack(static_cast<const opt::algo::SGD&>(*obj),arr+pos); break;
			case opt::algo::Name::SDM: pos+=pack(static_cast<const opt::algo::SDM&>(*obj),arr+pos); break;
			case opt::algo::Name::NAG: pos+=pack(static_cast<const opt::algo::NAG&>(*obj),arr+pos); break;
			case opt::algo::Name::ADAGRAD: pos+=pack(static_cast<const opt::algo::ADAGRAD&>(*obj),arr+pos); break;
			case opt::algo::Name::ADADELTA: pos+=pack(static_cast<const opt::algo::ADADELTA&>(*obj),arr+pos); break;
			case opt::algo::Name::RMSPROP: pos+=pack(static_cast<const opt::algo::RMSPROP&>(*obj),arr+pos); break;
			case opt::algo::Name::ADAM: pos+=pack(static_cast<const opt::algo::ADAM&>(*obj),arr+pos); break;
			case opt::algo::Name::ADAMW: pos+=pack(static_cast<const opt::algo::ADAMW&>(*obj),arr+pos); break;
			case opt::algo::Name::ADAB: pos+=pack(static_cast<const opt::algo::ADAB&>(*obj),arr+pos); break;
			case opt::algo::Name::YOGI: pos+=pack(static_cast<const opt::algo::YOGI&>(*obj),arr+pos); break;
			case opt::algo::Name::NOGI: pos+=pack(static_cast<const opt::algo::NOGI&>(*obj),arr+pos); break;
			case opt::algo::Name::NADAM: pos+=pack(static_cast<const opt::algo::NADAM&>(*obj),arr+pos); break;
			case opt::algo::Name::AMSGRAD: pos+=pack(static_cast<const opt::algo::AMSGRAD&>(*obj),arr+pos); break;
			case opt::algo::Name::BFGS: pos+=pack(static_cast<const opt::algo::BFGS&>(*obj),arr+pos); break;
			case opt::algo::Name::RPROP: pos+=pack(static_cast<const opt::algo::RPROP&>(*obj),arr+pos); break;
			case opt::algo::Name::CG: pos+=pack(static_cast<const opt::algo::CG&>(*obj),arr+pos); break;
			default: throw std::invalid_argument("pack(const std::shared_ptr<opt::algo::Base>&,char*): Invalid algo name.\n"); break;
		}
	}
	return pos;
}
template <> int pack(const opt::algo::SGD& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::SGD&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	return pos;
}
template <> int pack(const opt::algo::SDM& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::SDM&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eta(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.dx().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::NAG& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::NAG&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eta(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.dx().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::ADAGRAD& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::ADAGRAD&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::ADADELTA& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::ADADELTA&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eta(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.mdx2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.dx().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::RMSPROP& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::RMSPROP&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::ADAM& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::ADAM&,char*)\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::ADAMW& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::ADAMW&,char*)\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.w(),sizeof(double)); pos+=sizeof(double);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::ADAB& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::ADAB&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::YOGI& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::YOGI&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::NOGI& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::NOGI&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::NADAM& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::NADAM&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::AMSGRAD& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::AMSGRAD&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.mgrad().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.mgrad2().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.mgrad2m().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::BFGS& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::BFGS&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.s().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.y().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.B().data(),sizeof(double)*obj.dim()*obj.dim()); pos+=sizeof(double)*obj.dim()*obj.dim();
		std::memcpy(arr+pos,obj.BOld().data(),sizeof(double)*obj.dim()*obj.dim()); pos+=sizeof(double)*obj.dim()*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::RPROP& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::RPROP&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.dx().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(arr+pos,obj.delta().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int pack(const opt::algo::CG& obj, char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"pack(const opt::algo::CG&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&obj.dim(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&obj.name(),sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);
	if(obj.dim()>0){
		std::memcpy(arr+pos,obj.cgd().data(),sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(std::shared_ptr<opt::algo::Base>& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(std::shared_ptr<opt::algo::Base>&,const char*)\n";
	int pos=0;
	int null=1;
	std::memcpy(&null,arr+pos,sizeof(int)); pos+=sizeof(int);
	if(!null){
		opt::algo::Name name=opt::algo::Name::UNKNOWN;
		std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
		opt::algo::make(obj,name);
		switch(name){
			case opt::algo::Name::SGD: pos+=unpack(static_cast<opt::algo::SGD&>(*obj),arr+pos); break;
			case opt::algo::Name::SDM: pos+=unpack(static_cast<opt::algo::SDM&>(*obj),arr+pos); break;
			case opt::algo::Name::NAG: pos+=unpack(static_cast<opt::algo::NAG&>(*obj),arr+pos); break;
			case opt::algo::Name::ADAGRAD: pos+=unpack(static_cast<opt::algo::ADAGRAD&>(*obj),arr+pos); break;
			case opt::algo::Name::ADADELTA: pos+=unpack(static_cast<opt::algo::ADADELTA&>(*obj),arr+pos); break;
			case opt::algo::Name::RMSPROP: pos+=unpack(static_cast<opt::algo::RMSPROP&>(*obj),arr+pos); break;
			case opt::algo::Name::ADAM: pos+=unpack(static_cast<opt::algo::ADAM&>(*obj),arr+pos); break;
			case opt::algo::Name::ADAMW: pos+=unpack(static_cast<opt::algo::ADAMW&>(*obj),arr+pos); break;
			case opt::algo::Name::ADAB: pos+=unpack(static_cast<opt::algo::ADAB&>(*obj),arr+pos); break;
			case opt::algo::Name::YOGI: pos+=unpack(static_cast<opt::algo::YOGI&>(*obj),arr+pos); break;
			case opt::algo::Name::NOGI: pos+=unpack(static_cast<opt::algo::NOGI&>(*obj),arr+pos); break;
			case opt::algo::Name::NADAM: pos+=unpack(static_cast<opt::algo::NADAM&>(*obj),arr+pos); break;
			case opt::algo::Name::AMSGRAD: pos+=unpack(static_cast<opt::algo::AMSGRAD&>(*obj),arr+pos); break;
			case opt::algo::Name::BFGS: pos+=unpack(static_cast<opt::algo::BFGS&>(*obj),arr+pos); break;
			case opt::algo::Name::RPROP: pos+=unpack(static_cast<opt::algo::RPROP&>(*obj),arr+pos); break;
			case opt::algo::Name::CG: pos+=unpack(static_cast<opt::algo::CG&>(*obj),arr+pos); break;
			default: throw std::invalid_argument("unpack(std::shared_ptr<opt::algo::Base>&,const char*): Invalid algo name.\n"); break;
		}
	} else obj.reset();
	return pos;
}
template <> int unpack(opt::algo::SGD& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::SGD&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	if(name!=opt::algo::Name::SGD) throw std::invalid_argument("serialize::unpack(opt::algo::SGD&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
	}
	return pos;
}
template <> int unpack(opt::algo::SDM& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::SDM&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	if(name!=opt::algo::Name::SDM) throw std::invalid_argument("serialize::unpack(opt::algo::SDM&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(&obj.eta(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(obj.dx().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::NAG& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::NAG&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	if(name!=opt::algo::Name::NAG) throw std::invalid_argument("serialize::unpack(opt::algo::NAG&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(&obj.eta(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(obj.dx().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::ADAGRAD& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::ADAGRAD&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::ADAGRAD) throw std::invalid_argument("serialize::unpack(opt::algo::ADAGRAD&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::ADADELTA& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::ADADELTA&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::ADADELTA) throw std::invalid_argument("serialize::unpack(opt::algo::ADADELTA&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.mdx2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.dx().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::RMSPROP& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::RMSPROP&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	if(name!=opt::algo::Name::RMSPROP) throw std::invalid_argument("serialize::unpack(opt::algo::RMSPROP&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::ADAM& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::ADAM&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::ADAM) throw std::invalid_argument("serialize::unpack(opt::algo::ADAM&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::ADAMW& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::ADAMW&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.w(),arr+pos,sizeof(double)); pos+=sizeof(double);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::ADAMW) throw std::invalid_argument("serialize::unpack(opt::algo::ADAMW&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::ADAB& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::ADAB&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::ADAB) throw std::invalid_argument("serialize::unpack(opt::algo::ADAB&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::YOGI& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::YOGI&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::YOGI) throw std::invalid_argument("serialize::unpack(opt::algo::YOGI&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::NOGI& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::NOGI&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::NOGI) throw std::invalid_argument("serialize::unpack(opt::algo::NOGI&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::NADAM& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::NADAM&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::NADAM) throw std::invalid_argument("serialize::unpack(opt::algo::NADAM&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::AMSGRAD& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::AMSGRAD&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::AMSGRAD) throw std::invalid_argument("serialize::unpack(opt::algo::AMSGRAD&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.mgrad().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.mgrad2().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.mgrad2m().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::BFGS& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::BFGS&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	if(name!=opt::algo::Name::BFGS) throw std::invalid_argument("serialize::unpack(opt::algo::BFGS&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.s().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.y().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.B().data(),arr+pos,sizeof(double)*obj.dim()*obj.dim()); pos+=sizeof(double)*obj.dim()*obj.dim();
		std::memcpy(obj.BOld().data(),arr+pos,sizeof(double)*obj.dim()*obj.dim()); pos+=sizeof(double)*obj.dim()*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::RPROP& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::RPROP&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	if(name!=opt::algo::Name::RPROP) throw std::invalid_argument("serialize::unpack(opt::algo::RPROP&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.dx().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
		std::memcpy(obj.delta().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}
template <> int unpack(opt::algo::CG& obj, const char* arr){
	if(OPT_ALGO_PRINT_FUNC>0) std::cout<<"unpack(opt::algo::CG&,const char*):\n";
	int pos=0;
	int dim=0; opt::algo::Name name;
	std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&name,arr+pos,sizeof(opt::algo::Name)); pos+=sizeof(opt::algo::Name);
	std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(name!=opt::algo::Name::CG) throw std::invalid_argument("serialize::unpack(opt::algo::CG&,const char*): Invalid name.");
	if(dim>0){
		obj.resize(dim);
		std::memcpy(obj.cgd().data(),arr+pos,sizeof(double)*obj.dim()); pos+=sizeof(double)*obj.dim();
	}
	return pos;
}

}