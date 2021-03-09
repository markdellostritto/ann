// c libraries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
#include <ctime>
// c++ libraries
#include <iostream>
// ann - math 
#include "math_special_ann.h"
#include "math_func_ann.h"
// ann - string
#include "string_ann.h"
// ann - print
#include "print.h"
// ann - nn
#include "nn.h"

namespace NeuralNet{

//***********************************************************************
// INITIALIZATION METHOD
//***********************************************************************

std::ostream& operator<<(std::ostream& out, const InitN::type& n){
	switch(n){
		case InitN::RAND: out<<"RAND"; break;
		case InitN::XAVIER: out<<"XAVIER"; break;
		case InitN::HE: out<<"HE"; break;
		case InitN::MEAN: out<<"MEAN"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* InitN::name(const InitN::type& n){
	switch(n){
		case InitN::RAND: return "RAND";
		case InitN::XAVIER: return "XAVIER";
		case InitN::HE: return "HE";
		case InitN::MEAN: return "MEAN";
		default: return "UNKNOWN";
	}
}

InitN::type InitN::read(const char* str){
	if(std::strcmp(str,"RAND")==0) return InitN::RAND;
	else if(std::strcmp(str,"XAVIER")==0) return InitN::XAVIER;
	else if(std::strcmp(str,"HE")==0) return InitN::HE;
	else if(std::strcmp(str,"MEAN")==0) return InitN::MEAN;
	else return InitN::UNKNOWN;
}

//***********************************************************************
// TRANSFER FUNCTIONS - NAMES
//***********************************************************************

std::ostream& operator<<(std::ostream& out, const TransferN::type& tf){
	switch(tf){
		case TransferN::LINEAR: out<<"LINEAR"; break;
		case TransferN::SIGMOID: out<<"SIGMOID"; break;
		case TransferN::TANH: out<<"TANH"; break;
		case TransferN::ISRU: out<<"ISRU"; break;
		case TransferN::ARCTAN: out<<"ARCTAN"; break;
		case TransferN::SOFTSIGN: out<<"SOFTSIGN"; break;
		case TransferN::RELU: out<<"RELU"; break;
		case TransferN::SOFTPLUS: out<<"SOFTPLUS"; break;
		case TransferN::SOFTPLUS2: out<<"SOFTPLUS2"; break;
		case TransferN::ELU: out<<"ELU"; break;
		case TransferN::GELU: out<<"GELU"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* TransferN::name(const TransferN::type& tf){
	switch(tf){
		case TransferN::LINEAR: return "LINEAR";
		case TransferN::SIGMOID: return "SIGMOID";
		case TransferN::TANH: return "TANH";
		case TransferN::ISRU: return "ISRU";
		case TransferN::ARCTAN: return "ARCTAN";
		case TransferN::SOFTSIGN: return "SOFTSIGN";
		case TransferN::RELU: return "RELU";
		case TransferN::SOFTPLUS: return "SOFTPLUS";
		case TransferN::SOFTPLUS2: return "SOFTPLUS2";
		case TransferN::ELU: return "ELU";
		case TransferN::GELU: return "GELU";
		default: return "UNKNOWN";
	}
}

TransferN::type TransferN::read(const char* str){
	if(std::strcmp(str,"LINEAR")==0) return TransferN::LINEAR;
	else if(std::strcmp(str,"SIGMOID")==0) return TransferN::SIGMOID;
	else if(std::strcmp(str,"TANH")==0) return TransferN::TANH;
	else if(std::strcmp(str,"ISRU")==0) return TransferN::ISRU;
	else if(std::strcmp(str,"ARCTAN")==0) return TransferN::ARCTAN;
	else if(std::strcmp(str,"SOFTSIGN")==0) return TransferN::SOFTSIGN;
	else if(std::strcmp(str,"RELU")==0) return TransferN::RELU;
	else if(std::strcmp(str,"SOFTPLUS")==0) return TransferN::SOFTPLUS;
	else if(std::strcmp(str,"SOFTPLUS2")==0) return TransferN::SOFTPLUS2;
	else if(std::strcmp(str,"ELU")==0) return TransferN::ELU;
	else if(std::strcmp(str,"GELU")==0) return TransferN::GELU;
	else return TransferN::UNKNOWN;
}

//***********************************************************************
// TRANSFER FUNCTIONS
//***********************************************************************

void TransferFFDV::f_lin(VecXd& f, VecXd& d){
	for(int i=0; i<d.size(); ++i) d[i]=1.0;
}

void TransferFFDV::f_sigmoid(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>=0){
			const double expf=exp(-f[i]);
			f[i]=1.0/(1.0+expf);
			d[i]=1.0/((1.0+expf)*(1.0+1.0/expf));
		} else {
			const double expf=exp(f[i]);
			f[i]=expf/(expf+1.0);
			d[i]=1.0/((1.0+1.0/expf)*(1.0+expf));
		}
	}	
}

void TransferFFDV::f_tanh(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i) f[i]=tanh(f[i]);
	for(int i=0; i<size; ++i) d[i]=1.0-f[i]*f[i];
}

void TransferFFDV::f_isru(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		const double isr=1.0/sqrt(1.0+f[i]*f[i]);
		f[i]=f[i]*isr;
		d[i]=isr*isr*isr;
	}
}

void TransferFFDV::f_arctan(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		d[i]=(2.0/math::constant::PI)/(1.0+f[i]*f[i]);
		f[i]=(2.0/math::constant::PI)*atan(f[i]);
	}
}

void TransferFFDV::f_softsign(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		const double inv=1.0/(1.0+fabs(f[i]));
		f[i]=f[i]*inv;
		d[i]=inv*inv;
	}
}

void TransferFFDV::f_relu(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>0){
			d[i]=1.0;
		} else {
			f[i]=0.0;
			d[i]=0.0;
		}
	}
}

void TransferFFDV::f_softplus(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>=1.0){
			//f(x)=x+ln(1+exp(-x))
			const double expf=exp(-f[i]);
			f[i]+=math::special::logp1(expf);
			d[i]=1.0/(1.0+expf);
		} else {
			//f(x)=ln(1+exp(x))
			const double expf=exp(f[i]);
			f[i]=math::special::logp1(expf);
			d[i]=expf/(expf+1.0);
		}
	}
}

void TransferFFDV::f_softplus2(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>=1.0){
			//f(x)=x+ln(1+exp(-x))
			const double expf=exp(-f[i]);
			f[i]+=math::special::logp1(expf)-math::constant::LOG2;
			d[i]=1.0/(1.0+expf);
		} else {
			//f(x)=ln(1+exp(x))-ln(2)
			const double expf=exp(f[i]);
			f[i]=math::special::logp1(expf)-math::constant::LOG2;
			d[i]=expf/(expf+1.0);
		}
	}
}

void TransferFFDV::f_elu(VecXd& f, VecXd& d){
	const int size=f.size();
	for(int i=0; i<size; ++i){
		if(f[i]>0){
			d[i]=1.0;
		} else {
			const double expf=exp(f[i]);
			f[i]=expf-1.0;
			d[i]=expf;
		}
	}	
}

void TransferFFDV::f_gelu(VecXd& f, VecXd& d){
	const int size=f.size();
	const double rad2pii=1.0/(math::constant::Rad2*math::constant::RadPI);
	for(int i=0; i<size; ++i){
		const double erff=0.5*(1.0+erf(f[i]/math::constant::Rad2));
		d[i]=erff+f[i]*exp(-0.5*f[i]*f[i])*rad2pii;
		f[i]*=erff;
	}
}

//***********************************************************************
// ANN CLASS
//***********************************************************************

//==== operators ====

/**
* print network to screen
* @param out - output stream
* @param nn - neural network
* @return output stream
*/
std::ostream& operator<<(std::ostream& out, const ANN& nn){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ANN",str)<<"\n";
	out<<"nn       = "; for(int n=0; n<nn.node_.size(); ++n) out<<nn.node_[n].size()<<" "; out<<"\n";
	out<<"size     = "<<nn.size()<<"\n";
	out<<"transfer = "<<nn.tfType_<<"\n";
	out<<print::title("ANN",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

/**
* pack network parameters into serial array
* @param nn - neural network
* @param v - vector storing nn parameters
* @return v - vector storing nn parameters
*/
VecXd& operator>>(const ANN& nn, VecXd& v){
	if(NN_PRINT_FUNC>0) std::cout<<"operator>>(const ANN&, VecXd&):\n";
	int count=0;
	v=VecXd::Zero(nn.size());
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<nn.bias(l).size(); ++n) v[count++]=nn.bias(l)(n);
	}
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<nn.edge(l).size(); ++n) v[count++]=nn.edge(l)(n);
	}
	return v;
}

/**
* unpack network parameters from serial array
* @param nn - neural network
* @param v - vector storing nn parameters
* @return nn - neural network
*/
ANN& operator<<(ANN& nn, const VecXd& v){
	if(NN_PRINT_FUNC>0) std::cout<<"operator<<(ANN&,const VecXd&):\n";
	if(nn.size()!=v.size()) throw std::invalid_argument("Invalid size: vector and network mismatch.");
	int count=0;
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<nn.bias(l).size(); ++n) nn.bias(l)(n)=v[count++];
	}
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<nn.edge(l).size(); ++n) nn.edge(l)(n)=v[count++];
	}
	return nn;
}

//==== member functions ====

/**
* set the default values
*/
void ANN::defaults(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::defaults():\n";
	//input/output
		in_.resize(0);
		out_.resize(0);
		inw_.resize(0);
		inb_.resize(0);
		outw_.resize(0);
		outb_.resize(0);
	//node weights and biases
		nlayer_=0;
		node_.clear();
		bias_.clear();
		edge_.clear();
	//gradients - nodes
		dadz_.clear();
	//transfer functions
		tfType_=TransferN::UNKNOWN;
		tffdv_.clear();
}

/**
* clear all values
*/
void ANN::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::clear():\n";
	//input/output
		in_.resize(0);
		out_.resize(0);
		inw_.resize(0);
		inb_.resize(0);
		outw_.resize(0);
		outb_.resize(0);
	//node weights and biases
		nlayer_=0;
		node_.clear();
		bias_.clear();
		edge_.clear();
	//gradients - nodes
		dadz_.clear();
	//transfer functions
		tffdv_.clear();
}

/**
* compute and return the size of the network - the number of adjustable parameters
* @return the size of the network - the number of adjustable parameters
*/
int ANN::size()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::size():\n";
	int s=0;
	for(int n=0; n<nlayer_; ++n) s+=bias_[n].size();
	for(int n=0; n<nlayer_; ++n) s+=edge_[n].size();
	return s;
}

/**
* compute and return the number of bias parameters 
* @return the number of bias parameters 
*/
int ANN::nBias()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::nBias():\n";
	int s=0;
	for(int n=0; n<nlayer_; ++n) s+=bias_[n].size();
	return s;
}

/**
* compute and return the number of weight parameters 
* @return the number of weight parameters 
*/
int ANN::nWeight()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::nWeight():\n";
	int s=0;
	for(int n=0; n<nlayer_; ++n) s+=edge_[n].size();
	return s;
}

/**
* resize the network - no hidden layers
* @param nIn - number of inputs of the newtork
* @param nOut - the number of outputs of the network
*/
void ANN::resize(const ANNInit& init, int nIn, int nOut){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNInit&,int,int):\n";
	if(nIn<=0) throw std::invalid_argument("ANN::resize(int,int): Invalid output size.");
	if(nOut<=0) throw std::invalid_argument("ANN::resize(int,int): Invalid output size.");
	std::vector<int> nn(2);
	nn[0]=nIn; nn[1]=nOut;
	resize(init,nn);
}

/**
* resize the network - given separate hidden layers and output layer
* @param nIn - number of inputs of the newtork
* @param nOut - the number of outputs of the network
* @param nNodes - the number of nodes in each hidden layer
*/
void ANN::resize(const ANNInit& init, int nIn, const std::vector<int>& nNodes, int nOut){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNInit&,int,const std::vector<int>&,int):\n";
	if(nOut<=0) throw std::invalid_argument("ANN::resize(const ANNInit&,int,const std::vector<int>&,int): Invalid output size.");
	std::vector<int> nn(nNodes.size()+2);
	nn.front()=nIn;
	for(int n=0; n<nNodes.size(); ++n) nn[n+1]=nNodes[n];
	nn.back()=nOut;
	resize(init,nn);
}

/**
* resize the network - given combined hidden layers and output layer
* @param nNodes - the number of nodes in each layer of the network
*/
void ANN::resize(const ANNInit& init, const std::vector<int>& nNodes){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNInit&,const std::vector<int>&):\n";
	//initialize the random number generator
		if(init.sigma()<=0) throw std::invalid_argument("ANN::resize(const std::vector<int>&): Invalid initialization deviation");
		RNG::CG2 cg2; cg2.init(init.seed()<=0?std::time(NULL):init.seed());
	//clear the network
		clear();
	//check parameters
		for(int n=0; n<nNodes.size(); ++n){
			if(nNodes[n]<=0) throw std::invalid_argument("ANN::resize(const std::vector<int>&): Invalid layer size.");
		}
	//input/output
		in_=VecXd::Zero(nNodes.front());
		out_=VecXd::Zero(nNodes.back());
	//pre/post conditioning
		inw_=VecXd::Constant(in_.size(),1);
		inb_=VecXd::Constant(in_.size(),0);
		outw_=VecXd::Constant(out_.size(),1);
		outb_=VecXd::Constant(out_.size(),0);
	//number of layers
		nlayer_=nNodes.size()-1;//number of weights, i.e. connections b/w layers, thus 1 less than size of nNodes
		if(nlayer_<1) throw std::invalid_argument("ANN::resize(const std::vector<int>&): Invalid number of layers.");
	//nodes
		node_.resize(nlayer_+1);
		for(int n=0; n<nlayer_+1; ++n){
			node_[n]=VecXd::Zero(nNodes[n]);
		}
	//bias
		bias_.resize(nlayer_);
		for(int n=0; n<nlayer_; ++n) bias_[n]=VecXd::Zero(nNodes[n+1]);
		for(int n=0; n<nlayer_; ++n){
			for(int m=0; m<bias_[n].size(); ++m){
				bias_[n][m]=2.0*(cg2.randf()-0.5)*init.bInit();
			}
		}
	//edges
		edge_.resize(nlayer_);
		//edge(n) * layer(n) -> layer(n+1), thus size(edge) = (layer(n+1) rows * layer(n) cols)
		for(int n=0; n<nlayer_; ++n) edge_[n]=MatXd::Zero(nNodes[n+1],nNodes[n]);
		if(init.distT()==RNG::DistN::NORMAL){
			RNG::DistNormal dist(0.0,init.sigma());
			for(int n=0; n<nlayer_; ++n){
				for(int m=0; m<edge_[n].size(); ++m){
					edge_[n].data()[m]=dist(cg2);
				}
			}
		} else if(init.distT()==RNG::DistN::EXP){
			RNG::DistExp dist(init.sigma());
			for(int n=0; n<nlayer_; ++n){
				for(int m=0; m<edge_[n].size(); ++m){
					edge_[n].data()[m]=dist(cg2);
				}
			}
		} else throw std::invalid_argument("ANN::resize(const std::vector<int>&): Invalid probability distribution.");
		switch(init.initType()){
			case InitN::RAND:   for(int n=0; n<nlayer_; ++n) edge_[n]*=init.wInit(); break;
			case InitN::XAVIER: for(int n=0; n<nlayer_; ++n) edge_[n]*=init.wInit()*std::sqrt(1.0/nNodes[n]); break;
			case InitN::HE:     for(int n=0; n<nlayer_; ++n) edge_[n]*=init.wInit()*std::sqrt(2.0/nNodes[n]); break;
			case InitN::MEAN:   for(int n=0; n<nlayer_; ++n) edge_[n]*=init.wInit()*std::sqrt(2.0/(nNodes[n+1]+nNodes[n])); break;
			default: throw std::invalid_argument("ANN::resize(const std::vector<int>&): Invalid initialization scheme."); break;
		}
	//gradients - nodes
		dadz_.resize(nlayer_);
		for(int n=0; n<nlayer_; ++n){
			dadz_[n]=VecXd::Zero(nNodes[n+1]);//no transfer function for input layer
		}
	//transfer functions
		switch(tfType_){
			case TransferN::LINEAR:   tffdv_.resize(nlayer_,TransferFFDV::f_lin); break;
			case TransferN::SIGMOID:  tffdv_.resize(nlayer_,TransferFFDV::f_sigmoid); break;
			case TransferN::TANH:     tffdv_.resize(nlayer_,TransferFFDV::f_tanh); break;
			case TransferN::ISRU:     tffdv_.resize(nlayer_,TransferFFDV::f_isru); break;
			case TransferN::ARCTAN:   tffdv_.resize(nlayer_,TransferFFDV::f_arctan); break;
			case TransferN::SOFTSIGN: tffdv_.resize(nlayer_,TransferFFDV::f_softsign); break;
			case TransferN::SOFTPLUS: tffdv_.resize(nlayer_,TransferFFDV::f_softplus); break;
			case TransferN::SOFTPLUS2:tffdv_.resize(nlayer_,TransferFFDV::f_softplus2); break;
			case TransferN::RELU:     tffdv_.resize(nlayer_,TransferFFDV::f_relu); break;
			case TransferN::ELU:      tffdv_.resize(nlayer_,TransferFFDV::f_elu); break;
			case TransferN::GELU:     tffdv_.resize(nlayer_,TransferFFDV::f_gelu); break;
			default: throw std::invalid_argument("ANN::resize(int,const std::vector<int>&): Invalid transfer function."); break;
		}
		tffdv_.back()=TransferFFDV::f_lin;//final layer is typically linear
}

/**
* compute the error associated the output, given the target output
* @param out - the target output of the network (not the actual output of the network)
* @return the error of the output of the network (out_) w.r.t. the target output (out)
*/
double ANN::error(const VecXd& out)const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::error(const VecXd&):\n";
	return 0.5*(out_-out).squaredNorm();
}

/**
* compute the regularization error
* @return the regularization error - 1/2 the sum of the squares of the weights
*/
double ANN::error_lambda()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::error_lambda():\n";
	double err=0;
	for(int l=0; l<nlayer_; ++l){
		err+=0.5*edge_[l].squaredNorm();//lambda error - quadratic
	}
	//return error
	return err;
}

/**
* compute dcdo - dc/do - gradient of the cost function w.r.t. the output
* @param out - the target output of the network (not the actual output of the network)
* @param grad - stores the gradient of the cost function w.r.t. the output
* @return grad - the gradient of the cost function w.r.t. the output
*/
VecXd& ANN::dcdo(const VecXd& out, VecXd& dcdo_)const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::dcdo(const VecXd&,VecXd&):\n";
	dcdo_.noalias()=(out_-out);
	return dcdo_;
}

/**
* compute the regularization gradient
* @param grad - stores the regularization gradient w.r.t. each parameter of the network
* @return grad - the regularization gradient w.r.t. each parameter of the network
*/
VecXd& ANN::grad_lambda(VecXd& grad)const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::grad_lambda(VecXd&):\n";
	int count=0;
	//gradient w.r.t bias
	for(int l=0; l<nlayer_; ++l){
		for(int n=0; n<bias_[l].size(); ++n){
			grad[count++]=0.0;
		}
	}
	//gradient w.r.t. edges
	for(int l=0; l<nlayer_; ++l){
		for(int m=0; m<edge_[l].cols(); ++m){
			for(int n=0; n<edge_[l].rows(); ++n){
				grad[count++]=edge_[l](n,m);//edge(l,n,m) - quadratic
			}
		}
	}
	//return the gradient
	return grad;
}

/**
* execute the network
* @return out_ - the output of the network
*/
const VecXd& ANN::execute(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::execute():\n";
	//scale the input
	node_.front().noalias()=inw_.cwiseProduct(in_+inb_);
	//hidden layers
	for(int l=0; l<nlayer_; ++l){
		node_[l+1]=bias_[l];
		node_[l+1].noalias()+=edge_[l]*node_[l];
		(*tffdv_[l])(node_[l+1],dadz_[l]);
	}
	//scale the output
	out_=outb_;
	out_.noalias()+=node_.back().cwiseProduct(outw_);
	//return the output
	return out_;
}

//==== static functions ====

/**
* write the network to file
* @param file - the file name where the network is to be written
* @param nn - the neural network to be written
*/
void ANN::write(const char* file, const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::write(const char*,const ANN&):\n";
	//local variables
	FILE* writer=NULL;
	//open the file
	writer=std::fopen(file,"w");
	if(writer!=NULL){
		ANN::write(writer,nn);
		std::fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for writing.\n"));
}

/**
* write the network to file
* @param writer - file pointer
* @param nn - the neural network to be written
*/
void ANN::write(FILE* writer, const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::write(FILE*,const ANN&):\n";
	//print the configuration
	fprintf(writer,"nn ");
	for(int i=0; i<nn.nlayer()+1; ++i) fprintf(writer,"%i ",nn.nNodes(i));
	fprintf(writer,"\n");
	//print the transfer function
	fprintf(writer,"t_func %s\n",TransferN::name(nn.tfType()));
	//print the scaling layers
	fprintf(writer,"inw ");
	for(int i=0; i<nn.nIn(); ++i) fprintf(writer,"%.15f ",nn.inw()[i]);
	fprintf(writer,"\n");
	fprintf(writer,"outw ");
	for(int i=0; i<nn.nOut(); ++i) fprintf(writer,"%.15f ",nn.outw()[i]);
	fprintf(writer,"\n");
	//print the biasing layers
	fprintf(writer,"inb ");
	for(int i=0; i<nn.nIn(); ++i) fprintf(writer,"%.15f ",nn.inb()[i]);
	fprintf(writer,"\n");
	fprintf(writer,"outb ");
	for(int i=0; i<nn.nOut(); ++i) fprintf(writer,"%.15f ",nn.outb()[i]);
	fprintf(writer,"\n");
	//print the biases
	for(int n=0; n<nn.nlayer(); ++n){
		fprintf(writer,"bias[%i] ",n+1);
		for(int i=0; i<nn.bias(n).size(); ++i){
			fprintf(writer,"%.15f ",nn.bias(n)[i]);
		}
		fprintf(writer,"\n");
	}
	//print the edge weights
	for(int n=0; n<nn.nlayer(); ++n){
		fprintf(writer,"weight[%i,%i] ",n,n+1);
		for(int i=0; i<nn.edge(n).rows(); ++i){
			for(int j=0; j<nn.edge(n).cols(); ++j){
				fprintf(writer,"%.15f ",nn.edge(n)(i,j));
			}
		}
		fprintf(writer,"\n");
	}
}

/**
* read the network from file
* @param file - the file name where the network is to be read
* @param nn - the neural network to be read
*/
void ANN::read(const char* file, ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::read(const char*,ANN&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		ANN::read(reader,nn);
		std::fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for reading.\n"));
}

/**
* read the network from file
* @param reader - file pointer
* @param nn - the neural network to be read
*/
void ANN::read(FILE* reader, ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::read(FILE*,ANN&):\n";
	//==== local variables ====
	const int MAX=5000;
	const int N_DIGITS=32;//max number of digits in number
	int b_max=0;//max number of biases for a given layer
	int w_max=0;//max number of weights for a given layer
	char* input=new char[MAX];
	char* b_str=NULL;//bias string
	char* w_str=NULL;//weight string
	char* i_str=NULL;//input string
	char* o_str=NULL;//output string
	std::vector<int> nodes;
	std::vector<std::string> strlist;
	ANNInit init;
	//==== clear the network ====
	if(NN_PRINT_STATUS>0) std::cout<<"clearing the network\n";
	nn.clear();
	//==== load the configuration ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading configuration\n";
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	if(strlist.size()<2) throw std::invalid_argument("ANN::read(FILE*,ANN&): Invalid network configuration.");
	const int nlayer=strlist.size()-2;//"nn" nIn nh0 nh1 nh2 ... nOut
	nodes.resize(nlayer+1);
	for(int i=1; i<strlist.size(); ++i) nodes[i-1]=std::atoi(strlist[i].c_str());
	if(NN_PRINT_DATA>0){for(int i=0; i<nodes.size(); ++i) std::cout<<nodes[i]<<" "; std::cout<<"\n";}
	//==== set the transfer function ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading transfer function\n";
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	nn.tfType()=TransferN::read(strlist[1].c_str());
	if(nn.tfType()==TransferN::UNKNOWN) throw std::invalid_argument("ANN::read(FILE*,ANN&): Invalid transfer function.");
	//==== resize the nueral newtork ====
	if(NN_PRINT_STATUS>0) std::cout<<"resizing neural network\n";
	if(NN_PRINT_STATUS>0) {for(int i=0; i<nodes.size(); ++i) std::cout<<nodes[i]<<" "; std::cout<<"\n";}
	nn.resize(init,nodes);
	if(NN_PRINT_STATUS>1) std::cout<<"nn = "<<nn<<"\n";
	w_max=nn.nNodes(0)*nn.nIn();
	for(int i=0; i<nn.nlayer(); ++i) b_max=(b_max>nn.nNodes(i))?b_max:nn.nNodes(i);
	for(int i=1; i<nn.nlayer(); ++i) w_max=(w_max>nn.nNodes(i)*nn.nNodes(i-1))?w_max:nn.nNodes(i)*nn.nNodes(i-1);
	if(NN_PRINT_DATA>0) std::cout<<"b_max "<<b_max<<" w_max "<<w_max<<"\n";
	b_str=new char[b_max*N_DIGITS];
	w_str=new char[w_max*N_DIGITS];
	i_str=new char[nn.nIn()*N_DIGITS];
	o_str=new char[nn.nOut()*N_DIGITS];
	//==== read the scaling layers ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading scaling layers\n";
	string::split(fgets(i_str,nn.nIn()*N_DIGITS,reader),string::WS,strlist);
	for(int j=1; j<strlist.size(); ++j) nn.inw()[j-1]=std::atof(strlist[j].c_str());
	string::split(fgets(o_str,nn.nOut()*N_DIGITS,reader),string::WS,strlist);
	for(int j=1; j<strlist.size(); ++j) nn.outw()[j-1]=std::atof(strlist[j].c_str());
	//==== read the biasing layers ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading biasing layers\n";
	string::split(fgets(i_str,nn.nIn()*N_DIGITS,reader),string::WS,strlist);
	for(int j=1; j<strlist.size(); ++j) nn.inb()[j-1]=std::atof(strlist[j].c_str());
	string::split(fgets(o_str,nn.nOut()*N_DIGITS,reader),string::WS,strlist);
	for(int j=1; j<strlist.size(); ++j) nn.outb()[j-1]=std::atof(strlist[j].c_str());
	//==== read in the biases ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading biases\n";
	for(int n=0; n<nn.nlayer(); ++n){
		string::split(fgets(b_str,b_max*N_DIGITS,reader),string::WS,strlist);
		for(int i=0; i<nn.bias(n).size(); ++i){
			nn.bias(n)[i]=std::atof(strlist[i+1].c_str());
		}
	}
	//==== read in the edge weights ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading weights\n";
	for(int n=0; n<nn.nlayer(); ++n){
		string::split(fgets(w_str,w_max*N_DIGITS,reader),string::WS,strlist);
		int count=0;
		for(int i=0; i<nn.edge(n).rows(); ++i){
			for(int j=0; j<nn.edge(n).cols(); ++j){
				nn.edge(n)(i,j)=std::atof(strlist[++count].c_str());
			}
		}
	}
	//==== free local variables ====
	if(input!=NULL) delete[] input;
	if(b_str!=NULL) delete[] b_str;
	if(w_str!=NULL) delete[] w_str;
	if(i_str!=NULL) delete[] i_str;
	if(o_str!=NULL) delete[] o_str;
}

//==== operators ====

/**
* check equality of two networks
* @param n1 - neural network - first
* @param n2 - neural network - second
* @return equality of n1 and n2
*/
bool operator==(const ANN& n1, const ANN& n2){
	if(n1.tfType()!=n2.tfType()) return false;
	else if(n1.nlayer()!=n2.nlayer()) return false;
	else if(n1.nIn()!=n2.nIn()) return false;
	else {
		//number of layers
		for(int i=0; i<n1.nlayer(); ++i){
			if(n1.nNodes(i)!=n2.nNodes(i)) return false;
		}
		//pre-/post-conditioning
		for(int i=0; i<n1.nIn(); ++i){
			if(n1.inw()[i]!=n2.inw()[i]) return false;
			if(n1.inb()[i]!=n2.inb()[i]) return false;
		}
		for(int i=0; i<n1.nOut(); ++i){
			if(n1.outw()[i]!=n2.outw()[i]) return false;
			if(n1.outb()[i]!=n2.outb()[i]) return false;
		}
		//bias
		for(int i=0; i<n1.nlayer(); ++i){
			double diff=(n1.bias(i)-n2.bias(i)).norm();
			if(diff>math::constant::ZERO) return false;
		}
		//edge
		for(int i=0; i<n1.nlayer(); ++i){
			double diff=(n1.edge(i)-n2.edge(i)).norm();
			if(diff>math::constant::ZERO) return false;
		}
		//same
		return true;
	}
}

//***********************************************************************
// ANNInit
//***********************************************************************

void ANNInit::defaults(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANNInit::defaults():\n";
	bInit_=0.001;
	wInit_=1;
	sigma_=1.0;
	distT_=RNG::DistN::NORMAL;
	initType_=InitN::RAND;
	seed_=-1;
}

std::ostream& operator<<(std::ostream& out, const ANNInit& init){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ANN_INIT",str)<<"\n";
	out<<"b-init = "<<init.bInit_<<"\n";
	out<<"w-init = "<<init.wInit_<<"\n";
	out<<"sigma  = "<<init.sigma_<<"\n";
	out<<"dist   = "<<init.distT_<<"\n";
	out<<"init   = "<<init.initType_<<"\n";
	out<<"seed   = "<<init.seed_<<"\n";
	out<<print::title("ANN_INIT",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//***********************************************************************
// Cost
//***********************************************************************

/**
* clear all local data
*/
void Cost::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::clear():\n";
	dcdo_.resize(0);
	dcdz_.clear();
}

/**
* resize data for a given neural network
*/
void Cost::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::resize(const ANN&):\n";
	dcdo_=VecXd::Zero(nn.out().size());
	dcdz_.resize(nn.nlayer());
	for(int n=0; n<nn.nlayer(); ++n){
		dcdz_[n]=VecXd::Zero(nn.node(n+1).size());
	}
	grad_.resize(nn.size());
}

/**
* compute value and gradient of error
* @param nn - the neural network for which we will compute the error and gradient
* @param out - the target output of the network (not the actual output of the network)
* @return grad - the gradient of the cost function w.r.t. each parameter of the network
*/
double Cost::error(const ANN& nn, const VecXd& out){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::error(const ANN&,const VecXd&):\n";
	//compute the error
	const double err=0.5*(nn.out()-out).squaredNorm();
	//compute the gradient of the error function
	dcdo_.noalias()=(nn.out()-out);
	if(NN_PRINT_DATA>1) std::cout<<"dcdo_ = "<<dcdo_<<"\n";
	//compute delta for the output layer
	const int size=nn.outw().size();
	for(int i=0; i<size; ++i) dcdz_[nn.nlayer()-1][i]=nn.outw()[i]*dcdo_[i]*nn.dadz(nn.nlayer()-1)[i];
	//back-propogate the error
	for(int l=nn.nlayer()-1; l>0; --l){
		dcdz_[l-1].noalias()=nn.dadz(l-1).cwiseProduct(nn.edge(l).transpose()*dcdz_[l]);
	}
	int count=0;
	//gradient w.r.t bias
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. bias\n";
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<dcdz_[l].size(); ++n){
			grad_[count++]=dcdz_[l][n];//bias(l,n)
		}
	}
	//gradient w.r.t. edges
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. edges\n";
	for(int l=0; l<nn.nlayer(); ++l){
		for(int m=0; m<nn.edge(l).cols(); ++m){
			for(int n=0; n<nn.edge(l).rows(); ++n){
				grad_[count++]=dcdz_[l][n]*nn.node(l)(m);//edge(l,n,m)
			}
		}
	}
	return err;
}

/**
* compute gradient of error given the derivative of the cost function w.r.t. the output (dcdo)
* @param nn - the neural network for which we will compute the gradient
* @param dcdo - the derivative of the cost function w.r.t. the output
* @return grad - the gradient of the cost function w.r.t. each parameter of the network
*/
const VecXd& Cost::grad(const ANN& nn, const VecXd& dcdo){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::grad(const ANN&):\n";
	//store the gradient of the error function
	dcdo_=dcdo;
	if(NN_PRINT_DATA>1) std::cout<<"dcdo_ = "<<dcdo_<<"\n";
	//compute delta for the output layer
	const int size=nn.outw().size();
	for(int i=0; i<size; ++i) dcdz_[nn.nlayer()-1][i]=nn.outw()[i]*dcdo_[i]*nn.dadz(nn.nlayer()-1)[i];
	//back-propogate the error
	for(int l=nn.nlayer()-1; l>0; --l){
		dcdz_[l-1].noalias()=nn.dadz(l-1).cwiseProduct(nn.edge(l).transpose()*dcdz_[l]);
	}
	int count=0;
	//gradient w.r.t bias
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. bias\n";
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<dcdz_[l].size(); ++n){
			grad_[count++]=dcdz_[l][n];//bias(l,n)
		}
	}
	//gradient w.r.t. edges
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. edges\n";
	for(int l=0; l<nn.nlayer(); ++l){
		for(int m=0; m<nn.edge(l).cols(); ++m){
			for(int n=0; n<nn.edge(l).rows(); ++n){
				grad_[count++]=dcdz_[l][n]*nn.node(l)(m);//edge(l,n,m)
			}
		}
	}
	//return the gradient
	return grad_;
}

//***********************************************************************
// DOutDVal
//***********************************************************************

/**
* clear all local data
*/
void DOutDVal::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"DOutDVal::clear():\n";
	dodi_.resize(0,0);
	doda_.clear();
}

/**
* resize data for a given neural network
*/
void DOutDVal::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DOutDVal::resize(const ANN&):\n";
	dodi_=MatXd::Zero(nn.out().size(),nn.in().size());
	doda_.resize(nn.nlayer()+1);
	for(int n=0; n<nn.nlayer()+1; ++n){
		doda_[n]=MatXd::Zero(nn.out().size(),nn.node(n).size());
	}
}

/**
* compute the gradient of output w.r.t. all other node values (e.g. doda_ and dodi_)
*/
void DOutDVal::grad(const ANN& nn){
	//back-propogate the gradient (n.b. do/dz_{o}=outw_ "gradient of out_ w.r.t. the input of out_ is outw_")
	doda_.back()=nn.outw().asDiagonal();
	for(int l=nn.nlayer()-1; l>=0; --l){
		doda_[l].noalias()=doda_[l+1]*(nn.dadz(l).asDiagonal()*nn.edge(l));
	}
	//compute gradient of out_ w.r.t. in_ (effect of input scaling)
	dodi_.noalias()=doda_[0]*nn.inw().asDiagonal();
}

//***********************************************************************
// DOutDP
//***********************************************************************

/**
* clear all local data
*/
void DOutDP::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"DOutDP::clear():\n";
	dodz_.clear();
}

/**
* resize data for a given neural network
*/
void DOutDP::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DOutDP::resize(const ANN&):\n";
	dodz_.resize(nn.nlayer());
	for(int n=0; n<nn.nlayer(); ++n){
		dodz_[n]=MatXd::Zero(nn.out().size(),nn.bias(n).size());
	}
	dodw_.resize(nn.nlayer());
	for(int i=0; i<nn.nlayer(); ++i){
		dodw_[i].resize(nn.nOut());
		for(int j=0; j<nn.nOut(); ++j){
			dodw_[i][j]=MatXd::Zero(nn.edge(i).rows(),nn.edge(i).cols());
		}
	}
}

/**
* compute the gradient of output w.r.t. parameters
*/
void DOutDP::grad(const ANN& nn){
	//back-propogate the gradient (n.b. do/dz_{o}=outw_ "gradient of out_ w.r.t. the input of out_ is outw_")
	dodz_.back()=nn.outw().cwiseProduct(nn.dadz(nn.nlayer()-1)).asDiagonal();
	for(int l=nn.nlayer()-1; l>0; --l){
		dodz_[l-1].noalias()=dodz_[l]*nn.edge(l)*nn.dadz(l-1).asDiagonal();
	}
	//compute the gradient of the output w.r.t. the weights
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<nn.nOut(); ++n){
			for(int j=0; j<nn.edge(l).cols(); ++j){
				for(int i=0; i<nn.edge(l).rows(); ++i){
					dodw_[l][n](i,j)=dodz_[l](n,i)*nn.node(l)[j];
				}
			}
		}
	}
}

//***********************************************************************
// D2OutDPDVal
//***********************************************************************

/**
* clear all local data
*/
void D2OutDPDVal::clear(){
	nnc_.clear();
	dOutDVal_.clear();
	d2odpda_.clear();
}

/**
* resize data for a given neural network
*/
void D2OutDPDVal::resize(const ANN& nn){
	//gradient of the ouput with respect to the input
	dOutDVal_.resize(nn);
	//second derivative
	d2odpda_.resize(nn.size());
}

/**
* compute the gradient of the gradient of output w.r.t. the weights
*/
void D2OutDPDVal::grad(const ANN& nn){
	//local variables
	int count=0;
	//make copy of the network 
	nnc_=nn;
	//loop over all biases
	for(int l=0; l<nnc_.nlayer(); ++l){
		for(int n=0; n<nnc_.bias(l).size(); ++n){
			const double delta=nnc_.bias(l)[n]/100.0;
			//point 1
			nnc_.bias(l)[n]=nn.bias(l)[n]-delta;
			nnc_.execute();
			dOutDVal_.grad(nnc_);
			pt1_=dOutDVal_.dodi();
			//point 2
			nnc_.bias(l)[n]=nn.bias(l)[n]+delta;
			nnc_.execute();
			dOutDVal_.grad(nnc_);
			pt2_=dOutDVal_.dodi();
			//gradient
			d2odpda_[count++].noalias()=0.5*(pt2_-pt1_)/delta;
		}
	}
	//loop over all weights
	for(int l=0; l<nnc_.nlayer(); ++l){
		for(int n=0; n<nn.edge(l).size(); ++n){
			const double delta=nnc_.edge(l)(n)/100.0;
			//point 1
			nnc_.edge(l)(n)=nn.edge(l)(n)-delta;
			nnc_.execute();
			dOutDVal_.grad(nnc_);
			pt1_=dOutDVal_.dodi();
			//point 2
			nnc_.edge(l)(n)=nn.edge(l)(n)+delta;
			nnc_.execute();
			dOutDVal_.grad(nnc_);
			pt2_=dOutDVal_.dodi();
			//gradient
			d2odpda_[count++].noalias()=0.5*(pt2_-pt1_)/delta;
		}
	}
}

}

//***********************************************************************
// serialization
//***********************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NeuralNet::ANNInit& obj){
		if(NN_PRINT_FUNC>0) std::cout<<"nbytes(const NeuralNet::ANNInit&):\n";
		int size=0;
		size+=sizeof(double);//bInit_
		size+=sizeof(double);//wInit_
		size+=sizeof(double);//sigma_
		size+=sizeof(RNG::DistN::type);
		size+=sizeof(NeuralNet::InitN::type);
		size+=sizeof(int);//seed_
		return size;
	}
	
	template <> int nbytes(const NeuralNet::ANN& obj){
		if(NN_PRINT_FUNC>0) std::cout<<"nbytes(const NeuralNet::ANN&):\n";
		int size=0;
		size+=sizeof(int);//nlayer_
		size+=sizeof(int)*(obj.nlayer()+1);//number of nodes in each layer
		size+=sizeof(NeuralNet::TransferN::type);//transfer function type
		for(int l=0; l<obj.nlayer(); ++l) size+=obj.bias(l).size()*sizeof(double);//bias
		for(int l=0; l<obj.nlayer(); ++l) size+=obj.edge(l).size()*sizeof(double);//edge
		size+=obj.nIn()*sizeof(double);//pre-scale
		size+=obj.nIn()*sizeof(double);//pre-bias
		size+=obj.nOut()*sizeof(double);//post-scale
		size+=obj.nOut()*sizeof(double);//post-bias
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NeuralNet::ANNInit& obj, char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"pack(const NeuralNet::ANNInit&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.bInit(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.wInit(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.sigma(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.distT(),sizeof(RNG::DistN::type)); pos+=sizeof(RNG::DistN::type);
		std::memcpy(arr+pos,&obj.initType(),sizeof(NeuralNet::InitN::type)); pos+=sizeof(NeuralNet::InitN::type);
		std::memcpy(arr+pos,&obj.sigma(),sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
	template <> int pack(const NeuralNet::ANN& obj, char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"pack(const NeuralNet::ANN&,char*):\n";
		int pos=0;
		int tempInt=0;
		//nlayer
		std::memcpy(arr+pos,&(tempInt=obj.nlayer()),sizeof(int)); pos+=sizeof(int);
		//number of nodes in each layer
		for(int l=0; l<obj.nlayer()+1; ++l){
			std::memcpy(arr+pos,&(tempInt=obj.nNodes(l)),sizeof(int)); pos+=sizeof(int);
		}
		//transfer function type
		std::memcpy(arr+pos,&(obj.tfType()),sizeof(NeuralNet::TransferN::type)); pos+=sizeof(NeuralNet::TransferN::type);
		//bias
		for(int l=0; l<obj.nlayer(); ++l){
			for(int n=0; n<obj.bias(l).size(); ++n){
				std::memcpy(arr+pos,&(obj.bias(l)(n)),sizeof(double)); pos+=sizeof(double);
			}
		}
		//edge
		for(int l=0; l<obj.nlayer(); ++l){
			for(int n=0; n<obj.edge(l).size(); ++n){
				std::memcpy(arr+pos,&(obj.edge(l)(n)),sizeof(double)); pos+=sizeof(double);
			}
		}
		//pre-scale
		for(int i=0; i<obj.nIn(); ++i){
			std::memcpy(arr+pos,&(obj.inw()[i]),sizeof(double)); pos+=sizeof(double);
		}
		//pre-bias
		for(int i=0; i<obj.nIn(); ++i){
			std::memcpy(arr+pos,&(obj.inb()[i]),sizeof(double)); pos+=sizeof(double);
		}
		//post-scale
		for(int i=0; i<obj.nOut(); ++i){
			std::memcpy(arr+pos,&(obj.outw()[i]),sizeof(double)); pos+=sizeof(double);
		}
		//post-bias
		for(int i=0; i<obj.nOut(); ++i){
			std::memcpy(arr+pos,&(obj.outb()[i]),sizeof(double)); pos+=sizeof(double);
		}
		//return bytes written
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NeuralNet::ANNInit& obj, const char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"unpack(NeuralNet::ANNInit&,const char*):\n";
		//local variables
		int pos=0;
		std::memcpy(&obj.bInit(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.wInit(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.sigma(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.distT(),arr+pos,sizeof(RNG::DistN::type)); pos+=sizeof(RNG::DistN::type);
		std::memcpy(&obj.initType(),arr+pos,sizeof(NeuralNet::InitN::type)); pos+=sizeof(NeuralNet::InitN::type);
		std::memcpy(&obj.sigma(),arr+pos,sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
	template <> int unpack(NeuralNet::ANN& obj, const char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"unpack(NeuralNet::ANN&,const char*):\n";
		//local variables
		int pos=0;
		int nlayer=0,nIn=0;
		std::vector<int> nNodes;
		//nlayer
		std::memcpy(&nlayer,arr+pos,sizeof(int)); pos+=sizeof(int);
		nNodes.resize(nlayer+1,0);
		//number of nodes in each layer
		for(int i=0; i<nlayer+1; ++i){
			std::memcpy(&nNodes[i],arr+pos,sizeof(int)); pos+=sizeof(int);
		}
		//transfer function type
		std::memcpy(&(obj.tfType()),arr+pos,sizeof(NeuralNet::TransferN::type)); pos+=sizeof(NeuralNet::TransferN::type);
		//resize the network
		NeuralNet::ANNInit init;
		obj.resize(init,nNodes);
		//bias
		for(int l=0; l<obj.nlayer(); ++l){
			for(int n=0; n<obj.bias(l).size(); ++n){
				std::memcpy(&(obj.bias(l)(n)),arr+pos,sizeof(double)); pos+=sizeof(double);
			}
		}
		//edge
		for(int l=0; l<obj.nlayer(); ++l){
			for(int n=0; n<obj.edge(l).size(); ++n){
				std::memcpy(&(obj.edge(l)(n)),arr+pos,sizeof(double)); pos+=sizeof(double);
			}
		}
		//pre-scale
		for(int i=0; i<obj.nIn(); ++i){
			std::memcpy(&(obj.inw()[i]),arr+pos,sizeof(double)); pos+=sizeof(double);
		}
		//pre-bias
		for(int i=0; i<obj.nIn(); ++i){
			std::memcpy(&(obj.inb()[i]),arr+pos,sizeof(double)); pos+=sizeof(double);
		}
		//post-scale
		for(int i=0; i<obj.nOut(); ++i){
			std::memcpy(&(obj.outw()[i]),arr+pos,sizeof(double)); pos+=sizeof(double);
		}
		//post-bias
		for(int i=0; i<obj.nOut(); ++i){
			std::memcpy(&(obj.outb()[i]),arr+pos,sizeof(double)); pos+=sizeof(double);
		}
		//return bytes read
		return pos;
	};
	
}