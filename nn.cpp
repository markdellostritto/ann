#include "nn.hpp"

namespace NN{

//***********************************************************************
// TRANSFER FUNCTIONS - NAMES
//***********************************************************************

std::ostream& operator<<(std::ostream& out, const TransferN::type& tf){
	switch(tf){
		case TransferN::TANH: out<<"TANH"; break;
		case TransferN::SIGMOID: out<<"SIGMOID"; break;
		case TransferN::LINEAR: out<<"LINEAR"; break;
		case TransferN::SOFTPLUS: out<<"SOFTPLUS"; break;
		case TransferN::RELU: out<<"RELU"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

TransferN::type TransferN::read(const char* str){
	if(std::strcmp(str,"TANH")==0) return TransferN::TANH;
	else if(std::strcmp(str,"SIGMOID")==0) return TransferN::SIGMOID;
	else if(std::strcmp(str,"LINEAR")==0) return TransferN::LINEAR;
	else if(std::strcmp(str,"SOFTPLUS")==0) return TransferN::SOFTPLUS;
	else if(std::strcmp(str,"RELU")==0) return TransferN::RELU;
	else throw std::invalid_argument("Invalid transfer function name.");
}

void TransferFFDV::f_tanh(Eigen::VectorXd& f, Eigen::VectorXd& d)noexcept{
	const int size=f.size();
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	for(int i=size-1; i>=0; --i) f[i]=std::tanh(f[i]);
	for(int i=size-1; i>=0; --i) d[i]=1.0-f[i]*f[i];
	#elif (defined __ICC || defined __INTEL_COMPILER)
	for(int i=size-1; i>=0; --i) f[i]=tanh(f[i]);
	for(int i=size-1; i>=0; --i) d[i]=1.0-f[i]*f[i];
	#endif
}

void TransferFFDV::f_sigmoid(Eigen::VectorXd& f, Eigen::VectorXd& d)noexcept{
	const int size=f.size();
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	for(int i=size-1; i>=0; --i){
		if(f[i]>=0){
			const double expf=std::exp(-f[i]);
			f[i]=1.0/(1.0+expf);
			d[i]=1.0/((1.0+expf)*(1.0+1.0/expf));
		} else {
			const double expf=std::exp(f[i]);
			f[i]=expf/(expf+1.0);
			d[i]=1.0/((1.0+1.0/expf)*(1.0+expf));
		}
	}
	#elif (defined __ICC || defined __INTEL_COMPILER)
	for(int i=size-1; i>=0; --i){
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
	#endif			
}

void TransferFFDV::f_lin(Eigen::VectorXd& f, Eigen::VectorXd& d)noexcept{
	for(int i=d.size()-1; i>=0; --i) d[i]=1.0;
}

void TransferFFDV::f_softplus(Eigen::VectorXd& f, Eigen::VectorXd& d)noexcept{
	const int size=f.size();
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	for(int i=size-1; i>=0; --i){
		if(f[i]>=1.0){
			const double expf=std::exp(-f[i]);
			f[i]+=special::logp1(expf);
			d[i]=1.0/(1.0+expf);
		} else {
			const double expf=std::exp(f[i]);
			const double g=expf/(expf+2.0);
			const double g2=g*g;
			f[i]=2.0*g*(1.0+g2*(1.0/3.0+g2*(1.0/5.0+g2*(1.0/7.0+g2*1.0/9.0))));
			d[i]=expf/(expf+1.0);
		}
	}
	#elif (defined __ICC || defined __INTEL_COMPILER)
	for(int i=size-1; i>=0; --i){
		if(f[i]>=1.0){
			const double expf=exp(-f[i]);
			f[i]+=special::logp1(expf);
			d[i]=1.0/(1.0+expf);
		} else {
			const double expf=exp(f[i]);
			const double g=expf/(expf+2.0);
			const double g2=g*g;
			f[i]=2.0*g*(1.0+g2*(1.0/3.0+g2*(1.0/5.0+g2*(1.0/7.0+g2*1.0/9.0))));
			d[i]=expf/(expf+1.0);
		}
	}
	#endif
}

void TransferFFDV::f_relu(Eigen::VectorXd& f, Eigen::VectorXd& d)noexcept{
	const int size=f.size();
	for(int i=size-1; i>=0; --i){
		if(f[i]>0){
			d[i]=1.0;
		} else {
			f[i]=0.0;
			d[i]=0.0;
		}
	}
}

//***********************************************************************
// NETWORK CLASS
//***********************************************************************

//operators

std::ostream& operator<<(std::ostream& out, const Network& nn){
	out<<"**************************************************\n";
	out<<"*********************** NN ***********************\n";
	out<<"nn        = "<<nn.input_.size()<<" "; for(unsigned int n=0; n<nn.node_.size(); ++n) out<<nn.node_[n].size()<<" "; out<<"\n";
	out<<"size      = "<<nn.size()<<"\n";
	out<<"transfer  = "<<nn.tfType_<<"\n";
	out<<"lambda    = "<<nn.lambda_<<"\n";
	out<<"b-init    = "<<nn.bInit_<<"\n";
	out<<"w-init    = "<<nn.wInit_<<"\n";
	out<<"*********************** NN ***********************\n";
	out<<"**************************************************";
	return out;
}

Eigen::VectorXd& operator>>(const Network& nn, Eigen::VectorXd& v){
	unsigned int count=0;
	v=Eigen::VectorXd::Zero(nn.size());
	for(unsigned int l=0; l<nn.nlayer(); ++l){
		for(unsigned int n=0; n<nn.nNodes(l); ++n){
			v[count++]=nn.bias(l,n);
		}
	}
	for(unsigned int l=0; l<nn.nlayer(); ++l){
		for(unsigned int n=0; n<nn.edge(l).rows(); ++n){
			for(unsigned int m=0; m<nn.edge(l).cols(); ++m){
				v[count++]=nn.edge(l,n,m);
			}
		}
	}
}

Network& operator<<(Network& nn, const Eigen::VectorXd& v){
	if(nn.size()!=v.size()) throw std::invalid_argument("Invalid size: vector and network mismatch.");
	unsigned int count=0;
	for(unsigned int l=0; l<nn.nlayer(); ++l){
		for(unsigned int n=0; n<nn.nNodes(l); ++n){
			nn.bias(l,n)=v[count++];
		}
	}
	for(unsigned int l=0; l<nn.nlayer(); ++l){
		for(unsigned int n=0; n<nn.edge(l).rows(); ++n){
			for(unsigned int m=0; m<nn.edge(l).cols(); ++m){
				nn.edge(l,n,m)=v[count++];
			}
		}
	}
	return nn;
}

//member functions

//set the default values
void Network::defaults(){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::defaults():\n";
	//network dimensions
		nlayer_=0;
	//initialize
		bInit_=0.001;
		wInit_=1;
	//node weights and biases
		input_.resize(0);
		sinput_.resize(0);
		output_.resize(0);
		node_.clear();
		bias_.clear();
		edge_.clear();
	//gradients
		grad_.resize(0);
		dndz_.clear();
		delta_.clear();
		dOut_.clear();
	//transfer functions
		tfType_=TransferN::UNKNOWN;
		tffdv_.clear();
	//conditioning
		preScale_.resize(0);
		postScale_.resize(0);
		preBias_.resize(0);
		postBias_.resize(0);
	//regularization
		lambda_=0;
}

//clear all values
void Network::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::clear():\n";
	//network dimensions
		nlayer_=0;
	//node weights and biases
		input_.resize(0);
		sinput_.resize(0);
		output_.resize(0);
		node_.clear();
		bias_.clear();
		edge_.clear();
	//gradients
		grad_.resize(0);
		dndz_.clear();
		delta_.clear();
	//transfer functions
		tffdv_.clear();
	//conditioning
		preScale_.resize(0);
		postScale_.resize(0);
		preBias_.resize(0);
		postBias_.resize(0);
}

//the size of the network - the number of adjustable parameters
unsigned int Network::size()const{
	if(NN_PRINT_FUNC>0) std::cout<<"Network::size():\n";
	unsigned int s=0;
	for(int n=bias_.size()-1; n>=0; --n) s+=bias_[n].size();
	for(int n=edge_.size()-1; n>=0; --n) s+=edge_[n].size();
	return s;
}

//resize the network - no hidden layers
void Network::resize(unsigned int nInput, unsigned int nOutput){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::resize(unsigned int,unsigned int):\n";
	if(nInput==0) throw std::invalid_argument("Invalid output size.");
	if(nOutput==0) throw std::invalid_argument("Invalid output size.");
	std::vector<unsigned int> nn(1,nOutput);
	resize(nInput,nn);
}

//resize the network - given separate hidden layers and output layer
void Network::resize(unsigned int nInput, const std::vector<unsigned int>& nNodes, unsigned int nOutput){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::resize(unsigned int,const std::vector<unsigned int>&,unsigned int):\n";
	if(nOutput==0) throw std::invalid_argument("Invalid output size.");
	std::vector<unsigned int> nn(nNodes.size()+1);
	for(unsigned int n=0; n<nNodes.size(); ++n) nn[n]=nNodes[n];
	nn.back()=nOutput;
	resize(nInput,nn);
}

//resize the network - given combined hidden layers and output layer
void Network::resize(unsigned int nInput, const std::vector<unsigned int>& nNodes){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::resize(unsigned int,const std::vector<unsigned int>&):\n";
	//initialize the random number generator
		std::srand(std::time(NULL));
	//clear the network
		clear();
	//check parameters
		if(nInput==0) throw std::invalid_argument("Invalid input size.");
		for(unsigned int n=0; n<nNodes.size(); ++n){
			if(nNodes[n]==0) throw std::invalid_argument("Invalid layer size.");
		}
	//input/output
		input_=Eigen::VectorXd::Zero(nInput);
		sinput_=Eigen::VectorXd::Zero(nInput);
		output_=Eigen::VectorXd::Zero(nNodes.back());
	//nodes
		nlayer_=nNodes.size();
		node_.resize(nlayer_);
		for(unsigned int n=0; n<nlayer_; ++n) node_[n]=Eigen::VectorXd::Zero(nNodes[n]);
	//bias
		bias_.resize(nlayer_);
		for(unsigned int n=0; n<nlayer_; ++n) bias_[n]=Eigen::VectorXd::Random(nNodes[n])*bInit_;
	//edges
		edge_.resize(nlayer_);
		edge_[0]=Eigen::MatrixXd::Random(nNodes[0],nInput)*wInit_;
		for(unsigned int n=1; n<nlayer_; ++n) edge_[n]=Eigen::MatrixXd::Random(nNodes[n],nNodes[n-1])*std::sqrt(2.0/node_[n-1].size())*wInit_;
	//dOut
		dOut_.resize(nlayer_+2);//number of layers + scaling of input and ouput
		dOut_[0]=Eigen::MatrixXd::Zero(nNodes.back(),nInput);
		dOut_[0+1]=Eigen::MatrixXd::Zero(nNodes.back(),nInput);
		for(unsigned int n=1; n<nlayer_; ++n) dOut_[n+1]=Eigen::MatrixXd::Zero(nNodes.back(),nNodes[n-1]);
		dOut_.back()=Eigen::MatrixXd::Zero(nNodes.back(),nNodes.back());
	//pre/post conditioning
		preScale_=Eigen::VectorXd::Constant(nInput,1);
		postScale_=Eigen::VectorXd::Constant(nNodes.back(),1);
		preBias_=Eigen::VectorXd::Constant(nInput,0);
		postBias_=Eigen::VectorXd::Constant(nNodes.back(),0);
	//transfer functions
		switch(tfType_){
			case TransferN::LINEAR: tffdv_.resize(nlayer_,TransferFFDV::f_lin); break;
			case TransferN::SIGMOID: tffdv_.resize(nlayer_,TransferFFDV::f_sigmoid); break;
			case TransferN::TANH: tffdv_.resize(nlayer_,TransferFFDV::f_tanh); break;
			case TransferN::SOFTPLUS: tffdv_.resize(nlayer_,TransferFFDV::f_softplus); break;
			case TransferN::RELU: tffdv_.resize(nlayer_,TransferFFDV::f_relu); break;
			default: throw std::invalid_argument("Invalid transfer function."); break;
		}
		//final layer is typically linear, though this can be changed after initialization
		tffdv_.back()=TransferFFDV::f_lin;
	//gradients
		grad_.resize(nNodes.back());
		dndz_.resize(nlayer_);
		for(unsigned int n=0; n<nlayer_; ++n) dndz_[n]=Eigen::VectorXd::Zero(nNodes[n]);
		delta_.resize(nlayer_);
		for(unsigned int n=0; n<nlayer_; ++n) delta_[n]=Eigen::VectorXd::Zero(nNodes[n]);
}

//reset the node, bias, edge, input/output, bias/scaling values
void Network::reset(){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::reset():\n";
	//nodes
		for(unsigned int n=0; n<nlayer_; ++n) node_[n]=Eigen::VectorXd::Zero(node_[n].size());
	//bias
		for(unsigned int n=0; n<nlayer_; ++n) bias_[n]=Eigen::VectorXd::Random(bias_[n].size())*bInit_;
	//edges
		edge_[0]=Eigen::MatrixXd::Random(edge_[0].rows(),edge_[0].cols())*wInit_;
		for(unsigned int n=1; n<nlayer_; ++n) edge_[n]=Eigen::MatrixXd::Random(edge_[n].rows(),edge_[n].cols())*std::sqrt(2.0/node_[n-1].size())*wInit_;
	//dOut
		for(unsigned int n=0; n<dOut_.size(); ++n) dOut_[n]=Eigen::MatrixXd::Zero(dOut_[n].rows(),dOut_[n].cols());
	//pre/post conditioning
		preScale_=Eigen::VectorXd::Constant(input_.size(),1);
		postScale_=Eigen::VectorXd::Constant(output_.size(),1);
		preBias_=Eigen::VectorXd::Constant(input_.size(),0);
		postBias_=Eigen::VectorXd::Constant(output_.size(),0);
	
}

//compute the error associated the output, given the target output
double Network::error(const Eigen::VectorXd& output)const{
	if(NN_PRINT_FUNC>0) std::cout<<"Network::error(const Eigen::VectorXd&):\n";
	//compute error
	double err=0.5*(output_-output).squaredNorm();
	if(lambda_>0){
		//find the number of weights
		double nw=0;
		for(int l=nlayer_-1; l>=0; --l) nw+=edge_[l].size();
		//compute lambda error
		for(int l=nlayer_-1; l>=0; --l) err+=0.5*lambda_*edge_[l].squaredNorm()/nw;
	}
	return err;
}

//compute the regularization error
double Network::error_lambda()const{
	if(NN_PRINT_FUNC>0) std::cout<<"Network::error_lambda():\n";
	double nw=0,err=0;
	for(int l=nlayer_-1; l>=0; --l){
		nw+=edge_[l].size();//number of weights
		err+=0.5*lambda_*edge_[l].squaredNorm();//lambda error - quadratic
	}
	//return error
	return err/nw;
}

//compute dcda, the derivative of the quadratic const function w.r.t. the inputs
Eigen::VectorXd& Network::dcda(const Eigen::VectorXd& output, Eigen::VectorXd& grad)const{
	if(NN_PRINT_FUNC>0) std::cout<<"Network::dcda(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	grad.noalias()=(output_-output);
	return grad;
}

//compute value and gradient of error at a given point
double Network::error(const Eigen::VectorXd& output, Eigen::VectorXd& grad){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::error(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	if(NN_PRINT_DATA>1){
		std::cout<<"output_ = "<<output_.transpose()<<"\n";
		std::cout<<"output  = "<<output.transpose()<<"\n";
		std::cout<<"error   = "<<0.5*(output_-output).squaredNorm()<<"\n";
	}
	//compute the error
	double err=0.5*(output_-output).squaredNorm();
	//compute the gradient of the error function
	grad_.noalias()=(output_-output);
	if(NN_PRINT_DATA>1) std::cout<<"grad_ = "<<grad_<<"\n";
	//compute delta for the output layer
	delta_.back().noalias()=grad_.cwiseProduct(dndz_.back());
	//back-propogate the error
	for(int l=nlayer_-1; l>0; --l){
		delta_[l-1].noalias()=dndz_[l-1].cwiseProduct(edge_[l].transpose()*delta_[l]);
	}
	unsigned int count=0;
	//gradient w.r.t bias
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. bias\n";
	for(unsigned int l=0; l<nlayer_; ++l){
		for(unsigned int n=0; n<delta_[l].size(); ++n){
			grad[count++]=delta_[l][n];
		}
	}
	//gradient w.r.t. edges
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. edges\n";
	for(unsigned int n=0; n<edge_[0].rows(); ++n){
		for(unsigned int m=0; m<edge_[0].cols(); ++m){
			grad[count++]=delta_[0][n]*sinput_[m];//edge(0,n,m)
		}
	}
	for(unsigned int l=1; l<nlayer_; ++l){
		for(unsigned int n=0; n<edge_[l].rows(); ++n){
			for(unsigned int m=0; m<edge_[l].cols(); ++m){
				grad[count++]=delta_[l][n]*node_[l-1](m);//edge(l,n,m)
			}
		}
	}
	//regularization
	if(lambda_>0){
		if(NN_PRINT_STATUS>1) std::cout<<"computing regularization contribution\n";
		//compute the number of weights
		double nw=0;
		for(int l=nlayer_-1; l>=0; --l) nw+=edge_[l].size();
		//compute the regularization error
		for(int l=nlayer_-1; l>=0; --l) err+=0.5*lambda_*edge_[l].squaredNorm()/nw;
		//compute the regularization gradient
		count=0;
		for(unsigned int n=0; n<delta_.size(); ++n) count+=delta_[n].size();
		for(unsigned int l=0; l<edge_.size(); ++l){
			for(unsigned int n=0; n<edge_[l].rows(); ++n){
				for(unsigned int m=0; m<edge_[l].cols(); ++m){
					grad[count++]+=lambda_*edge_[l](n,m)/nw;//edge(l,n,m) - quadratic
				}
			}
		}
	}
	//return the error
	return err;
}

//compute gradient of error at a given point, given the derivative of the gradient w.r.t. the output (dcda)
Eigen::VectorXd& Network::grad(const Eigen::VectorXd& dcda, Eigen::VectorXd& grad){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::grad(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	//store the gradient of the error function
	grad_=dcda;
	if(NN_PRINT_DATA>1) std::cout<<"grad_ = "<<grad_<<"\n";
	//compute delta for the output layer
	delta_.back().noalias()=grad_.cwiseProduct(dndz_.back());
	//back-propogate the error
	for(int l=nlayer_-1; l>0; --l){
		delta_[l-1].noalias()=dndz_[l-1].cwiseProduct(edge_[l].transpose()*delta_[l]);
	}
	unsigned int count=0;
	//gradient w.r.t bias
	for(unsigned int l=0; l<nlayer_; ++l){
		for(unsigned int n=0; n<delta_[l].size(); ++n){
			grad[count++]=delta_[l][n];
		}
	}
	//gradient w.r.t. edges
	for(unsigned int n=0; n<edge_[0].rows(); ++n){
		for(unsigned int m=0; m<edge_[0].cols(); ++m){
			grad[count++]=delta_[0][n]*sinput_[m];//edge(0,n,m)
		}
	}
	for(unsigned int l=1; l<nlayer_; ++l){
		for(unsigned int n=0; n<edge_[l].rows(); ++n){
			for(unsigned int m=0; m<edge_[l].cols(); ++m){
				grad[count++]=delta_[l][n]*node_[l-1](m);//edge(l,n,m)
			}
		}
	}
	//regularization
	if(lambda_>0){
		//compute the number of weights
		double nw=0;
		for(unsigned int n=0; n<edge_.size(); ++n) nw+=edge_[n].size();
		count=0;
		//compute the regularization gradient
		for(unsigned int n=0; n<delta_.size(); ++n) count+=delta_[n].size();
		for(unsigned int l=0; l<edge_.size(); ++l){
			for(unsigned int n=0; n<edge_[l].rows(); ++n){
				for(unsigned int m=0; m<edge_[l].cols(); ++m){
					grad[count++]=lambda_*edge_[l](n,m)/nw;//edge(l,n,m) - quadratic
				}
			}
		}
	}
	//return the gradient
	return grad;
}

//compute gradient of error at a given point, given the derivative of the gradient w.r.t. the output (dcda) - no regularization
Eigen::VectorXd& Network::grad_nol(const Eigen::VectorXd& dcda, Eigen::VectorXd& grad){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::grad_nol(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	//store the gradient of the error function
	grad_=dcda;
	if(NN_PRINT_DATA>1) std::cout<<"grad_ = "<<grad_<<"\n";
	//compute delta for the output layer
	delta_.back().noalias()=grad_.cwiseProduct(dndz_.back());
	//back-propogate the error
	for(int l=nlayer_-1; l>0; --l){
		delta_[l-1].noalias()=dndz_[l-1].cwiseProduct(edge_[l].transpose()*delta_[l]);
	}
	unsigned int count=0;
	//gradient w.r.t bias
	for(unsigned int l=0; l<nlayer_; ++l){
		for(unsigned int n=0; n<delta_[l].size(); ++n){
			grad[count++]=delta_[l][n];
		}
	}
	//gradient w.r.t. edges
	for(unsigned int n=0; n<edge_[0].rows(); ++n){
		for(unsigned int m=0; m<edge_[0].cols(); ++m){
			grad[count++]=delta_[0][n]*sinput_[m];//edge(0,n,m)
		}
	}
	for(unsigned int l=1; l<nlayer_; ++l){
		for(unsigned int n=0; n<edge_[l].rows(); ++n){
			for(unsigned int m=0; m<edge_[l].cols(); ++m){
				grad[count++]=delta_[l][n]*node_[l-1](m);//edge(l,n,m)
			}
		}
	}
	//return the gradient
	return grad;
}

//compute the regularization gradient
Eigen::VectorXd& Network::grad_lambda(Eigen::VectorXd& grad)const{
	if(NN_PRINT_FUNC>0) std::cout<<"Network::grad_lambda(Eigen::VectorXd&):\n";
	unsigned int count=0;
	//gradient w.r.t bias
	for(unsigned int l=0; l<nlayer_; ++l){
		for(unsigned int n=0; n<delta_[l].size(); ++n){
			grad[count++]=0.0;
		}
	}
	//gradient w.r.t. edges
	double nw=0;
	for(unsigned int l=0; l<edge_.size(); ++l){
		nw+=edge_[l].size();
		for(unsigned int n=0; n<edge_[l].rows(); ++n){
			for(unsigned int m=0; m<edge_[l].cols(); ++m){
				grad[count++]=lambda_*edge_[l](n,m);//edge(l,n,m) - quadratic
			}
		}
	}
	//normalize gradient
	grad/=nw;
	//return the gradient
	return grad;
}

//compute the gradient of output node n on all other nodes, the gradients are stored in "delta_"
void Network::grad_out(){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::grad_out():\n";
	//compute delta for the output scaling layer
	dOut_[nlayer_+1].diagonal()=postScale_;
	//back-propogate the gradient
	for(int l=nlayer_; l>0; --l){
		dOut_[l].noalias()=dOut_[l+1]*(dndz_[l-1].asDiagonal()*edge_[l-1]);
	}
	//compute delta for the input scaling layer
	dOut_[0].noalias()=dOut_[0+1]*preScale_.asDiagonal();
}

//execute the network
const Eigen::VectorXd& Network::execute(){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::execute():\n";
	//scale the input
	sinput_.noalias()=preScale_.cwiseProduct(input_+preBias_);
	//first layer
	node_.front()=bias_.front();
	node_.front().noalias()+=edge_.front()*sinput_;
	(*tffdv_.front())(node_.front(),dndz_.front());
	//subsequent layers
	for(int l=1; l<node_.size(); ++l){
		node_[l]=bias_[l];
		node_[l].noalias()+=edge_[l]*node_[l-1];
		(*tffdv_[l])(node_[l],dndz_[l]);
	}
	//scale the output
	output_=postBias_;
	output_.noalias()+=node_.back().cwiseProduct(postScale_);
	//return the output
	return output_;
}

//static functions

//write the network to file
void Network::write(const char* file, const Network& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::write(const char*,const Network&):\n";
	//local variables
	FILE* writer=NULL;
	//open the file
	writer=std::fopen(file,"w");
	if(writer!=NULL){
		Network::write(writer,nn);
		std::fclose(writer);
		writer=NULL;
	} else std::cout<<"WARNING: Could not open \""<<file<<"\" for printing.\n";
}

//write the network to file
void Network::write(FILE* writer, const Network& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::write(FILE*,const Network&):\n";
	//print the configuration
	fprintf(writer,"nn ");
	fprintf(writer,"%i ",nn.nInput());
	for(unsigned int i=0; i<nn.nlayer(); ++i) fprintf(writer,"%i ",nn.nlayer(i));
	fprintf(writer,"\n");
	//print the transfer function
	switch(nn.tfType()){
		case TransferN::TANH: fprintf(writer,"t_func TANH\n"); break;
		case TransferN::SIGMOID: fprintf(writer,"t_func SIGMOID\n"); break;
		case TransferN::LINEAR: fprintf(writer,"t_func LINEAR\n"); break;
	}
	//print the scaling layers
	fprintf(writer,"input-scale ");
	for(unsigned int i=0; i<nn.nInput(); ++i) fprintf(writer,"%f ",nn.preScale()[i]);
	fprintf(writer,"\n");
	fprintf(writer,"output-scale ");
	for(unsigned int i=0; i<nn.nOutput(); ++i) fprintf(writer,"%f ",nn.postScale()[i]);
	fprintf(writer,"\n");
	//print the biasing layers
	fprintf(writer,"input-bias ");
	for(unsigned int i=0; i<nn.nInput(); ++i) fprintf(writer,"%f ",nn.preBias()[i]);
	fprintf(writer,"\n");
	fprintf(writer,"output-bias ");
	for(unsigned int i=0; i<nn.nOutput(); ++i) fprintf(writer,"%f ",nn.postBias()[i]);
	fprintf(writer,"\n");
	//print the biases
	for(unsigned int n=0; n<nn.nlayer(); ++n){
		fprintf(writer,"bias[%i] ",n+1);
		for(unsigned int i=0; i<nn.nlayer(n); ++i){
			fprintf(writer,"%f ",nn.bias(n,i));
		}
		fprintf(writer,"\n");
	}
	//print the edge weights
	for(unsigned int n=0; n<nn.nlayer(); ++n){
		fprintf(writer,"weight[%i,%i] ",n,n+1);
		for(unsigned int i=0; i<nn.edge(n).rows(); ++i){
			for(unsigned int j=0; j<nn.edge(n).cols(); ++j){
				fprintf(writer,"%f ",nn.edge(n,i,j));
			}
		}
		fprintf(writer,"\n");
	}
}

//read the network from file
void Network::read(const char* file, Network& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::read(const char*,Network&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		Network::read(reader,nn);
		std::fclose(reader);
		reader=NULL;
	} else std::cout<<"WARNING: Could not open \""<<file<<"\" for reading.\n";
}

//read the network from file
void Network::read(FILE* reader, Network& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::read(FILE*,Network&):\n";
	//==== local variables ====
	const unsigned int MAX=5000;
	const unsigned int N_DIGITS=16;//max number of digits in number
	unsigned int b_max=0;//max number of biases for a given layer
	unsigned int w_max=0;//max number of weights for a given layer
	char* input=new char[MAX];
	char* b_str=NULL;//bias string
	char* w_str=NULL;//weight string
	std::vector<unsigned int> nh;
	std::vector<std::string> strlist;
	//==== clear the network ====
	if(NN_PRINT_STATUS>0) std::cout<<"clearing the network\n";
	nn.clear();
	//==== load the configuration ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading configuration\n";
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	if(strlist.size()<2) throw std::invalid_argument("Invalid network configuration.");
	nh.resize(strlist.size()-2);//nh nInput nh0 nh1 nh2 ... nhN
	unsigned int nInput=std::atoi(strlist[1].c_str());
	for(unsigned int i=2; i<strlist.size(); ++i) nh[i-2]=std::atoi(strlist[i].c_str());
	if(NN_PRINT_DATA>0){std::cout<<"nn "<<nInput<<" "; for(unsigned int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<"\n";}
	//==== set the transfer function ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading transfer function\n";
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	nn.tfType()=TransferN::read(strlist[1].c_str());
	if(nn.tfType()==TransferN::UNKNOWN) throw std::invalid_argument("Invalid transfer function.");
	//==== resize the nueral newtork ====
	if(NN_PRINT_STATUS>0) std::cout<<"resizing neural network\n";
	if(NN_PRINT_STATUS>0) {std::cout<<nInput<<" "; for(unsigned int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<"\n";}
	nn.resize(nInput,nh);
	if(NN_PRINT_STATUS>1) std::cout<<"nn = "<<nn<<"\n";
	w_max=nn.nNodes(0)*nn.nInput();
	for(unsigned int i=0; i<nn.nlayer(); ++i) b_max=(b_max>nn.nNodes(i))?b_max:nn.nNodes(i);
	for(unsigned int i=1; i<nn.nlayer(); ++i) w_max=(w_max>nn.nNodes(i)*nn.nNodes(i-1))?w_max:nn.nNodes(i)*nn.nNodes(i-1);
	if(NN_PRINT_DATA>0) std::cout<<"b_max "<<b_max<<" w_max "<<w_max<<"\n";
	b_str=new char[b_max*N_DIGITS];
	w_str=new char[w_max*N_DIGITS];
	//==== read the scaling layers ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading scaling layers\n";
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	for(unsigned int j=0; j<strlist.size(); ++j) nn.preScale(j)=std::atof(strlist[j+1].c_str());
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	for(unsigned int j=0; j<strlist.size(); ++j) nn.postScale(j)=std::atof(strlist[j+1].c_str());
	//==== read the biasing layers ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading biasing layers\n";
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	for(unsigned int j=0; j<strlist.size(); ++j) nn.preBias(j)=std::atof(strlist[j+1].c_str());
	string::split(fgets(input,MAX,reader),string::WS,strlist);
	for(unsigned int j=0; j<strlist.size(); ++j) nn.postBias(j)=std::atof(strlist[j+1].c_str());
	//==== read in the biases ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading biases\n";
	for(unsigned int n=0; n<nn.nlayer(); ++n){
		string::split(fgets(b_str,b_max*N_DIGITS,reader),string::WS,strlist);
		for(unsigned int i=0; i<nn.nlayer(n); ++i){
			nn.bias(n,i)=std::atof(strlist[i+1].c_str());
		}
	}
	//==== read in the edge weights ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading weights\n";
	for(unsigned int n=0; n<nn.nlayer(); ++n){
		string::split(fgets(w_str,w_max*N_DIGITS,reader),string::WS,strlist);
		unsigned int count=0;
		for(unsigned int i=0; i<nn.edge(n).rows(); ++i){
			for(unsigned int j=0; j<nn.edge(n).cols(); ++j){
				nn.edge(n,i,j)=std::atof(strlist[++count].c_str());
			}
		}
	}
	//==== free local variables ====
	if(input!=NULL) delete[] input;
	if(b_str!=NULL) delete[] b_str;
	if(w_str!=NULL) delete[] w_str;
}

//operators

bool operator==(const Network& n1, const Network& n2){
	if(n1.tfType()!=n2.tfType()) return false;
	else if(n1.lambda()!=n2.lambda()) return false;
	else if(n1.nlayer()!=n2.nlayer()) return false;
	else if(n1.nInput()!=n2.nInput()) return false;
	else {
		//number of layers
		for(unsigned int i=0; i<n1.nlayer(); ++i){
			if(n1.nlayer(i)!=n2.nlayer(i)) return false;
		}
		//pre-/post-conditioning
		for(unsigned int i=0; i<n1.nInput(); ++i){
			if(n1.preScale(i)!=n2.preScale(i)) return false;
			if(n1.preBias(i)!=n2.preBias(i)) return false;
		}
		for(unsigned int i=0; i<n1.nOutput(); ++i){
			if(n1.postScale(i)!=n2.postScale(i)) return false;
			if(n1.postBias(i)!=n2.postBias(i)) return false;
		}
		//bias
		for(unsigned int i=0; i<n1.nlayer(); ++i){
			double diff=(n1.bias(i)-n2.bias(i)).norm();
			if(diff>num_const::ZERO) return false;
		}
		//edge
		for(unsigned int i=0; i<n1.nlayer(); ++i){
			double diff=(n1.edge(i)-n2.edge(i)).norm();
			if(diff>num_const::ZERO) return false;
		}
		//same
		return true;
	}
}

}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> unsigned int nbytes(const NN::Network& obj){
		unsigned int N=0;
		N+=sizeof(unsigned int);//nlayer_
		N+=sizeof(unsigned int);//nInput_
		N+=sizeof(unsigned int)*obj.nlayer();//number of nodes in each layer
		N+=sizeof(NN::TransferN::type);//transfer function type
		N+=sizeof(double);//lambda
		for(unsigned int l=0; l<obj.nlayer(); ++l) N+=obj.bias(l).size()*sizeof(double);//bias
		for(unsigned int l=0; l<obj.nlayer(); ++l) N+=obj.edge(l).rows()*obj.edge(l).cols()*sizeof(double);//edge
		N+=obj.nInput()*sizeof(double);//pre-scale
		N+=obj.nInput()*sizeof(double);//pre-bias
		N+=obj.nOutput()*sizeof(double);//post-scale
		N+=obj.nOutput()*sizeof(double);//post-bias
		return N;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> void pack(const NN::Network& obj, char* arr){
		unsigned int pos=0;
		unsigned int tempInt=0;
		//nlayer
		std::memcpy(arr+pos,&(tempInt=obj.nlayer()),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		//nInput
		std::memcpy(arr+pos,&(tempInt=obj.nInput()),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		//number of nodes in each layer
		for(unsigned int l=0; l<obj.nlayer(); ++l){
			std::memcpy(arr+pos,&(tempInt=obj.nNodes(l)),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//transfer function type
		std::memcpy(arr+pos,&(obj.tfType()),sizeof(NN::TransferN::type)); pos+=sizeof(NN::TransferN::type);
		//lambda
		std::memcpy(arr+pos,&(obj.lambda()),sizeof(double)); pos+=sizeof(double);
		//bias
		for(unsigned int l=0; l<obj.nlayer(); ++l){
			for(unsigned int n=0; n<obj.bias(l).size(); ++n){
				std::memcpy(arr+pos,&(obj.bias(l,n)),sizeof(double)); pos+=sizeof(double);
			}
		}
		//edge
		for(unsigned int l=0; l<obj.nlayer(); ++l){
			for(unsigned int n=0; n<obj.edge(l).cols(); ++n){
				for(unsigned int m=0; m<obj.edge(l).rows(); ++m){
					std::memcpy(arr+pos,&(obj.edge(l,m,n)),sizeof(double)); pos+=sizeof(double);
				}
			}
		}
		//pre-scale
		for(unsigned int i=0; i<obj.nInput(); ++i){
			std::memcpy(arr+pos,&(obj.preScale(i)),sizeof(double)); pos+=sizeof(double);
		}
		//pre-bias
		for(unsigned int i=0; i<obj.nInput(); ++i){
			std::memcpy(arr+pos,&(obj.preBias(i)),sizeof(double)); pos+=sizeof(double);
		}
		//post-scale
		for(unsigned int i=0; i<obj.nOutput(); ++i){
			std::memcpy(arr+pos,&(obj.postScale(i)),sizeof(double)); pos+=sizeof(double);
		}
		//post-bias
		for(unsigned int i=0; i<obj.nOutput(); ++i){
			std::memcpy(arr+pos,&(obj.postBias(i)),sizeof(double)); pos+=sizeof(double);
		}
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> void unpack(NN::Network& obj, const char* arr){
		//local variables
		unsigned int pos=0;
		unsigned int nlayer=0,nInput=0;
		std::vector<unsigned int> nNodes;
		//nlayer
		std::memcpy(&nlayer,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		nNodes.resize(nlayer,0);
		//nInput
		std::memcpy(&nInput,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		//number of nodes in each layer
		for(unsigned int i=0; i<nlayer; ++i){
			std::memcpy(&nNodes[i],arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//transfer function type
		std::memcpy(&(obj.tfType()),arr+pos,sizeof(NN::TransferN::type)); pos+=sizeof(NN::TransferN::type);
		//lambda
		std::memcpy(&(obj.lambda()),arr+pos,sizeof(double)); pos+=sizeof(double);
		//resize the network
		obj.resize(nInput,nNodes);
		//bias
		for(unsigned int l=0; l<obj.nlayer(); ++l){
			for(unsigned int n=0; n<obj.bias(l).size(); ++n){
				std::memcpy(&(obj.bias(l,n)),arr+pos,sizeof(double)); pos+=sizeof(double);
			}
		}
		//edge
		for(unsigned int l=0; l<obj.nlayer(); ++l){
			for(unsigned int n=0; n<obj.edge(l).cols(); ++n){
				for(unsigned int m=0; m<obj.edge(l).rows(); ++m){
					std::memcpy(&(obj.edge(l,m,n)),arr+pos,sizeof(double)); pos+=sizeof(double);
				}
			}
		}
		//pre-scale
		for(unsigned int i=0; i<obj.nInput(); ++i){
			std::memcpy(&(obj.preScale(i)),arr+pos,sizeof(double)); pos+=sizeof(double);
		}
		//pre-bias
		for(unsigned int i=0; i<obj.nInput(); ++i){
			std::memcpy(&(obj.preBias(i)),arr+pos,sizeof(double)); pos+=sizeof(double);
		}
		//post-scale
		for(unsigned int i=0; i<obj.nOutput(); ++i){
			std::memcpy(&(obj.postScale(i)),arr+pos,sizeof(double)); pos+=sizeof(double);
		}
		//post-bias
		for(unsigned int i=0; i<obj.nOutput(); ++i){
			std::memcpy(&(obj.postBias(i)),arr+pos,sizeof(double)); pos+=sizeof(double);
		}
	};
	
}