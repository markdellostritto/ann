#include "nn.hpp"

namespace NN{

//***********************************************************************
// TRANSFER FUNCTIONS - NAMES
//***********************************************************************

std::ostream& operator<<(std::ostream& out, const TransferN::type& tf){
	if(tf==TransferN::TANH) out<<"TANH";
	else if(tf==TransferN::SIGMOID) out<<"SIGMOID";
	else if(tf==TransferN::LINEAR) out<<"LINEAR";
	return out;
}

TransferN::type TransferN::load(const char* str){
	if(std::strcmp(str,"TANH")==0) return TransferN::TANH;
	else if(std::strcmp(str,"SIGMOID")==0) return TransferN::SIGMOID;
	else if(std::strcmp(str,"LINEAR")==0) return TransferN::LINEAR;
	else throw std::invalid_argument("Invalid transfer function name.");
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

void Network::defaults(){
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
		tf_.clear();
		tfd_.clear();
	//conditioning
		//preCond_=false;
		//postCond_=false;
		preScale_.resize(0);
		postScale_.resize(0);
		preBias_.resize(0);
		postBias_.resize(0);
	//regularization
		lambda_=0;
}

void Network::clear(){
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
		tf_.clear();
		tfd_.clear();
	//conditioning
		preScale_.resize(0);
		postScale_.resize(0);
		preBias_.resize(0);
		postBias_.resize(0);
}

unsigned int Network::size()const{
	unsigned int s=0;
	for(unsigned int n=0; n<bias_.size(); ++n) s+=bias_[n].size();
	for(unsigned int n=0; n<edge_.size(); ++n) s+=edge_[n].rows()*edge_[n].cols();
	return s;
}

void Network::resize(unsigned int nInput, unsigned int nOutput){
	if(nInput==0) throw std::invalid_argument("Invalid output size.");
	if(nOutput==0) throw std::invalid_argument("Invalid output size.");
	std::vector<unsigned int> nn(1,nOutput);
	resize(nInput,nn);
}

void Network::resize(unsigned int nInput, const std::vector<unsigned int>& nNodes, unsigned int nOutput){
	if(nOutput==0) throw std::invalid_argument("Invalid output size.");
	std::vector<unsigned int> nn(nNodes.size()+1);
	for(unsigned int n=0; n<nNodes.size(); ++n) nn[n]=nNodes[n];
	nn.back()=nOutput;
	resize(nInput,nn);
}

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
		if(tfType_==TransferN::LINEAR){
			tf_.resize(nlayer_,TransferF::f_lin);
			tfd_.resize(nlayer_,TransferFD::f_lin);
		} else if(tfType_==TransferN::SIGMOID){
			tf_.resize(nlayer_,TransferF::f_sigmoid);
			tfd_.resize(nlayer_,TransferFD::f_sigmoid);
		} else if(tfType_==TransferN::TANH){
			tf_.resize(nlayer_,TransferF::f_tanh);
			tfd_.resize(nlayer_,TransferFD::f_tanh);
		} else throw std::invalid_argument("Invalid transfer function.");
		//final layer is typically linear, though this can be changed after initialization
		tf_.back()=TransferF::f_lin;
		tfd_.back()=TransferFD::f_lin;
	//gradients
		grad_.resize(nNodes.back());
		dndz_.resize(nlayer_);
		for(unsigned int n=0; n<nlayer_; ++n) dndz_[n]=Eigen::VectorXd::Zero(nNodes[n]);
		delta_.resize(nlayer_);
		for(unsigned int n=0; n<nlayer_; ++n) delta_[n]=Eigen::VectorXd::Zero(nNodes[n]);
}

double Network::error(const Eigen::VectorXd& output){
	double err=0.5*(output_-output).squaredNorm();
	if(lambda_>0) for(unsigned int l=0; l<nlayer_; ++l) err+=0.5*lambda_*edge_[l].squaredNorm();
	return err;
}

Eigen::VectorXd& Network::dcda(const Eigen::VectorXd& output, Eigen::VectorXd& grad){
	grad.noalias()=(output_-output);
	return grad;
}

//quadratic cost function - calculate value and gradient of error at a given point
double Network::error(const Eigen::VectorXd& output, Eigen::VectorXd& grad){
	if(NN_PRINT_FUNC) std::cout<<"Network::error(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	//calculate the error
	double err=0.5*(output_-output).squaredNorm();
	if(NN_PRINT_DATA>1){
		std::cout<<"output_ = "<<output_.transpose()<<"\n";
		std::cout<<"output = "<<output.transpose()<<"\n";
		std::cout<<"error = "<<err<<"\n";
	}
	if(lambda_>0) for(unsigned int l=0; l<nlayer_; ++l) err+=0.5*lambda_*edge_[l].squaredNorm();
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
	//regularization - gradient
	if(lambda_>0){
		count=0;
		for(unsigned int n=0; n<delta_.size(); ++n) count+=delta_[n].size();
		for(unsigned int l=0; l<edge_.size(); ++l){
			for(unsigned int n=0; n<edge_[l].rows(); ++n){
				for(unsigned int m=0; m<edge_[l].cols(); ++m){
					grad[count++]+=lambda_*edge_[l](n,m);//edge(l,n,m)
				}
			}
		}
	}
	//return the error
	return err;
}

//quadratic cost function - calculate gradient of error at a given point, given the derivative of the gradient w.r.t. the output (dcda)
Eigen::VectorXd& Network::grad(const Eigen::VectorXd& dcda, Eigen::VectorXd& grad){
	if(NN_PRINT_FUNC>0) std::cout<<"grad(Network&,const Eigen::VectorXd&,Eigen::VectorXd&):\n";
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
	//regularization - gradient
	if(lambda_>0){
		count=0;
		for(unsigned int n=0; n<delta_.size(); ++n) count+=delta_[n].size();
		for(unsigned int l=0; l<edge_.size(); ++l){
			for(unsigned int n=0; n<edge_[l].rows(); ++n){
				for(unsigned int m=0; m<edge_[l].cols(); ++m){
					grad[count++]+=lambda_*edge_[l](n,m);//edge(l,n,m)
				}
			}
		}
	}
	//return the gradient
	return grad;
}

//calculate the gradient of output node n on all other nodes, the gradients are stored in "delta_"
void Network::grad_out(){
	if(NN_PRINT_FUNC>0) std::cout<<"grad_out():\n";
	//compute delta for the output scaling layer
	dOut_[nlayer_+1].diagonal()=postScale_;
	//back-propogate the error
	for(int l=nlayer_; l>0; --l){
		Eigen::MatrixXd tempMat=edge_[l-1].array().colwise()*dndz_[l-1].array();
		dOut_[l].noalias()=dOut_[l+1]*tempMat;
	}
	//compute delta for the input scaling layer
	dOut_[0]=dOut_[0+1]*preScale_.asDiagonal();
}

const Eigen::VectorXd& Network::execute(){
	//scale the input
	sinput_.noalias()=preScale_.cwiseProduct(input_+preBias_);
	//first layer
	node_.front()=bias_.front();
	node_.front().noalias()+=edge_.front()*sinput_;
	for(unsigned int n=0; n<node_.front().size(); ++n) dndz_.front()[n]=tfd_.front()(node_.front()[n]);
	for(unsigned int n=0; n<node_.front().size(); ++n) node_.front()[n]=tf_.front()(node_.front()[n]);
	//for(unsigned int n=0; n<node_.front().size(); ++n) dndz_.front()[n]=(*tfd_.front())(node_.front()[n]);
	//for(unsigned int n=0; n<node_.front().size(); ++n) node_.front()[n]=(*tf_.front())(node_.front()[n]);
	//subsequent layers
	for(unsigned int l=1; l<node_.size(); ++l){
		node_[l]=bias_[l];
		node_[l].noalias()+=edge_[l]*node_[l-1];
		for(unsigned int n=0; n<node_[l].size(); ++n) dndz_[l][n]=tfd_[l](node_[l][n]);
		for(unsigned int n=0; n<node_[l].size(); ++n) node_[l][n]=tf_[l](node_[l][n]);
		//for(unsigned int n=0; n<node_[l].size(); ++n) dndz_[l][n]=(*tfd_[l])(node_[l][n]);
		//for(unsigned int n=0; n<node_[l].size(); ++n) node_[l][n]=(*tf_[l])(node_[l][n]);
	}
	//scale the output
	output_=postBias_;
	output_.noalias()+=node_.back().cwiseProduct(postScale_);
	//return the output
	return output_;
}

const Eigen::VectorXd& Network::execute(const Eigen::VectorXd& input){
	input_=input;
	return execute();
}

//static functions

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

void Network::write(FILE* writer, const Network& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::write(FILE*,const Network&):\n";
	//print the configuration
	fprintf(writer,"nn ");
	fprintf(writer,"%i ",nn.nInput());
	for(unsigned int i=0; i<nn.nlayer(); ++i) fprintf(writer,"%i ",nn.nlayer(i));
	fprintf(writer,"\n");
	//print the transfer function
	if(nn.tfType()==TransferN::TANH) fprintf(writer,"t_func TANH\n");
	else if(nn.tfType()==TransferN::SIGMOID) fprintf(writer,"t_func SIGMOID\n");
	else if(nn.tfType()==TransferN::LINEAR) fprintf(writer,"t_func LINEAR\n");
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

void Network::read(FILE* reader, Network& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"Network::read(FILE*,Network&):\n";
	//local variables
	const unsigned int MAX=5000;
	char* input=(char*)malloc(sizeof(char)*MAX);
	std::vector<unsigned int> nh;
	std::vector<std::string> strlist;
	//clear the network
	if(NN_PRINT_STATUS>0) std::cout<<"clearing the network...\n";
	nn.clear();
	//load the configuration
	if(NN_PRINT_STATUS>0) std::cout<<"loading configuration...\n";
	fgets(input,MAX,reader);
	strlist.resize(string::substrN(input,string::WS)-1); std::strtok(input,string::WS);
	for(unsigned int j=0; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
	nh.resize(strlist.size()-2);
	unsigned int nInput=std::atoi(strlist.front().c_str());
	unsigned int nOutput=std::atoi(strlist.back().c_str());
	for(unsigned int i=1; i<strlist.size()-1; ++i) nh[i-1]=std::atoi(strlist[i].c_str());
	//set the transfer function
	if(NN_PRINT_STATUS>0) std::cout<<"loading transfer function...\n";
	fgets(input,MAX,reader);
	strlist.resize(string::substrN(input,string::WS)-1); std::strtok(input,string::WS);
	for(unsigned int j=0; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
	nn.tfType()=TransferN::load(strlist[0].c_str());
	if(nn.tfType()==TransferN::UNKNOWN) throw std::invalid_argument("Invalid transfer function.");
	//resize the nueral newtork
	if(NN_PRINT_STATUS>0) std::cout<<"resizing neural network...\n";
	if(NN_PRINT_STATUS>0) {std::cout<<nInput<<" "; for(unsigned int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nOutput<<"\n";}
	nn.resize(nInput,nh,nOutput);
	if(NN_PRINT_STATUS>1) std::cout<<"nn = "<<nn<<"\n";
	//read the scaling layers
	if(NN_PRINT_STATUS>0) std::cout<<"loading scaling layers...\n";
	fgets(input,MAX,reader);
	strlist.resize(string::substrN(input,string::WS)-1); std::strtok(input,string::WS);
	for(unsigned int j=0; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
	for(unsigned int j=0; j<strlist.size(); ++j) nn.preScale(j)=std::atof(strlist[j].c_str());
	fgets(input,MAX,reader);
	strlist.resize(string::substrN(input,string::WS)-1); std::strtok(input,string::WS);
	for(unsigned int j=0; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
	for(unsigned int j=0; j<strlist.size(); ++j) nn.postScale(j)=std::atof(strlist[j].c_str());
	//read the biasing layers
	if(NN_PRINT_STATUS>0) std::cout<<"loading biasing layers...\n";
	fgets(input,MAX,reader);
	strlist.resize(string::substrN(input,string::WS)-1); std::strtok(input,string::WS);
	for(unsigned int j=0; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
	for(unsigned int j=0; j<strlist.size(); ++j) nn.preBias(j)=std::atof(strlist[j].c_str());
	fgets(input,MAX,reader);
	strlist.resize(string::substrN(input,string::WS)-1); std::strtok(input,string::WS);
	for(unsigned int j=0; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
	for(unsigned int j=0; j<strlist.size(); ++j) nn.postBias(j)=std::atof(strlist[j].c_str());
	//read in the biases
	if(NN_PRINT_STATUS>0) std::cout<<"loading biases...\n";
	for(unsigned int n=0; n<nn.nlayer(); ++n){
		fgets(input,MAX,reader);
		strlist.resize(string::substrN(input,string::WS)-1); std::strtok(input,string::WS);
		for(unsigned int j=0; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
		for(unsigned int i=0; i<nn.nlayer(n); ++i){
			nn.bias(n,i)=std::atof(strlist[i].c_str());
		}
	}
	//read in the edge weights
	if(NN_PRINT_STATUS>0) std::cout<<"loading weights...\n";
	for(unsigned int n=0; n<nn.nlayer(); ++n){
		fgets(input,MAX,reader);
		strlist.resize(string::substrN(input,string::WS)-1); std::strtok(input,string::WS);
		for(unsigned int j=0; j<strlist.size(); ++j) strlist[j]=std::string(std::strtok(NULL,string::WS));
		unsigned int count=0;
		for(unsigned int i=0; i<nn.edge(n).rows(); ++i){
			for(unsigned int j=0; j<nn.edge(n).cols(); ++j){
				nn.edge(n,i,j)=std::atof(strlist[count++].c_str());
			}
		}
	}
	//free local variables
	free(input);
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
		N+=sizeof(int);//transfer function type
		N+=sizeof(double);//lambda
		for(unsigned int l=0; l<obj.nlayer(); ++l) N+=obj.bias(l).size()*sizeof(double);
		for(unsigned int l=0; l<obj.nlayer(); ++l) N+=obj.edge(l).rows()*obj.edge(l).cols()*sizeof(double);
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
		std::memcpy(arr+pos,&(tempInt=obj.nlayer()),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		std::memcpy(arr+pos,&(tempInt=obj.nInput()),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		for(unsigned int l=0; l<obj.nlayer(); ++l){
			std::memcpy(arr+pos,&(tempInt=obj.nNodes(l)),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//transfer function type
		std::memcpy(arr+pos,&(obj.tfType()),sizeof(obj.tfType())); pos+=sizeof(obj.tfType());
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
		//load network configuration
		std::memcpy(&nlayer,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		nNodes.resize(nlayer,0);
		std::memcpy(&nInput,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		for(unsigned int i=0; i<nlayer; ++i){
			std::memcpy(&nNodes[i],arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
		//transfer function type
		std::memcpy(&(obj.tfType()),arr+pos,sizeof(obj.tfType())); pos+=sizeof(obj.tfType());
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