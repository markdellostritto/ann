#include "nn_pot_train_omp.hpp"

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const NNPotOpt& obj){
	unsigned int size=0;
	//elements
		size+=sizeof(unsigned int);//nElements
	//optimization
		size+=sizeof(OPT_METHOD::type);
		if(obj.algo_==OPT_METHOD::SGD) size+=nbytes(static_cast<const SGD&>(*obj.opt_));
		else if(obj.algo_==OPT_METHOD::SDM) size+=nbytes(static_cast<const SDM&>(*obj.opt_));
		else if(obj.algo_==OPT_METHOD::NAG) size+=nbytes(static_cast<const NAG&>(*obj.opt_));
		else if(obj.algo_==OPT_METHOD::ADAGRAD) size+=nbytes(static_cast<const ADAGRAD&>(*obj.opt_));
		else if(obj.algo_==OPT_METHOD::ADADELTA) size+=nbytes(static_cast<const ADADELTA&>(*obj.opt_));
		else if(obj.algo_==OPT_METHOD::RMSPROP) size+=nbytes(static_cast<const RMSPROP&>(*obj.opt_));
		else if(obj.algo_==OPT_METHOD::ADAM) size+=nbytes(static_cast<const ADAM&>(*obj.opt_));
		else if(obj.algo_==OPT_METHOD::BFGS) size+=nbytes(static_cast<const BFGS&>(*obj.opt_));
		else if(obj.algo_==OPT_METHOD::LM) size+=nbytes(static_cast<const LM&>(*obj.opt_));
		else if(obj.algo_==OPT_METHOD::RPROP) size+=nbytes(static_cast<const RPROP&>(*obj.opt_));
		else throw std::runtime_error("Invalid optimaztion method.");
	//nn
		size+=sizeof(bool);//pre-conditioning
		size+=nbytes(obj.nnpot_);
	//error
		size+=sizeof(double);//error_val_min_
		size+=sizeof(double);//error_train_min_
	//return the size
		return size;
}

//**********************************************
// packing
//**********************************************

template <> void pack(const NNPotOpt& obj, char* arr){
	unsigned int pos=0;
	//elements
		std::memcpy(arr+pos,&obj.nElements_,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	//optimization
		std::memcpy(arr+pos,&obj.algo_,sizeof(OPT_METHOD::type)); pos+=sizeof(OPT_METHOD::type);
		if(obj.algo_==OPT_METHOD::SGD){
			pack(static_cast<const SGD&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const SGD&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::SDM){
			pack(static_cast<const SDM&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const SDM&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::NAG){
			pack(static_cast<const NAG&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const NAG&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::ADAGRAD){
			pack(static_cast<const ADAGRAD&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const ADAGRAD&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::ADADELTA){
			pack(static_cast<const ADADELTA&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const ADADELTA&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::RMSPROP){
			pack(static_cast<const RMSPROP&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const RMSPROP&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::ADAM){
			pack(static_cast<const ADAM&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const ADAM&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::BFGS){
			pack(static_cast<const BFGS&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const BFGS&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::LM){
			pack(static_cast<const LM&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const LM&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::RPROP){
			pack(static_cast<const RPROP&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const RPROP&>(*obj.opt_));
		} else throw std::runtime_error("Invalid optimaztion method.");
	//nn
		std::memcpy(arr+pos,&obj.preCond_,sizeof(bool)); pos+=sizeof(bool);
		pack(obj.nnpot_,arr+pos); pos+=nbytes(obj.nnpot_);
	//error
		std::memcpy(arr+pos,&obj.error_val_min_,sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.error_train_min_,sizeof(double)); pos+=sizeof(double);
}

//**********************************************
// unpacking
//**********************************************

template <> void unpack(NNPotOpt& obj, const char* arr){
	unsigned int pos=0;
	//elements
		std::memcpy(&obj.nElements_,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	//optimization
		std::memcpy(&obj.algo_,arr+pos,sizeof(OPT_METHOD::type)); pos+=sizeof(OPT_METHOD::type);
		if(obj.algo_==OPT_METHOD::SGD){
			unpack(static_cast<SGD&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const SGD&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::SDM){
			unpack(static_cast<SDM&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const SDM&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::NAG){
			unpack(static_cast<NAG&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const NAG&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::ADAGRAD){
			unpack(static_cast<ADAGRAD&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const ADAGRAD&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::ADADELTA){
			unpack(static_cast<ADADELTA&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const ADADELTA&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::RMSPROP){
			unpack(static_cast<RMSPROP&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const RMSPROP&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::ADAM){
			unpack(static_cast<ADAM&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const ADAM&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::BFGS){
			unpack(static_cast<BFGS&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const BFGS&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::LM){
			unpack(static_cast<LM&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const LM&>(*obj.opt_));
		} else if(obj.algo_==OPT_METHOD::RPROP){
			unpack(static_cast<RPROP&>(*obj.opt_),arr+pos);
			pos+=nbytes(static_cast<const RPROP&>(*obj.opt_));
		} else throw std::runtime_error("Invalid optimaztion method.");
	//nn
		std::memcpy(&obj.preCond_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		unpack(obj.nnpot_,arr+pos); pos+=nbytes(obj.nnpot_);
		obj.nnpotv_.resize(obj.nThreads_,obj.nnpot_);
		obj.error_train_thread_.resize(obj.nThreads_,0);
		obj.error_val_thread_.resize(obj.nThreads_,0);
	//error
		std::memcpy(&obj.error_val_min_,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.error_train_min_,arr+pos,sizeof(double)); pos+=sizeof(double);
}
	
}

//************************************************************
// NNPotOpt - Neural Network Potential - Optimization
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPotOpt& nnPotOpt){
	out<<"**************************************************\n";
	out<<"****************** NN - POT - OPT ****************\n";
	out<<"P_BATCH      = "<<nnPotOpt.pBatch_<<"\n";
	out<<"N_BATCH      = "<<nnPotOpt.nBatch_<<"\n";
	out<<"N_PRINT      = "<<nnPotOpt.nPrint_<<"\n";
	out<<"N_WRITE      = "<<nnPotOpt.nWrite_<<"\n";
	out<<"RESTART      = "<<nnPotOpt.restart_<<"\n";
	out<<"RESTART_FILE = "<<nnPotOpt.restart_file_<<"\n";
	out<<"N_THREADS    = "<<nnPotOpt.nThreads_<<"\n";
	out<<"****************** NN - POT - OPT ****************\n";
	out<<"**************************************************";
	return out;
}

void NNPotOpt::defaults(){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::defaults():\n";
	//simulation data
		if(strucTrain_!=NULL) strucTrain_->clear(); strucTrain_=NULL;
		if(strucVal_!=NULL) strucVal_->clear(); strucVal_=NULL;
		if(strucTest_!=NULL) strucTest_->clear(); strucTest_=NULL;
	//elements
		nElements_=0;
		nAtoms_.clear();
		gElement_.clear();
		pElement_.clear();
	//batch
		batch_.clear();
		indices_.clear();
	//nn
		nParams_=0;
		nnpotv_.clear();
	//input/output
		restart_=false;
		restart_file_="nn_pot_train";
		nPrint_=0;
		nWrite_=0;
	//optimization
		algo_=OPT_METHOD::UNKNOWN;
		opt_.reset(new Opt());
	//parallel
		nThreads_=1;
		#ifdef _OPENMP
			nThreads_=omp_get_max_threads();
		#endif
	//file i/o
		file_error=std::string("nn_pot_error.dat");
	//error
		error_val_min_=0;
		error_train_min_=0;
		error_train_thread_.resize(nThreads_,0);
		error_val_thread_.resize(nThreads_,0);
}

void NNPotOpt::clear(){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::clear():\n";
	//simulation data
		if(strucTrain_!=NULL) strucTrain_->clear(); strucTrain_=NULL;
		if(strucVal_!=NULL) strucVal_->clear(); strucVal_=NULL;
		if(strucTest_!=NULL) strucTest_->clear(); strucTest_=NULL;
	//elements
		nElements_=0;
		nAtoms_.clear();
		gElement_.clear();
		pElement_.clear();
	//batch
		batch_.clear();
		indices_.clear();
	//nn
		nParams_=0;
		nnpot_.clear();
		nnpotv_.clear();
	//error
		error_val_min_=0;
		error_train_min_=0;
		error_train_thread_.resize(nThreads_,0);
		error_val_thread_.resize(nThreads_,0);
}

void NNPotOpt::write_restart(const char* file){
	if(NN_POT_TRAIN_DEBUG>1) std::cout<<"NNPotOpt::write_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* writer=NULL;
	unsigned int nBytes=0;
	bool error=false;
	try{
		//open file
		writer=fopen(file,"wb");
		if(writer==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: ")+file);
		//allocate buffer
		nBytes=serialize::nbytes(*this);
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("Could not allocate memory.");
		//write to buffer
		serialize::pack(*this,arr);
		//write to file
		fwrite(arr,nBytes,1,writer);
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(writer); writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in NNPotOpt::write_restart(const char*):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(writer!=NULL) fclose(writer);
	if(error) throw std::runtime_error("Failed to write");
}

void NNPotOpt::read_restart(const char* file){
	if(NN_POT_TRAIN_DEBUG>1) std::cout<<"NNPotOpt::read_restart(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* reader=NULL;
	unsigned int nBytes=0;
	bool error=false;
	try{
		//open file
		reader=fopen(file,"rb");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: ")+std::string(file));
		//find size
		std::fseek(reader,0,SEEK_END);
		nBytes=std::ftell(reader);
		std::fseek(reader,0,SEEK_SET);
		//allocate buffer
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("Could not allocate memory.");
		//read from file
		fread(arr,nBytes,1,reader);
		//read from buffer
		serialize::unpack(*this,arr);
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(reader); reader=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in NNPotOpt::read_restart(const char*):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(reader!=NULL) fclose(reader);
	if(error) throw std::runtime_error("Failed to read");
}

void NNPotOpt::train(unsigned int batchSize){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPotOpt::train(NNPot&,std::vector<Structure>&,unsigned int):\n";
	//====== local function variables ======
	//elements
		std::vector<double> an;//atomic number
	//bias
		double avg_out=0,stddev_out=0,max_out=0,min_out=0;
		VecList avg_in;//average of the inputs for each element (nnpot_.nSpecies_ x nInput_)
		VecList max_in;//max of the inputs for each element (nnpot_.nSpecies_ x nInput_)
		VecList min_in;//min of the inputs for each element (nnpot_.nSpecies_ x nInput_)
		VecList stddev_in;//average of the stddev for each element (nnpot_.nSpecies_ x nInput_)
	//timing
		clock_t start,stop;
		double time_train;
	//misc
		unsigned int count=0;
	
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Training NN potential...\n";
	
	//====== check the parameters ======
	if(strucTrain_==NULL) throw std::runtime_error("NULL POINTER: no training simulations.");
	else if(strucTrain_->size()==0) throw std::invalid_argument("No training data provided.");
	
	//====== set the number of atoms of each element ======
	if(nElements_==0) nElements_=nnpot_.nSpecies();
	else if(nElements_!=nnpot_.nSpecies()) throw std::invalid_argument("Invalid number of elements in the potential.");
	nAtoms_.resize(nnpot_.nSpecies());
	for(unsigned int i=0; i<strucTrain_->size(); ++i){
		for(unsigned int j=0; j<(*strucTrain_)[i].nSpecies(); ++j){
			nAtoms_[nnpot_.speciesIndex((*strucTrain_)[i].atomNames(j))]+=(*strucTrain_)[i].nAtoms(j);
		}
	}
	std::cout<<"Species - Names   = "; for(unsigned int i=0; i<nnpot_.nSpecies(); ++i) std::cout<<nnpot_.speciesName(i)<<" "; std::cout<<"\n";
	std::cout<<"Species - Numbers = "; for(unsigned int i=0; i<nAtoms_.size(); ++i) std::cout<<nAtoms_[i]<<" "; std::cout<<"\n";
	
	//====== set the indices and batch size ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting indices and batch...\n";
	indices_.resize(strucTrain_->size());
	for(unsigned int i=0; i<indices_.size(); ++i) indices_[i]=i;
	batch_.resize(batchSize,0);
	for(unsigned int i=0; i<batch_.size(); ++i) batch_[i]=i;
	
	//====== precondition the input ======
	if(preCond_){
		if(NN_POT_PRINT_STATUS>-1) std::cout<<"Pre-conditioning input...\n";
		//local variables
		avg_in.resize(nnpot_.nSpecies());
		max_in.resize(nnpot_.nSpecies());
		min_in.resize(nnpot_.nSpecies());
		stddev_in.resize(nnpot_.nSpecies());
		for(unsigned int i=0; i<nnpot_.nSpecies(); ++i){
			avg_in[i]=Eigen::VectorXd::Zero(nnpot_.nInput_[i]);
			max_in[i]=Eigen::VectorXd::Zero(nnpot_.nInput_[i]);
			min_in[i]=Eigen::VectorXd::Zero(nnpot_.nInput_[i]);
			stddev_in[i]=Eigen::VectorXd::Zero(nnpot_.nInput_[i]);
		}
		std::vector<double> N(nnpot_.nSpecies());//total number of inputs for each element 
		//compute the max/min
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Compute the max/min...\n";
		for(unsigned int i=0; i<(*strucTrain_)[0].nSpecies(); ++i){
			//find the index of the current species
			unsigned int index=nnpot_.speciesMap_[(*strucTrain_)[0].atomNames(i)];
			//loop over all bases
			for(unsigned int k=0; k<nnpot_.nInput_[i]; ++k){
				//find the max and min
				max_in[index][k]=(*strucTrain_)[0].symm(i,0)[k];
				min_in[index][k]=(*strucTrain_)[0].symm(i,0)[k];
			}
		}
		for(unsigned int n=0; n<strucTrain_->size(); ++n){
			//loop over all species
			for(unsigned int i=0; i<(*strucTrain_)[n].nSpecies(); ++i){
				//find the index of the current species
				unsigned int index=nnpot_.speciesMap_[(*strucTrain_)[n].atomNames(i)];
				//loop over all atoms of the species
				for(unsigned int j=0; j<(*strucTrain_)[n].nAtoms(i); ++j){
					//loop over all bases
					for(unsigned int k=0; k<nnpot_.nInput_[i]; ++k){
						//find the max and min
						if((*strucTrain_)[n].symm(i,j)[k]>max_in[index][k]) max_in[index][k]=(*strucTrain_)[n].symm(i,j)[k];
						if((*strucTrain_)[n].symm(i,j)[k]<min_in[index][k]) min_in[index][k]=(*strucTrain_)[n].symm(i,j)[k];
					}
				}
			}
		}
		//compute the average - loop over all simulations
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Compute the average...\n";
		for(unsigned int n=0; n<strucTrain_->size(); ++n){
			//loop over all species
			for(unsigned int i=0; i<(*strucTrain_)[n].nSpecies(); ++i){
				//find the index of the current species
				unsigned int index=nnpot_.speciesMap_[(*strucTrain_)[n].atomNames(i)];
				//loop over all atoms of the species
				for(unsigned int j=0; j<(*strucTrain_)[n].nAtoms(i); ++j){
					//add the inputs to the average
					avg_in[index].noalias()+=(*strucTrain_)[n].symm(i,j); ++N[index];
				}
			}
		}
		//normalize average
		for(unsigned int i=0; i<avg_in.size(); ++i) avg_in[i]/=N[i];
		//compute the stddev - loop over all simulations
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Compute the stddev...\n";
		for(unsigned int n=0; n<strucTrain_->size(); ++n){
			//loop over all species
			for(unsigned int i=0; i<(*strucTrain_)[n].nSpecies(); ++i){
				//find the index of the current species
				unsigned int index=nnpot_.speciesMap_[(*strucTrain_)[n].atomNames(i)];
				//loop over all atoms of a species
				for(unsigned int j=0; j<(*strucTrain_)[n].nAtoms(i); ++j){
					stddev_in[index].noalias()+=(avg_in[index]-(*strucTrain_)[n].symm(i,j)).cwiseProduct(avg_in[index]-(*strucTrain_)[n].symm(i,j));
				}
			}
		}
		//normalize the stddev
		for(unsigned int i=0; i<stddev_in.size(); ++i){
			for(unsigned int j=0; j<stddev_in[i].size(); ++j){
				stddev_in[i][j]=std::sqrt(stddev_in[i][j]/(N[i]-1.0));
			}
		}
		//set the preconditioning vectors
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Set the preconditiong vectors...\n";
		preBias_=avg_in; for(unsigned int i=0; i<preBias_.size(); ++i) preBias_[i]*=-1;
		preScale_=stddev_in;
		for(unsigned int i=0; i<preScale_.size(); ++i){
			for(unsigned int j=0; j<preScale_[i].size(); ++j){
				if(preScale_[i][j]==0) preScale_[i][j]=1;
				else preScale_[i][j]=1.0/(3.0*preScale_[i][j]+1e-6);
			}
		}
	} else {
		preBias_.resize(nnpot_.nSpecies());
		preScale_.resize(nnpot_.nSpecies());
		for(unsigned int i=0; i<nnpot_.nSpecies(); ++i){
			preBias_[i]=Eigen::VectorXd::Constant(nnpot_.nInput_[i],0.0);
			preScale_[i]=Eigen::VectorXd::Constant(nnpot_.nInput_[i],1.0);
		}
	}
	
	//====== set the bias for each of the species ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting the bias for each species...\n";
	for(unsigned int n=0; n<nnpot_.nSpecies(); ++n){
		for(unsigned int i=0; i<nnpot_.nn_[n].nInput(); ++i) nnpot_.nn_[n].preBias(i)=preBias_[n][i];
		for(unsigned int i=0; i<nnpot_.nn_[n].nInput(); ++i) nnpot_.nn_[n].preScale(i)=preScale_[n][i];
		nnpot_.nn_[n].postBias(0)=0.0;
		nnpot_.nn_[n].postScale(0)=1.0;
	}
	
	//====== initialize the optimization data ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Initializing the optimization data...\n";
	//resize pElement_ and gElement_
	pElement_.resize(nnpot_.nSpecies());
	gElement_.resize(nnpot_.nSpecies());
	for(unsigned int n=0; n<nnpot_.nSpecies(); ++n){
		nnpot_.nn_[n]>>pElement_[n];
		nnpot_.nn_[n]>>gElement_[n];//this just resizes the gradients
		gElement_[n].setZero();
		nParams_+=pElement_[n].size();
	}
	//set initial parameters
	if(opt_->dim()==0){
		std::cout<<"Starting from scratch...\n";
		opt_->resize(nParams_);
	} else {
		std::cout<<"Restarting optimization...\n";
		if(nParams_!=opt_->dim()) throw std::runtime_error(
			std::string("Network has ")+std::to_string(nParams_)+std::string(" while opt has ")
			+std::to_string(opt_->dim())+std::string(" parameters.")
		);
		count=0;
		for(unsigned int n=0; n<pElement_.size(); ++n){
			for(unsigned int m=0; m<pElement_[n].size(); ++m){
				pElement_[n][m]=opt_->x()[count];
				gElement_[n][m]=opt_->grad()[count];
				++count;
			}
		}
	}
	//set local storage vectors (so starting values aren't overwritten)
	double val=opt_->val();
	Eigen::VectorXd p=Eigen::VectorXd::Zero(nParams_);
	Eigen::VectorXd g=Eigen::VectorXd::Zero(nParams_);
	count=0;
	for(unsigned int n=0; n<pElement_.size(); ++n){
		for(unsigned int m=0; m<pElement_[n].size(); ++m){
			p[count]=pElement_[n][m];
			g[count]=gElement_[n][m];
			++count;
		}
	}
	gElementT_.resize(nThreads_);
	for(unsigned int nt=0; nt<nThreads_; ++nt){
		gElementT_[nt].resize(nnpot_.nSpecies());
		for(unsigned int n=0; n<nnpot_.nSpecies(); ++n){
			gElementT_[nt][n]=gElement_[n];
		}
	}
	gTemp_.resize(nThreads_);
	for(unsigned int nt=0; nt<nThreads_; ++nt){
		gTemp_[nt].resize(nnpot_.nSpecies());
		for(unsigned int n=0; n<nnpot_.nSpecies(); ++n){
			gTemp_[nt][n]=gElement_[n];
		}
	}
	
	//====== print the potential ======
	std::cout<<nnpot_<<"\n";
	
	//====== distribute the potential ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Distributing the potential...\n";
	nnpotv_.resize(nThreads_);
	for(unsigned int i=0; i<nThreads_; ++i) nnpotv_[i]=nnpot_;
	
	//====== set the objective function ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting the objective function...\n";
	std::function<double(NNPotOpt&, const Eigen::VectorXd&, Eigen::VectorXd&)> func = &NNPotOpt::error;
	
	//====== print optimization data ======
	if(NN_POT_PRINT_DATA>-1){
		std::cout<<"**************************************************\n";
		std::cout<<"******************* OPT - DATA *******************\n";
		std::cout<<"AVG - INPUT     = \n";
		for(unsigned int i=0; i<avg_in.size(); ++i) std::cout<<"\t"<<avg_in[i].transpose()<<"\n";
		std::cout<<"MAX - INPUT     = \n";
		for(unsigned int i=0; i<max_in.size(); ++i) std::cout<<"\t"<<max_in[i].transpose()<<"\n";
		std::cout<<"MIN - INPUT     = \n";
		for(unsigned int i=0; i<min_in.size(); ++i) std::cout<<"\t"<<min_in[i].transpose()<<"\n";
		std::cout<<"STDDEV - INPUT  = \n";
		for(unsigned int i=0; i<stddev_in.size(); ++i) std::cout<<"\t"<<stddev_in[i].transpose()<<"\n";
		std::cout<<"PRE-BIAS        = \n";
		for(unsigned int i=0; i<preBias_.size(); ++i) std::cout<<"\t"<<preBias_[i].transpose()<<"\n";
		std::cout<<"PRE-SCALE       = \n";
		for(unsigned int i=0; i<preScale_.size(); ++i) std::cout<<"\t"<<preScale_[i].transpose()<<"\n";
		std::cout<<"scaling-output:\n";
		std::cout<<"MAX - OUTPUT    = "<<max_out<<"\n";
		std::cout<<"MIN - OUTPUT    = "<<min_out<<"\n";
		std::cout<<"AVG - OUTPUT    = "<<avg_out<<"\n";
		std::cout<<"STDDEV - OUTPUT = "<<stddev_out<<"\n";
		std::cout<<"******************* OPT - DATA *******************\n";
		std::cout<<"**************************************************\n";
	}
	
	//====== execute the optimization ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Executing the optimization...\n";
	//open the error file
	if(!restart_) writer_error=fopen(file_error.c_str(),"w");
	else writer_error=fopen(file_error.c_str(),"a");
	if(writer_error==NULL) throw std::runtime_error("I/O Error: Could not open error record file.");
	//initialize the max and min error
	error_train_min_=FLT_MAX;
	error_val_min_=FLT_MAX;
	//train the nn potential
	start=std::clock();
	opt_->opts<NNPotOpt>(func,*this,p,g,val);
	stop=std::clock();
	time_train=((double)(stop-start))/CLOCKS_PER_SEC;
	//close the error file
	fclose(writer_error);
	writer_error=NULL;
	
	//====== unpack final parameters into element arrays ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Unpacking final parameters into element arrays...\n";
	count=0;
	for(unsigned int n=0; n<pElement_.size(); ++n){
		for(unsigned int m=0; m<pElement_[n].size(); ++m){
			pElement_[n][m]=p[count];
			gElement_[n][m]=g[count];
			++count;
		}
	}
	
	//====== pack final parameters into neural network ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Packing final parameters into neural network...\n";
	for(unsigned int n=0; n<nnpot_.nSpecies(); ++n) nnpot_.nn_[n]<<pElement_[n];
	
	if(NN_POT_PRINT_DATA>-1){
		std::cout<<"**************************************************\n";
		std::cout<<"******************* OPT - SUMM *******************\n";
		std::cout<<"N-STEPS       = "<<opt_->nStep()<<"\n";
		std::cout<<"OPT-VAL       = "<<opt_->val()<<"\n";
		std::cout<<"ERR-TRAIN-MIN = "<<error_train_min_<<"\n";
		std::cout<<"ERR-VAL-MIN   = "<<error_val_min_<<"\n";
		std::cout<<"TIME-TRAIN    = "<<time_train<<"\n";
		if(NN_POT_PRINT_DATA>0){std::cout<<"p = "; for(int i=0; i<p.size(); ++i) std::cout<<p[i]<<" "; std::cout<<"\n";}
		std::cout<<"******************* OPT - SUMM *******************\n";
		std::cout<<"**************************************************\n";
	}
}

double NNPotOpt::error(const Eigen::VectorXd& x, Eigen::VectorXd& gradTot){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPotOpt::error(const Eigen::VectorXd&,Eigen::VectorXd&):\n";
	//====== local variables ======
	//utility
		unsigned int count=0;
	//error
		double error_train=0;
		double error_val=0;
	//gradients
		VecList gElementLoc_=gElement_;
	
	//====== reset the error ======
	for(unsigned int nt=0; nt<error_train_thread_.size(); ++nt) error_train_thread_[nt]=0;
	for(unsigned int nt=0; nt<error_val_thread_.size(); ++nt) error_val_thread_[nt]=0;
	
	//====== unpack total parameters into element arrays ======
	if(NN_POT_PRINT_STATUS>1) std::cout<<"Unpacking total parameters into parameter arrays...\n";
	count=0;
	for(unsigned int n=0; n<pElement_.size(); ++n){
		for(unsigned int m=0; m<pElement_[n].size(); ++m){
			pElement_[n][m]=x[count++];
		}
	}
	
	//====== unpack arrays into element nn's ======
	if(NN_POT_PRINT_STATUS>0) std::cout<<"Unpacking arrays into element nn's...\n";
	for(unsigned int nt=0; nt<nThreads_; ++nt){
		for(unsigned int n=0; n<nnpotv_[nt].nSpecies(); ++n){
			nnpotv_[nt].nn_[n]<<pElement_[n];
		}
	}
	
	//====== reset the gradients ======
	if(NN_POT_PRINT_STATUS>1) std::cout<<"Resetting gradients...\n";
	for(unsigned int nt=0; nt<gElementT_.size(); ++nt){
		for(unsigned int i=0; i<gElementT_[nt].size(); ++i) gElementT_[nt][i].setZero();
	}
	for(unsigned int nt=0; nt<gTemp_.size(); ++nt){
		for(unsigned int i=0; i<gTemp_[nt].size(); ++i) gTemp_[nt][i].setZero();
	}
	
	//====== randomize the batch ======
	if(batch_.size()<strucTrain_->size()){
		if(NN_POT_PRINT_STATUS>1) std::cout<<"Randomizing the batch...\n";
		std::random_shuffle(indices_.begin(),indices_.end());
		for(unsigned int i=0; i<batch_.size(); ++i) batch_[i]=indices_[i];
		std::sort(batch_.begin(),batch_.end());
		if(NN_POT_PRINT_STATUS>1){for(unsigned int i=0; i<batch_.size(); ++i) std::cout<<"batch["<<i<<"] = "<<batch_[i]<<"\n";}
	}
	
	//====== compute training error and gradient ======
	if(NN_POT_PRINT_STATUS>1) std::cout<<"Computing training error and gradient...\n";
	/*#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads()) firstprivate(dcda,gElementLoc_)
	for(unsigned int i=0; i<batch_.size(); ++i){
		unsigned int TN=0;
		#ifdef _OPENMP
			TN=omp_get_thread_num();
		#endif
		//set the local simulation reference
		const Structure& siml=(*strucTrain_)[batch_[i]];
		//compute the energy
		double energy=0;
		for(unsigned int n=0; n<siml.nSpecies(); ++n){
			//find the element index in the nn
			const unsigned int index=nnpotv_[TN].speciesIndex(siml.atomNames(n));
			//loop over all atoms of the species
			for(unsigned int m=0; m<siml.nAtoms(n); ++m){
				//execute the network
				nnpotv_[TN].nn_[index].execute(siml.symm(n,m));
				//add the energy to the total
				energy+=nnpotv_[TN].nn_[index].output()[0]+nnpotv_[TN].energyAtom(index);
			}
		}
		//add to the error
		error_train_thread_[TN]+=0.5*(energy-siml.energy())*(energy-siml.energy());
		//compute the gradient
		dcda[0]=(energy-siml.energy());
		//compute the gradients
		for(unsigned int n=0; n<siml.nSpecies(); ++n){
			//find the element index in the nn
			const unsigned int index=nnpotv_[TN].speciesIndex(siml.atomNames(n));
			for(unsigned int m=0; m<siml.nAtoms(n); ++m){
				//execute the network
				nnpotv_[TN].nn_[index].execute(siml.symm(n,m));
				//compute the gradient
				nnpotv_[TN].nn_[index].grad(dcda,gElementLoc_[n]);
				//add the gradient to the total
				gElementT_[TN][index].noalias()+=gElementLoc_[n];
			}
		}
	}*/
	Eigen::VectorXd identity(1); identity[0]=1;
	#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads()) firstprivate(gElementLoc_)
	for(unsigned int i=0; i<batch_.size(); ++i){
		unsigned int TN=0;
		#ifdef _OPENMP
			TN=omp_get_thread_num();
		#endif
		for(unsigned int j=0; j<gTemp_[TN].size(); ++j) gTemp_[TN][j].setZero();
		//set the local simulation reference
		const Structure& siml=(*strucTrain_)[batch_[i]];
		//compute the energy
		double energy=0;
		for(unsigned int n=0; n<siml.nSpecies(); ++n){
			//find the element index in the nn
			const unsigned int index=nnpotv_[TN].speciesIndex(siml.atomNames(n));
			//loop over all atoms of the species
			for(unsigned int m=0; m<siml.nAtoms(n); ++m){
				//execute the network
				nnpotv_[TN].nn_[index].execute(siml.symm(n,m));
				//add the energy to the total
				energy+=nnpotv_[TN].nn_[index].output()[0]+nnpotv_[TN].energyAtom(index);
				//compute the gradient
				nnpotv_[TN].nn_[index].grad(identity,gElementLoc_[n]);
				//add the gradient to the total
				gTemp_[TN][index].noalias()+=gElementLoc_[n];
			}
		}
		//add to the error
		error_train_thread_[TN]+=0.5*(energy-siml.energy())*(energy-siml.energy());
		//compute the gradient
		const double dcda=(energy-siml.energy());
		//multiply the element gradients by dcda
		for(unsigned int j=0; j<gElementT_[TN].size(); ++j) gElementT_[TN][j].noalias()+=gTemp_[TN][j]*dcda;
	}
	//consolidate the error and gradient
	for(unsigned int i=0; i<gElement_.size(); ++i){
		gElement_[i].setZero();
		for(unsigned int nt=0; nt<gElementT_.size(); ++nt){
			gElement_[i].noalias()+=gElementT_[nt][i];
		}
	}
	error_train=0;
	for(unsigned int nt=0; nt<error_train_thread_.size(); ++nt) error_train+=error_train_thread_[nt];
	
	//====== compute validation error and gradient ======
	if(strucVal_!=NULL){
		if(NN_POT_PRINT_STATUS>1) std::cout<<"Computing validation error and gradient...\n";
		#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
		for(unsigned int i=0; i<strucVal_->size(); ++i){
			unsigned int TN=0;
			#ifdef _OPENMP
				TN=omp_get_thread_num();
			#endif
			//set the local simulation reference
			const Structure& siml=(*strucVal_)[i];
			//compute the energy
			double energy=0;
			for(unsigned int n=0; n<siml.nSpecies(); ++n){
				//find the element index in the nn
				const unsigned int index=nnpotv_[TN].speciesIndex(siml.atomNames(n));
				//loop over all atoms of the species
				for(unsigned int m=0; m<siml.nAtoms(n); ++m){
					//execute the network
					nnpotv_[TN].nn_[index].execute(siml.symm(n,m));
					//add the energy to the total
					energy+=nnpotv_[TN].nn_[index].output()[0]+nnpotv_[TN].energyAtom(index);
				}
			}
			//scale the energy
			energy=energy;
			//add to the error
			error_val_thread_[TN]+=0.5*(energy-siml.energy())*(energy-siml.energy());
		}
		//consolidate the error
		error_val=0;
		for(unsigned int nt=0; nt<nThreads_; ++nt) error_val+=error_val_thread_[nt];
		//normalize the error
		error_val/=(strucVal_->size()>0)?strucVal_->size():1;
	}
	
	//====== print energy ======
	if(NN_POT_PRINT_DATA>0 && opt_->nStep()%nPrint_==0){
		std::vector<double> energy_train(strucTrain_->size());
		std::vector<double> energy_exact(strucTrain_->size());
		std::cout<<"======================================\n";
		std::cout<<"============== ENERGIES ==============\n";
		#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
		for(unsigned int i=0; i<strucTrain_->size(); ++i){
			unsigned int TN=0;
			#ifdef _OPENMP
				TN=omp_get_thread_num();
			#endif
			//set the local simulation reference
			const Structure& siml=(*strucTrain_)[batch_[i]];
			//compute the energy
			double energy=0;
			for(unsigned int n=0; n<siml.nSpecies(); ++n){
				//find the element index in the nn
				const unsigned int index=nnpotv_[TN].speciesIndex(siml.atomNames(n));
				//loop over all atoms of the species
				for(unsigned int m=0; m<siml.nAtoms(n); ++m){
					//execute the network
					nnpotv_[TN].nn_[index].execute(siml.symm(n,m));
					//add the energy to the total
					energy+=nnpotv_[TN].nn_[index].output()[0]+nnpotv_[TN].energyAtom(index);
				}
			}
			//scale the energy
			energy_train[i]=energy;
			energy_exact[i]=siml.energy();
		}
		for(unsigned int i=0; i<strucTrain_->size(); ++i){
			std::cout<<"sim"<<i<<" "<<energy_exact[i]<<" "<<energy_train[i]<<" "<<0.5*(energy_train[i]-energy_exact[i])*(energy_train[i]-energy_exact[i])<<"\n";
		}
		std::cout<<"============== ENERGIES ==============\n";
		std::cout<<"======================================\n";
	}
	
	//====== normalize the error and gradient ======
	if(NN_POT_PRINT_STATUS>1) std::cout<<"Normalizing the error and gradient...\n";
	error_train/=batch_.size();
	for(unsigned int i=0; i<gElement_.size(); ++i) gElement_[i]/=nAtoms_[i];
	
	//====== pack element gradients into total gradient ======
	if(NN_POT_PRINT_STATUS>1) std::cout<<"Packing the element gradients into the total gradient...\n";
	count=0;
	for(unsigned int n=0; n<gElement_.size(); ++n){
		for(unsigned int m=0; m<gElement_[n].size(); ++m){
			gradTot[count++]=gElement_[n][m];
		}
	}
	
	//====== store the error ======
	if(NN_POT_PRINT_STATUS>1) std::cout<<"Storing the error...\n";
	if(opt_->nStep()%nPrint_==0) fprintf(writer_error,"%6i %12.10f %12.10f \n",opt_->nStep(),error_train,error_val);
	
	//====== caclulate the validation stopping criterion ======
	error_val_min_=(error_val_min_>error_val)?error_val:error_val_min_;
	error_train_min_=(error_train_min_>error_train)?error_train:error_train_min_;
	
	//====== print the optimization data ======
	if(opt_->nStep()%nPrint_==0) printf("opt %8i err_t %12.10f err_v %12.10f\n",opt_->nStep(),error_train,error_val);
	
	//====== write the basis and potentials ======
	if(opt_->nStep()%nWrite_==0){
		nnpotv_[0].write(opt_->nStep());
		std::string file=restart_file_+".restart."+std::to_string(opt_->nStep());
		this->write_restart(file.c_str());
	}
	
	//====== return the error ======
	if(NN_POT_PRINT_STATUS>1) std::cout<<"Returning the error...\n";
	return error_train;
}

//************************************************************
// MAIN
//************************************************************

int main(int argc, char* argv[]){
	//======== global variables ========
	//mode
		struct MODE{
			enum type{
				TRAIN,
				TEST,
				UNKNOWN
			};
		};
		MODE::type mode=MODE::TRAIN;
	//nn potential
		NNPot::Init nnPotInit;
	//nn potential - opt
		NNPotOpt nnPotOpt;//nn potential optimization data
		std::vector<double> energy_;
		std::vector<std::string> name_;
	//simulations
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.specie=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.symm=true;
		FILE_FORMAT::type format;
		unsigned int nSpecies=0;//number of atomic species
		std::vector<std::string> speciesNames;//names of atomic species
		std::vector<std::string> data_train;//file for training data
		std::vector<std::string> data_val;//file for validation data
		std::vector<std::string> data_test;//file for test data
		std::vector<std::string> files_train;//training data files
		std::vector<std::string> files_val;//validation data files
		std::vector<std::string> files_test;//test data files
		std::vector<Structure> struc_train;//training data
		std::vector<Structure> struc_val;//validation data
		std::vector<Structure> struc_test;//test data
	//linear regression
		double m=0,b=0,r2;
	//timing
		clock_t start,stop;//starting/stopping time
		double time_energy_train=0;//time to compute energies - training
		double time_energy_val=0;  //time to compute energies - validation
		double time_energy_test=0; //time to compute energies - test
		double time_force_train=0; //time to compute forces - training
		double time_force_val=0;   //time to compute forces - validation
		double time_force_test=0;  //time to compute forces - test
		double time_symm_train=0;  //time to compute symmetry functions - training
		double time_symm_val=0;    //time to compute symmetry functions - validation
		double time_symm_test=0;   //time to compute symmetry functions - test
	//random
		int seed=-1;//seed for random number generator (negative => current time)
	//file i/o
		FILE* reader=NULL;
		FILE* writer=NULL;
		std::vector<std::string> strlist;
		std::vector<std::string> fileNNPot;
		std::vector<std::string> fileName;
		char* paramfile=(char*)malloc(sizeof(char)*string::M);
		char* datafile=(char*)malloc(sizeof(char)*string::M);
		char* input=(char*)malloc(sizeof(char)*string::M);
		char* temp=(char*)malloc(sizeof(char)*string::M);
	//writing
		bool write_corr=true;
		bool write_basis=true;
		bool write_energy=false;
		bool write_force=false;
	//units
		units::System::type unitsys=units::System::UNKNOWN;
		
	try{
		//************************************************************************************
		// LOADING/INITIALIZATION
		//************************************************************************************
		
		//======== check the arguments ========
		if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
		
		//======== load the parameter file ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading parameter file...\n";
		std::strcpy(paramfile,argv[1]);
		
		//======== open the parameter file ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Opening parameter file...\n";
		reader=fopen(paramfile,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+paramfile);
		
		//======== read in the parameters ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading in parameters...\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);
			if(string::split(input,string::WS,strlist)==0) continue;
			string::to_upper(strlist.at(0));
			//data and execution mode
			if(strlist.at(0)=="MODE"){
				string::to_upper(strlist.at(1));
				if(strlist.at(1)=="TRAIN") mode=MODE::TRAIN;
				else if(strlist.at(1)=="TEST") mode=MODE::TEST;	
				else throw std::invalid_argument("Invalid mode.");
			} else if(strlist.at(0)=="DATA_TRAIN"){
				data_train.push_back(strlist.at(1));
			} else if(strlist.at(0)=="DATA_VAL"){
				data_val.push_back(strlist.at(1));
			} else if(strlist.at(0)=="DATA_TEST"){
				data_test.push_back(strlist.at(1));
			}
			//neural network potential - initialization
			if(strlist.at(0)=="NR"){//number of radial basis functions
				nnPotInit.nR=std::atoi(strlist.at(1).c_str());
			} else if(strlist.at(0)=="NA"){//number of angular basis functions
				nnPotInit.nA=std::atoi(strlist.at(1).c_str());
			} else if(strlist.at(0)=="PHIRN"){//type of radial basis functions
				nnPotInit.phiRN=PhiRN::load(string::to_upper(strlist.at(1)).c_str());
			} else if(strlist.at(0)=="PHIAN"){//type of angular basis functions
				nnPotInit.phiAN=PhiAN::load(string::to_upper(strlist.at(1)).c_str());
			} else if(strlist.at(0)=="R_CUT"){//distance cutoff
				nnPotInit.rc=std::atof(strlist.at(1).c_str());
			} else if(strlist.at(0)=="R_MIN"){//distance cutoff
				nnPotInit.rm=std::atof(strlist.at(1).c_str());
			} else if(strlist.at(0)=="CUTOFF"){//type of cutoff function
				nnPotInit.tcut=CutoffN::load(string::to_upper(strlist.at(1)).c_str());
			} else if(strlist.at(0)=="LAMBDA"){//regularization parameter
				nnPotInit.lambda=std::atof(strlist.at(1).c_str());
			} else if(strlist.at(0)=="N_HIDDEN"){//number of hidden layers
				int nl=strlist.size()-1;
				if(nl<=0) throw std::invalid_argument("Invalid hidden layer configuration.");
				std::vector<unsigned int> nh(nl);
				for(unsigned int i=0; i<nl; ++i){
					nh.at(i)=std::atoi(strlist.at(i+1).c_str());
					if(nh.at(i)==0) throw std::invalid_argument("Invalid hidden layer configuration.");
				}
				nnPotInit.nh=nh;
			} else if(strlist.at(0)=="TRANSFER"){//transfer function
				nnPotInit.tfType=NN::TransferN::load(string::to_upper(strlist.at(1)).c_str());
			} 
			//neural network potential optimization
			if(strlist.at(0)=="PRE_COND"){//whether to precondition the inputs
				nnPotOpt.preCond_=string::boolean(strlist.at(1).c_str());
			} else if(strlist.at(0)=="N_BATCH"){//size of the batch
				nnPotOpt.nBatch_=std::atoi(strlist.at(1).c_str());
			} else if(strlist.at(0)=="P_BATCH"){//batch percentage
				nnPotOpt.pBatch_=std::atof(strlist.at(1).c_str());
			} else if(strlist.at(0)=="N_PRINT"){
				nnPotOpt.nPrint_=std::atoi(strlist.at(1).c_str());
			} else if(strlist.at(0)=="N_WRITE"){
				nnPotOpt.nWrite_=std::atoi(strlist.at(1).c_str());
			} else if(strlist.at(0)=="RESTART"){
				nnPotOpt.restart_file_=strlist.at(1);
				nnPotOpt.restart_=true;
			} else if(strlist.at(0)=="ALGO"){
				nnPotOpt.algo_=OPT_METHOD::load(string::to_upper(strlist.at(1)).c_str());
			} 
			//general
			if(strlist.at(0)=="SEED"){
				seed=std::atoi(strlist.at(1).c_str());
			} else if(strlist.at(0)=="ENERGY"){
				name_.push_back(strlist.at(1));
				energy_.push_back(std::atof(strlist.at(2).c_str()));
			} else if(strlist.at(0)=="READ"){
				fileName.push_back(strlist.at(1));
				fileNNPot.push_back(strlist.at(2));
			} else if(strlist.at(0)=="FORMAT"){
				format=FILE_FORMAT::read(string::to_upper(strlist.at(1)).c_str());
			} else if(strlist.at(0)=="UNITS"){
				unitsys=units::System::read(string::to_upper(strlist.at(1)).c_str());
			}
			//writing
			if(strlist.at(0)=="WRITE_CORR"){
				write_corr=string::boolean(strlist.at(1).c_str());
			} else if(strlist.at(0)=="WRITE_BASIS"){
				write_basis=string::boolean(strlist.at(1).c_str());
			} else if(strlist.at(0)=="WRITE_ENERGY"){
				write_energy=string::boolean(strlist.at(1).c_str());
			} else if(strlist.at(0)=="WRITE_FORCE"){
				write_force=string::boolean(strlist.at(1).c_str());
			}
		}
		
		//======== load optimization object ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Loading optimization object...\n";
		if(nnPotOpt.algo_==OPT_METHOD::SGD){
			nnPotOpt.opt_.reset(new SGD());
			read(static_cast<SGD&>(*nnPotOpt.opt_),reader);
		} else if(nnPotOpt.algo_==OPT_METHOD::SDM){
			nnPotOpt.opt_.reset(new SDM());
			read(static_cast<SDM&>(*nnPotOpt.opt_),reader);
		} else if(nnPotOpt.algo_==OPT_METHOD::NAG){
			nnPotOpt.opt_.reset(new NAG());
			read(static_cast<NAG&>(*nnPotOpt.opt_),reader);
		} else if(nnPotOpt.algo_==OPT_METHOD::ADAGRAD){
			nnPotOpt.opt_.reset(new ADAGRAD());
			read(static_cast<ADAGRAD&>(*nnPotOpt.opt_),reader);
		} else if(nnPotOpt.algo_==OPT_METHOD::ADADELTA){
			nnPotOpt.opt_.reset(new ADADELTA());
			read(static_cast<ADADELTA&>(*nnPotOpt.opt_),reader);
		} else if(nnPotOpt.algo_==OPT_METHOD::RMSPROP){
			nnPotOpt.opt_.reset(new RMSPROP());
			read(static_cast<RMSPROP&>(*nnPotOpt.opt_),reader);
		} else if(nnPotOpt.algo_==OPT_METHOD::ADAM){
			nnPotOpt.opt_.reset(new ADAM());
			read(static_cast<ADAM&>(*nnPotOpt.opt_),reader);
		} else if(nnPotOpt.algo_==OPT_METHOD::BFGS){
			nnPotOpt.opt_.reset(new BFGS());
			read(static_cast<BFGS&>(*nnPotOpt.opt_),reader);
		} else if(nnPotOpt.algo_==OPT_METHOD::LM){
			nnPotOpt.opt_.reset(new LM());
			read(static_cast<LM&>(*nnPotOpt.opt_),reader);
		} else if(nnPotOpt.algo_==OPT_METHOD::RPROP){
			nnPotOpt.opt_.reset(new RPROP());
			read(static_cast<RPROP&>(*nnPotOpt.opt_),reader);
		} else throw std::invalid_argument("Invalid optimization algorithm.");
		nnPotOpt.opt_->nPrint()=0;
		
		//======== close parameter file ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Closing parameter file...\n";
		fclose(reader);
		reader=NULL;
		
		//======== initialize the random number generator ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Initializing random number generator...\n";
		if(seed<0) std::srand(std::time(NULL));
		else std::srand(seed);
		
		//======== check the parameters ========
		if(mode==MODE::TRAIN && data_train.size()==0) throw std::invalid_argument("No training data provided.");
		else if(mode==MODE::TEST && data_test.size()==0) throw std::invalid_argument("No test data provided.");
		if(nnPotOpt.pBatch_<0 || nnPotOpt.pBatch_>1) throw std::invalid_argument("Invalid batch size.");
		if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		
		//======== set the unit system
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Setting the unit system...\n";
		units::consts::init(unitsys);
		
		//======== read restart file ========
		if(nnPotOpt.restart_){
			if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading restart file...\n";
			std::string file=nnPotOpt.restart_file_+".restart";
			nnPotOpt.read_restart(file.c_str());
		}
		
		//======== print parameters ========
		std::cout<<"**************************************************\n";
		std::cout<<"*************** GENERAL PARAMETERS ***************\n";
		std::cout<<"\tSEED       = "<<seed<<"\n";
		std::cout<<"\tFORMAT     = "<<format<<"\n";
		std::cout<<"\tUNITS      = "<<unitsys<<"\n";
		std::cout<<"\tENERGIES   = \n"; 
		for(unsigned int i=0; i<energy_.size(); ++i) std::cout<<"\t\t"<<name_[i]<<" "<<energy_[i]<<"\n";
		std::cout<<"\tFILES_POT  = \n"; 
		for(unsigned int i=0; i<fileNNPot.size(); ++i) std::cout<<"\t\t"<<fileName[i]<<" "<<fileNNPot[i]<<"\n";
		if(mode==MODE::TRAIN) std::cout<<"\tMODE       = TRAIN\n";
		else if(mode==MODE::TEST) std::cout<<"\tMODE       = TEST\n";
		std::cout<<"\tDATA_TRAIN = \n"; for(unsigned int i=0; i<data_train.size(); ++i) std::cout<<"\t\t"<<data_train[i]<<"\n";
		std::cout<<"\tDATA_VAL   = \n"; for(unsigned int i=0; i<data_val.size(); ++i) std::cout<<"\t\t"<<data_val[i]<<"\n";
		std::cout<<"\tDATA_TEST  = \n"; for(unsigned int i=0; i<data_test.size(); ++i) std::cout<<"\t\t"<<data_test[i]<<"\n";
		std::cout<<"*************** GENERAL PARAMETERS ***************\n";
		std::cout<<"**************************************************\n";
		std::cout<<"**************************************************\n";
		std::cout<<"******************** WRITING ********************\n";
		std::cout<<"WRITE_CORR   = "<<write_corr<<"\n";
		std::cout<<"WRITE_BASIS  = "<<write_basis<<"\n";
		std::cout<<"WRITE_ENERGY = "<<write_energy<<"\n";
		std::cout<<"WRITE_FORCE  = "<<write_force<<"\n";
		std::cout<<"******************** WRITING ********************\n";
		std::cout<<"**************************************************\n";
		if(nnPotOpt.algo_==OPT_METHOD::SGD) std::cout<<static_cast<SGD&>(*nnPotOpt.opt_)<<"\n";
		else if(nnPotOpt.algo_==OPT_METHOD::SDM) std::cout<<static_cast<SDM&>(*nnPotOpt.opt_)<<"\n";
		else if(nnPotOpt.algo_==OPT_METHOD::NAG) std::cout<<static_cast<NAG>(*nnPotOpt.opt_)<<"\n";
		else if(nnPotOpt.algo_==OPT_METHOD::ADAGRAD) std::cout<<static_cast<ADAGRAD&>(*nnPotOpt.opt_)<<"\n";
		else if(nnPotOpt.algo_==OPT_METHOD::ADADELTA) std::cout<<static_cast<ADADELTA&>(*nnPotOpt.opt_)<<"\n";
		else if(nnPotOpt.algo_==OPT_METHOD::RMSPROP) std::cout<<static_cast<RMSPROP&>(*nnPotOpt.opt_)<<"\n";
		else if(nnPotOpt.algo_==OPT_METHOD::ADAM) std::cout<<static_cast<ADAM&>(*nnPotOpt.opt_)<<"\n";
		else if(nnPotOpt.algo_==OPT_METHOD::BFGS) std::cout<<static_cast<BFGS&>(*nnPotOpt.opt_)<<"\n";
		else if(nnPotOpt.algo_==OPT_METHOD::LM) std::cout<<static_cast<LM&>(*nnPotOpt.opt_)<<"\n";
		else if(nnPotOpt.algo_==OPT_METHOD::RPROP) std::cout<<static_cast<RPROP&>(*nnPotOpt.opt_)<<"\n";
		
		//======== load the training data ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading training data...\n";
		for(unsigned i=0; i<data_train.size(); ++i){
			//open the data file
			if(NN_POT_TRAIN_DEBUG>1) std::cout<<"Data file "<<i<<": "<<data_train[i]<<"\n";
			reader=fopen(data_train[i].c_str(),"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data_train[i]);
			//read in the data
			while(fgets(input,string::M,reader)!=NULL){
				if(!string::empty(input)) files_train.push_back(std::string(string::trim(input)));
			}
			//close the file
			fclose(reader);
			reader=NULL;
		}
		//======== load the test data ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading test data...\n";
		for(unsigned int i=0; i<data_test.size(); ++i){
			//open the data file
			if(NN_POT_TRAIN_DEBUG>1) std::cout<<"Data file "<<i<<": "<<data_test[i]<<"\n";
			reader=fopen(data_test[i].c_str(),"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data_test[i]);
			//read in the data
			while(fgets(input,string::M,reader)!=NULL){
				if(!string::empty(input)) files_test.push_back(std::string(string::trim(input)));
			}
			//close the file
			fclose(reader);
			reader=NULL;
		}
		//======== load the training data ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading validation data...\n";
		for(unsigned int i=0; i<data_val.size(); ++i){
			//open the data file
			if(NN_POT_TRAIN_DEBUG>1) std::cout<<"Data file "<<i<<": "<<data_val[i]<<"\n";
			reader=fopen(data_val[i].c_str(),"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data_val[i]);
			//read in the data
			while(fgets(input,string::M,reader)!=NULL){
				if(!string::empty(input)) files_val.push_back(std::string(string::trim(input)));
			}
			//close the file
			fclose(reader);
			reader=NULL;
		}
		
		//======== print the files ========
		if(NN_POT_TRAIN_DEBUG>0){
			if(files_train.size()>0){
				std::cout<<"**************************************************\n";
				std::cout<<"***************** FILES - TRAIN *****************\n";
				for(unsigned int i=0; i<files_train.size(); ++i) std::cout<<"\t"<<files_train[i]<<"\n";
				std::cout<<"***************** FILES - TRAIN *****************\n";
				std::cout<<"**************************************************\n";
			}
			if(files_val.size()>0){
				std::cout<<"**************************************************\n";
				std::cout<<"****************** FILES - VAL ******************\n";
				for(unsigned int i=0; i<files_val.size(); ++i) std::cout<<"\t"<<files_val[i]<<"\n";
				std::cout<<"****************** FILES - VAL ******************\n";
				std::cout<<"**************************************************\n";
			}
			if(files_test.size()>0){
				std::cout<<"**************************************************\n";
				std::cout<<"****************** FILES - TEST ******************\n";
				for(unsigned int i=0; i<files_test.size(); ++i) std::cout<<"\t"<<files_test[i]<<"\n";
				std::cout<<"****************** FILES - TEST ******************\n";
				std::cout<<"**************************************************\n";
			}
		}
		
		//======== load the simulations ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading training simulations...\n";
		if(files_train.size()>0){
			struc_train.resize(files_train.size());
			if(format==FILE_FORMAT::VASP_XML){
				for(unsigned int i=0; i<files_train.size(); ++i){
					VASP::XML::read(files_train[i].c_str(),0,atomT,struc_train[i]);
				}
			} else if(format==FILE_FORMAT::QE){
				for(unsigned int i=0; i<files_train.size(); ++i){
					QE::OUT::read(files_train[i].c_str(),atomT,struc_train[i]);
					std::cout<<"struc_train["<<i<<"] = \n"<<struc_train[i];
					for(unsigned int j=0; j<struc_train[i].nAtoms(); ++j){
						std::cout<<struc_train[i].name(j)<<" "<<struc_train[i].posn(j)[0]<<" "<<struc_train[i].posn(j)[1]<<" "<<struc_train[i].posn(j)[2]<<"\n";
					}
				}
			} else if(format==FILE_FORMAT::AME){
				for(unsigned int i=0; i<files_train.size(); ++i){
					AME::read(files_train[i].c_str(),atomT,struc_train[i]);
				}
			}
			if(NN_POT_TRAIN_DEBUG>1) for(unsigned int i=0; i<files_train.size(); ++i) std::cout<<"\t"<<files_train[i]<<" "<<struc_train[i].energy()<<"\n";
		}
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading validation simulations...\n";
		if(files_val.size()>0){
			struc_val.resize(files_val.size());
			if(format==FILE_FORMAT::VASP_XML){
				for(unsigned int i=0; i<files_val.size(); ++i){
					VASP::XML::read(files_val[i].c_str(),0,atomT,struc_val[i]);
				}
			} else if(format==FILE_FORMAT::QE){
				for(unsigned int i=0; i<files_val.size(); ++i){
					QE::OUT::read(files_val[i].c_str(),atomT,struc_val[i]);
				}
			} else if(format==FILE_FORMAT::AME){
				for(unsigned int i=0; i<files_val.size(); ++i){
					AME::read(files_val[i].c_str(),atomT,struc_val[i]);
				}
			}
			if(NN_POT_TRAIN_DEBUG>1) for(unsigned int i=0; i<files_val.size(); ++i) std::cout<<"\t"<<files_val[i]<<" "<<struc_val[i].energy()<<"\n";
		}
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading testing simulations...\n";
		if(files_test.size()>0){
			struc_test.resize(files_test.size());
			if(format==FILE_FORMAT::VASP_XML){
				for(unsigned int i=0; i<files_test.size(); ++i){
					VASP::XML::read(files_test[i].c_str(),0,atomT,struc_test[i]);
				}
			} else if(format==FILE_FORMAT::QE){
				for(unsigned int i=0; i<files_test.size(); ++i){
					QE::OUT::read(files_test[i].c_str(),atomT,struc_test[i]);
				}
			} else if(format==FILE_FORMAT::AME){
				for(unsigned int i=0; i<files_test.size(); ++i){
					AME::read(files_test[i].c_str(),atomT,struc_test[i]);
				}
			}
			if(NN_POT_TRAIN_DEBUG>1) for(unsigned int i=0; i<files_test.size(); ++i) std::cout<<"\t"<<files_test[i]<<" "<<struc_test[i].energy()<<"\n";
		}
		
		//======== check the simulations ========
		if(struc_test.size()>0){
			for(unsigned int i=0; i<struc_train.size(); ++i){
				if(std::isinf(struc_train[i].energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+files_train[i]+std::string(" has inf energy."));
				if(struc_train[i].energy()!=struc_train[i].energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+files_train[i]+std::string(" has nan energy."));
				if(struc_train[i].energy()==0) std::cout<<"*********** WARNING: Structure \""<<files_train[i]<<"\" has ZERO energy. ***********";
				for(unsigned int n=0; n<struc_train[i].nAtoms(); ++n){
					double force=struc_train[i].force(n).norm();
					if(std::isinf(force)) throw std::runtime_error(std::string("ERROR: Atom \"")+struc_train[i].name(n)+std::to_string(struc_train[i].index(n))
							+std::string("\" in \"")+files_train[i]+std::string(" has inf force."));
					if(force!=force) throw std::runtime_error(std::string("ERROR: Atom \"")+struc_train[i].name(n)+std::to_string(struc_train[i].index(n))
							+std::string("\" in \"")+files_train[i]+std::string(" has nan force."));
				}
			}
			if(NN_POT_TRAIN_DEBUG>1) for(unsigned int i=0; i<files_train.size(); ++i) std::cout<<"\t"<<files_train[i]<<" "<<struc_train[i].energy()<<"\n";
		}
		if(struc_val.size()>0){
			for(unsigned int i=0; i<struc_val.size(); ++i){
				if(std::isinf(struc_val[i].energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+files_val[i]+std::string(" has inf energy."));
				if(struc_val[i].energy()!=struc_val[i].energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+files_val[i]+std::string(" has nan energy."));
				if(struc_train[i].energy()==0) std::cout<<"*********** WARNING: Structure \""<<files_val[i]<<"\" has ZERO energy. ***********";
				for(unsigned int n=0; n<struc_val[i].nAtoms(); ++n){
					double force=struc_val[i].force(n).norm();
					if(std::isinf(force)) throw std::runtime_error(std::string("ERROR: Atom \"")+struc_val[i].name(n)+std::to_string(struc_val[i].index(n))
							+std::string("\" in \"")+files_val[i]+std::string(" has inf force."));
					if(force!=force) throw std::runtime_error(std::string("ERROR: Atom \"")+struc_val[i].name(n)+std::to_string(struc_val[i].index(n))
							+std::string("\" in \"")+files_val[i]+std::string(" has nan force."));
				}
			}
			if(NN_POT_TRAIN_DEBUG>1) for(unsigned int i=0; i<files_val.size(); ++i) std::cout<<"\t"<<files_val[i]<<" "<<struc_val[i].energy()<<"\n";
		}
		if(struc_test.size()>0){
			for(unsigned int i=0; i<struc_test.size(); ++i){
				if(std::isinf(struc_test[i].energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+files_test[i]+std::string(" has inf energy."));
				if(struc_test[i].energy()!=struc_test[i].energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+files_test[i]+std::string(" has nan energy."));
				if(struc_train[i].energy()==0) std::cout<<"*********** WARNING: Structure \""<<files_test[i]<<"\" has ZERO energy. ***********";
				for(unsigned int n=0; n<struc_test[i].nAtoms(); ++n){
					double force=struc_test[i].force(n).norm();
					if(std::isinf(force)) throw std::runtime_error(std::string("ERROR: Atom \"")+struc_test[i].name(n)+std::to_string(struc_test[i].index(n))
							+std::string("\" in \"")+files_test[i]+std::string(" has inf force."));
					if(force!=force) throw std::runtime_error(std::string("ERROR: Atom \"")+struc_test[i].name(n)+std::to_string(struc_test[i].index(n))
							+std::string("\" in \"")+files_test[i]+std::string(" has nan force."));
				}
			}
			if(NN_POT_TRAIN_DEBUG>1) for(unsigned int i=0; i<files_test.size(); ++i) std::cout<<"\t"<<files_test[i]<<" "<<struc_test[i].energy()<<"\n";
		}
		
		//======== statistical data - energies/forces/errors ========
		//data - train
			Accumulator1D<Max,Avg,Var> acc1d_energy_train_a;
			Accumulator1D<Max,Avg,Var> acc1d_energy_train_n;
			Accumulator1D<Max,Avg,Var> acc1d_energy_train_p;
			Accumulator1D<Max,Avg,Var> acc1d_force_train_a;
			Accumulator1D<Max,Avg,Var> acc1d_force_train_p;
			Accumulator2D<LinReg> acc2d_energy_train;
			Accumulator2D<LinReg> acc2d_forcex_train;
			Accumulator2D<LinReg> acc2d_forcey_train;
			Accumulator2D<LinReg> acc2d_forcez_train;
		//data - val	
			Accumulator1D<Max,Avg,Var> acc1d_energy_val_a;
			Accumulator1D<Max,Avg,Var> acc1d_energy_val_n;
			Accumulator1D<Max,Avg,Var> acc1d_energy_val_p;
			Accumulator1D<Max,Avg,Var> acc1d_force_val_a;
			Accumulator1D<Max,Avg,Var> acc1d_force_val_p;
			Accumulator2D<LinReg> acc2d_energy_val;
			Accumulator2D<LinReg> acc2d_forcex_val;
			Accumulator2D<LinReg> acc2d_forcey_val;
			Accumulator2D<LinReg> acc2d_forcez_val;
		//data - test
			Accumulator1D<Max,Avg,Var> acc1d_energy_test_a;
			Accumulator1D<Max,Avg,Var> acc1d_energy_test_n;
			Accumulator1D<Max,Avg,Var> acc1d_energy_test_p;
			Accumulator1D<Max,Avg,Var> acc1d_force_test_a;
			Accumulator1D<Max,Avg,Var> acc1d_force_test_p;
			Accumulator2D<LinReg> acc2d_energy_test;
			Accumulator2D<LinReg> acc2d_forcex_test;
			Accumulator2D<LinReg> acc2d_forcey_test;
			Accumulator2D<LinReg> acc2d_forcez_test;
			
		//======== set the batch size ========
		if(nnPotOpt.pBatch_>0) nnPotOpt.nBatch_=std::floor(nnPotOpt.pBatch_*struc_train.size());
		if(nnPotOpt.nBatch_==0) throw std::invalid_argument("Invalid batch size.");
		if(nnPotOpt.nBatch_>struc_train.size()) throw std::invalid_argument("Invalid batch size.");
		
		//======== set the data ========
		if(struc_train.size()>0) nnPotOpt.strucTrain_=&struc_train;
		if(struc_val.size()>0) nnPotOpt.strucVal_=&struc_val;
		if(struc_test.size()>0) nnPotOpt.strucTest_=&struc_test;
		
		//====== set the vacuum energy ======
		if(NN_POT_PRINT_STATUS>-1) std::cout<<"Set the vacuum energy...\n";
		nnPotOpt.nnpot_.resize(struc_train);
		for(unsigned int i=0; i<name_.size(); ++i){
			nnPotOpt.nnpot_.energyAtom(nnPotOpt.nnpot_.speciesIndex(name_[i]))=energy_[i];
		}
		
		//====== initialize the potential ======
		if(!nnPotOpt.restart_){
			if(NN_POT_PRINT_STATUS>-1) std::cout<<"Initializing potential...\n";
			nnPotOpt.nnpot_.init(nnPotInit);
		}
		
		//====== read neural network potentials ======
		for(unsigned int i=0; i<fileNNPot.size(); ++i){
			std::cout<<"Reading nn-pot for \""<<fileName[i]<<"\" from \""<<fileNNPot[i]<<"\"...\n";
			nnPotOpt.nnpot_.read(nnPotOpt.nnpot_.speciesIndex(fileName[i]),fileNNPot[i]);
		}
		
		//====== initialize the symmetry functions ======
		if(NN_POT_PRINT_STATUS>-1) std::cout<<"Initializing symmetry functions...\n";
		nnPotOpt.nnpot_.initSymm(struc_train);
		if(struc_val.size()>0) nnPotOpt.nnpot_.initSymm(struc_val);
		if(struc_test.size()>0) nnPotOpt.nnpot_.initSymm(struc_test);
		
		//====== print the optimization object ======
		std::cout<<nnPotOpt<<"\n";
		
		//====== set the inputs for each of the simulations ======
		nnPotOpt.nnpotv_.resize(nnPotOpt.nThreads_,nnPotOpt.nnpot_);
		//training
		start=std::clock();
		if(struc_train.size()>0){
			if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting the inputs (symmetry functions) - training...\n";
			#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
			for(unsigned int n=0; n<struc_train.size(); ++n){
				unsigned int TN=0;
				#ifdef _OPENMP
					TN=omp_get_thread_num();
				#endif
				std::cout<<"system-train["<<n<<"]\n";
				nnPotOpt.nnpotv_[TN].inputs_symm(struc_train[n]);
			}
		}
		stop=std::clock();
		time_symm_train=((double)(stop-start))/CLOCKS_PER_SEC;
		//validation
		start=std::clock();
		if(struc_val.size()>0){
			if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting the inputs (symmetry functions) - validation...\n";
			#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
			for(unsigned int n=0; n<struc_val.size(); ++n){
				unsigned int TN=0;
				#ifdef _OPENMP
					TN=omp_get_thread_num();
				#endif
				std::cout<<"system-val["<<n<<"]\n";
				nnPotOpt.nnpotv_[TN].inputs_symm(struc_val[n]);
			}
		}
		stop=std::clock();
		time_symm_val=((double)(stop-start))/CLOCKS_PER_SEC;
		//testing
		start=std::clock();
		if(struc_test.size()>0){
			if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting the inputs (symmetry functions) - testing...\n";
			#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
			for(unsigned int n=0; n<struc_test.size(); ++n){
				unsigned int TN=0;
				#ifdef _OPENMP
					TN=omp_get_thread_num();
				#endif
				std::cout<<"system-test["<<n<<"]\n";
				nnPotOpt.nnpotv_[TN].inputs_symm(struc_test[n]);
			}
		}
		stop=std::clock();
		time_symm_test=((double)(stop-start))/CLOCKS_PER_SEC;
		//print the inputs
		if(NN_POT_PRINT_DATA>0){
			std::cout<<"====================================\n";
			std::cout<<"============== INPUTS ==============\n";
			for(unsigned int n=0; n<struc_train.size(); ++n){
				std::cout<<"sim["<<n<<"] = \n";
				for(unsigned int i=0; i<struc_train[n].nAtoms(); ++i){
					std::cout<<struc_train[n].name(i)<<struc_train[n].index(i)+1<<" "<<struc_train[n].symm(i).transpose()<<"\n";
				}
			}
			FILE* writer=fopen("nn_pot_inputs.dat","w");
			if(writer!=NULL){
				for(unsigned int n=0; n<struc_train.size(); ++n){
					for(unsigned int i=0; i<struc_train[n].nAtoms(); ++i){
						fprintf(writer,"%s%i ",struc_train[n].name(i).c_str(),struc_train[n].index(i)+1);
						for(unsigned int j=0; j<struc_train[n].symm(i).size(); ++j){
							fprintf(writer,"%f ",struc_train[n].symm(i)[j]);
						}
						fprintf(writer,"\n");
					}
				}
				fclose(writer);
				writer=NULL;
			}
			std::cout<<"============== INPUTS ==============\n";
			std::cout<<"====================================\n";
		}
		
		//************************************************************************************
		// TRAINING
		//************************************************************************************
		
		//======== train the nn potential ========
		if(mode==MODE::TRAIN){
			if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Training the nn potential...\n";
			nnPotOpt.train(nnPotOpt.nBatch_);
		} else {
			nnPotOpt.nnpotv_.resize(nnPotOpt.nThreads_,nnPotOpt.nnpot_);
			for(unsigned int i=0; i<nnPotOpt.nThreads_; ++i) nnPotOpt.nnpotv_[i]=nnPotOpt.nnpot_;
		}
		
		//************************************************************************************
		// EVALUTION
		//************************************************************************************
		
		//======== calculate the final energies ========
		if(struc_train.size()>0){
			std::cout<<"Final energies - training set ... \n";
			std::vector<double> energy_nn(struc_train.size());
			std::vector<double> energy_exact(struc_train.size());
			start=std::clock();
			#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
			for(unsigned int n=0; n<struc_train.size(); ++n){
				unsigned int TN=0;
				#ifdef _OPENMP
					TN=omp_get_thread_num();
				#endif
				std::cout<<"system-train["<<n<<"]\n";
				energy_nn[n]=nnPotOpt.nnpotv_[TN].energy(struc_train[n],false);
				energy_exact[n]=struc_train[n].energy();
				
			}
			stop=std::clock();
			time_energy_train=((double)(stop-start))/CLOCKS_PER_SEC;
			for(unsigned int n=0; n<struc_train.size(); ++n){
				acc1d_energy_train_a.push(std::fabs(energy_exact[n]-energy_nn[n]));
				acc1d_energy_train_n.push(std::fabs(energy_exact[n]-energy_nn[n])/struc_train[n].nAtoms());
				acc1d_energy_train_p.push(std::fabs((energy_exact[n]-energy_nn[n])/energy_exact[n])*100.0);
				acc2d_energy_train.push(energy_exact[n],energy_nn[n]);
			}
			if(write_energy){
				const char* file="nn_pot_energy_train.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					for(unsigned int i=0; i<struc_train.size(); ++i) fprintf(writer,"%s %f %f\n",files_train[i].c_str(),energy_exact[i],energy_nn[i]);
					fclose(writer); writer=NULL;
				}
			}
		}
		if(struc_val.size()>0){
			std::cout<<"Final energies - validation set ... \n";
			std::vector<double> energy_nn(struc_val.size());
			std::vector<double> energy_exact(struc_val.size());
			start=std::clock();
			#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
			for(unsigned int n=0; n<struc_val.size(); ++n){
				unsigned int TN=0;
				#ifdef _OPENMP
					TN=omp_get_thread_num();
				#endif
				std::cout<<"system-val["<<n<<"]\n";
				energy_nn[n]=nnPotOpt.nnpotv_[TN].energy(struc_val[n],false);
				energy_exact[n]=struc_val[n].energy();
			}
			stop=std::clock();
			time_energy_val=((double)(stop-start))/CLOCKS_PER_SEC;
			for(unsigned int n=0; n<struc_val.size(); ++n){
				acc1d_energy_val_a.push(std::fabs(energy_exact[n]-energy_nn[n]));
				acc1d_energy_val_n.push(std::fabs(energy_exact[n]-energy_nn[n])/struc_val[n].nAtoms());
				acc1d_energy_val_p.push(std::fabs((energy_exact[n]-energy_nn[n])/energy_exact[n])*100.0);
				acc2d_energy_val.push(energy_exact[n],energy_nn[n]);
			}
			if(write_energy){
				const char* file="nn_pot_energy_val.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					for(unsigned int i=0; i<struc_val.size(); ++i) fprintf(writer,"%s %f %f\n",files_val[i].c_str(),energy_exact[i],energy_nn[i]);
					fclose(writer); writer=NULL;
				}
			}
		}
		if(struc_test.size()>0){
			std::cout<<"Final energies - test set ... \n";
			std::vector<double> energy_nn(struc_test.size());
			std::vector<double> energy_exact(struc_test.size());
			start=std::clock();
			#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
			for(unsigned int n=0; n<struc_test.size(); ++n){
				unsigned int TN=0;
				#ifdef _OPENMP
					TN=omp_get_thread_num();
				#endif
				std::cout<<"system-test["<<n<<"]\n";
				energy_nn[n]=nnPotOpt.nnpotv_[TN].energy(struc_test[n],false);
				energy_exact[n]=struc_test[n].energy();
			}
			stop=std::clock();
			time_energy_test=((double)(stop-start))/CLOCKS_PER_SEC;
			for(unsigned int n=0; n<struc_test.size(); ++n){
				acc1d_energy_test_a.push(std::fabs(energy_exact[n]-energy_nn[n]));
				acc1d_energy_test_n.push(std::fabs(energy_exact[n]-energy_nn[n])/struc_test[n].nAtoms());
				acc1d_energy_test_p.push(std::fabs((energy_exact[n]-energy_nn[n])/energy_exact[n])*100.0);
				acc2d_energy_test.push(energy_exact[n],energy_nn[n]);
			}
			if(write_energy){
				const char* file="nn_pot_energy_test.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					for(unsigned int i=0; i<struc_test.size(); ++i) fprintf(writer,"%s %f %f\n",files_test[i].c_str(),energy_exact[i],energy_nn[i]);
					fclose(writer); writer=NULL;
				}
			}
		}
		
		//======== calculate the final forces ========
		if(struc_train.size()>0){
			std::cout<<"Calculating final forces - training set...\n";
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_exact(struc_train.size());
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_nn(struc_train.size());
			for(unsigned int n=0; n<struc_train.size(); ++n){
				forces_exact[n].resize(struc_train[n].nAtoms());
				forces_nn[n].resize(struc_train[n].nAtoms());
				for(unsigned int j=0; j<struc_train[n].nAtoms(); ++j) forces_exact[n][j]=struc_train[n].force(j);
			}
			start=std::clock();
			#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
			for(unsigned int n=0; n<struc_train.size(); ++n){
				unsigned int TN=0;
				#ifdef _OPENMP
					TN=omp_get_thread_num();
				#endif
				std::cout<<"system-train["<<n<<"]\n";
				nnPotOpt.nnpotv_[TN].forces(struc_train[n],false);
			}
			stop=std::clock();
			time_force_train=((double)(stop-start))/CLOCKS_PER_SEC;
			for(unsigned int n=0; n<struc_train.size(); ++n){
				for(unsigned int j=0; j<struc_train[n].nAtoms(); ++j) forces_nn[n][j]=struc_train[n].force(j);
			}
			for(unsigned int i=0; i<struc_train.size(); ++i){
				for(unsigned int j=0; j<struc_train[i].nAtoms(); ++j){
					acc1d_force_train_a.push((forces_exact[i][j]-forces_nn[i][j]).norm());
					acc1d_force_train_p.push((forces_exact[i][j]-forces_nn[i][j]).norm()/forces_exact[i][j].norm()*100.0);
					acc2d_forcex_train.push(forces_exact[i][j][0],forces_nn[i][j][0]);
					acc2d_forcey_train.push(forces_exact[i][j][1],forces_nn[i][j][1]);
					acc2d_forcez_train.push(forces_exact[i][j][2],forces_nn[i][j][2]);
				}
			}
			if(write_force){
				const char* file="nn_pot_force_train.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					for(unsigned int i=0; i<struc_train.size(); ++i){
						for(unsigned int n=0; n<struc_train[i].nAtoms(); ++n){
							fprintf(writer,"%s %s%i %f %f %f %f %f %f\n",files_train[i].c_str(),
								struc_train[i].name(n).c_str(),struc_train[i].index(n),
								forces_exact[i][n][0],forces_exact[i][n][1],forces_exact[i][n][2],
								forces_nn[i][n][0],forces_nn[i][n][1],forces_nn[i][n][2]
							);
						}
					}
					fclose(writer); writer=NULL;
				}
			}
		}
		if(struc_val.size()>0){
			std::cout<<"Calculating final forces - validation set...\n";
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_exact(struc_val.size());
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_nn(struc_val.size());
			for(unsigned int n=0; n<struc_val.size(); ++n){
				forces_exact[n].resize(struc_val[n].nAtoms());
				forces_nn[n].resize(struc_val[n].nAtoms());
				for(unsigned int j=0; j<struc_val[n].nAtoms(); ++j) forces_exact[n][j]=struc_val[n].force(j);
			}
			start=std::clock();
			#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
			for(unsigned int n=0; n<struc_val.size(); ++n){
				unsigned int TN=0;
				#ifdef _OPENMP
					TN=omp_get_thread_num();
				#endif
				std::cout<<"system-val["<<n<<"]\n";
				nnPotOpt.nnpotv_[TN].forces(struc_val[n],false);
			}
			stop=std::clock();
			time_force_val=((double)(stop-start))/CLOCKS_PER_SEC;
			for(unsigned int n=0; n<struc_val.size(); ++n){
				for(unsigned int j=0; j<struc_val[n].nAtoms(); ++j) forces_nn[n][j]=struc_val[n].force(j);
			}
			for(unsigned int i=0; i<struc_val.size(); ++i){
				for(unsigned int j=0; j<struc_val[i].nAtoms(); ++j){
					acc1d_force_val_a.push((forces_exact[i][j]-forces_nn[i][j]).norm());
					acc1d_force_val_p.push((forces_exact[i][j]-forces_nn[i][j]).norm()/forces_exact[i][j].norm()*100.0);
					acc2d_forcex_val.push(forces_exact[i][j][0],forces_nn[i][j][0]);
					acc2d_forcey_val.push(forces_exact[i][j][1],forces_nn[i][j][1]);
					acc2d_forcez_val.push(forces_exact[i][j][2],forces_nn[i][j][2]);
				}
			}
			if(write_force){
				const char* file="nn_pot_force_val.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					for(unsigned int i=0; i<struc_val.size(); ++i){
						for(unsigned int n=0; n<struc_val[i].nAtoms(); ++n){
							fprintf(writer,"%s %s%i %f %f %f %f %f %f\n",files_val[i].c_str(),
								struc_val[i].name(n).c_str(),struc_val[i].index(n),
								forces_exact[i][n][0],forces_exact[i][n][1],forces_exact[i][n][2],
								forces_nn[i][n][0],forces_nn[i][n][1],forces_nn[i][n][2]
							);
						}
					}
					fclose(writer); writer=NULL;
				}
			}
		}
		if(struc_test.size()>0){
			std::cout<<"Calculating final forces - test set...\n";
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d> > forces_exact(struc_test.size());
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_nn(struc_test.size());
			for(unsigned int n=0; n<struc_test.size(); ++n){
				forces_exact[n].resize(struc_test[n].nAtoms());
				forces_nn[n].resize(struc_test[n].nAtoms());
				for(unsigned int j=0; j<struc_test[n].nAtoms(); ++j) forces_exact[n][j]=struc_test[n].force(j);
			}
			start=std::clock();
			#pragma omp parallel for schedule(dynamic) num_threads(omp_get_max_threads())
			for(unsigned int n=0; n<struc_test.size(); ++n){
				unsigned int TN=0;
				#ifdef _OPENMP
					TN=omp_get_thread_num();
				#endif
				std::cout<<"system-test["<<n<<"]\n";
				nnPotOpt.nnpotv_[TN].forces(struc_test[n],false);
			}
			stop=std::clock();
			time_force_test=((double)(stop-start))/CLOCKS_PER_SEC;
			for(unsigned int n=0; n<struc_test.size(); ++n){
				for(unsigned int j=0; j<struc_test[n].nAtoms(); ++j) forces_nn[n][j]=struc_test[n].force(j);
			}
			for(unsigned int i=0; i<struc_test.size(); ++i){
				for(unsigned int j=0; j<struc_test[i].nAtoms(); ++j){
					acc1d_force_test_a.push((forces_exact[i][j]-forces_nn[i][j]).norm());
					acc1d_force_test_p.push((forces_exact[i][j]-forces_nn[i][j]).norm()/forces_exact[i][j].norm()*100.0);
					acc2d_forcex_test.push(forces_exact[i][j][0],forces_nn[i][j][0]);
					acc2d_forcey_test.push(forces_exact[i][j][1],forces_nn[i][j][1]);
					acc2d_forcez_test.push(forces_exact[i][j][2],forces_nn[i][j][2]);
				}
			}
			if(write_force){
				const char* file="nn_pot_force_test.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					for(unsigned int i=0; i<struc_test.size(); ++i){
						for(unsigned int n=0; n<struc_test[i].nAtoms(); ++n){
							fprintf(writer,"%s %s%i %f %f %f %f %f %f\n",files_test[i].c_str(),
								struc_test[i].name(n).c_str(),struc_test[i].index(n),
								forces_exact[i][n][0],forces_exact[i][n][1],forces_exact[i][n][2],
								forces_nn[i][n][0],forces_nn[i][n][1],forces_nn[i][n][2]
							);
						}
					}
					fclose(writer); writer=NULL;
				}
			}
		}
		
		//************************************************************************************
		// OUTPUT
		//************************************************************************************
		
		//======== print the timing info ========
		std::cout<<"**************************************************\n";
		std::cout<<"******************* TIMING (S) *******************\n";
		std::cout<<"time - symm   - train = "<<time_symm_train<<"\n";
		std::cout<<"time - energy - train = "<<time_energy_train<<"\n";
		std::cout<<"time - force  - train = "<<time_force_train<<"\n";
		std::cout<<"time - symm   - val   = "<<time_symm_val<<"\n";
		std::cout<<"time - energy - val   = "<<time_energy_val<<"\n";
		std::cout<<"time - force  - val   = "<<time_force_val<<"\n";
		std::cout<<"time - symm   - test  = "<<time_symm_test<<"\n";
		std::cout<<"time - energy - test  = "<<time_energy_test<<"\n";
		std::cout<<"time - force  - test  = "<<time_force_test<<"\n";
		std::cout<<"******************* TIMING (S) *******************\n";
		std::cout<<"**************************************************\n";
		
		//======== print the error statistics - training ========
		if(mode==MODE::TRAIN){
		std::cout<<"**************************************************\n";
		std::cout<<"********* STATISTICS - ERROR - TRAINING *********\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tAVG    = "<<acc1d_energy_train_a.avg()<<" "<<acc1d_energy_train_n.avg()<<" "<<acc1d_energy_train_p.avg()<<"\n";
		std::cout<<"\tSTDDEV = "<<std::sqrt(acc1d_energy_train_a.var())<<" "<<std::sqrt(acc1d_energy_train_n.var())<<" "<<std::sqrt(acc1d_energy_train_p.var())<<"\n";
		std::cout<<"\tMAX    = "<<acc1d_energy_train_a.max()<<" "<<acc1d_energy_train_n.max()<<" "<<acc1d_energy_train_p.max()<<"\n";
		std::cout<<"\tM      = "<<acc2d_energy_train.m()<<"\n";
		std::cout<<"\tR2     = "<<acc2d_energy_train.r2()<<"\n";
		std::cout<<"FORCE:\n";
		std::cout<<"\tAVG    = "<<acc1d_force_train_a.avg()<<" "<<acc1d_force_train_p.avg()<<"\n";
		std::cout<<"\tSTDDEV = "<<std::sqrt(acc1d_force_train_a.var())<<" "<<std::sqrt(acc1d_force_train_p.var())<<"\n";
		std::cout<<"\tMAX    = "<<acc1d_force_train_a.max()<<" "<<acc1d_force_train_p.max()<<"\n";
		std::cout<<"\tM      = "<<acc2d_forcex_train.m() <<" "<<acc2d_forcey_train.m() <<" "<<acc2d_forcez_train.m() <<"\n";
		std::cout<<"\tR2     = "<<acc2d_forcex_train.r2()<<" "<<acc2d_forcey_train.r2()<<" "<<acc2d_forcez_train.r2()<<"\n";
		std::cout<<"********* STATISTICS - ERROR - TRAINING *********\n";
		std::cout<<"**************************************************\n";
		}
		
		//======== print the error statistics - validation ========
		if(struc_val.size()>0 && mode==MODE::TRAIN){
		std::cout<<"**************************************************\n";
		std::cout<<"******** STATISTICS - ERROR - VALIDATION ********\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tAVG    = "<<acc1d_energy_val_a.avg()<<" "<<acc1d_energy_val_n.avg()<<" "<<acc1d_energy_val_p.avg()<<"\n";
		std::cout<<"\tSTDDEV = "<<std::sqrt(acc1d_energy_val_a.var())<<" "<<std::sqrt(acc1d_energy_val_n.var())<<" "<<std::sqrt(acc1d_energy_val_p.var())<<"\n";
		std::cout<<"\tMAX    = "<<acc1d_energy_val_a.max()<<" "<<acc1d_energy_val_n.max()<<" "<<acc1d_energy_val_p.max()<<"\n";
		std::cout<<"\tM      = "<<acc2d_energy_val.m()<<"\n";
		std::cout<<"\tR2     = "<<acc2d_energy_val.r2()<<"\n";
		std::cout<<"FORCE:\n";
		std::cout<<"\tAVG    = "<<acc1d_force_val_a.avg()<<" "<<acc1d_force_val_p.avg()<<"\n";
		std::cout<<"\tSTDDEV = "<<std::sqrt(acc1d_force_val_a.var())<<" "<<std::sqrt(acc1d_force_val_p.var())<<"\n";
		std::cout<<"\tMAX    = "<<acc1d_force_val_a.max()<<" "<<acc1d_force_val_p.max()<<"\n";
		std::cout<<"\tM      = "<<acc2d_forcex_val.m() <<" "<<acc2d_forcey_val.m() <<" "<<acc2d_forcez_val.m() <<"\n";
		std::cout<<"\tR2     = "<<acc2d_forcex_val.r2()<<" "<<acc2d_forcey_val.r2()<<" "<<acc2d_forcez_val.r2()<<"\n";
		std::cout<<"******** STATISTICS - ERROR - VALIDATION ********\n";
		std::cout<<"**************************************************\n";
		}
		
		//======== print the error statistics - test ========
		if(struc_test.size()>0){
		std::cout<<"**************************************************\n";
		std::cout<<"*********** STATISTICS - ERROR - TEST ***********\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tAVG    = "<<acc1d_energy_test_a.avg()<<" "<<acc1d_energy_test_n.avg()<<" "<<acc1d_energy_test_p.avg()<<"\n";
		std::cout<<"\tSTDDEV = "<<std::sqrt(acc1d_energy_test_a.var())<<" "<<std::sqrt(acc1d_energy_test_n.var())<<" "<<std::sqrt(acc1d_energy_test_p.var())<<"\n";
		std::cout<<"\tMAX    = "<<acc1d_energy_test_a.max()<<" "<<acc1d_energy_test_n.max()<<" "<<acc1d_energy_test_p.max()<<"\n";
		std::cout<<"\tM      = "<<acc2d_energy_test.m()<<"\n";
		std::cout<<"\tR2     = "<<acc2d_energy_test.r2()<<"\n";
		std::cout<<"FORCE:\n";
		std::cout<<"\tAVG    = "<<acc1d_force_test_a.avg()<<" "<<acc1d_force_test_p.avg()<<"\n";
		std::cout<<"\tSTDDEV = "<<std::sqrt(acc1d_force_test_a.var())<<" "<<std::sqrt(acc1d_force_test_p.var())<<"\n";
		std::cout<<"\tMAX    = "<<acc1d_force_test_a.max()<<" "<<acc1d_force_test_p.max()<<"\n";
		std::cout<<"\tM      = "<<acc2d_forcex_test.m() <<" "<<acc2d_forcey_test.m() <<" "<<acc2d_forcez_test.m() <<"\n";
		std::cout<<"\tR2     = "<<acc2d_forcex_test.r2()<<" "<<acc2d_forcey_test.r2()<<" "<<acc2d_forcez_test.r2()<<"\n";
		std::cout<<"*********** STATISTICS - ERROR - TEST ***********\n";
		std::cout<<"**************************************************\n";
		}
		
		//======== print the basis functions ========
		if(write_basis){
			unsigned int N=200;
			for(unsigned int n=0; n<nnPotOpt.nnpot_.nSpecies(); ++n){
				for(unsigned int m=0; m<nnPotOpt.nnpot_.nSpecies(); ++m){
					std::string filename="basisR_"+nnPotOpt.nnpot_.speciesName(n)+"_"+nnPotOpt.nnpot_.speciesName(m)+".dat";
					FILE* writer=fopen(filename.c_str(),"w");
					if(writer!=NULL){
						for(unsigned int i=0; i<N; ++i){
							double dr=nnPotOpt.nnpot_.rc()*i/N;
							fprintf(writer,"%f ",dr);
							for(unsigned int j=0; j<nnPotOpt.nnpot_.basisR(n,m).nfR(); ++j){
								fprintf(writer,"%f ",nnPotOpt.nnpot_.basisR(n,m).fR(j)(dr));
							}
							fprintf(writer,"\n");
						}
						fclose(writer);
						writer=NULL;
					} else std::cout<<"WARNING: Could not open: \""<<filename<<"\"\n";
				}
			}
			for(unsigned int n=0; n<nnPotOpt.nnpot_.nSpecies(); ++n){
				for(unsigned int m=0; m<nnPotOpt.nnpot_.nSpecies(); ++m){
					for(unsigned int l=0; l<nnPotOpt.nnpot_.nSpecies(); ++l){
						std::string filename="basisA_"+nnPotOpt.nnpot_.speciesName(n)+"_"+nnPotOpt.nnpot_.speciesName(m)+"_"+nnPotOpt.nnpot_.speciesName(l)+".dat";
						FILE* writer=fopen(filename.c_str(),"w");
						if(writer!=NULL){
							for(unsigned int i=0; i<N; ++i){
								double angle=num_const::PI*i/N;
								fprintf(writer,"%f ",angle);
								for(unsigned int j=0; j<nnPotOpt.nnpot_.basisA(n,m,l).nfA(); ++j){
									fprintf(writer,"%f ",nnPotOpt.nnpot_.basisA(n,m,l).fA(j)(std::cos(angle),0.5,0.5,0.5));
								}
								fprintf(writer,"\n");
							}
							fclose(writer);
							writer=NULL;
						} else std::cout<<"WARNING: Could not open: \""<<filename<<"\"\n";
					}
				}
			}
		}
		
		//======== print the nn's ========
		std::cout<<"Printing the nn's...\n";
		nnPotOpt.nnpot_.write();
		
		//======== write restart file
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Writing restart file...\n";
		{
			std::string file=nnPotOpt.restart_file_+".restart";
			nnPotOpt.write_restart(file.c_str());
		}
		
	}catch(std::exception& e){
		std::cout<<"ERROR in nn_pot_train::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//free local variables
	free(input);
	free(temp);
	
	return 0;
}