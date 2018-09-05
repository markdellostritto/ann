#include "nn_pot_train_omp.hpp"

//************************************************************
// NNPotOpt - Neural Network Potential - Optimization
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPotOpt& nnPotOpt){
	out<<"**************************************************\n";
	out<<"****************** NN - POT - OPT ****************\n";
	out<<"N_PRINT             = "<<nnPotOpt.nPrint_<<"\n";
	out<<"N_WRITE             = "<<nnPotOpt.nWrite_<<"\n";
	out<<"N_THREADS           = "<<nnPotOpt.nThreads_<<"\n";
	out<<"THREAD_DIST_TRAIN   = "; for(unsigned int i=0; i<nnPotOpt.tdTrain_.size(); ++i) out<<nnPotOpt.tdTrain_[i]<<" "; out<<"\n";
	out<<"THREAD_OFFSET_TRAIN = "; for(unsigned int i=0; i<nnPotOpt.toTrain_.size(); ++i) out<<nnPotOpt.toTrain_[i]<<" "; out<<"\n";
	out<<"THREAD_DIST_VAL     = "; for(unsigned int i=0; i<nnPotOpt.tdVal_.size(); ++i) out<<nnPotOpt.tdVal_[i]<<" "; out<<"\n";
	out<<"THREAD_OFFSET_VAL   = "; for(unsigned int i=0; i<nnPotOpt.toVal_.size(); ++i) out<<nnPotOpt.toVal_[i]<<" "; out<<"\n";
	out<<"THREAD_DIST_TEST    = "; for(unsigned int i=0; i<nnPotOpt.tdTest_.size(); ++i) out<<nnPotOpt.tdTest_[i]<<" "; out<<"\n";
	out<<"THREAD_OFFSET_TEST  = "; for(unsigned int i=0; i<nnPotOpt.toTest_.size(); ++i) out<<nnPotOpt.toTest_[i]<<" "; out<<"\n";
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
		nAtoms_.clear();
		gElement_.clear();
		pElement_.clear();
	//batch
		batch_.clear();
		indices_.clear();
	//nn
		nParams_=0;
		nnpot_.clear();
	//input/output
		nPrint_=0;
		nWrite_=0;
	//parallel
		nThreads_=1;
		tdTrain_.clear();
		toTrain_.clear();
		tdVal_.clear();
		toVal_.clear();
		tdTest_.clear();
		toTest_.clear();
	//error
		opt_count_=0;
		memory_=5;
		error_val_min_=0;
		error_train_min_=0;
		gl_=0;
		pq_=0;
		progress_=0;
		tol_val_=0;
		error_train_vec_.clear();
		error_val_vec_.clear();
		error_train_thread_.clear();
		error_val_thread_.clear();
}

void NNPotOpt::clear(){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::clear():\n";
	//simulation data
		if(strucTrain_!=NULL) strucTrain_->clear(); strucTrain_=NULL;
		if(strucVal_!=NULL) strucVal_->clear(); strucVal_=NULL;
		if(strucTest_!=NULL) strucTest_->clear(); strucTest_=NULL;
	//elements
		nAtoms_.clear();
		gElement_.clear();
		pElement_.clear();
	//batch
		batch_.clear();
		indices_.clear();
	//nn
		nParams_=0;
		nnpot_.clear();
	//parallel
		nThreads_=1;
		tdTrain_.clear();
		toTrain_.clear();
		tdVal_.clear();
		toVal_.clear();
		tdTest_.clear();
		toTest_.clear();
	//error
		opt_count_=0;
		error_val_min_=0;
		error_train_min_=0;
		gl_=0;
		pq_=0;
		progress_=0;
		tol_val_=0;
		error_train_vec_.clear();
		error_val_vec_.clear();
		error_train_thread_.clear();
		error_val_thread_.clear();
}

void NNPotOpt::train(NNPot& nnpot, Opt& opt, unsigned int batchSize){
	if(NN_POT_PRINT_FUNC) std::cout<<"NNPotOpt::train(NNPot&,std::vector<Structure<AtomT> >&,Opt&,unsigned int):\n";
	//====== local function variables ======
	//optimization
		Eigen::VectorXd p;//parameters to be optimized
	//elements
		std::vector<double> an;//atomic number
	//bias
		double avg_out=0,stddev_out=0,max_out=0,min_out=0;
		VecList avg_in;//average of the inputs for each element (nnpot.nSpecies_ x nInput_)
		VecList max_in;//max of the inputs for each element (nnpot.nSpecies_ x nInput_)
		VecList min_in;//min of the inputs for each element (nnpot.nSpecies_ x nInput_)
		VecList stddev_in;//average of the stddev for each element (nnpot.nSpecies_ x nInput_)
	//timing
		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point stop;
		std::chrono::duration<double> time_symm;
		std::chrono::duration<double> time_train;
	//misc
		unsigned int count=0;
	
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Training NN potential...\n";
	
	//====== check the parameters ======
	if(strucTrain_==NULL) throw std::runtime_error("NULL POINTER: no training simulations.");
	else if(strucTrain_->size()==0) throw std::invalid_argument("No training data provided.");
	
	//====== initialize omp ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Initializing OMP...\n";
	nThreads_=1;
	#ifdef _OPENMP
		nThreads_=omp_get_max_threads();
	#endif
	parallel::gen_thread_dist(nThreads_,strucTrain_->size(),tdTrain_);
	parallel::gen_thread_offset(nThreads_,strucTrain_->size(),toTrain_);
	if(strucVal_!=NULL){
		parallel::gen_thread_dist(nThreads_,strucVal_->size(),tdVal_);
		parallel::gen_thread_offset(nThreads_,strucVal_->size(),toVal_);
	}
	if(strucTest_!=NULL){parallel::gen_thread_dist(nThreads_,strucTest_->size(),tdTest_);
		parallel::gen_thread_offset(nThreads_,strucTest_->size(),toTest_);
	}
	error_train_thread_.resize(nThreads_);
	error_val_thread_.resize(nThreads_);
	
	//====== print the optimizer ======
	std::cout<<*this<<"\n";
	
	//====== initialize the species ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Initializing species...\n";
	nnpot.initSpecies(*strucTrain_);
	
	//====== initialize the potential ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Initializing potential...\n";
	nnpot.init();
	
	//====== initialize the symmetry functions ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Initializing symmetry functions...\n";
	nnpot.initSymm(*strucTrain_);
	if(strucVal_!=NULL) nnpot.initSymm(*strucVal_);
	if(strucTest_!=NULL) nnpot.initSymm(*strucTest_);
	
	//====== set the number of atoms of each element ======
	nAtoms_.resize(nnpot.nSpecies());
	for(unsigned int i=0; i<strucTrain_->size(); ++i){
		for(unsigned int j=0; j<(*strucTrain_)[i].nSpecies(); ++j){
			nAtoms_[nnpot.speciesIndex((*strucTrain_)[i].atomNames(j))]+=(*strucTrain_)[i].nAtoms(j);
		}
	}
	std::cout<<"Species - Names   = "; for(unsigned int i=0; i<nnpot.nSpecies(); ++i) std::cout<<nnpot.speciesName(i)<<" "; std::cout<<"\n";
	std::cout<<"Species - Numbers = "; for(unsigned int i=0; i<nAtoms_.size(); ++i) std::cout<<nAtoms_[i]<<" "; std::cout<<"\n";
	
	//====== set the inputs for each of the simulations ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting the inputs (symmetry functions) - training...\n";
	start=std::chrono::high_resolution_clock::now();
	nnpot_.resize(nThreads_,nnpot);
	//for(unsigned int n=0; n<strucTrain_->size(); ++n){std::cout<<"system["<<n<<"]\n"; nnpot.inputs_symm((*strucTrain_)[n],0);}
	#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
	for(unsigned int nt=0; nt<nThreads_; ++nt){
		for(unsigned int n=toTrain_[nt]; n<toTrain_[nt]+tdTrain_[nt]; ++n){
			std::cout<<"system["<<n<<"]\n";
			nnpot_[nt].inputs_symm((*strucTrain_)[n]);
		}
	}
	if(strucVal_!=NULL){
		if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting the inputs (symmetry functions) - validation...\n";
		//for(unsigned int n=0; n<strucVal_->size(); ++n){std::cout<<"system["<<n<<"]\n"; nnpot.inputs_symm((*strucVal_)[n],0);}
		#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
		for(unsigned int nt=0; nt<nThreads_; ++nt){
			for(unsigned int n=toVal_[nt]; n<toVal_[nt]+tdVal_[nt]; ++n){
				std::cout<<"system["<<n<<"]\n";
				nnpot_[nt].inputs_symm((*strucVal_)[n]);
			}
		}
	}
	if(strucTest_!=NULL){
		if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting the inputs (symmetry functions) - testing...\n";
		//for(unsigned int n=0; n<strucTest_->size(); ++n){std::cout<<"system["<<n<<"]\n"; nnpot.inputs_symm((*strucTest_)[n],0);}
		#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
		for(unsigned int nt=0; nt<nThreads_; ++nt){
			for(unsigned int n=toTrain_[nt]; n<toTrain_[nt]+tdTrain_[nt]; ++n){
				std::cout<<"system["<<n<<"]\n";
				nnpot_[nt].inputs_symm((*strucTest_)[n]);
			}
		}
	}
	//print the inputs
	if(NN_POT_PRINT_DATA>0){
		std::cout<<"====================================\n";
		std::cout<<"============== INPUTS ==============\n";
		for(unsigned int n=0; n<strucTrain_->size(); ++n){
			std::cout<<"sim["<<n<<"] = \n";
			for(unsigned int i=0; i<(*strucTrain_)[n].nAtoms(); ++i){
				std::cout<<(*strucTrain_)[n].atom(i).name()<<(*strucTrain_)[n].atom(i).index()+1<<" "<<(*strucTrain_)[n].atom(i).symm().transpose()<<"\n";
			}
		}
		FILE* writer=fopen("nn_pot_inputs.dat","w");
		if(writer!=NULL){
			for(unsigned int n=0; n<strucTrain_->size(); ++n){
				for(unsigned int i=0; i<(*strucTrain_)[n].nAtoms(); ++i){
					fprintf(writer,"%s%i ",(*strucTrain_)[n].atom(i).name().c_str(),(*strucTrain_)[n].atom(i).index()+1);
					for(unsigned int j=0; j<(*strucTrain_)[n].atom(i).symm().size(); ++j){
						fprintf(writer,"%f ",(*strucTrain_)[n].atom(i).symm()[j]);
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
	stop=std::chrono::high_resolution_clock::now();
	time_symm=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
	
	//====== set the indices and batch size ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting indices and batch...\n";
	indices_.resize(strucTrain_->size());
	for(unsigned int i=0; i<indices_.size(); ++i) indices_[i]=i;
	batch_.resize(batchSize,0);
	for(unsigned int i=0; i<batch_.size(); ++i) batch_[i]=i;
	for(unsigned int i=0; i<batch_.size(); ++i) std::cout<<"batch["<<i<<"] = "<<batch_[i]<<"\n";
	
	//====== precondition the input ======
	if(nnpot.preCond()){
		if(NN_POT_PRINT_STATUS>-1) std::cout<<"Pre-conditioning input...\n";
		//local variables
		avg_in.resize(nnpot.nSpecies(),Eigen::VectorXd::Zero(nnpot.nInput_));
		max_in.resize(nnpot.nSpecies(),Eigen::VectorXd::Zero(nnpot.nInput_));
		min_in.resize(nnpot.nSpecies(),Eigen::VectorXd::Zero(nnpot.nInput_));
		stddev_in.resize(nnpot.nSpecies(),Eigen::VectorXd::Zero(nnpot.nInput_));
		std::vector<double> N(nnpot.nSpecies());//total number of inputs for each element 
		//compute the max/min
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Compute the max/min...\n";
		for(unsigned int i=0; i<(*strucTrain_)[0].nSpecies(); ++i){
			//find the index of the current species
			unsigned int index=nnpot.speciesMap_[(*strucTrain_)[0].atomNames(i)];
			//loop over all bases
			for(unsigned int k=0; k<nnpot.nInput_; ++k){
				//find the max and min
				max_in[index][k]=(*strucTrain_)[0].atom(i,0).symm()[k];
				min_in[index][k]=(*strucTrain_)[0].atom(i,0).symm()[k];
			}
		}
		for(unsigned int n=0; n<strucTrain_->size(); ++n){
			//loop over all species
			for(unsigned int i=0; i<(*strucTrain_)[n].nSpecies(); ++i){
				//find the index of the current species
				unsigned int index=nnpot.speciesMap_[(*strucTrain_)[n].atomNames(i)];
				//loop over all atoms of the species
				for(unsigned int j=0; j<(*strucTrain_)[n].nAtoms(i); ++j){
					//loop over all bases
					for(unsigned int k=0; k<nnpot.nInput_; ++k){
						//find the max and min
						if((*strucTrain_)[n].atom(i,j).symm()[k]>max_in[index][k]) max_in[index][k]=(*strucTrain_)[n].atom(i,j).symm()[k];
						if((*strucTrain_)[n].atom(i,j).symm()[k]<min_in[index][k]) min_in[index][k]=(*strucTrain_)[n].atom(i,j).symm()[k];
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
				unsigned int index=nnpot.speciesMap_[(*strucTrain_)[n].atomNames(i)];
				//loop over all atoms of the species
				for(unsigned int j=0; j<(*strucTrain_)[n].nAtoms(i); ++j){
					//add the inputs to the average
					avg_in[index].noalias()+=(*strucTrain_)[n].atom(i,j).symm(); ++N[index];
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
				unsigned int index=nnpot.speciesMap_[(*strucTrain_)[n].atomNames(i)];
				//loop over all atoms of a species
				for(unsigned int j=0; j<(*strucTrain_)[n].nAtoms(i); ++j){
					stddev_in[index].noalias()+=(avg_in[index]-(*strucTrain_)[n].atom(i,j).symm()).cwiseProduct(avg_in[index]-(*strucTrain_)[n].atom(i,j).symm());
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
		nnpot.preBias_=avg_in; for(unsigned int i=0; i<nnpot.preBias_.size(); ++i) nnpot.preBias_[i]*=-1;
		nnpot.preScale_=stddev_in;
		for(unsigned int i=0; i<nnpot.preScale_.size(); ++i){
			for(unsigned int j=0; j<nnpot.preScale_[i].size(); ++j){
				if(nnpot.preScale_[i][j]==0) nnpot.preScale_[i][j]=1;
				else nnpot.preScale_[i][j]=1.0/(3.0*nnpot.preScale_[i][j]+1e-6);
			}
		}
	} else {
		nnpot.preBias_.resize(nnpot.nSpecies(),Eigen::VectorXd::Constant(nnpot.nInput_,0.0));
		nnpot.preScale_.resize(nnpot.nSpecies(),Eigen::VectorXd::Constant(nnpot.nInput_,1.0));
	}
	//====== precondition the output ======
	if(nnpot.postCond()){
		if(NN_POT_PRINT_STATUS>-1) std::cout<<"Post-conditioning output...\n";
		max_out=(*strucTrain_)[0].energy();
		min_out=(*strucTrain_)[0].energy();
		for(unsigned int n=1; n<strucTrain_->size(); ++n){
			if(max_out>(*strucTrain_)[n].energy()) max_out=(*strucTrain_)[n].energy();
			if(min_out<(*strucTrain_)[n].energy()) min_out=(*strucTrain_)[n].energy();
		}
		for(unsigned int n=0; n<strucTrain_->size(); ++n){
			avg_out+=(*strucTrain_)[n].energy();
		}
		avg_out/=strucTrain_->size();
		for(unsigned int n=0; n<strucTrain_->size(); ++n){
			stddev_out+=((*strucTrain_)[n].energy()-avg_out)*((*strucTrain_)[n].energy()-avg_out);
		}
		if(strucTrain_->size()>1) stddev_out=std::sqrt(stddev_out/(strucTrain_->size()-1.0));
		else stddev_out=std::sqrt(stddev_out/strucTrain_->size());
		nnpot.postBias_=avg_out;
		nnpot.postScale_=3*stddev_out;
		if(nnpot.postScale_==0.0) nnpot.postScale_=1.0;
	} else {
		nnpot.postBias_=0.0;
		nnpot.postScale_=1.0;
	}
	
	//====== set the bias for each of the species ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Setting the bias for each species...\n";
	for(unsigned int n=0; n<nnpot.nSpecies(); ++n){
		for(unsigned int i=0; i<nnpot.nn_[n].nInput(); ++i) nnpot.nn_[n].preBias(i)=nnpot.preBias_[n][i];
		for(unsigned int i=0; i<nnpot.nn_[n].nInput(); ++i) nnpot.nn_[n].preScale(i)=nnpot.preScale_[n][i];
		nnpot.nn_[n].postBias(0)=0.0;
		nnpot.nn_[n].postScale(0)=1.0;
	}
	
	//====== initialize the optimization data ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Initializing the optimization data...\n";
	pElement_.resize(nnpot.nSpecies());
	gElement_.resize(nnpot.nSpecies());
	for(unsigned int n=0; n<nnpot.nSpecies(); ++n){
		nnpot.nn_[n]>>pElement_[n];
		nnpot.nn_[n]>>gElement_[n];//this just resizes the gradients
		nParams_+=pElement_[n].size();
	}
	p.resize(nParams_);
	count=0;
	for(unsigned int n=0; n<pElement_.size(); ++n){
		for(unsigned int m=0; m<pElement_[n].size(); ++m){
			p[count++]=pElement_[n][m];
		}
	}
	gElementT_.resize(nThreads_);
	for(unsigned int nt=0; nt<nThreads_; ++nt){
		gElementT_[nt].resize(nnpot.nSpecies());
		for(unsigned int n=0; n<nnpot.nSpecies(); ++n){
			nnpot.nn_[n]>>gElementT_[nt][n];//this just resizes the gradients
		}
	}
	
	//====== print the potential ======
	std::cout<<nnpot<<"\n";
	
	//====== distribute the potential ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Distributing the potential...\n";
	//nnpot_.resize(nThreads_,nnpot);
	for(unsigned int nt=0; nt<nThreads_; ++nt){
		nnpot_[nt]=nnpot;
		std::cout<<"nnpot_["<<nt<<"] = \n"<<nnpot_[nt]<<"\n";
	}
	
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
		for(unsigned int i=0; i<nnpot.preBias_.size(); ++i) std::cout<<"\t"<<nnpot.preBias_[i].transpose()<<"\n";
		std::cout<<"PRE-SCALE       = \n";
		for(unsigned int i=0; i<nnpot.preScale_.size(); ++i) std::cout<<"\t"<<nnpot.preScale_[i].transpose()<<"\n";
		std::cout<<"scaling-output:\n";
		std::cout<<"MAX - OUTPUT    = "<<max_out<<"\n";
		std::cout<<"MIN - OUTPUT    = "<<min_out<<"\n";
		std::cout<<"AVG - OUTPUT    = "<<avg_out<<"\n";
		std::cout<<"STDDEV - OUTPUT = "<<stddev_out<<"\n";
		std::cout<<"POST-BIAS       = "<<nnpot.postBias_<<"\n";
		std::cout<<"POST-SCALE      = "<<nnpot.postScale_<<"\n";
		std::cout<<"******************* OPT - DATA *******************\n";
		std::cout<<"**************************************************\n";
	}
	
	//====== execute the optimization ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Executing the optimization...\n";
	opt_count_=0;
	error_train_vec_.resize(opt.maxIter(),0);
	error_val_vec_.resize(opt.maxIter(),0);
	error_train_min_=std::numeric_limits<double>::max();
	error_val_min_=std::numeric_limits<double>::max();
	start=std::chrono::high_resolution_clock::now();
	opt.opt<NNPotOpt>(func,*this,p);
	stop=std::chrono::high_resolution_clock::now();
	time_train=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
	
	//====== unpack final parameters into element arrays ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Unpacking final parameters into element arrays...\n";
	count=0;
	for(unsigned int n=0; n<pElement_.size(); ++n){
		for(unsigned int m=0; m<pElement_[n].size(); ++m){
			pElement_[n][m]=p[count++];
		}
	}
	
	//====== pack final parameters into neural network ======
	if(NN_POT_PRINT_STATUS>-1) std::cout<<"Packing final parameters into neural network...\n";
	for(unsigned int n=0; n<nnpot.nSpecies(); ++n) nnpot.nn_[n]<<pElement_[n];
	
	if(NN_POT_PRINT_DATA>-1){
		std::cout<<"**************************************************\n";
		std::cout<<"******************* OPT - SUMM *******************\n";
		std::cout<<"N-STEPS    = "<<opt.nStep()<<"\n";
		std::cout<<"OPT-VAL    = "<<opt.val()<<"\n";
		std::cout<<"TIME-SYMM  = "<<time_symm.count()<<"\n";
		std::cout<<"TIME-TRAIN = "<<time_train.count()<<"\n";
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
		Eigen::VectorXd dcda=Eigen::VectorXd::Zero(1);
		Eigen::VectorXd gradLocal=Eigen::VectorXd::Zero(nnpot_[0].nParams_);
	
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
		for(unsigned int n=0; n<nnpot_[nt].nSpecies(); ++n){
			nnpot_[nt].nn_[n]<<pElement_[n];
		}
	}
	
	//====== reset the gradients ======
	if(NN_POT_PRINT_STATUS>1) std::cout<<"Resetting gradients...\n";
	for(unsigned int nt=0; nt<gElementT_.size(); ++nt){
		for(unsigned int i=0; i<gElementT_[nt].size(); ++i) gElementT_[nt][i].setZero();
	}
	for(unsigned int i=0; i<gElement_.size(); ++i) gElement_[i].setZero();
	
	//====== randomize the batch ======
	if(batch_.size()<strucTrain_->size()){
		if(NN_POT_PRINT_STATUS>1) std::cout<<"Randomizing the batch...\n";
		std::random_shuffle(indices_.begin(),indices_.end());
		for(unsigned int i=0; i<batch_.size(); ++i) batch_[i]=indices_[i];
		std::sort(batch_.begin(),batch_.end());
	}
	
	//====== compute training error and gradient ======
	if(NN_POT_PRINT_STATUS>1) std::cout<<"Computing training error and gradient...\n";
	#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads()) firstprivate(dcda,gradLocal)
	for(unsigned int nt=0; nt<nThreads_; ++nt){
		for(unsigned int i=toTrain_[nt]; i<toTrain_[nt]+tdTrain_[nt]; ++i){
			//set the local simulation reference
			Structure<AtomT>& siml=(*strucTrain_)[batch_[i]];
			//compute the energy
			double energy=0;
			for(unsigned int n=0; n<siml.nSpecies(); ++n){
				//find the element index in the nn
				unsigned int index=nnpot_[nt].speciesIndex(siml.atomNames(n));
				//loop over all atoms of the species
				for(unsigned int m=0; m<siml.nAtoms(n); ++m){
					//execute the network
					nnpot_[nt].nn_[index].execute(siml.atom(n,m).symm());
					//add the energy to the total
					energy+=nnpot_[nt].nn_[index].output()[0];
				}
			}
			//scale the energy
			energy=energy*nnpot_[nt].postScale_+nnpot_[nt].postBias_;
			//add to the error
			error_train_thread_[nt]+=0.5*(energy-siml.energy())*(energy-siml.energy());
			//compute the gradient
			dcda[0]=(energy-siml.energy());
			//compute the gradients
			for(unsigned int n=0; n<siml.nSpecies(); ++n){
				//find the element index in the nn
				unsigned int index=nnpot_[nt].speciesIndex(siml.atomNames(n));
				for(unsigned int m=0; m<siml.nAtoms(n); ++m){
					//execute the network
					nnpot_[nt].nn_[index].execute(siml.atom(n,m).symm());
					//compute the gradient
					nnpot_[nt].nn_[index].grad(dcda,gradLocal);
					//add the gradient to the total
					gElementT_[nt][index].noalias()+=gradLocal;
				}
			}
		}
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
		#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
		for(unsigned int nt=0; nt<nThreads_; ++nt){
			for(unsigned int i=toVal_[nt]; i<toVal_[nt]+tdVal_[nt]; ++i){
				//set the local simulation reference
				Structure<AtomT>& siml=(*strucVal_)[i];
				//compute the energy
				double energy=0;
				for(unsigned int n=0; n<siml.nSpecies(); ++n){
					//find the element index in the nn
					unsigned int index=nnpot_[nt].speciesIndex(siml.atomNames(n));
					//loop over all atoms of the species
					for(unsigned int m=0; m<siml.nAtoms(n); ++m){
						//execute the network
						nnpot_[nt].nn_[index].execute(siml.atom(n,m).symm());
						//add the energy to the total
						energy+=nnpot_[nt].nn_[index].output()[0];
					}
				}
				//scale the energy
				energy=energy*nnpot_[0].postScale_+nnpot_[0].postBias_;
				//add to the error
				error_val_thread_[nt]+=0.5*(energy-siml.energy())*(energy-siml.energy());
			}
		}
		//consolidate the error
		error_val=0;
		for(unsigned int nt=0; nt<nThreads_; ++nt) error_val+=error_val_thread_[nt];
		//normalize the error
		error_val/=(strucVal_->size()>0)?strucVal_->size():1;
	}
	
	//====== print energy ======
	if(NN_POT_PRINT_DATA>0 && opt_count_%nPrint_==0){
		std::vector<double> energyExact((*strucTrain_).size(),0);
		std::vector<double> energyNN((*strucTrain_).size(),0);
		std::cout<<"============== ENERGIES ==============\n";
		for(unsigned int i=0; i<(*strucTrain_).size(); ++i){
			Structure<AtomT>& siml=(*strucTrain_)[i];
			energyExact[i]=siml.energy();
			double energy=0;
			for(unsigned int n=0; n<siml.nSpecies(); ++n){
				//find the nn index
				unsigned int index=nnpot_[0].speciesIndex(siml.atomNames(n));
				//calculate the taomic energies
				for(unsigned int m=0; m<siml.nAtoms(n); ++m){
					//execute the network
					nnpot_[0].nn_[index].execute(siml.atom(n,m).symm());
					//add the energy to the total
					energy+=nnpot_[0].nn_[index].output()[0];
				}
			}
			energyNN[i]=energy*nnpot_[0].postScale_+nnpot_[0].postBias_;
			std::cout<<"sim"<<i<<" "<<energyExact[i]<<" "<<energyNN[i]<<" "<<0.5*(energyNN[i]-energyExact[i])*(energyNN[i]-energyExact[i])<<"\n";
		}
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
	error_train_vec_[opt_count_]=error_train;
	error_val_vec_[opt_count_]=error_val;
	
	//====== caclulate the validation stopping criterion ======
	error_val_min_=(error_val_min_>error_val)?error_val:error_val_min_;
	error_train_min_=(error_train_min_>error_train)?error_train:error_train_min_;
	if(strucVal_!=NULL){
		if(NN_POT_PRINT_STATUS>1) std::cout<<"Computing the progress quotient...\n";
		//calculate the generalization loss
		gl_=100*(error_val/error_val_min_-1.0);
		//calculate the sum over the recent training errors
		unsigned int beg=0;
		if(opt_count_>=memory_) beg=opt_count_-memory_;
		double error_train_sum=0;
		for(unsigned int i=beg; i<opt_count_; ++i){
			error_train_sum+=error_train_vec_[i];
		}
		//calculate the training progress
		progress_=1000*(error_train_sum/(memory_*error_train_min_)-1.0);
		//calculate the progress quotient
		pq_=gl_/progress_;
	}
	
	//====== print the optimization data ======
	if(opt_count_%nPrint_==0) std::cout<<"opt "<<opt_count_<<" error_train "<<error_train<<" error_val "<<error_val<<" gl "<<gl_<<" pr "<<progress_<<" pq "<<pq_<<"\n";
	//if(opt_count_%nPrint_==0) std::cout<<"error_train_min_ "<<error_train_min_<<" error_val_min_ "<<error_val_min_<<"\n";
	
	//====== increment the opt count ======
	++opt_count_;
	
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
				UNKOWN
			};
		};
		MODE::type mode=MODE::TRAIN;
	//nn potential
		NNPot nnPot;
	//nn potential - opt
		Opt opt;//optimization class
		NNPotOpt nnPotOpt;//nn potential optimization data
		unsigned int nBatch=0;//size of training batch - number
		double pBatch=0;//size of training batch - percent
	//simulations
		unsigned int nSpecies=0;//number of atomic species
		std::vector<std::string> speciesNames;//names of atomic species
		std::vector<std::string> data_train;//file for training data
		std::vector<std::string> data_val;//file for validation data
		std::vector<std::string> data_test;//file for test data
		std::vector<std::string> files_train;//training data files
		std::vector<std::string> files_val;//validation data files
		std::vector<std::string> files_test;//test data files
		std::vector<Structure<AtomT> > struc_train;//training data
		std::vector<Structure<AtomT> > struc_val;//validation data
		std::vector<Structure<AtomT> > struc_test;//test data
	//linear regression
		double m=0,b=0,r2;
	//timing
		std::chrono::high_resolution_clock::time_point start;//starting time
		std::chrono::high_resolution_clock::time_point stop;//stopping time
		std::chrono::duration<double> time_energy_train;//time to calculate the energies of the training set
		std::chrono::duration<double> time_force_train;//time to calculate the forces of the training set
		std::chrono::duration<double> time_energy_val;//time to calculate the energies of the validation set
		std::chrono::duration<double> time_force_val;//time to calculate the forces of the validation set
		std::chrono::duration<double> time_energy_test;//time to calculate the energies of the test set
		std::chrono::duration<double> time_force_test;//time to calculate the forces of the test set
	//random
		int seed=-1;//seed for random number generator (negative => current time)
	//file i/o
		FILE* reader=NULL;
		FILE* writer=NULL;
		char* paramfile=(char*)malloc(sizeof(char)*string::M);
		char* datafile=(char*)malloc(sizeof(char)*string::M);
		char* input=(char*)malloc(sizeof(char)*string::M);
		char* temp=(char*)malloc(sizeof(char)*string::M);
		
	try{
		//======== check the arguments ========
		if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
		
		//======== load the parameter file ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Loading parameter file...\n";
		std::strcpy(paramfile,argv[1]);
		
		//======== open the parameter file ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Opening parameter file...\n";
		reader=fopen(paramfile,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Could not open parameter file.");
		
		//======== read in the parameters ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Reading in parameters...\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);
			string::trim_all(string::to_upper(string::copy_left(temp,input,"=")));
			//data and execution mode
			if(std::strcmp(temp,"MODE")==0){
				string::to_upper(string::trim(std::strcpy(temp,std::strpbrk(input,"=")+1)));
				if(std::strcmp(temp,"TRAIN")==0) mode=MODE::TRAIN;
				else if(std::strcmp(temp,"TEST")==0) mode=MODE::TEST;	
				else throw std::invalid_argument("Invalid mode.");
			} else if(std::strcmp(temp,"DATA_TRAIN")==0){
				data_train.push_back(std::string(string::trim(std::strcpy(temp,std::strpbrk(input,"=")+1))));
			} else if(std::strcmp(temp,"DATA_VAL")==0){
				data_val.push_back(std::string(string::trim(std::strcpy(temp,std::strpbrk(input,"=")+1))));
			} else if(std::strcmp(temp,"DATA_TEST")==0){
				data_test.push_back(std::string(string::trim(std::strcpy(temp,std::strpbrk(input,"=")+1))));
			}
			//optimization
			if(std::strcmp(temp,"ALGO")==0){
				opt.algo()=OPT_METHOD::load(string::to_upper(string::trim(std::strcpy(temp,std::strpbrk(input,"=")+1))));
			} else if(std::strcmp(temp,"GAMMA")==0){
				opt.gamma()=std::atof(std::strpbrk(input,"=")+1);
				if(opt.gamma()<=0) throw std::invalid_argument("Invalid descent parameter.");
			} else if(std::strcmp(temp,"ETA")==0){
				opt.eta()=std::atof(std::strpbrk(input,"=")+1);
				if(opt.eta()<=0) throw std::invalid_argument("Invalid descent parameter.");
			} else if(std::strcmp(temp,"PERIOD")==0){
				opt.period()=std::atoi(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"OPT_VAL")==0){
				opt.optVal()=OPT_VAL::load(string::to_upper(string::trim(std::strcpy(temp,std::strpbrk(input,"=")+1))));
			} else if(std::strcmp(temp,"TOL")==0){
				opt.tol()=std::atof(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"MAX_ITER")==0){
				opt.maxIter()=std::atof(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"N_PRINT")==0){
				opt.nPrint()=std::atoi(std::strpbrk(input,"=")+1);
			}
			//neural network potential
			if(std::strcmp(temp,"NR")==0){//number of radial basis functions
				nnPot.nR()=std::atoi(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"NA")==0){//number of angular basis functions
				nnPot.nA()=std::atoi(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"PHIRN")==0){//type of radial basis functions
				nnPot.phiRN()=PhiRN::load(string::to_upper(string::trim_all(std::strpbrk(input,"=")+1)));
			} else if(std::strcmp(temp,"PHIAN")==0){//type of angular basis functions
				nnPot.phiAN()=PhiAN::load(string::to_upper(string::trim_all(std::strpbrk(input,"=")+1)));
			} else if(std::strcmp(temp,"R_CUT")==0){//distance cutoff
				nnPot.rc()=std::atof(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"R_MIN")==0){//distance cutoff
				nnPot.rm()=std::atof(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"CUTOFF")==0){//type of cutoff function
				nnPot.tcut()=CutoffN::load(string::to_upper(string::trim(std::strcpy(temp,std::strpbrk(input,"=")+1))));
			} else if(std::strcmp(temp,"LAMBDA")==0){//regularization parameter
				nnPot.lambda()=std::atof(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"N_HIDDEN")==0){//number of hidden layers
				unsigned int nl=string::substrN(std::strpbrk(input,"=")+1,string::WS);
				if(nl==0) throw std::invalid_argument("Invalid hidden layer configuration.");
				std::vector<unsigned int> nh(nl);
				std::strtok(input,"=");
				for(unsigned int i=0; i<nl; ++i){
					nh[i]=std::atoi(std::strtok(NULL,string::WS));
					if(nh[i]==0) throw std::invalid_argument("Invalid hidden layer configuration.");
				}
				nnPot.nh()=nh;
			} else if(std::strcmp(temp,"TRANSFER")==0){//transfer function
				nnPot.tfType()=NN::TransferN::load(string::to_upper(string::trim(std::strcpy(temp,std::strpbrk(input,"=")+1))));
			} else if(std::strcmp(temp,"PRE_COND")==0){//whether to precondition the inputs
				nnPot.preCond()=string::boolean(string::trim_all(std::strpbrk(input,"=")+1));
			} else if(std::strcmp(temp,"POST_COND")==0){//whether to precondition the outputs
				nnPot.postCond()=string::boolean(string::trim_all(std::strpbrk(input,"=")+1));
			}else if(std::strcmp(temp,"N_BATCH")==0){//size of the batch
				nBatch=std::atoi(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"P_BATCH")==0){//batch percentage
				pBatch=std::atof(std::strpbrk(input,"=")+1);
			} 
			//neural network potential optimization
			if(std::strcmp(temp,"N_PRINT")==0){
				nnPotOpt.nPrint_=std::atoi(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"N_WRITE")==0){
				nnPotOpt.nWrite_=std::atoi(std::strpbrk(input,"=")+1);
			}
			//general
			if(std::strcmp(temp,"SEED")==0){
				seed=std::atoi(std::strpbrk(input,"=")+1);
			}
		}
		
		//======== close parameter file ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Closing parameter file...\n";
		fclose(reader);
		reader=NULL;
		
		//======== print parameters ========
		std::cout<<"**************************************************\n";
		std::cout<<"*************** GENERAL PARAMETERS ***************\n";
		std::cout<<"\tSEED       = "<<seed<<"\n";
		if(mode==MODE::TRAIN) std::cout<<"\tMODE       = TRAIN\n";
		else if(mode==MODE::TEST) std::cout<<"\tMODE       = TEST\n";
		std::cout<<"\tP_BATCH    = "<<pBatch<<"\n";
		std::cout<<"\tN_BATCH    = "<<nBatch<<"\n";
		std::cout<<"\tDATA_TRAIN = \n"; for(unsigned int i=0; i<data_train.size(); ++i) std::cout<<"\t\t"<<data_train[i]<<"\n";
		std::cout<<"\tDATA_VAL   = \n"; for(unsigned int i=0; i<data_val.size(); ++i) std::cout<<"\t\t"<<data_val[i]<<"\n";
		std::cout<<"\tDATA_TEST  = \n"; for(unsigned int i=0; i<data_test.size(); ++i) std::cout<<"\t\t"<<data_test[i]<<"\n";
		std::cout<<"*************** GENERAL PARAMETERS ***************\n";
		std::cout<<"**************************************************\n";
		std::cout<<opt<<"\n";
		
		//======== initialize the random number generator ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Initializing random number generator...\n";
		if(seed<0) std::srand(std::time(NULL));
		else std::srand(seed);
		
		//======== check the parameters ========
		if(mode==MODE::TRAIN && data_train.size()==0) throw std::invalid_argument("No training data provided.");
		else if(mode==MODE::TEST && data_test.size()==0) throw std::invalid_argument("No test data provided.");
		if(pBatch<0 || pBatch>1) throw std::invalid_argument("Invalid batch size.");
		
		//======== load the trainin data ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Loading training data...\n";
		for(unsigned i=0; i<data_train.size(); ++i){
			//open the data file
			if(NN_POT_TRAIN_DEBUG>1) std::cout<<"Data file "<<i<<": "<<data_train[i]<<"\n";
			reader=fopen(data_train[i].c_str(),"r");
			if(reader==NULL) throw std::runtime_error("I/O Error: Could not open data file.");
			//read in the data
			while(fgets(input,string::M,reader)!=NULL){
				if(!string::empty(input)) files_train.push_back(std::string(string::trim(input)));
			}
			//close the file
			fclose(reader);
			reader=NULL;
		}
		//======== load the test data ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Loading test data...\n";
		for(unsigned int i=0; i<data_test.size(); ++i){
			//open the data file
			if(NN_POT_TRAIN_DEBUG>1) std::cout<<"Data file "<<i<<": "<<data_test[i]<<"\n";
			reader=fopen(data_test[i].c_str(),"r");
			if(reader==NULL) throw std::runtime_error("I/O Error: Could not open data file.");
			//read in the data
			while(fgets(input,string::M,reader)!=NULL){
				if(!string::empty(input)) files_test.push_back(std::string(string::trim(input)));
			}
			//close the file
			fclose(reader);
			reader=NULL;
		}
		//======== load the training data ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Loading validation data...\n";
		for(unsigned int i=0; i<data_val.size(); ++i){
			//open the data file
			if(NN_POT_TRAIN_DEBUG>1) std::cout<<"Data file "<<i<<": "<<data_val[i]<<"\n";
			reader=fopen(data_val[i].c_str(),"r");
			if(reader==NULL) throw std::runtime_error("I/O Error: Could not open data file.");
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
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Loading training simulations...\n";
		if(files_train.size()>0){
			struc_train.resize(files_train.size());
			for(unsigned int i=0; i<files_train.size(); ++i) VASP::XML::load(files_train[i].c_str(),struc_train[i]);
			if(NN_POT_TRAIN_DEBUG>1) for(unsigned int i=0; i<files_train.size(); ++i) std::cout<<"\t"<<files_train[i]<<" "<<struc_train[i].energy()<<"\n";
		}
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Loading validation simulations...\n";
		if(files_val.size()>0){
			struc_val.resize(files_val.size());
			for(unsigned int i=0; i<files_val.size(); ++i) VASP::XML::load(files_val[i].c_str(),struc_val[i]);
			if(NN_POT_TRAIN_DEBUG>1) for(unsigned int i=0; i<files_val.size(); ++i) std::cout<<"\t"<<files_val[i]<<" "<<struc_val[i].energy()<<"\n";
		}
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Loading testing simulations...\n";
		if(files_test.size()>0){
			struc_test.resize(files_test.size());
			for(unsigned int i=0; i<files_test.size(); ++i) VASP::XML::load(files_test[i].c_str(),struc_test[i]);
			if(NN_POT_TRAIN_DEBUG>1) for(unsigned int i=0; i<files_test.size(); ++i) std::cout<<"\t"<<files_test[i]<<" "<<struc_test[i].energy()<<"\n";
		}
		
		//======== find the total number of species and names ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Determining species number and names...\n";
		nSpecies=0; speciesNames.clear();
		for(unsigned int i=0; i<files_train.size(); ++i){
			for(unsigned int n=0; n<struc_train[i].nSpecies(); ++n){
				bool match=false;
				for(unsigned int m=0; m<nSpecies; ++m){
					if(struc_train[i].atomNames(n)==speciesNames[m]){
						match=true; break;
					}
				}
				if(!match){
					speciesNames.push_back(struc_train[i].atomNames(n));
					++nSpecies;
				}
			}
		}
		if(NN_POT_TRAIN_DEBUG>0){
			std::cout<<"N_SPECIES = "<<nSpecies<<"\n";
			std::cout<<"SPECIES_NAMES = "; for(unsigned int i=0; i<speciesNames.size(); ++i) std::cout<<speciesNames[i]<<" "; std::cout<<"\n";
		}
		
		//======== set the batch size ========
		if(pBatch>0) nBatch=std::floor(pBatch*struc_train.size());
		if(nBatch==0) throw std::invalid_argument("Invalid batch size.");
		if(nBatch>struc_train.size()) throw std::invalid_argument("Invalid batch size.");
		
		//======== set the data ========
		if(struc_train.size()>0) nnPotOpt.strucTrain_=&struc_train;
		if(struc_val.size()>0) nnPotOpt.strucVal_=&struc_val;
		if(struc_test.size()>0) nnPotOpt.strucTest_=&struc_test;
		
		//======== train the nn potential ========
		if(NN_POT_TRAIN_DEBUG>0) std::cout<<"Training the nn potential...\n";
		nnPotOpt.train(nnPot,opt,nBatch);
		
		//======== calculate the final energies - training set ========
		std::cout<<"Final energies - training set ... \n";
		std::vector<double> energy_exact_train(struc_train.size(),0);
		std::vector<double> energy_nn_train(struc_train.size(),0);
		std::vector<double> error_energy_train(struc_train.size(),0);
		std::vector<double> error_energy_p_train(struc_train.size(),0);
		start=std::chrono::high_resolution_clock::now();
		for(unsigned int n=0; n<struc_train.size(); ++n){
			energy_nn_train[n]=nnPot.energy(struc_train[n]);
			energy_exact_train[n]=struc_train[n].energy();
			error_energy_train[n]=std::fabs(energy_nn_train[n]-energy_exact_train[n]);
			error_energy_p_train[n]=std::fabs((energy_exact_train[n]-energy_nn_train[n])/energy_exact_train[n])*100.0;
		}
		stop=std::chrono::high_resolution_clock::now();
		time_energy_train=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		
		//======== calculate the final energies - validation set ========
		std::vector<double> energy_exact_val(struc_val.size(),0);
		std::vector<double> energy_nn_val(struc_val.size(),0);
		std::vector<double> error_energy_val(struc_val.size(),0);
		std::vector<double> error_energy_p_val(struc_val.size(),0);
		if(struc_val.size()>0){
			std::cout<<"Final energies - validation set ... \n";
			start=std::chrono::high_resolution_clock::now();
			for(unsigned int n=0; n<struc_val.size(); ++n){
				energy_nn_val[n]=nnPot.energy(struc_val[n]);
				energy_exact_val[n]=struc_val[n].energy();
				error_energy_val[n]=std::fabs(energy_nn_val[n]-energy_exact_val[n]);
				error_energy_p_val[n]=std::fabs((energy_exact_val[n]-energy_nn_val[n])/energy_exact_val[n])*100.0;
			}
			stop=std::chrono::high_resolution_clock::now();
			time_energy_val=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		}
		
		//======== calculate the final energies - test set ========
		std::vector<double> energy_exact_test(struc_test.size(),0);
		std::vector<double> energy_nn_test(struc_test.size(),0);
		std::vector<double> error_energy_test(struc_test.size(),0);
		std::vector<double> error_energy_p_test(struc_test.size(),0);
		if(struc_test.size()>0){
			std::cout<<"Final energies - test set ... \n";
			start=std::chrono::high_resolution_clock::now();
			for(unsigned int n=0; n<struc_test.size(); ++n){
				energy_nn_test[n]=nnPot.energy(struc_test[n]);
				energy_exact_test[n]=struc_test[n].energy();
				error_energy_test[n]=std::fabs(energy_nn_test[n]-energy_exact_test[n]);
				error_energy_p_test[n]=std::fabs((energy_exact_test[n]-energy_nn_test[n])/energy_exact_test[n])*100.0;
			}
			stop=std::chrono::high_resolution_clock::now();
			time_energy_test=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		}
		
		//======== calculate the final forces - training set ========
		std::cout<<"Calculating final forces - training set...\n";
		std::vector<std::vector<Eigen::Vector3d> > forces_exact_train(struc_train.size());
		std::vector<std::vector<Eigen::Vector3d> > forces_nn_train(struc_train.size());
		std::vector<double> error_force_train(struc_train.size(),0);
		std::vector<double> error_force_p_train(struc_train.size(),0);
		start=std::chrono::high_resolution_clock::now();
		#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
		for(unsigned int nt=0; nt<nnPotOpt.nThreads_; ++nt){
			for(unsigned int i=nnPotOpt.toTrain_[nt]; i<nnPotOpt.toTrain_[nt]+nnPotOpt.tdTrain_[nt]; ++i){
				std::cout<<"system-train["<<i<<"]\n";
				forces_exact_train[i].resize(struc_train[i].nAtoms());
				forces_nn_train[i].resize(struc_train[i].nAtoms());
				for(unsigned int j=0; j<struc_train[i].nAtoms(); ++j) forces_exact_train[i][j]=struc_train[i].atom(j).force();
				nnPotOpt.nnpot_[nt].forces(struc_train[i]);
				for(unsigned int j=0; j<struc_train[i].nAtoms(); ++j) forces_nn_train[i][j]=struc_train[i].atom(j).force();
			}
		}
		for(unsigned int i=0; i<struc_train.size(); ++i){
			for(unsigned int j=0; j<struc_train[i].nAtoms(); ++j){
				error_force_train[i]+=(forces_exact_train[i][j]-forces_nn_train[i][j]).squaredNorm();
				error_force_p_train[i]+=(forces_exact_train[i][j]-forces_nn_train[i][j]).squaredNorm()/forces_exact_train[i][j].squaredNorm();
			}
			error_force_train[i]=std::sqrt(error_force_train[i])/struc_train[i].nAtoms();
			error_force_p_train[i]=std::sqrt(error_force_p_train[i])/struc_train[i].nAtoms();
		}
		stop=std::chrono::high_resolution_clock::now();
		time_force_train=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		
		//======== calculate the final forces - validation set ========
		std::vector<std::vector<Eigen::Vector3d> > forces_exact_val(struc_val.size());
		std::vector<std::vector<Eigen::Vector3d> > forces_nn_val(struc_val.size());
		std::vector<double> error_force_val(struc_val.size(),0);
		std::vector<double> error_force_p_val(struc_val.size(),0);
		if(struc_val.size()>0){
			std::cout<<"Calculating final forces - validation set...\n";
			start=std::chrono::high_resolution_clock::now();
			#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
			for(unsigned int nt=0; nt<nnPotOpt.nThreads_; ++nt){
				for(unsigned int i=nnPotOpt.toVal_[nt]; i<nnPotOpt.toVal_[nt]+nnPotOpt.tdVal_[nt]; ++i){
					std::cout<<"system-train["<<i<<"]\n";
					forces_exact_val[i].resize(struc_val[i].nAtoms());
					forces_nn_val[i].resize(struc_val[i].nAtoms());
					for(unsigned int j=0; j<struc_val[i].nAtoms(); ++j) forces_exact_val[i][j]=struc_val[i].atom(j).force();
					nnPotOpt.nnpot_[nt].forces(struc_val[i]);
					for(unsigned int j=0; j<struc_val[i].nAtoms(); ++j) forces_nn_val[i][j]=struc_val[i].atom(j).force();
				}
			}
			for(unsigned int i=0; i<struc_val.size(); ++i){
				for(unsigned int j=0; j<struc_val[i].nAtoms(); ++j){
					error_force_val[i]+=(forces_exact_val[i][j]-forces_nn_val[i][j]).squaredNorm();
					error_force_p_val[i]+=(forces_exact_val[i][j]-forces_nn_val[i][j]).squaredNorm()/forces_exact_val[i][j].squaredNorm();
				}
				error_force_val[i]=std::sqrt(error_force_val[i])/struc_val[i].nAtoms();
				error_force_p_val[i]=std::sqrt(error_force_p_val[i])/struc_val[i].nAtoms();
			}
			stop=std::chrono::high_resolution_clock::now();
			time_force_train=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		}
		
		//======== calculate the final forces - test set ========
		std::vector<std::vector<Eigen::Vector3d> > forces_exact_test(struc_test.size());
		std::vector<std::vector<Eigen::Vector3d> > forces_nn_test(struc_test.size());
		std::vector<double> error_force_test(struc_test.size(),0);
		std::vector<double> error_force_p_test(struc_test.size(),0);
		if(struc_test.size()>0){
			std::cout<<"Calculating final forces - test set...\n";
			start=std::chrono::high_resolution_clock::now();
			#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
			for(unsigned int nt=0; nt<nnPotOpt.nThreads_; ++nt){
				for(unsigned int i=nnPotOpt.toTest_[nt]; i<nnPotOpt.toTest_[nt]+nnPotOpt.tdTest_[nt]; ++i){
					std::cout<<"system-train["<<i<<"]\n";
					forces_exact_test[i].resize(struc_test[i].nAtoms());
					forces_nn_test[i].resize(struc_test[i].nAtoms());
					for(unsigned int j=0; j<struc_test[i].nAtoms(); ++j) forces_exact_test[i][j]=struc_test[i].atom(j).force();
					nnPotOpt.nnpot_[nt].forces(struc_test[i]);
					for(unsigned int j=0; j<struc_test[i].nAtoms(); ++j) forces_nn_test[i][j]=struc_test[i].atom(j).force();
				}
			}
			for(unsigned int i=0; i<struc_val.size(); ++i){
				for(unsigned int j=0; j<struc_val[i].nAtoms(); ++j){
					error_force_test[i]+=(forces_exact_test[i][j]-forces_nn_test[i][j]).squaredNorm();
					error_force_p_test[i]+=(forces_exact_test[i][j]-forces_nn_test[i][j]).squaredNorm()/forces_exact_test[i][j].squaredNorm();
				}
				error_force_test[i]=std::sqrt(error_force_test[i])/struc_test[i].nAtoms();
				error_force_p_test[i]=std::sqrt(error_force_p_test[i])/struc_test[i].nAtoms();
			}
			stop=std::chrono::high_resolution_clock::now();
			time_force_test=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		}
		
		//======== print the timing info ========
		std::cout<<"**************************************************\n";
		std::cout<<"********************* TIMING *********************\n";
		std::cout<<"time - energy - train = "<<time_energy_train.count()<<"\n";
		std::cout<<"time - force  - train = "<<time_force_train.count()<<"\n";
		std::cout<<"time - energy - val   = "<<time_energy_val.count()<<"\n";
		std::cout<<"time - force  - val   = "<<time_force_val.count()<<"\n";
		std::cout<<"time - energy - test  = "<<time_energy_test.count()<<"\n";
		std::cout<<"time - force  - test  = "<<time_force_test.count()<<"\n";
		std::cout<<"********************* TIMING *********************\n";
		std::cout<<"**************************************************\n";
		
		//======== print the error statistics - training ========
		{
		std::cout<<"**************************************************\n";
		std::cout<<"********* STATISTICS - ERROR - TRAINING *********\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tAVG        = "<<stats::average(error_energy_train)<<" "<<stats::average(error_energy_p_train)<<"\n";
		std::cout<<"\tSTDDEV     = "<<stats::stddev(error_energy_train)<<" "<<stats::stddev(error_energy_p_train)<<"\n";
		std::cout<<"\tMAX        = "<<stats::max(error_energy_train)<<" "<<stats::max(error_energy_p_train)<<"\n";
		r2=stats::lin_reg(energy_exact_train,energy_nn_train,m,b);
		std::cout<<"\tM          = "<<m<<"\n";
		std::cout<<"\tR2         = "<<r2<<"\n";
		std::cout<<"FORCE:\n";
		std::cout<<"\tAVG        = "<<stats::average(error_force_train)<<" "<<stats::average(error_force_p_train)<<"\n";
		std::cout<<"\tSTDDEV     = "<<stats::stddev(error_force_train)<<" "<<stats::stddev(error_force_p_train)<<"\n";
		std::cout<<"\tMAX        = "<<stats::max(error_force_train)<<" "<<stats::max(error_force_p_train)<<"\n";
		double mx,my,mz,r2x,r2y,r2z;
		unsigned int nfTot=0,count;
		for(unsigned int i=0; i<struc_train.size(); ++i) nfTot+=struc_train[i].nAtoms();
		std::vector<double> fc_exact(nfTot),fc_nn(nfTot);
		count=0; for(int i=0; i<struc_train.size(); ++i) for(int j=0; j<struc_train[i].nAtoms(); ++j) fc_exact[count++]=forces_exact_train[i][j][0];
		count=0; for(int i=0; i<struc_train.size(); ++i) for(int j=0; j<struc_train[i].nAtoms(); ++j) fc_nn[count++]=forces_nn_train[i][j][0];
		r2x=stats::lin_reg(fc_exact,fc_nn,mx,b);
		count=0; for(int i=0; i<struc_train.size(); ++i) for(int j=0; j<struc_train[i].nAtoms(); ++j) fc_exact[count++]=forces_exact_train[i][j][1];
		count=0; for(int i=0; i<struc_train.size(); ++i) for(int j=0; j<struc_train[i].nAtoms(); ++j) fc_nn[count++]=forces_nn_train[i][j][1];
		r2y=stats::lin_reg(fc_exact,fc_nn,my,b);
		count=0; for(int i=0; i<struc_train.size(); ++i) for(int j=0; j<struc_train[i].nAtoms(); ++j) fc_exact[count++]=forces_exact_train[i][j][2];
		count=0; for(int i=0; i<struc_train.size(); ++i) for(int j=0; j<struc_train[i].nAtoms(); ++j) fc_nn[count++]=forces_nn_train[i][j][2];
		r2z=stats::lin_reg(fc_exact,fc_nn,mz,b);
		std::cout<<"\t(MX,MY,MZ) = ("<<mx<<","<<my<<","<<mz<<")\n";
		std::cout<<"\t(RX,RY,RZ) = ("<<r2x<<","<<r2y<<","<<r2z<<")\n";
		std::cout<<"********* STATISTICS - ERROR - TRAINING *********\n";
		std::cout<<"**************************************************\n";
		}
		
		//======== print the error statistics - validation ========
		if(struc_val.size()>0){
		std::cout<<"**************************************************\n";
		std::cout<<"******** STATISTICS - ERROR - VALIDATION ********\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tAVG        = "<<stats::average(error_energy_val)<<" "<<stats::average(error_energy_p_val)<<"\n";
		std::cout<<"\tSTDDEV     = "<<stats::stddev(error_energy_val)<<" "<<stats::stddev(error_energy_p_val)<<"\n";
		std::cout<<"\tMAX        = "<<stats::max(error_energy_val)<<" "<<stats::max(error_energy_p_val)<<"\n";
		r2=stats::lin_reg(energy_exact_val,energy_nn_val,m,b);
		std::cout<<"\tM          = "<<m<<"\n";
		std::cout<<"\tR2         = "<<r2<<"\n";
		std::cout<<"FORCE:\n";
		std::cout<<"\tAVG        = "<<stats::average(error_force_val)<<" "<<stats::average(error_force_p_val)<<"\n";
		std::cout<<"\tSTDDEV     = "<<stats::stddev(error_force_val)<<" "<<stats::stddev(error_force_p_val)<<"\n";
		std::cout<<"\tMAX        = "<<stats::max(error_force_val)<<" "<<stats::max(error_force_p_val)<<"\n";
		double mx,my,mz,r2x,r2y,r2z;
		unsigned int nfTot=0,count;
		for(unsigned int i=0; i<struc_val.size(); ++i) nfTot+=struc_val[i].nAtoms();
		std::vector<double> fc_exact(nfTot),fc_nn(nfTot);
		count=0; for(int i=0; i<struc_val.size(); ++i) for(int j=0; j<struc_val[i].nAtoms(); ++j) fc_exact[count++]=forces_exact_val[i][j][0];
		count=0; for(int i=0; i<struc_val.size(); ++i) for(int j=0; j<struc_val[i].nAtoms(); ++j) fc_nn[count++]=forces_nn_val[i][j][0];
		r2x=stats::lin_reg(fc_exact,fc_nn,mx,b);
		count=0; for(int i=0; i<struc_val.size(); ++i) for(int j=0; j<struc_val[i].nAtoms(); ++j) fc_exact[count++]=forces_exact_val[i][j][1];
		count=0; for(int i=0; i<struc_val.size(); ++i) for(int j=0; j<struc_val[i].nAtoms(); ++j) fc_nn[count++]=forces_nn_val[i][j][1];
		r2y=stats::lin_reg(fc_exact,fc_nn,my,b);
		count=0; for(int i=0; i<struc_val.size(); ++i) for(int j=0; j<struc_val[i].nAtoms(); ++j) fc_exact[count++]=forces_exact_val[i][j][2];
		count=0; for(int i=0; i<struc_val.size(); ++i) for(int j=0; j<struc_val[i].nAtoms(); ++j) fc_nn[count++]=forces_nn_val[i][j][2];
		r2z=stats::lin_reg(fc_exact,fc_nn,mz,b);
		std::cout<<"\t(MX,MY,MZ) = ("<<mx<<","<<my<<","<<mz<<")\n";
		std::cout<<"\t(RX,RY,RZ) = ("<<r2x<<","<<r2y<<","<<r2z<<")\n";
		std::cout<<"******** STATISTICS - ERROR - VALIDATION ********\n";
		std::cout<<"**************************************************\n";
		}
		
		//======== print the energies - training set ========
		std::cout<<"Printing the energies - training set...\n";
		writer=fopen("nn_pot_energy_train.dat","w");
		if(writer!=NULL){
			fprintf(writer,"sim energy_exact energy_nn\n");
			for(unsigned int i=0; i<struc_train.size(); ++i) fprintf(writer,"%s %f %f\n",files_train[i].c_str(),energy_exact_train[i],energy_nn_train[i]);
			fclose(writer); writer=NULL;
		} else std::cout<<"WARNING: Could not open: \"nn_pot_energy_train.dat\"\n";
		
		//======== print the energies - validation set ========
		if(struc_val.size()>0){
			std::cout<<"Printing the energies - validation set...\n";
			writer=fopen("nn_pot_energy_val.dat","w");
			if(writer!=NULL){
				fprintf(writer,"sim energy_exact energy_nn\n");
				for(unsigned int i=0; i<struc_val.size(); ++i) fprintf(writer,"%s %f %f\n",files_val[i].c_str(),energy_exact_val[i],energy_nn_val[i]);
				fclose(writer); writer=NULL;
			} else std::cout<<"WARNING: Could not open: \"nn_pot_energy_val.dat\"\n";
		}
		
		//======== print the forces ========
		std::cout<<"Printing the forces - training set...\n";
		writer=fopen("nn_pot_force_train.dat","w");
		if(writer!=NULL){
			fprintf(writer,"sim atom f_exact_x f_exact_y f_exact_z f_nn_x f_nn_y f_nn_z\n");
			for(unsigned int i=0; i<struc_train.size(); ++i){
				for(unsigned int j=0; j<struc_train[i].nAtoms(); ++j){
					fprintf(writer,"%s %s%i %f %f %f %f %f %f\n",files_train[i].c_str(),struc_train[i].atom(j).name().c_str(),struc_train[i].atom(j).index()+1,
						forces_exact_train[i][j][0],forces_exact_train[i][j][1],forces_exact_train[i][j][2],
						forces_nn_train[i][j][0],forces_nn_train[i][j][1],forces_nn_train[i][j][2]
					);
				}
			}
		} else std::cout<<"WARNING: Could not open: \"nn_pot_force_train.dat\"\n";
		
		//======== print the forces ========
		if(struc_val.size()>0){
			std::cout<<"Printing the forces - validation set...\n";
			writer=fopen("nn_pot_force_val.dat","w");
			if(writer!=NULL){
				fprintf(writer,"sim atom f_exact_x f_exact_y f_exact_z f_nn_x f_nn_y f_nn_z\n");
				for(unsigned int i=0; i<struc_val.size(); ++i){
					for(unsigned int j=0; j<struc_val[i].nAtoms(); ++j){
						fprintf(writer,"%s %s%i %f %f %f %f %f %f\n",files_val[i].c_str(),struc_val[i].atom(j).name().c_str(),struc_val[i].atom(j).index()+1,
							forces_exact_val[i][j][0],forces_exact_val[i][j][1],forces_exact_val[i][j][2],
							forces_nn_val[i][j][0],forces_nn_val[i][j][1],forces_nn_val[i][j][2]
						);
					}
				}
			} else std::cout<<"WARNING: Could not open: \"nn_pot_force_val.dat\"\n";
		}
		
		//======== print the energies ========
		std::cout<<"Printing the error...\n";
		writer=fopen("nn_pot_error.dat","w");
		if(writer!=NULL){
			fprintf(writer,"iteration error_train error_val\n");
			for(unsigned int i=0; i<opt.maxIter(); ++i){
				fprintf(writer,"%i %f %f\n",i,nnPotOpt.error_train_vec_[i],nnPotOpt.error_val_vec_[i]);
			}
			fclose(writer); writer=NULL;
		} else std::cout<<"WARNING: Could not open: \"nn_pot_error.dat\"\n";
		
		//======== print the nn's ========
		std::cout<<"Printing the nn's...\n";
		for(unsigned int i=0; i<nnPot.nn().size(); ++i){
			std::string filename="nn_";
			filename+=nnPot.speciesName(i);
			filename+=".dat";
			std::cout<<"filename = "<<filename<<"\n";
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