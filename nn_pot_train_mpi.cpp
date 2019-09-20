#include "nn_pot_train_mpi.hpp"

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const NNPotOpt& obj){
	if(NN_POT_TRAIN_PRINT_FUNC>0) std::cout<<"nbytes(const NNPotOpt&)\n";
	unsigned int size=0;
	//elements
		size+=sizeof(unsigned int);//nElements
	//optimization
		size+=nbytes(obj.data_);
		switch(obj.data_.algo()){
			case Opt::ALGO::SGD: size+=nbytes(static_cast<const Opt::SGD&>(*obj.model_)); break;
			case Opt::ALGO::SDM: size+=nbytes(static_cast<const Opt::SDM&>(*obj.model_)); break;
			case Opt::ALGO::NAG: size+=nbytes(static_cast<const Opt::NAG&>(*obj.model_)); break;
			case Opt::ALGO::ADAGRAD: size+=nbytes(static_cast<const Opt::ADAGRAD&>(*obj.model_)); break;
			case Opt::ALGO::ADADELTA: size+=nbytes(static_cast<const Opt::ADADELTA&>(*obj.model_)); break;
			case Opt::ALGO::RMSPROP: size+=nbytes(static_cast<const Opt::RMSPROP&>(*obj.model_)); break;
			case Opt::ALGO::ADAM: size+=nbytes(static_cast<const Opt::ADAM&>(*obj.model_)); break;
			case Opt::ALGO::NADAM: size+=nbytes(static_cast<const Opt::NADAM&>(*obj.model_)); break;
			case Opt::ALGO::BFGS: size+=nbytes(static_cast<const Opt::BFGS&>(*obj.model_)); break;
			case Opt::ALGO::RPROP: size+=nbytes(static_cast<const Opt::RPROP&>(*obj.model_)); break;
			default: throw std::runtime_error("Invalid optimization method."); break;
		}
	//nn
		size+=sizeof(bool);//pre-conditioning
		size+=nbytes(obj.nnpot_);
	//return the size
		return size;
}

//**********************************************
// packing
//**********************************************

template <> void pack(const NNPotOpt& obj, char* arr){
	if(NN_POT_TRAIN_PRINT_FUNC>0) std::cout<<"pack(const NNPotOpt&,char*)\n";
	unsigned int pos=0;
	//elements
		std::memcpy(arr+pos,&obj.nElements_,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	//optimization
		pack(obj.data_,arr+pos); pos+=nbytes(obj.data_);
		switch(obj.data_.algo()){
			case Opt::ALGO::SGD:
				pack(static_cast<const Opt::SGD&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::SGD&>(*obj.model_));
			break;
			case Opt::ALGO::SDM:
				pack(static_cast<const Opt::SDM&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::SDM&>(*obj.model_));
			break;
			case Opt::ALGO::NAG:
				pack(static_cast<const Opt::NAG&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::NAG&>(*obj.model_));
			break;
			case Opt::ALGO::ADAGRAD:
				pack(static_cast<const Opt::ADAGRAD&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::ADAGRAD&>(*obj.model_));
			break;
			case Opt::ALGO::ADADELTA:
				pack(static_cast<const Opt::ADADELTA&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::ADADELTA&>(*obj.model_));
			break;
			case Opt::ALGO::RMSPROP:
				pack(static_cast<const Opt::RMSPROP&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::RMSPROP&>(*obj.model_));
			break;
			case Opt::ALGO::ADAM:
				pack(static_cast<const Opt::ADAM&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::ADAM&>(*obj.model_));
			break;
			case Opt::ALGO::NADAM:
				pack(static_cast<const Opt::NADAM&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::NADAM&>(*obj.model_));
			break;
			case Opt::ALGO::BFGS:
				pack(static_cast<const Opt::BFGS&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::BFGS&>(*obj.model_));
			break;
			case Opt::ALGO::RPROP:
				pack(static_cast<const Opt::RPROP&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::RPROP&>(*obj.model_));
			break;
			default:
				throw std::runtime_error("Invalid optimization method.");
			break;
		}
	//nn
		std::memcpy(arr+pos,&obj.preCond_,sizeof(bool)); pos+=sizeof(bool);
		pack(obj.nnpot_,arr+pos); pos+=nbytes(obj.nnpot_);
}

//**********************************************
// unpacking
//**********************************************

template <> void unpack(NNPotOpt& obj, const char* arr){
	if(NN_POT_TRAIN_PRINT_FUNC>0) std::cout<<"unpack(const NNPotOpt&,char*)\n";
	unsigned int pos=0;
	//elements
		std::memcpy(&obj.nElements_,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	//optimization
		unpack(obj.data_,arr+pos); pos+=nbytes(obj.data_);
		switch(obj.data_.algo()){
			case Opt::ALGO::SGD:
				obj.model_.reset(new Opt::SGD());
				unpack(static_cast<Opt::SGD&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::SGD&>(*obj.model_));
			break;
			case Opt::ALGO::SDM:
				obj.model_.reset(new Opt::SDM());
				unpack(static_cast<Opt::SDM&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::SDM&>(*obj.model_));
			break;
			case Opt::ALGO::NAG:
				obj.model_.reset(new Opt::NAG());
				unpack(static_cast<Opt::NAG&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::NAG&>(*obj.model_));
			break;
			case Opt::ALGO::ADAGRAD:
				obj.model_.reset(new Opt::ADAGRAD());
				unpack(static_cast<Opt::ADAGRAD&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::ADAGRAD&>(*obj.model_));
			break;
			case Opt::ALGO::ADADELTA:
				obj.model_.reset(new Opt::ADADELTA());
				unpack(static_cast<Opt::ADADELTA&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::ADADELTA&>(*obj.model_));
			break;
			case Opt::ALGO::RMSPROP:
				obj.model_.reset(new Opt::RMSPROP());
				unpack(static_cast<Opt::RMSPROP&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::RMSPROP&>(*obj.model_));
			break;
			case Opt::ALGO::ADAM:
				obj.model_.reset(new Opt::ADAM());
				unpack(static_cast<Opt::ADAM&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::ADAM&>(*obj.model_));
			break;
			case Opt::ALGO::NADAM:
				obj.model_.reset(new Opt::NADAM());
				unpack(static_cast<Opt::NADAM&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::NADAM&>(*obj.model_));
			break;
			case Opt::ALGO::BFGS:
				obj.model_.reset(new Opt::BFGS());
				unpack(static_cast<Opt::BFGS&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::BFGS&>(*obj.model_));
			break;
			case Opt::ALGO::RPROP:
				obj.model_.reset(new Opt::RPROP());
				unpack(static_cast<Opt::RPROP&>(*obj.model_),arr+pos);
				pos+=nbytes(static_cast<const Opt::RPROP&>(*obj.model_));
			break;
			default:
				throw std::runtime_error("Invalid optimization method.");
			break;
		}
	//nn
		std::memcpy(&obj.preCond_,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		unpack(obj.nnpot_,arr+pos); pos+=nbytes(obj.nnpot_);
}
	
}

bool compare_pair(const std::pair<unsigned int,double>& p1, const std::pair<unsigned int,double>& p2){
	return p1.first<p2.first;
}

//************************************************************
// NNPotOpt - Neural Network Potential - Optimization
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPotOpt& nnPotOpt){
	out<<"**************************************************\n";
	out<<"****************** NN - POT - OPT ****************\n";
	out<<"P_BATCH      = "<<nnPotOpt.pBatch_<<"\n";
	out<<"N_BATCH      = "<<nnPotOpt.nBatch_<<"\n";
	out<<"CHARGE       = "<<nnPotOpt.charge_<<"\n";
	out<<"CALC_FORCE   = "<<nnPotOpt.calcForce_<<"\n";
	out<<"RESTART      = "<<nnPotOpt.restart_<<"\n";
	out<<"RESTART_FILE = "<<nnPotOpt.restart_file_<<"\n";
	out<<"ATOMS        = \n";
	for(unsigned int i=0; i<nnPotOpt.atoms_.size(); ++i){
		std::cout<<"\t"<<nnPotOpt.atoms_[i]<<"\n";
	}
	out<<"BASIS_RADIAL_READ = \n";
	for(unsigned int i=0; i<nnPotOpt.atoms_basis_radial_.size(); ++i){
		std::cout<<"\t"<<nnPotOpt.atoms_basis_radial_[i]<<" "<<nnPotOpt.files_basis_radial_[i]<<"\n";
	}
	out<<"BASIS_ANGULAR_READ = \n";
	for(unsigned int i=0; i<nnPotOpt.atoms_basis_angular_.size(); ++i){
		std::cout<<"\t"<<nnPotOpt.atoms_basis_angular_[i].first<<" "<<nnPotOpt.atoms_basis_angular_[i].second
			<<" "<<nnPotOpt.files_basis_angular_[i]<<"\n";
	}
	out<<"****************** NN - POT - OPT ****************\n";
	out<<"**************************************************";
	return out;
}

NNPotOpt::NNPotOpt(){
	if(NN_POT_TRAIN_PRINT_FUNC>0) std::cout<<"NNPot::NNPotOpt():\n";
	strucTrain_=NULL;
	strucVal_=NULL;
	strucTest_=NULL;
	writer_error_=NULL;
	defaults();
};

void NNPotOpt::defaults(){
	if(NN_POT_TRAIN_PRINT_FUNC>0) std::cout<<"NNPot::defaults():\n";
	//simulation data
		if(strucTrain_!=NULL) strucTrain_->clear();
		if(strucVal_!=NULL) strucVal_->clear(); 
		if(strucTest_!=NULL) strucTest_->clear(); 
		strucTrain_=NULL;
		strucVal_=NULL;
		strucTest_=NULL;
		nTrain_=0;
		nVal_=0;
		nTest_=0;
	//elements
		atoms_.clear();
		nElements_=0;
		nAtoms_.clear();
		gElement_.clear();
		pElement_.clear();
		gTemp_.clear();
		gTempSum_.clear();
	//batch
		nBatch_=0;
		pBatch_=1.0;
		batch_.clear();
		indices_.clear();
	//nn
		nParams_=0;
		nnpot_.clear();
		preBias_.clear();
		preScale_.clear();
		preCond_=false;
		charge_=false;
	//input/output
		calcForce_=true;
		restart_=false;
		restart_file_="nn_pot_train";
		atoms_basis_radial_.clear();
		atoms_basis_angular_.clear();
		files_basis_radial_.clear();
		files_basis_angular_.clear();
	//optimization
		//model_.reset(new Model());
		identity_=Eigen::VectorXd::Identity(1,1);
	//error
		error_train_=0;
		error_val_=0;
		error_lambda_=0;
	//file i/o
		if(writer_error_!=NULL) fclose(writer_error_);
		writer_error_=NULL;
		file_error_=std::string("nn_pot_error.dat");
}

void NNPotOpt::clear(){
	if(NN_POT_TRAIN_PRINT_FUNC>0) std::cout<<"NNPot::clear():\n";
	//simulation data
		if(strucTrain_!=NULL) strucTrain_->clear();
		if(strucVal_!=NULL) strucVal_->clear(); 
		if(strucTest_!=NULL) strucTest_->clear(); 
		strucTrain_=NULL;
		strucVal_=NULL;
		strucTest_=NULL;
		nTrain_=0;
		nVal_=0;
		nTest_=0;
	//elements
		atoms_.clear();
		nElements_=0;
		nAtoms_.clear();
		gElement_.clear();
		pElement_.clear();
		gTemp_.clear();
		gTempSum_.clear();
	//batch
		batch_.clear();
		indices_.clear();
	//nn
		nParams_=0;
		nnpot_.clear();
		preBias_.clear();
		preScale_.clear();
	//input/output
		atoms_basis_radial_.clear();
		atoms_basis_angular_.clear();
		files_basis_radial_.clear();
		files_basis_angular_.clear();
	//optimization
		//model_.reset(new Opt());
		data_.clear();
		identity_=Eigen::VectorXd::Identity(1,1);
	//error
		error_train_=0;
		error_val_=0;
		error_lambda_=0;
	//file i/o
		if(writer_error_!=NULL) fclose(writer_error_);
		writer_error_=NULL;
}

void NNPotOpt::write_restart(const char* file){
	if(NN_POT_TRAIN_PRINT_FUNC>1) std::cout<<"NNPotOpt::write_restart(const char*):\n";
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
	if(NN_POT_TRAIN_PRINT_FUNC>0) std::cout<<"NNPotOpt::read_restart(const char*):\n";
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
	if(NN_POT_TRAIN_PRINT_FUNC>0) std::cout<<"NNPotOpt::train(NNPot&,std::vector<Structure>&,unsigned int):\n";
	//====== local function variables ======
	//bias
		VecList avg_in;//average of the inputs for each element (nnpot_.nSpecies_ x nInput_)
		VecList max_in;//max of the inputs for each element (nnpot_.nSpecies_ x nInput_)
		VecList min_in;//min of the inputs for each element (nnpot_.nSpecies_ x nInput_)
		VecList dev_in;//average of the stddev for each element (nnpot_.nSpecies_ x nInput_)
	//timing
		clock_t start,stop;
		double time_train;
	//mpi
		int nprocs=1,rank=0;
		MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
		MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	//misc
		unsigned int count=0;
	
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"training NN potential\n";
	
	//====== check the parameters ======
	if(strucTrain_==NULL) throw std::runtime_error("NULL POINTER: no training structures.");
	else if(strucTrain_->size()==0) throw std::invalid_argument("No training data provided.");
	if(strucVal_==NULL) throw std::runtime_error("NULL POINTER: no validation structures.");
	else if(strucVal_->size()==0) throw std::invalid_argument("No validation data provided.");
	
	//====== compute the number of atoms of each element ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"computing the number of atoms of each element\n";
	if(nElements_==0) nElements_=nnpot_.nAtoms();
	else if(nElements_!=nnpot_.nAtoms()) throw std::invalid_argument("Invalid number of elements in the potential.");
	//compute the number of atoms in each structure
	nAtoms_.resize(nElements_);
	for(unsigned int i=0; i<nElements_; ++i) nAtoms_[i]=0;
	for(unsigned int i=0; i<strucTrain_->size(); ++i){
		for(unsigned int j=0; j<(*strucTrain_)[i].nSpecies(); ++j){
			nAtoms_[nnpot_.atom_index((*strucTrain_)[i].atomNames(j))]+=(*strucTrain_)[i].nAtoms(j);
		}
	}
	//consolidate and bcast total number of atoms
	std::vector<unsigned int> tempv(nElements_);
	MPI_Allreduce(nAtoms_.data(),tempv.data(),nElements_,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	nAtoms_=tempv; tempv.clear();
	if(NN_POT_TRAIN_PRINT_DATA>-1 && rank==0){
		std::cout<<"Atoms - Names   = "; for(unsigned int i=0; i<nnpot_.nAtoms(); ++i) std::cout<<nnpot_.atom(i).name()<<" "; std::cout<<"\n";
		std::cout<<"Atoms - Numbers = "; for(unsigned int i=0; i<nAtoms_.size(); ++i) std::cout<<nAtoms_[i]<<" "; std::cout<<"\n";
	}
	
	//====== set the indices and batch size ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"setting indices and batch\n";
	//indices
	indices_.resize(strucTrain_->size());
	for(int i=indices_.size()-1; i>=0; --i) indices_[i]=i;
	//batch
	batch_.resize(batchSize,0);
	for(int i=batch_.size()-1; i>=0; --i) batch_[i]=i;
	
	//====== collect input statistics ======
	//local variables
	std::vector<unsigned int> N(nnpot_.nAtoms());//total number of inputs for each element 
	avg_in.resize(nElements_);
	max_in.resize(nElements_);
	min_in.resize(nElements_);
	dev_in.resize(nElements_);
	for(unsigned int n=0; n<nElements_; ++n){
		avg_in[n]=Eigen::VectorXd::Zero(nnpot_.nInput(n));
		max_in[n]=Eigen::VectorXd::Zero(nnpot_.nInput(n));
		min_in[n]=Eigen::VectorXd::Zero(nnpot_.nInput(n));
		dev_in[n]=Eigen::VectorXd::Zero(nnpot_.nInput(n));
	}
	//compute the max/min
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"compute the max/min\n";
	for(unsigned int i=0; i<(*strucTrain_)[0].nSpecies(); ++i){
		//find the index of the current species
		const unsigned int index=nnpot_.atomMap_[string::hash((*strucTrain_)[0].atomNames(i))];
		//loop over all bases
		for(unsigned int k=0; k<nnpot_.nInput(i); ++k){
			//find the max and min
			max_in[index][k]=(*strucTrain_)[0].symm(i,0)[k];
			min_in[index][k]=(*strucTrain_)[0].symm(i,0)[k];
		}
	}
	for(unsigned int n=0; n<strucTrain_->size(); ++n){
		const Structure& strucl=(*strucTrain_)[n];
		//loop over all species
		for(unsigned int i=0; i<strucl.nSpecies(); ++i){
			//find the index of the current species
			const unsigned int index=nnpot_.atomMap_[string::hash(strucl.atomNames(i))];
			//loop over all atoms of the species
			for(unsigned int j=0; j<strucl.nAtoms(i); ++j){
				//loop over all bases
				for(unsigned int k=0; k<nnpot_.nInput(i); ++k){
					//find the max and min
					if(strucl.symm(i,j)[k]>max_in[index][k]) max_in[index][k]=strucl.symm(i,j)[k];
					if(strucl.symm(i,j)[k]<min_in[index][k]) min_in[index][k]=strucl.symm(i,j)[k];
				}
			}
		}
	}
	//accumulate the min/max
	for(unsigned int i=0; i<min_in.size(); ++i){
		for(unsigned int j=0; j<min_in[i].size(); ++j){
			double temp=0;
			MPI_Allreduce(&min_in[i][j],&temp,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
			min_in[i][j]=temp;
		}
	}
	for(unsigned int i=0; i<max_in.size(); ++i){
		for(unsigned int j=0; j<max_in[i].size(); ++j){
			double temp=0;
			MPI_Allreduce(&max_in[i][j],&temp,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
			max_in[i][j]=temp;
		}
	}
	//compute the average - loop over all structures
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"compute the average\n";
	for(unsigned int n=0; n<strucTrain_->size(); ++n){
		const Structure& strucl=(*strucTrain_)[n];
		//loop over all species
		for(unsigned int i=0; i<strucl.nSpecies(); ++i){
			//find the index of the current species
			const unsigned int index=nnpot_.atomMap_[string::hash(strucl.atomNames(i))];
			//loop over all atoms of the species
			for(unsigned int j=0; j<strucl.nAtoms(i); ++j){
				//add the inputs to the average
				avg_in[index].noalias()+=strucl.symm(i,j);
				//increment the count
				++N[index];
			}
		}
	}
	//accumulate the number
	for(unsigned int i=0; i<avg_in.size(); ++i){
		unsigned int Ntot=0;
		MPI_Allreduce(&N[i],&Ntot,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
		N[i]=Ntot;
	}
	//accumulate the average
	for(unsigned int i=0; i<avg_in.size(); ++i){
		for(unsigned int j=0; j<avg_in[i].size(); ++j){
			double temp=0;
			MPI_Allreduce(&avg_in[i][j],&temp,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			avg_in[i][j]=temp/N[i];
		}
	}
	//compute the stddev - loop over all structures
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"compute the stddev\n";
	for(unsigned int n=0; n<strucTrain_->size(); ++n){
		const Structure& strucl=(*strucTrain_)[n];
		//loop over all species
		for(unsigned int i=0; i<strucl.nSpecies(); ++i){
			//find the index of the current species
			unsigned int index=nnpot_.atomMap_[string::hash(strucl.atomNames(i))];
			//loop over all atoms of a species
			for(unsigned int j=0; j<strucl.nAtoms(i); ++j){
				dev_in[index].noalias()+=(avg_in[index]-strucl.symm(i,j)).cwiseProduct(avg_in[index]-strucl.symm(i,j));
			}
		}
	}
	//accumulate the stddev
	for(unsigned int i=0; i<dev_in.size(); ++i){
		for(unsigned int j=0; j<dev_in[i].size(); ++j){
			double temp=0;
			MPI_Allreduce(&dev_in[i][j],&temp,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			dev_in[i][j]=std::sqrt(temp/(N[i]-1.0));
		}
	}
	
	//====== precondition the input ======
	if(preCond_){
		if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"pre-conditioning input\n";
		//set the preconditioning vectors
		preBias_=avg_in; for(unsigned int i=0; i<preBias_.size(); ++i) preBias_[i]*=-1;
		preScale_=dev_in;
		for(unsigned int i=0; i<preScale_.size(); ++i){
			for(unsigned int j=0; j<preScale_[i].size(); ++j){
				if(preScale_[i][j]==0) preScale_[i][j]=1;
				else preScale_[i][j]=1.0/(3.0*preScale_[i][j]+1e-6);
			}
		}
	} else {
		preBias_.resize(nElements_);
		preScale_.resize(nElements_);
		for(unsigned int n=0; n<nElements_; ++n){
			preBias_[n]=Eigen::VectorXd::Constant(nnpot_.nInput(n),0.0);
			preScale_[n]=Eigen::VectorXd::Constant(nnpot_.nInput(n),1.0);
		}
	}
	
	//====== set the bias for each of the species ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"setting the bias for each species\n";
	for(unsigned int n=0; n<nElements_; ++n){
		for(unsigned int i=0; i<nnpot_.nn(n).nInput(); ++i) nnpot_.nn(n).preBias(i)=preBias_[n][i];
		for(unsigned int i=0; i<nnpot_.nn(n).nInput(); ++i) nnpot_.nn(n).preScale(i)=preScale_[n][i];
		nnpot_.nn(n).postBias(0)=0.0;
		nnpot_.nn(n).postScale(0)=1.0;
	}
	
	//====== initialize the optimization data ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"initializing the optimization data\n";
	//set parameters for each element
	pElement_.resize(nElements_);
	gElement_.resize(nElements_);
	gTemp_.resize(nElements_);
	gTempSum_.resize(nElements_);
	for(unsigned int n=0; n<nElements_; ++n){
		nnpot_.nn(n)>>pElement_[n];//resizes vector and sets values
		nnpot_.nn(n)>>gElement_[n];//resizes vector
		nnpot_.nn(n)>>gTemp_[n];   //resizes vector
		nnpot_.nn(n)>>gTempSum_[n];//resizes vector
		gElement_[n]=Eigen::VectorXd::Random(gElement_[n].size())*1e-6;
		nParams_+=pElement_[n].size();
	}
	//set initial parameters
	if(restart_){
		//restart
		if(rank==0) std::cout<<"restarting optimization\n";
		if(nParams_!=data_.dim()) throw std::runtime_error(
			std::string("Network has ")+std::to_string(nParams_)+std::string(" while opt has ")
			+std::to_string(data_.dim())+std::string(" parameters.")
		);
	} else {
		//from scratch
		if(rank==0) std::cout<<"starting from scratch\n";
		//resize the optimization objects
		data_.init(nParams_);
		model_->init(nParams_);
		//set initial values
		count=0;
		for(unsigned int n=0; n<pElement_.size(); ++n){
			for(unsigned int m=0; m<pElement_[n].size(); ++m){
				data_.p()[count]=pElement_[n][m];
				data_.g()[count]=gElement_[n][m];
				++count;
			}
		}
	}
	
	//====== print the potential ======
	if(rank==0) std::cout<<nnpot_<<"\n";
	
	//====== print optimization data ======
	if(NN_POT_TRAIN_PRINT_DATA>-1 && rank==0){
		std::cout<<"**************************************************\n";
		std::cout<<"******************* OPT - DATA *******************\n";
		std::cout<<"N-PARAMS    = \n"<<nParams_<<"\n";
		std::cout<<"AVG - INPUT = \n"; for(unsigned int i=0; i<avg_in.size(); ++i) std::cout<<"\t"<<avg_in[i].transpose()<<"\n";
		std::cout<<"MAX - INPUT = \n"; for(unsigned int i=0; i<max_in.size(); ++i) std::cout<<"\t"<<max_in[i].transpose()<<"\n";
		std::cout<<"MIN - INPUT = \n"; for(unsigned int i=0; i<min_in.size(); ++i) std::cout<<"\t"<<min_in[i].transpose()<<"\n";
		std::cout<<"DEV - INPUT = \n"; for(unsigned int i=0; i<dev_in.size(); ++i) std::cout<<"\t"<<dev_in[i].transpose()<<"\n";
		std::cout<<"PRE-BIAS    = \n"; for(unsigned int i=0; i<preBias_.size(); ++i) std::cout<<"\t"<<preBias_[i].transpose()<<"\n";
		std::cout<<"PRE-SCALE   = \n"; for(unsigned int i=0; i<preScale_.size(); ++i) std::cout<<"\t"<<preScale_[i].transpose()<<"\n";
		std::cout<<"******************* OPT - DATA *******************\n";
		std::cout<<"**************************************************\n";
	}
	
	//====== execute the optimization ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"executing the optimization\n";
	//open the error file
	if(rank==0){
		if(!restart_){
			writer_error_=fopen(file_error_.c_str(),"w");
			fprintf(writer_error_,"#STEP ERROR_RMS_TRAIN ERROR_RMS_VAL\n");
		} else {
			writer_error_=fopen(file_error_.c_str(),"a");
		}
		if(writer_error_==NULL) throw std::runtime_error("I/O Error: Could not open error record file.");
	}
	//optimization variables
	bool fbreak=false;
	identity_.resize(1); identity_[0]=1;
	//compute the total number of training structures in the batch
	unsigned int nTrainBatch_=0;
	MPI_Reduce(&nBatch_,&nTrainBatch_,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	//bcast parameters
	MPI_Bcast(data_.p().data(),data_.p().size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
	//train the nn potential
	start=std::clock();
	for(unsigned int iter=0; iter<data_.max(); ++iter){
		//double tempf;
		double error_train_sum_=0,error_val_sum_=0;
		//compute the value and gradient
		error(data_.p());
		for(int n=nElements_-1; n>=0; --n) gTemp_[n].setZero();
		//accumulate error
		MPI_Reduce(&error_train_,&error_train_sum_,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&error_val_,&error_val_sum_,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		//accumulate gradient
		for(int n=nElements_-1; n>=0; --n){
			MPI_Reduce(gElement_[n].data(),gTemp_[n].data(),gElement_[n].size(),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		}
		if(rank==0){
			//compute error
			error_train_=error_train_sum_/nTrainBatch_;
			error_val_=error_val_sum_/nVal_;
			//compute gradient
			for(int n=nElements_-1; n>=0; --n) gElement_[n]=gTemp_[n];
			//compute regularization error
			if(NN_POT_TRAIN_PRINT_STATUS>1) std::cout<<"computing regularization error\n";
			error_lambda_=0;
			for(int n=nElements_-1; n>=0; --n){
				if(nnpot_.nn(n).lambda()>0) error_lambda_+=nnpot_.nn(n).error_lambda()/nTrainBatch_;
			}
			//compute regularization gradient
			if(NN_POT_TRAIN_PRINT_STATUS>1) std::cout<<"computing regularization gradient\n";
			for(int n=nElements_-1; n>=0; --n){
				if(nnpot_.nn(n).lambda()>0) gElement_[n].noalias()+=nnpot_.nn(n).grad_lambda(gTemp_[n]);
			}
			//pack the gradient
			count=0;
			for(unsigned int n=0; n<nElements_; ++n){
				gElement_[n]/=nTrainBatch_;
				for(unsigned int i=0; i<gElement_[n].size(); ++i){
					data_.g()[count++]=gElement_[n][i];
				}
			}
			//print error
			if(data_.step()%data_.nPrint()==0) printf("opt %8i err_t %12.10f err_v %12.10f\n",
				data_.step(),std::sqrt(2.0*error_train_),std::sqrt(2.0*error_val_));
			//write error
			if(data_.step()%data_.nPrint()==0) fprintf(writer_error_,"%6i %12.10f %12.10f\n",
				data_.step(),std::sqrt(2.0*error_train_),std::sqrt(2.0*error_val_));
			//write the basis and potentials
			if(data_.step()%data_.nWrite()==0){
				if(NN_POT_TRAIN_PRINT_STATUS>1) std::cout<<"writing the restart file and potentials\n";
				nnpot_.tail()=std::string(".")+std::to_string(data_.step());
				nnpot_.write();
				std::string file=restart_file_+".restart."+std::to_string(data_.step());
				this->write_restart(file.c_str());
			}
			//compute the new position
			data_.val()=error_train_+error_lambda_;
			model_->step(data_);
			//compute the difference
			data_.dv()=std::fabs(data_.val()-data_.valOld());
			data_.dp()=(data_.p()-data_.pOld()).norm();
			//set the new "old" values
			data_.valOld()=data_.val();//set "old" value
			data_.pOld()=data_.p();//set "old" p value
			data_.gOld()=data_.g();//set "old" g value
			//check the break condition
			switch(data_.optVal()){
				case Opt::VAL::FTOL_REL: if(data_.dv()<data_.tol()) fbreak=true; break;
				case Opt::VAL::XTOL_REL: if(data_.dp()<data_.tol()) fbreak=true; break;
				case Opt::VAL::FTOL_ABS: if(data_.val()<data_.tol()) fbreak=true; break;
			}
		}
		//bcast parameters
		MPI_Bcast(data_.p().data(),data_.p().size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
		//bcast break condition
		MPI_Bcast(&fbreak,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		if(fbreak) break;
		//increment step
		++data_.step();
	}
	stop=std::clock();
	time_train=((double)(stop-start))/CLOCKS_PER_SEC;
	//close the error file
	if(rank==0){
		fclose(writer_error_);
		writer_error_=NULL;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	//====== unpack final parameters into element arrays ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"unpacking final parameters into element arrays\n";
	count=0;
	for(unsigned int n=0; n<nElements_; ++n){
		for(unsigned int m=0; m<pElement_[n].size(); ++m){
			pElement_[n][m]=data_.p()[count];
			gElement_[n][m]=data_.g()[count];
			++count;
		}
	}
	
	//====== pack final parameters into neural network ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"packing final parameters into neural network\n";
	for(unsigned int n=0; n<nElements_; ++n) nnpot_.nn_[n]<<pElement_[n];
	
	if(NN_POT_TRAIN_PRINT_DATA>-1 && rank==0){
		std::cout<<"**************************************************\n";
		std::cout<<"******************* OPT - SUMM *******************\n";
		std::cout<<"N-STEPS    = "<<data_.step()<<"\n";
		std::cout<<"OPT-VAL    = "<<data_.val()<<"\n";
		std::cout<<"TIME-TRAIN = "<<time_train<<"\n";
		if(NN_POT_TRAIN_PRINT_DATA>1){
			std::cout<<"p = "; for(int i=0; i<data_.p().size(); ++i) std::cout<<data_.p()[i]<<" "; std::cout<<"\n";
		}
		std::cout<<"******************* OPT - SUMM *******************\n";
		std::cout<<"**************************************************\n";
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

double NNPotOpt::error(const Eigen::VectorXd& x){
	if(NN_POT_TRAIN_PRINT_FUNC>0) std::cout<<"NNPotOpt::error(const Eigen::VectorXd&):\n";
	//====== local variables ======
	//mpi
		int rank=0;
		MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	//number of atoms
		std::vector<unsigned int> nAtomsl(nElements_);
	
	//====== reset the error ======
	error_train_=0;
	
	//====== unpack total parameters into element arrays ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"unpacking total parameters into element arrays\n";
	unsigned int count=0;
	for(unsigned int n=0; n<pElement_.size(); ++n){
		for(unsigned int m=0; m<pElement_[n].size(); ++m){
			pElement_[n][m]=x[count++];
		}
	}
	
	//====== unpack arrays into element nn's ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"unpacking arrays into element nn's\n";
	for(int n=nElements_-1; n>=0; --n) nnpot_.nn_[n]<<pElement_[n];
	
	//====== reset the gradients ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"resetting gradients\n";
	for(int n=nElements_-1; n>=0; --n) gElement_[n].setZero();
	
	//====== randomize the batch ======
	if(batch_.size()<strucTrain_->size()){
		//randomize the batch
		if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"randomizing the batch\n";
		std::random_shuffle(indices_.begin(),indices_.end());
		for(unsigned int i=0; i<batch_.size(); ++i) batch_[i]=indices_[i];
		std::sort(batch_.begin(),batch_.end());
		if(NN_POT_TRAIN_PRINT_DATA>1 && rank==0){for(unsigned int i=0; i<batch_.size(); ++i) std::cout<<"batch["<<i<<"] = "<<batch_[i]<<"\n";}
	}
	
	//====== compute training error and gradient ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"computing training error and gradient\n";
	for(unsigned int i=0; i<batch_.size(); ++i){
		for(int j=nElements_-1; j>=0; --j) gTempSum_[j].setZero();
		for(int j=nElements_-1; j>=0; --j) nAtomsl[j]=0;
		//set the local simulation reference
		const Structure& strucl=(*strucTrain_)[batch_[i]];
		//compute the energy
		double energy=0;
		for(int n=strucl.nAtoms()-1; n>=0; --n){
			//find the element index in the nn
			const unsigned int index=nnpot_.atom_index(strucl.name(n));
			//execute the network
			nnpot_.nn(index).execute(strucl.symm(n));
			//add the atom energy to the total
			energy+=nnpot_.nn(index).output()[0]+nnpot_.atom(index).energy();
			//compute the gradient - here dcda (first argument) is one, dcda is pulled out and multiplied later
			nnpot_.nn(index).grad_nol(identity_,gTemp_[index]);
			//add the gradient to the total
			gTempSum_[index].noalias()+=gTemp_[index];
			//increment the number of atoms
			++nAtomsl[index];
		}
		//compute the gradient of cost (c) w.r.t. output (a) normalized by number of atoms
		const double dcda=(energy-strucl.energy())/strucl.nAtoms();
		//add to the error
		error_train_+=0.5*dcda*dcda;
		//multiply the element gradients by error in energy normalized by number of atoms of the species
		for(int j=nElements_-1; j>=0; --j){
			if(nAtomsl[j]>0){
				gElement_[j].noalias()+=gTempSum_[j]*(energy-strucl.energy())/nAtomsl[j];
			}
		}
	}
	
	//====== compute validation error and gradient ======
	if(strucVal_!=NULL && (data_.step()%data_.nWrite()==0 || data_.step()%data_.nPrint()==0)){
		error_val_=0;
		if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"computing validation error and gradient\n";
		for(unsigned int i=0; i<strucVal_->size(); ++i){
			//set the local simulation reference
			const Structure& strucl=(*strucVal_)[i];
			//compute the energy
			double energy=0;
			for(int n=strucl.nAtoms()-1; n>=0; --n){
				//find the element index in the nn
				const unsigned int index=nnpot_.atom_index(strucl.name(n));
				//execute the network
				nnpot_.nn(index).execute(strucl.symm(n));
				//add the energy to the total
				energy+=nnpot_.nn(index).output()[0]+nnpot_.atom(index).energy();
			}
			//add to the error - normalized by the number of atoms
			error_val_+=0.5*(energy-strucl.energy())*(energy-strucl.energy())/(strucl.nAtoms()*strucl.nAtoms());
		}
	}
	
	//====== print energy ======
	if(NN_POT_TRAIN_PRINT_DATA>0 && rank==0 && data_.step()%data_.nPrint()==0){
		std::vector<double> energy_train(strucTrain_->size());
		std::vector<double> energy_exact(strucTrain_->size());
		std::cout<<"======================================\n";
		std::cout<<"============== ENERGIES ==============\n";
		for(unsigned int i=0; i<strucTrain_->size(); ++i){
			//set the local simulation reference
			const Structure& strucl=(*strucTrain_)[batch_[i]];
			//compute the energy
			double energy=0;
			for(int n=strucl.nAtoms()-1; n>=0; --n){
				//find the element index in the nn
				const unsigned int index=nnpot_.atom_index(strucl.name(n));
				//execute the network
				nnpot_.nn(index).execute(strucl.symm(n));
				//add the energy to the total
				energy+=nnpot_.nn(index).output()[0]+nnpot_.atom(index).energy();
			}
			//scale the energy
			energy_train[i]=energy/strucl.nAtoms();
			energy_exact[i]=strucl.energy()/strucl.nAtoms();
		}
		for(unsigned int i=0; i<strucTrain_->size(); ++i){
			std::cout<<"struc "<<i<<" "<<energy_exact[i]<<" "<<energy_train[i]<<" "<<0.5*(energy_train[i]-energy_exact[i])*(energy_train[i]-energy_exact[i])<<"\n";
		}
		std::cout<<"============== ENERGIES ==============\n";
		std::cout<<"======================================\n";
	}
	
	//====== return the error ======
	if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"returning the error\n";
	return error_train_;
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
		Opt::Data opt_param;//object storing opt parameters read from file, used when restarting
		Opt::Model* model_param_=NULL;//optimization - data
	//structures - format
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.specie=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.symm=true; atomT.charge=false;
		FILE_FORMAT::type format;//format of training data
	//structures - data
		std::vector<std::string> data_train;  //data files - training
		std::vector<std::string> data_val;    //data files - validation
		std::vector<std::string> data_test;   //data files - testing
		std::vector<std::string> files_train; //structure files - training
		std::vector<std::string> files_val;   //structure files - validation
		std::vector<std::string> files_test;  //structure files - testing
		std::vector<Structure> struc_train;   //structures - training
		std::vector<Structure> struc_val;     //structures - validation
		std::vector<Structure> struc_test;    //structures - testing
	//units
		units::System::type unitsys=units::System::UNKNOWN;
	//ewald
		Ewald3D::Coulomb ewald;//ewald object
		double prec=0;//ewald precision
		std::vector<double> ewald_train; //ewald energies - training
		std::vector<double> ewald_val;   //ewald energies - validation
		std::vector<double> ewald_test;  //ewald energies - testing
	//mpi distribution
		int* thread_dist_train=NULL;   //subset - number of structures  - global - training
		int* thread_dist_val=NULL;     //subset - number of structures  - global - validation
		int* thread_dist_test=NULL;    //subset - number of structures  - global - testing
		int* thread_offset_train=NULL; //subset - offset for structures - global - training
		int* thread_offset_val=NULL;   //subset - offset for structures - global - validation
		int* thread_offset_test=NULL;  //subset - offset for structures - global - testing
		unsigned int subset_train=0;   //subset - number of structures  - local  - training
		unsigned int subset_val=0;     //subset - number of structures  - local  - validation
		unsigned int subset_test=0;    //subset - number of structures  - local  - testing
		unsigned int offset_train=0;   //subset - offset for structures - local  - training
		unsigned int offset_val=0;     //subset - offset for structures - local  - validation
		unsigned int offset_test=0;    //subset - offset for structures - local  - testing
	//timing
		clock_t start,stop;//starting/stopping time
		clock_t start_wall,stop_wall;//starting/stopping wall time
		double time_wall=0;         //total wall time
		double time_energy_train=0; //compute time - energies - training
		double time_energy_val=0;   //compute time - energies - validation
		double time_energy_test=0;  //compute time - energies - test
		double time_force_train=0;  //compute time - forces - training
		double time_force_val=0;    //compute time - forces - validation
		double time_force_test=0;   //compute time - forces - test
		double time_symm_train=0;   //compute time - symmetry functions - training
		double time_symm_val=0;     //compute time - symmetry functions - validation
		double time_symm_test=0;    //compute time - symmetry functions - test
	//random
		int seed=-1;//seed for random number generator (negative => current time)
	//file i/o
		FILE* reader=NULL;
		std::vector<std::string> strlist;
		std::vector<std::string> fileNNPot;
		std::vector<std::string> fileName;
		char* paramfile=new char[string::M];
		char* input=new char[string::M];
	//writing
		bool write_corr=true;    //writing - correlation functions
		bool write_basis=true;   //writing - basis functions
		bool write_energy=false; //writing - energies
		bool write_ewald=false;  //writing - ewald energies
		bool write_force=false;  //writing - forces
	//mpi
		int rank=0;
		int nprocs=1;
		
	try{
		//************************************************************************************
		// LOADING/INITIALIZATION
		//************************************************************************************
		
		//======== initialize mpi ========
		if(rank==0) std::cout<<"initializing mpi\n";
		MPI_Init(&argc,&argv);
		
		//======== start wall clock ========
		if(rank==0) start_wall=std::clock();
		
		//======== set mpi data ========
		MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
		MPI_Comm_rank(MPI_COMM_WORLD,&rank);
		if(rank==0){
			std::cout<<"**************************************************\n";
			std::cout<<"********************** MPI **********************\n";
			std::cout<<"nprocs "<<nprocs<<"\n"<<std::flush;
		}
		MPI_Barrier(MPI_COMM_WORLD);
		std::cout<<"reporting from process "<<rank<<" out of "<<nprocs-1<<"\n";
		std::cout<<std::flush;
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank==0){
			std::cout<<"********************** MPI **********************\n";
			std::cout<<"**************************************************\n";
			std::cout<<std::flush;
		}
		MPI_Barrier(MPI_COMM_WORLD);
		
		//======== print compiler version ========
		if(rank==0){
			std::cout<<"compiler    = "<<compiler::version()<<"\n";
			std::cout<<"standard    = "<<compiler::standard()<<"\n";
			std::cout<<"instruction = "<<compiler::instruction()<<"\n";
		}
		
		//======== rank 0 reads from file ========
		if(rank==0){
			
			//======== check the arguments ========
			if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
			
			//======== load the parameter file ========
			if(NN_POT_TRAIN_PRINT_STATUS>0) std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//======== open the parameter file ========
			if(NN_POT_TRAIN_PRINT_STATUS>0) std::cout<<"opening parameter file\n";
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+paramfile);
			
			//======== read in the parameters ========
			if(NN_POT_TRAIN_PRINT_STATUS>0) std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL){
				string::trim_right(input,string::COMMENT);//trim comments
				if(string::split(input,string::WS,strlist)==0) continue;//skip if empty
				string::to_upper(strlist.at(0));//convert tag to upper case
				if(strlist.size()<2) throw std::runtime_error("Parameter tag without corresponding value.");
				//general
				if(strlist.at(0)=="SEED"){//random number seed
					seed=std::atoi(strlist.at(1).c_str());
				} else if(strlist.at(0)=="READ_POT"){//read potential file
					if(strlist.size()!=3) throw std::runtime_error("Invalid potential format.");
					fileName.push_back(strlist.at(1));
					fileNNPot.push_back(strlist.at(2));
				} else if(strlist.at(0)=="FORMAT"){//simulation format
					format=FILE_FORMAT::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="UNITS"){//units
					unitsys=units::System::read(string::to_upper(strlist.at(1)).c_str());
				}
				//data and execution mode
				if(strlist.at(0)=="MODE"){//mode of calculation
					string::to_upper(strlist.at(1));
					if(strlist.at(1)=="TRAIN") mode=MODE::TRAIN;
					else if(strlist.at(1)=="TEST") mode=MODE::TEST;	
					else throw std::invalid_argument("Invalid mode.");
				} else if(strlist.at(0)=="DATA_TRAIN"){//data - training
					data_train.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DATA_VAL"){//data - validation
					data_val.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DATA_TEST"){//data - testing
					data_test.push_back(strlist.at(1));
				} else if(strlist.at(0)=="ATOM"){//atom - name/mass/energy
					nnPotOpt.atoms_.push_back(Atom());
					if(strlist.size()==4 || strlist.size()==5){
						nnPotOpt.atoms_.back().name()=string::trim_all(std::strcpy(input,strlist.at(1).c_str()));
						nnPotOpt.atoms_.back().id()=string::hash(nnPotOpt.atoms_.back().name());
						nnPotOpt.atoms_.back().mass()=std::atof(strlist.at(2).c_str());
						nnPotOpt.atoms_.back().energy()=std::atof(strlist.at(3).c_str());
						if(strlist.size()==5) nnPotOpt.atoms_.back().charge()=std::atof(strlist.at(4).c_str());
					} else throw std::runtime_error("Invalid atom format.");
				} 
				//neural network potential
				if(strlist.at(0)=="R_CUT"){//distance cutoff
					nnPotOpt.nnpot_.rc()=std::atof(strlist.at(1).c_str());
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
					nnPotInit.tfType=NN::TransferN::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="CHARGE"){//whether charge contributions to energy are included (n.y.i.)
					nnPotOpt.charge_=string::boolean(strlist.at(1).c_str());
				}
				//ewald
				if(strlist.at(0)=="PREC"){//precision of ewald calculation
					prec=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="EPS"){//dielectric constant of bounding medium
					ewald.eps()=std::atof(strlist.at(1).c_str());
				}
				//neural network potential optimization
				if(strlist.at(0)=="PRE_COND"){//whether to precondition the inputs
					nnPotOpt.preCond_=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="N_BATCH"){//size of the batch
					nnPotOpt.nBatch_=std::atoi(strlist.at(1).c_str());
				} else if(strlist.at(0)=="P_BATCH"){//batch percentage
					nnPotOpt.pBatch_=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="CALC_FORCE"){//compute force at end
					nnPotOpt.calcForce_=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="READ_RESTART"){//read restart file
					nnPotOpt.restart_file_=strlist.at(1);
					nnPotOpt.restart_=true;
				} 
				//reading
				if(strlist.at(0)=="READ_BASIS_RADIAL"){//radial basis
					if(strlist.size()!=3) throw std::runtime_error("Invalid radial basis format.");
					nnPotOpt.atoms_basis_radial_.push_back(strlist.at(1));
					nnPotOpt.files_basis_radial_.push_back(strlist.at(2));
				} else if(strlist.at(0)=="READ_BASIS_ANGULAR"){//angular basis
					if(strlist.size()!=4) throw std::runtime_error("Invalid angular basis format.");
					nnPotOpt.atoms_basis_angular_.push_back(std::pair<std::string,std::string>(strlist.at(1),strlist.at(2)));
					nnPotOpt.files_basis_angular_.push_back(strlist.at(3));
				}
				//writing
				if(strlist.at(0)=="WRITE_CORR"){//whether to write correlation of inputs
					write_corr=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="WRITE_BASIS"){//whether to write the basis (function of distance/angle)
					write_basis=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="WRITE_ENERGY"){//whether to write the final energies
					write_energy=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="WRITE_FORCE"){//whether to write the final forces
					write_force=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="WRITE_EWALD"){//whether to write the final ewald energies
					write_ewald=string::boolean(strlist.at(1).c_str());
				} 
			}
			
			//======== read optimization data =========
			if(NN_POT_TRAIN_PRINT_STATUS>0) std::cout<<"reading optimization data\n";
			Opt::read(nnPotOpt.data_,reader);
			
			//======== read optimization model ========
			if(NN_POT_TRAIN_PRINT_STATUS>0) std::cout<<"reading optimization model\n";
			switch(nnPotOpt.data_.algo()){
				case Opt::ALGO::SGD:
					nnPotOpt.model_.reset(new Opt::SGD());
					Opt::read(static_cast<Opt::SGD&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::SGD(static_cast<const Opt::SGD&>(*nnPotOpt.model_));
				break;
				case Opt::ALGO::SDM:
					nnPotOpt.model_.reset(new Opt::SDM());
					Opt::read(static_cast<Opt::SDM&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::SDM(static_cast<const Opt::SDM&>(*nnPotOpt.model_));
				break;
				case Opt::ALGO::NAG:
					nnPotOpt.model_.reset(new Opt::NAG());
					Opt::read(static_cast<Opt::NAG&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::NAG(static_cast<const Opt::NAG&>(*nnPotOpt.model_));
				break;
				case Opt::ALGO::ADAGRAD:
					nnPotOpt.model_.reset(new Opt::ADAGRAD());
					Opt::read(static_cast<Opt::ADAGRAD&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::ADAGRAD(static_cast<const Opt::ADAGRAD&>(*nnPotOpt.model_));
				break;
				case Opt::ALGO::ADADELTA:
					nnPotOpt.model_.reset(new Opt::ADADELTA());
					Opt::read(static_cast<Opt::ADADELTA&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::ADADELTA(static_cast<const Opt::ADADELTA&>(*nnPotOpt.model_));
				break;
				case Opt::ALGO::RMSPROP:
					nnPotOpt.model_.reset(new Opt::RMSPROP());
					Opt::read(static_cast<Opt::RMSPROP&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::RMSPROP(static_cast<const Opt::RMSPROP&>(*nnPotOpt.model_));
				break;
				case Opt::ALGO::ADAM:
					nnPotOpt.model_.reset(new Opt::ADAM());
					Opt::read(static_cast<Opt::ADAM&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::ADAM(static_cast<const Opt::ADAM&>(*nnPotOpt.model_));
				break;
				case Opt::ALGO::NADAM:
					nnPotOpt.model_.reset(new Opt::NADAM());
					Opt::read(static_cast<Opt::NADAM&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::NADAM(static_cast<const Opt::NADAM&>(*nnPotOpt.model_));
				break;
				case Opt::ALGO::BFGS:
					nnPotOpt.model_.reset(new Opt::BFGS());
					Opt::read(static_cast<Opt::BFGS&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::BFGS(static_cast<const Opt::BFGS&>(*nnPotOpt.model_));
				break;
				case Opt::ALGO::RPROP:
					nnPotOpt.model_.reset(new Opt::RPROP());
					Opt::read(static_cast<Opt::RPROP&>(*nnPotOpt.model_),reader);
					model_param_=new Opt::RPROP(static_cast<const Opt::RPROP&>(*nnPotOpt.model_));
				break;
				default:
					throw std::invalid_argument("Invalid optimization algorithm.");
				break;
			}
			
			//======== set the optimization paremeters ========
			opt_param.max()=nnPotOpt.data_.max();
			opt_param.nPrint()=nnPotOpt.data_.nPrint();
			opt_param.tol()=nnPotOpt.data_.tol();
			
			//======== close parameter file ========
			if(NN_POT_TRAIN_PRINT_STATUS>0) std::cout<<"closing parameter file\n";
			fclose(reader);
			reader=NULL;
			
			//======== set charge flag ========
			if(nnPotOpt.charge_) atomT.charge=true;
		
			//======== check the parameters ========
			if(mode==MODE::TRAIN && data_train.size()==0) throw std::invalid_argument("No training data provided.");
			else if(mode==MODE::TEST && data_test.size()==0) throw std::invalid_argument("No test data provided.");
			if(nnPotOpt.pBatch_<0 || nnPotOpt.pBatch_>1) throw std::invalid_argument("Invalid batch size.");
			if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			
		}
		
		//======== bcast the paramters ========
		if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"broadcasting parameters\n";
		//general parameters
		MPI_Bcast(&unitsys,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);
		//nn_pot_opt
		MPI_Bcast(&nnPotOpt.nBatch_,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&nnPotOpt.pBatch_,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(&nnPotOpt.preCond_,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		MPI_Bcast(&nnPotOpt.charge_,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		MPI_Bcast(&nnPotOpt.restart_,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		MPI_Bcast(&nnPotOpt.calcForce_,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		//file i/o
		MPI_Bcast(&format,1,MPI_INT,0,MPI_COMM_WORLD);
		//writing
		MPI_Bcast(&write_corr,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		MPI_Bcast(&write_basis,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		MPI_Bcast(&write_energy,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		MPI_Bcast(&write_force,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		MPI_Bcast(&write_ewald,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		//mode
		MPI_Bcast(&mode,1,MPI_INT,0,MPI_COMM_WORLD);
		//ewald
		MPI_Bcast(&nnPotOpt.charge_,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
		if(nnPotOpt.charge_) atomT.charge=true;
		
		//======== set the unit system ========
		if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"setting the unit system\n";
		units::consts::init(unitsys);
		
		//======== initialize the random number generator ========
		if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"initializing random number generator\n";
		if(seed<0) std::srand(std::time(NULL));
		else std::srand(seed);
		
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
		
		//======== print parameters ========
		if(rank==0){
			std::cout<<"**************************************************\n";
			std::cout<<"*************** GENERAL PARAMETERS ***************\n";
			std::cout<<"\tSEED       = "<<seed<<"\n";
			std::cout<<"\tFORMAT     = "<<format<<"\n";
			std::cout<<"\tUNITS      = "<<unitsys<<"\n";
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
		}
		MPI_Barrier(MPI_COMM_WORLD);
		
		//************************************************************************************
		// READ DATA
		//************************************************************************************
		
		//======== rank 0 reads the data files (lists of structure files) ========
		if(rank==0){
			//==== read the training data ====
			if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"reading training data\n";
			for(unsigned i=0; i<data_train.size(); ++i){
				//open the data file
				if(NN_POT_TRAIN_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data_train[i]<<"\n";
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
			//==== read the validation data ====
			if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"reading validation data\n";
			for(unsigned int i=0; i<data_val.size(); ++i){
				//open the data file
				if(NN_POT_TRAIN_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data_val[i]<<"\n";
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
			//==== read the test data ====
			if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"reading testing data\n";
			for(unsigned int i=0; i<data_test.size(); ++i){
				//open the data file
				if(NN_POT_TRAIN_PRINT_DATA>0) std::cout<<"data file "<<i<<": "<<data_test[i]<<"\n";
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
		
			//==== print the files ====
			if(NN_POT_TRAIN_PRINT_DATA>-1){
				if(files_train.size()>0){
					std::cout<<"**************************************************\n";
					std::cout<<"***************** FILES - TRAIN *****************\n";
					for(int i=0; i<files_train.size(); ++i) std::cout<<"\t"<<files_train[i]<<"\n";
					std::cout<<"***************** FILES - TRAIN *****************\n";
					std::cout<<"**************************************************\n";
				}
				if(files_val.size()>0){
					std::cout<<"**************************************************\n";
					std::cout<<"****************** FILES - VAL ******************\n";
					for(int i=0; i<files_val.size(); ++i) std::cout<<"\t"<<files_val[i]<<"\n";
					std::cout<<"****************** FILES - VAL ******************\n";
					std::cout<<"**************************************************\n";
				}
				if(files_test.size()>0){
					std::cout<<"**************************************************\n";
					std::cout<<"****************** FILES - TEST ******************\n";
					for(int i=0; i<files_test.size(); ++i) std::cout<<"\t"<<files_test[i]<<"\n";
					std::cout<<"****************** FILES - TEST ******************\n";
					std::cout<<"**************************************************\n";
				}
			}
		}
		
		//======== bcast the file names =======
		if(NN_POT_TRAIN_PRINT_STATUS>-1 && rank==0) std::cout<<"bcasting file names\n";
		//bcast names
		mpi_util::bcast(files_train);
		mpi_util::bcast(files_val);
		mpi_util::bcast(files_test);
		//set number of structures
		nnPotOpt.nTrain_=files_train.size();
		nnPotOpt.nVal_=files_val.size();
		nnPotOpt.nTest_=files_test.size();
		//print number of structures
		if(rank==0){
			std::cout<<"ntrain = "<<nnPotOpt.nTrain_<<"\n";
			std::cout<<"nval   = "<<nnPotOpt.nVal_<<"\n";
			std::cout<<"ntest  = "<<nnPotOpt.nTest_<<"\n";
		}
		
		//==== gen thread dist + offset ====
		//thread dist
		thread_dist_train=new int[nprocs];
		thread_dist_val=new int[nprocs];
		thread_dist_test=new int[nprocs];
		parallel::gen_thread_dist(nprocs,nnPotOpt.nTrain_,thread_dist_train);
		parallel::gen_thread_dist(nprocs,nnPotOpt.nVal_,thread_dist_val);
		parallel::gen_thread_dist(nprocs,nnPotOpt.nTest_,thread_dist_test);
		subset_train=parallel::thread_subset(nnPotOpt.nTrain_,rank,nprocs);
		subset_val=parallel::thread_subset(nnPotOpt.nVal_,rank,nprocs);
		subset_test=parallel::thread_subset(nnPotOpt.nTest_,rank,nprocs);
		//thread offset
		thread_offset_train=new int[nprocs];
		thread_offset_val=new int[nprocs];
		thread_offset_test=new int[nprocs];
		parallel::gen_thread_offset(nprocs,nnPotOpt.nTrain_,thread_offset_train);
		parallel::gen_thread_offset(nprocs,nnPotOpt.nVal_,thread_offset_val);
		parallel::gen_thread_offset(nprocs,nnPotOpt.nTest_,thread_offset_test);
		offset_train=parallel::thread_offset(nnPotOpt.nTrain_,rank,nprocs);
		offset_val=parallel::thread_offset(nnPotOpt.nVal_,rank,nprocs);
		offset_test=parallel::thread_offset(nnPotOpt.nTest_,rank,nprocs);
		if(rank==0){
			std::cout<<"thread_dist_train   = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_dist_train[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_val     = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_dist_val[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_test    = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_dist_test[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_train = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_offset_train[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_val   = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_offset_val[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_test  = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_offset_test[i]<<" "; std::cout<<"\n";
		}
		//==== gen indices (random shuffle) ====
		std::vector<unsigned int> indices_train(nnPotOpt.nTrain_,0);
		std::vector<unsigned int> indices_val(nnPotOpt.nVal_,0);
		std::vector<unsigned int> indices_test(nnPotOpt.nTest_,0);
		if(rank==0){
			for(unsigned int i=0; i<indices_train.size(); ++i) indices_train[i]=i;
			for(unsigned int i=0; i<indices_val.size(); ++i) indices_val[i]=i;
			for(unsigned int i=0; i<indices_test.size(); ++i) indices_test[i]=i;
			std::random_shuffle(indices_train.begin(),indices_train.end());
			std::random_shuffle(indices_val.begin(),indices_val.end());
			std::random_shuffle(indices_test.begin(),indices_test.end());
		}
		//==== bcast randomized indices ====
		mpi_util::bcast(indices_train);
		mpi_util::bcast(indices_val);
		mpi_util::bcast(indices_test);
		
		//======== read the structures ========
		//==== training structures ====
		if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"reading training structures - rank "<<rank<<"\n";
		if(files_train.size()>0){
			struc_train.resize(subset_train);
			if(format==FILE_FORMAT::VASP_XML){
				for(unsigned int i=0; i<subset_train; ++i){
					VASP::XML::read(files_train[indices_train[offset_train+i]].c_str(),0,atomT,struc_train[i]);
				}
			} else if(format==FILE_FORMAT::QE){
				for(unsigned int i=0; i<subset_train; ++i){
					QE::OUT::read(files_train[indices_train[offset_train+i]].c_str(),atomT,struc_train[i]);
				}
			} else if(format==FILE_FORMAT::AME){
				for(unsigned int i=0; i<subset_train; ++i){
					AME::read(files_train[indices_train[offset_train+i]].c_str(),atomT,struc_train[i]);
				}
			}
			if(NN_POT_TRAIN_PRINT_DATA>1){
				for(unsigned int i=0; i<files_train.size(); ++i){
					std::cout<<"\t"<<files_train[indices_train[offset_train+i]]<<" "<<struc_train[i].energy()<<"\n";
				}
			}
		}
		//==== validation structures ====
		if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"reading validation structures - rank "<<rank<<"\n";
		if(files_val.size()>0){
			struc_val.resize(subset_val);
			if(format==FILE_FORMAT::VASP_XML){
				for(unsigned int i=0; i<subset_val; ++i){
					VASP::XML::read(files_val[indices_val[offset_val+i]].c_str(),0,atomT,struc_val[i]);
				}
			} else if(format==FILE_FORMAT::QE){
				for(unsigned int i=0; i<subset_val; ++i){
					QE::OUT::read(files_val[indices_val[offset_val+i]].c_str(),atomT,struc_val[i]);
				}
			} else if(format==FILE_FORMAT::AME){
				for(unsigned int i=0; i<subset_val; ++i){
					AME::read(files_val[indices_val[offset_val+i]].c_str(),atomT,struc_val[i]);
				}
			}
			if(NN_POT_TRAIN_PRINT_DATA>1){
				for(unsigned int i=0; i<subset_val; ++i){
					std::cout<<"\t"<<files_val[indices_val[offset_val+i]]<<" "<<struc_val[i].energy()<<"\n";
				}
			}
		}
		//==== testing structures ====
		if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"reading testing structures - rank "<<rank<<"\n";
		if(files_test.size()>0){
			struc_test.resize(subset_test);
			if(format==FILE_FORMAT::VASP_XML){
				for(unsigned int i=0; i<subset_test; ++i){
					VASP::XML::read(files_test[indices_test[offset_test+i]].c_str(),0,atomT,struc_test[i]);
				}
			} else if(format==FILE_FORMAT::QE){
				for(unsigned int i=0; i<subset_test; ++i){
					QE::OUT::read(files_test[indices_test[offset_test+i]].c_str(),atomT,struc_test[i]);
				}
			} else if(format==FILE_FORMAT::AME){
				for(unsigned int i=0; i<subset_test; ++i){
					AME::read(files_test[indices_test[offset_test+i]].c_str(),atomT,struc_test[i]);
				}
			}
			if(NN_POT_TRAIN_PRINT_DATA>1){
				for(unsigned int i=0; i<subset_test; ++i){
					std::cout<<"\t"<<files_test[indices_test[offset_test+i]]<<" "<<struc_test[i].energy()<<"\n";
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		
		//======== check the structures ========
		if(NN_POT_TRAIN_PRINT_STATUS>-1 && rank==0) std::cout<<"checking the structures\n";
		//==== training structures ====
		for(unsigned int i=0; i<struc_train.size(); ++i){
			const std::string filename=files_train[indices_train[offset_train+i]];
			if(struc_train[i].nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
			if(std::isinf(struc_train[i].energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
			if(struc_train[i].energy()!=struc_train[i].energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
			if(std::fabs(struc_train[i].energy())<num_const::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
			for(unsigned int n=0; n<struc_train[i].nAtoms(); ++n){
				const double force=struc_train[i].force(n).norm();
				if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<struc_train[i].name(n)<<struc_train[i].index(n)<<"\" in \""<<filename<<" has inf force.";
				if(force!=force) std::cout<<"WARNING: Atom \""<<struc_train[i].name(n)<<struc_train[i].index(n)<<"\" in \""<<filename<<" has nan force.";
			}
		}
		if(NN_POT_TRAIN_PRINT_DATA>1) for(unsigned int i=0; i<struc_train.size(); ++i) std::cout<<"\t"<<files_train[indices_train[offset_train+i]]<<" "<<struc_train[i].energy()<<"\n";
		//==== validation structures ====
		for(unsigned int i=0; i<struc_val.size(); ++i){
			const std::string filename=files_val[indices_val[offset_val+i]];
			if(struc_val[i].nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
			if(std::isinf(struc_val[i].energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
			if(struc_val[i].energy()!=struc_val[i].energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
			if(std::fabs(struc_val[i].energy())<num_const::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
			for(unsigned int n=0; n<struc_val[i].nAtoms(); ++n){
				const double force=struc_val[i].force(n).norm();
				if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<struc_val[i].name(n)<<struc_val[i].index(n)<<"\" in \""<<filename<<" has inf force.";
				if(force!=force) std::cout<<"WARNING: Atom \""<<struc_val[i].name(n)<<struc_val[i].index(n)<<"\" in \""<<filename<<" has nan force.";
			}
		}
		if(NN_POT_TRAIN_PRINT_DATA>1) for(unsigned int i=0; i<struc_val.size(); ++i) std::cout<<"\t"<<files_val[offset_val+i]<<" "<<struc_val[i].energy()<<"\n";
		//==== testing structures ====
		for(unsigned int i=0; i<struc_test.size(); ++i){
			const std::string filename=files_test[indices_test[offset_test+i]];
			if(struc_test[i].nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
			if(std::isinf(struc_test[i].energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
			if(struc_test[i].energy()!=struc_test[i].energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
			if(std::fabs(struc_test[i].energy())<num_const::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
			for(unsigned int n=0; n<struc_test[i].nAtoms(); ++n){
				const double force=struc_test[i].force(n).norm();
				if(std::isinf(force)) std::cout<<"WARNING: Atom \""<<struc_test[i].name(n)<<struc_test[i].index(n)<<"\" in \""<<filename<<" has inf force.";
				if(force!=force) std::cout<<"WARNING: Atom \""<<struc_test[i].name(n)<<struc_test[i].index(n)<<"\" in \""<<filename<<" has nan force.";
			}
		}
		if(NN_POT_TRAIN_PRINT_DATA>1) for(unsigned int i=0; i<struc_test.size(); ++i) std::cout<<"\t"<<files_test[indices_test[offset_test+i]]<<" "<<struc_test[i].energy()<<"\n";
		
		//======== set the batch size ========
		if(nnPotOpt.pBatch_>0) nnPotOpt.nBatch_=std::floor(nnPotOpt.pBatch_*struc_train.size());
		if(nnPotOpt.nBatch_==0) throw std::invalid_argument("Invalid batch size.");
		if(nnPotOpt.nBatch_>struc_train.size()) throw std::invalid_argument("Invalid batch size.");
		
		//======== set the data - optimization object ========
		if(struc_train.size()>0) nnPotOpt.strucTrain_=&struc_train;
		if(struc_val.size()>0) nnPotOpt.strucVal_=&struc_val;
		if(struc_test.size()>0) nnPotOpt.strucTest_=&struc_test;
		
		//************************************************************************************
		// READ/INITIALIZE NN-POT
		//************************************************************************************
		
		//======== initialize the potential (rank 0) ========
		if(rank==0){
			if(!nnPotOpt.restart_){
				//resize the potential
				if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"resizing potential\n";
				nnPotOpt.nnpot_.resize(nnPotOpt.atoms_);
				//read the radial basis
				if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"reading radial basis\n";
				for(unsigned int i=0; i<nnPotOpt.atoms_basis_radial_.size(); ++i){
					if(NN_POT_TRAIN_PRINT_STATUS>0) std::cout<<"reading radial basis for atom "<<nnPotOpt.atoms_basis_radial_[i]<<"\n";
					const unsigned int index=nnPotOpt.nnpot_.atom_index(nnPotOpt.atoms_basis_radial_[i]);
					for(unsigned int j=0; j<nnPotOpt.nnpot_.nAtoms(); ++j){
						BasisR::read(nnPotOpt.files_basis_radial_[i].c_str(),nnPotOpt.nnpot_.basisR(j,index));
					}
				}
				//read the angular basis
				for(unsigned int i=0; i<nnPotOpt.atoms_basis_angular_.size(); ++i){
					if(NN_POT_TRAIN_PRINT_STATUS>0) std::cout<<"reading radial basis for atom "<<nnPotOpt.atoms_basis_angular_[i].first<<" "<<nnPotOpt.atoms_basis_angular_[i].second<<"\n";
					const unsigned int index1=nnPotOpt.nnpot_.atom_index(nnPotOpt.atoms_basis_angular_[i].first);
					const unsigned int index2=nnPotOpt.nnpot_.atom_index(nnPotOpt.atoms_basis_angular_[i].second);
					for(unsigned int j=0; j<nnPotOpt.nnpot_.nAtoms(); ++j){
						BasisA::read(nnPotOpt.files_basis_angular_[i].c_str(),nnPotOpt.nnpot_.basisA(j,index1,index2));
					}
				}
				//initialize the potential
				if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"initializing potential\n";
				nnPotOpt.nnpot_.init(nnPotInit);
			}
			
			//======== read neural network potentials ========
			if(!nnPotOpt.restart_){
				for(unsigned int i=0; i<fileNNPot.size(); ++i){
					if(NN_POT_TRAIN_PRINT_STATUS>0) std::cout<<"reading nn-pot for \""<<fileName[i]<<"\" from \""<<fileNNPot[i]<<"\"\n";
					nnPotOpt.nnpot_.read(nnPotOpt.nnpot_.atom_index(fileName[i]),fileNNPot[i]);
				}
			}
			
			//======== read restart file ========
			if(nnPotOpt.restart_){
				if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"reading restart file\n";
				std::string file=nnPotOpt.restart_file_+".restart";
				nnPotOpt.read_restart(file.c_str());
			}
			
			//======== print the potential ========
			std::cout<<nnPotOpt.nnpot_<<"\n";
		}
		
		//======== set optimization data ========
		if(rank==0) std::cout<<"setting optimization data\n";
		if(rank==0){
			//set parameters which are allowed to change when restarting
			nnPotOpt.data_.max()=opt_param.max();
			nnPotOpt.data_.nPrint()=opt_param.nPrint();
			nnPotOpt.data_.tol()=opt_param.tol();
			switch(nnPotOpt.data_.algo()){
				case Opt::ALGO::SGD:{
					Opt::SGD& nnModel_=static_cast<Opt::SGD&>(*nnPotOpt.model_);
					Opt::SGD& pModel_=static_cast<Opt::SGD&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
				}break;
				case Opt::ALGO::SDM:{
					Opt::SDM& nnModel_=static_cast<Opt::SDM&>(*nnPotOpt.model_);
					Opt::SDM& pModel_=static_cast<Opt::SDM&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.eta()>0) nnModel_.eta()=pModel_.eta();
				}break;
				case Opt::ALGO::NAG:{
					Opt::NAG& nnModel_=static_cast<Opt::NAG&>(*nnPotOpt.model_);
					Opt::NAG& pModel_=static_cast<Opt::NAG&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.eta()>0) nnModel_.eta()=pModel_.eta();
				}break;
				case Opt::ALGO::ADAGRAD:{
					Opt::ADAGRAD& nnModel_=static_cast<Opt::ADAGRAD&>(*nnPotOpt.model_);
					Opt::ADAGRAD& pModel_=static_cast<Opt::ADAGRAD&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
				}break;
				case Opt::ALGO::ADADELTA:{
					Opt::ADADELTA& nnModel_=static_cast<Opt::ADADELTA&>(*nnPotOpt.model_);
					Opt::ADADELTA& pModel_=static_cast<Opt::ADADELTA&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					if(pModel_.eta()>0) nnModel_.eta()=pModel_.eta();
				}break;
				case Opt::ALGO::RMSPROP:{
					Opt::RMSPROP& nnModel_=static_cast<Opt::RMSPROP&>(*nnPotOpt.model_);
					Opt::RMSPROP& pModel_=static_cast<Opt::RMSPROP&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
				}break;
				case Opt::ALGO::ADAM:{
					Opt::ADAM& nnModel_=static_cast<Opt::ADAM&>(*nnPotOpt.model_);
					Opt::ADAM& pModel_=static_cast<Opt::ADAM&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::ALGO::NADAM:{
					Opt::NADAM& nnModel_=static_cast<Opt::NADAM&>(*nnPotOpt.model_);
					Opt::NADAM& pModel_=static_cast<Opt::NADAM&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
					nnModel_.decay()=pModel_.decay();
				}break;
				case Opt::ALGO::BFGS:{
					Opt::BFGS& nnModel_=static_cast<Opt::BFGS&>(*nnPotOpt.model_);
					Opt::BFGS& pModel_=static_cast<Opt::BFGS&>(*model_param_);
					if(pModel_.gamma()>0) nnModel_.gamma()=pModel_.gamma();
				}break;
				case Opt::ALGO::RPROP:
					//no parameters
				break;
			}
		}
		
		//======== print optimization data ========
		if(rank==0){
			switch(nnPotOpt.data_.algo()){
				case Opt::ALGO::SGD: std::cout<<static_cast<Opt::SGD&>(*nnPotOpt.model_)<<"\n"; break;
				case Opt::ALGO::SDM: std::cout<<static_cast<Opt::SDM&>(*nnPotOpt.model_)<<"\n"; break;
				case Opt::ALGO::NAG: std::cout<<static_cast<Opt::NAG&>(*nnPotOpt.model_)<<"\n"; break;
				case Opt::ALGO::ADAGRAD: std::cout<<static_cast<Opt::ADAGRAD&>(*nnPotOpt.model_)<<"\n"; break;
				case Opt::ALGO::ADADELTA: std::cout<<static_cast<Opt::ADADELTA&>(*nnPotOpt.model_)<<"\n"; break;
				case Opt::ALGO::RMSPROP: std::cout<<static_cast<Opt::RMSPROP&>(*nnPotOpt.model_)<<"\n"; break;
				case Opt::ALGO::ADAM: std::cout<<static_cast<Opt::ADAM&>(*nnPotOpt.model_)<<"\n"; break;
				case Opt::ALGO::NADAM: std::cout<<static_cast<Opt::NADAM&>(*nnPotOpt.model_)<<"\n"; break;
				case Opt::ALGO::BFGS: std::cout<<static_cast<Opt::BFGS&>(*nnPotOpt.model_)<<"\n"; break;
				case Opt::ALGO::RPROP: std::cout<<static_cast<Opt::RPROP&>(*nnPotOpt.model_)<<"\n"; break;
			}
			std::cout<<nnPotInit<<"\n";
			std::cout<<nnPotOpt<<"\n";
			std::cout<<ewald<<"\n";
		}
		MPI_Barrier(MPI_COMM_WORLD);
		
		//======== bcast the optimization data ========
		if(rank==0) std::cout<<"bcasting optimization data\n";
		mpi_util::bcast(nnPotOpt.data_);
		if(rank==0) std::cout<<"bcasting optimization model\n";
		MPI_Bcast(&nnPotOpt.data_.algo(),1,MPI_INT,0,MPI_COMM_WORLD);
		switch(nnPotOpt.data_.algo()){
			case Opt::ALGO::SGD:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::SGD());
				mpi_util::bcast(static_cast<Opt::SGD&>(*nnPotOpt.model_));
			break;
			case Opt::ALGO::SDM:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::SDM());
				mpi_util::bcast(static_cast<Opt::SDM&>(*nnPotOpt.model_));
			break;
			case Opt::ALGO::NAG:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::NAG());
				mpi_util::bcast(static_cast<Opt::NAG&>(*nnPotOpt.model_));
			break;
			case Opt::ALGO::ADAGRAD:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::ADAGRAD());
				mpi_util::bcast(static_cast<Opt::ADAGRAD&>(*nnPotOpt.model_));
			break;
			case Opt::ALGO::ADADELTA:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::ADADELTA());
				mpi_util::bcast(static_cast<Opt::ADADELTA&>(*nnPotOpt.model_));
			break;
			case Opt::ALGO::RMSPROP:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::RMSPROP());
				mpi_util::bcast(static_cast<Opt::RMSPROP&>(*nnPotOpt.model_));
			break;
			case Opt::ALGO::ADAM:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::ADAM());
				mpi_util::bcast(static_cast<Opt::ADAM&>(*nnPotOpt.model_));
			break;
			case Opt::ALGO::NADAM:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::NADAM());
				mpi_util::bcast(static_cast<Opt::NADAM&>(*nnPotOpt.model_));
			break;
			case Opt::ALGO::BFGS:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::BFGS());
				mpi_util::bcast(static_cast<Opt::BFGS&>(*nnPotOpt.model_));
			break;
			case Opt::ALGO::RPROP:
				if(rank!=0) nnPotOpt.model_.reset(new Opt::RPROP());
				mpi_util::bcast(static_cast<Opt::RPROP&>(*nnPotOpt.model_));
			break;
		}
		
		//======== bcast the potential ========
		if(rank==0) std::cout<<"bcasting the potential\n";
		mpi_util::bcast(nnPotOpt.nnpot_);
		
		//======== compute ewald energies ========
		if(nnPotOpt.charge_){
			//==== set charges - training ====
			if(rank==0) std::cout<<"setting charges - training\n";
			for(int i=0; i<struc_train.size(); ++i){
				for(int n=0; n<struc_train[i].nAtoms(); ++n){
					for(int j=0; j<nnPotOpt.nnpot_.nAtoms(); ++j){
						if(nnPotOpt.nnpot_.atom(j).name()==struc_train[i].name(n)){
							struc_train[i].charge(n)=nnPotOpt.nnpot_.atom(j).charge();
							break;
						}
					}
				}
			}
			//==== set charges - validation ====
			if(rank==0) std::cout<<"setting charges - validation\n";
			for(int i=0; i<struc_val.size(); ++i){
				for(int n=0; n<struc_val[i].nAtoms(); ++n){
					for(int j=0; j<nnPotOpt.nnpot_.nAtoms(); ++j){
						if(nnPotOpt.nnpot_.atom(j).name()==struc_val[i].name(n)){
							struc_val[i].charge(n)=nnPotOpt.nnpot_.atom(j).charge();
							break;
						}
					}
				}
			}
			//==== set charges - testing ====
			if(rank==0) std::cout<<"setting charges - testing\n";
			for(int i=0; i<struc_test.size(); ++i){
				for(int n=0; n<struc_test[i].nAtoms(); ++n){
					for(int j=0; j<nnPotOpt.nnpot_.nAtoms(); ++j){
						if(nnPotOpt.nnpot_.atom(j).name()==struc_test[i].name(n)){
							struc_test[i].charge(n)=nnPotOpt.nnpot_.atom(j).charge();
							break;
						}
					}
				}
			}
			//==== compute energies - training ====
			if(rank==0) std::cout<<"computing ewald energies - training\n";
			ewald_train.resize(struc_train.size(),0);
			for(int i=struc_train.size()-1; i>=0; --i){
				ewald.init(struc_train[i],prec);
				ewald_train[i]=ewald.energy(struc_train[i]);
				struc_train[i].energy()-=ewald_train[i];
			}
			//==== compute energies - validation ====
			if(rank==0) std::cout<<"computing ewald energies - validation\n";
			ewald_val.resize(struc_val.size(),0);
			for(int i=struc_val.size()-1; i>=0; --i){
				ewald.init(struc_val[i],prec);
				ewald_val[i]=ewald.energy(struc_val[i]);
				struc_val[i].energy()-=ewald_val[i];
			}
			//==== compute energies - testing ====
			if(rank==0) std::cout<<"computing ewald energies - testing\n";
			ewald_test.resize(struc_test.size(),0);
			for(int i=struc_test.size()-1; i>=0; --i){
				ewald.init(struc_test[i],prec);
				ewald_test[i]=ewald.energy(struc_test[i]);
				struc_test[i].energy()-=ewald_test[i];
			}
		}
		
		//======== initialize the symmetry functions ========
		if(NN_POT_TRAIN_PRINT_STATUS>-1 && rank==0) std::cout<<"initializing symmetry functions - training set\n";
		for(int i=struc_train.size()-1; i>=0; --i) nnPotOpt.nnpot_.init_symm(struc_train[i]);
		if(NN_POT_TRAIN_PRINT_STATUS>-1 && rank==0) std::cout<<"initializing symmetry functions - validation set\n";
		for(int i=struc_val.size()-1; i>=0; --i) nnPotOpt.nnpot_.init_symm(struc_val[i]);
		if(NN_POT_TRAIN_PRINT_STATUS>-1 && rank==0) std::cout<<"initializing symmetry functions - test set\n";
		for(int i=struc_test.size()-1; i>=0; --i) nnPotOpt.nnpot_.init_symm(struc_test[i]);
		
		//======== print the optimization object ========
		if(rank==0) std::cout<<nnPotOpt<<"\n";
		std::cout<<std::flush;
		MPI_Barrier(MPI_COMM_WORLD);
		
		//************************************************************************************
		// SET INPUTS
		//************************************************************************************
		
		//==== training ====
		start=std::clock();
		if(struc_train.size()>0){
			if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"setting the inputs (symmetry functions) - training - process "<<rank<<"\n";
			for(unsigned int n=0; n<struc_train.size(); ++n){
				std::cout<<"structure-train["<<n<<"]\n";
				nnPotOpt.nnpot_.inputs_symm(struc_train[n]);
			}
		}
		stop=std::clock();
		time_symm_train=((double)(stop-start))/CLOCKS_PER_SEC;
		//==== validation ====
		start=std::clock();
		if(struc_val.size()>0){
			if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"setting the inputs (symmetry functions) - validation - process "<<rank<<"\n";
			for(unsigned int n=0; n<struc_val.size(); ++n){
				std::cout<<"structure-val["<<n<<"]\n";
				nnPotOpt.nnpot_.inputs_symm(struc_val[n]);
			}
		}
		stop=std::clock();
		time_symm_val=((double)(stop-start))/CLOCKS_PER_SEC;
		//==== testing ====
		start=std::clock();
		if(struc_test.size()>0){
			if(NN_POT_TRAIN_PRINT_STATUS>-1) std::cout<<"setting the inputs (symmetry functions) - testing - process "<<rank<<"\n";
			for(unsigned int n=0; n<struc_test.size(); ++n){
				std::cout<<"structure-test["<<n<<"]\n";
				nnPotOpt.nnpot_.inputs_symm(struc_test[n]);
			}
		}
		stop=std::clock();
		time_symm_test=((double)(stop-start))/CLOCKS_PER_SEC;
		//==== print the inputs ====
		if(NN_POT_TRAIN_PRINT_DATA>1){
			std::cout<<"====================================\n";
			std::cout<<"============== INPUTS ==============\n";
			for(unsigned int n=0; n<struc_train.size(); ++n){
				std::cout<<"struc["<<n<<"] = \n";
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
		MPI_Barrier(MPI_COMM_WORLD);
		
		//************************************************************************************
		// TRAINING
		//************************************************************************************
		
		//======== train the nn potential ========
		if(mode==MODE::TRAIN){
			if(NN_POT_TRAIN_PRINT_STATUS>0 && rank==0) std::cout<<"training the nn potential\n";
			nnPotOpt.train(nnPotOpt.nBatch_);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		
		//************************************************************************************
		// EVALUTION
		//************************************************************************************
		
		//======== compute the final energies ========
		//==== training systems ====
		if(struc_train.size()>0){
			if(rank==0) std::cout<<"final energies - training set\n";
			//local variables
			double* energy_nn=new double[struc_train.size()];
			double* energy_exact=new double[struc_train.size()];
			unsigned int* natoms=new unsigned int[struc_train.size()];
			double* energy_nn_t=NULL;
			double* energy_exact_t=NULL;
			unsigned int* natoms_t=NULL;
			//compute energies
			start=std::clock();
			for(unsigned int n=0; n<struc_train.size(); ++n){
				std::cout<<"structure-train["<<rank<<"]["<<n<<"]\n";
				energy_nn[n]=nnPotOpt.nnpot_.energy(struc_train[n],false);
				energy_exact[n]=struc_train[n].energy();
				natoms[n]=struc_train[n].nAtoms();
			}
			stop=std::clock();
			time_energy_train=((double)(stop-start))/CLOCKS_PER_SEC;
			MPI_Barrier(MPI_COMM_WORLD);
			//gather energies
			if(rank==0){
				energy_nn_t=new double[nnPotOpt.nTrain_];
				energy_exact_t=new double[nnPotOpt.nTrain_];
				natoms_t=new unsigned int[nnPotOpt.nTrain_];
			}
			MPI_Gatherv(energy_nn,subset_train,MPI_DOUBLE,energy_nn_t,thread_dist_train,thread_offset_train,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(energy_exact,subset_train,MPI_DOUBLE,energy_exact_t,thread_dist_train,thread_offset_train,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(natoms,subset_train,MPI_INT,natoms_t,thread_dist_train,thread_offset_train,MPI_INT,0,MPI_COMM_WORLD);
			//accumulate statistics
			if(rank==0){
				for(unsigned int n=0; n<nnPotOpt.nTrain_; ++n){
					acc1d_energy_train_a.push(std::fabs(energy_exact_t[n]-energy_nn_t[n]));
					acc1d_energy_train_n.push(std::fabs(energy_exact_t[n]-energy_nn_t[n])/natoms_t[n]);
					acc1d_energy_train_p.push(std::fabs((energy_exact_t[n]-energy_nn_t[n])/energy_exact_t[n])*100.0);
					acc2d_energy_train.push(energy_exact_t[n]/natoms_t[n],energy_nn_t[n]/natoms_t[n]);
				}
			}
			//write energies
			if(rank==0 && write_energy){
				const char* file="nn_pot_energy_train.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					std::vector<std::pair<unsigned int,double> > energy_exact_pair(nnPotOpt.nTrain_);
					std::vector<std::pair<unsigned int,double> > energy_nn_pair(nnPotOpt.nTrain_);
					for(unsigned int n=0; n<nnPotOpt.nTrain_; ++n){
						energy_exact_pair[n].first=indices_train[n];
						energy_exact_pair[n].second=energy_exact_t[n];
						energy_nn_pair[n].first=indices_train[n];
						energy_nn_pair[n].second=energy_nn_t[n];
					}
					std::sort(energy_exact_pair.begin(),energy_exact_pair.end(),compare_pair);
					std::sort(energy_nn_pair.begin(),energy_nn_pair.end(),compare_pair);
					fprintf(writer,"#STRUCTURE ENERGY_EXACT ENERGY_NN\n");
					for(unsigned int n=0; n<nnPotOpt.nTrain_; ++n){
						fprintf(writer,"%s %f %f\n",files_train[n].c_str(),energy_exact_pair[n].second,energy_nn_pair[n].second);
					}
					fclose(writer);
					writer=NULL;
				}
			}
			//free memory
			delete[] energy_nn;
			delete[] energy_exact;
			delete[] natoms;
			if(rank==0){
				delete[] energy_nn_t;
				delete[] energy_exact_t;
				delete[] natoms_t;
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		//==== validation systems ====
		if(struc_val.size()>0){
			if(rank==0) std::cout<<"final energies - validation set\n";
			//local variables
			double* energy_nn=new double[struc_val.size()];
			double* energy_exact=new double[struc_val.size()];
			unsigned int* natoms=new unsigned int[struc_val.size()];
			double* energy_nn_t=NULL;
			double* energy_exact_t=NULL;
			unsigned int* natoms_t=NULL;
			//compute energies
			start=std::clock();
			for(unsigned int n=0; n<struc_val.size(); ++n){
				std::cout<<"structure-val["<<rank<<"]["<<n<<"]\n";
				energy_nn[n]=nnPotOpt.nnpot_.energy(struc_val[n],false);
				energy_exact[n]=struc_val[n].energy();
				natoms[n]=struc_val[n].nAtoms();
			}
			stop=std::clock();
			time_energy_val=((double)(stop-start))/CLOCKS_PER_SEC;
			MPI_Barrier(MPI_COMM_WORLD);
			//gather energies
			if(rank==0){
				energy_nn_t=new double[nnPotOpt.nVal_];
				energy_exact_t=new double[nnPotOpt.nVal_];
				natoms_t=new unsigned int[nnPotOpt.nVal_];
			}
			MPI_Gatherv(energy_nn,subset_val,MPI_DOUBLE,energy_nn_t,thread_dist_val,thread_offset_val,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(energy_exact,subset_val,MPI_DOUBLE,energy_exact_t,thread_dist_val,thread_offset_val,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(natoms,subset_val,MPI_INT,natoms_t,thread_dist_val,thread_offset_val,MPI_INT,0,MPI_COMM_WORLD);
			//accumulate statistics
			const unsigned int tag=0;
			if(rank==0){
				for(unsigned int n=0; n<nnPotOpt.nVal_; ++n){
					acc1d_energy_val_a.push(std::fabs(energy_exact_t[n]-energy_nn_t[n]));
					acc1d_energy_val_n.push(std::fabs(energy_exact_t[n]-energy_nn_t[n])/natoms_t[n]);
					acc1d_energy_val_p.push(std::fabs((energy_exact_t[n]-energy_nn_t[n])/energy_exact_t[n])*100.0);
					acc2d_energy_val.push(energy_exact_t[n]/natoms_t[n],energy_nn_t[n]/natoms_t[n]);
				}
			}
			//write energies
			if(rank==0 && write_energy){
				const char* file="nn_pot_energy_val.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					std::vector<std::pair<unsigned int,double> > energy_exact_pair(nnPotOpt.nVal_);
					std::vector<std::pair<unsigned int,double> > energy_nn_pair(nnPotOpt.nVal_);
					for(unsigned int n=0; n<nnPotOpt.nVal_; ++n){
						energy_exact_pair[n].first=indices_val[n];
						energy_exact_pair[n].second=energy_exact_t[n];
						energy_nn_pair[n].first=indices_val[n];
						energy_nn_pair[n].second=energy_nn_t[n];
					}
					std::sort(energy_exact_pair.begin(),energy_exact_pair.end(),compare_pair);
					std::sort(energy_nn_pair.begin(),energy_nn_pair.end(),compare_pair);
					fprintf(writer,"#STRUCTURE ENERGY_EXACT ENERGY_NN\n");
					for(unsigned int n=0; n<nnPotOpt.nVal_; ++n){
						fprintf(writer,"%s %f %f\n",files_val[n].c_str(),energy_exact_pair[n].second,energy_nn_pair[n].second);
					}
					fclose(writer);
					writer=NULL;
				}
			}
			//free memory
			delete[] energy_nn;
			delete[] energy_exact;
			delete[] natoms;
			if(rank==0){
				delete[] energy_nn_t;
				delete[] energy_exact_t;
				delete[] natoms_t;
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		//==== test systems ====
		if(struc_test.size()>0){
			if(rank==0) std::cout<<"final energies - testing set\n";
			//local variables
			double* energy_nn=new double[struc_test.size()];
			double* energy_exact=new double[struc_test.size()];
			unsigned int* natoms=new unsigned int[struc_test.size()];
			double* energy_nn_t=NULL;
			double* energy_exact_t=NULL;
			unsigned int* natoms_t=NULL;
			//compute energies
			start=std::clock();
			for(unsigned int n=0; n<struc_test.size(); ++n){
				std::cout<<"structure-test["<<rank<<"]["<<n<<"]\n";
				energy_nn[n]=nnPotOpt.nnpot_.energy(struc_test[n],false);
				energy_exact[n]=struc_test[n].energy();
				natoms[n]=struc_test[n].nAtoms();
			}
			stop=std::clock();
			time_energy_test=((double)(stop-start))/CLOCKS_PER_SEC;
			MPI_Barrier(MPI_COMM_WORLD);
			//gather energies
			if(rank==0){
				energy_nn_t=new double[nnPotOpt.nTest_];
				energy_exact_t=new double[nnPotOpt.nTest_];
				natoms_t=new unsigned int[nnPotOpt.nTest_];
			}
			MPI_Gatherv(energy_nn,subset_test,MPI_DOUBLE,energy_nn_t,thread_dist_test,thread_offset_test,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(energy_exact,subset_test,MPI_DOUBLE,energy_exact_t,thread_dist_test,thread_offset_test,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(natoms,subset_test,MPI_INT,natoms_t,thread_dist_test,thread_offset_test,MPI_INT,0,MPI_COMM_WORLD);
			//accumulate statistics
			const unsigned int tag=0;
			if(rank==0){
				for(unsigned int n=0; n<nnPotOpt.nTest_; ++n){
					acc1d_energy_test_a.push(std::fabs(energy_exact_t[n]-energy_nn_t[n]));
					acc1d_energy_test_n.push(std::fabs(energy_exact_t[n]-energy_nn_t[n])/natoms_t[n]);
					acc1d_energy_test_p.push(std::fabs((energy_exact_t[n]-energy_nn_t[n])/energy_exact_t[n])*100.0);
					acc2d_energy_test.push(energy_exact_t[n]/natoms_t[n],energy_nn_t[n]/natoms_t[n]);
				}
			}
			//write energies
			if(rank==0 && write_energy){
				const char* file="nn_pot_energy_test.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					std::vector<std::pair<unsigned int,double> > energy_exact_pair(nnPotOpt.nTest_);
					std::vector<std::pair<unsigned int,double> > energy_nn_pair(nnPotOpt.nTest_);
					for(unsigned int n=0; n<nnPotOpt.nTest_; ++n){
						energy_exact_pair[n].first=indices_test[n];
						energy_exact_pair[n].second=energy_exact_t[n];
						energy_nn_pair[n].first=indices_test[n];
						energy_nn_pair[n].second=energy_nn_t[n];
					}
					std::sort(energy_exact_pair.begin(),energy_exact_pair.end(),compare_pair);
					std::sort(energy_nn_pair.begin(),energy_nn_pair.end(),compare_pair);
					fprintf(writer,"#STRUCTURE ENERGY_EXACT ENERGY_NN\n");
					for(unsigned int n=0; n<nnPotOpt.nTest_; ++n){
						fprintf(writer,"%s %f %f\n",files_test[n].c_str(),energy_exact_pair[n].second,energy_nn_pair[n].second);
					}
					fclose(writer);
					writer=NULL;
				}
			}
			//free memory
			delete[] energy_nn;
			delete[] energy_exact;
			delete[] natoms;
			if(rank==0){
				delete[] energy_nn_t;
				delete[] energy_exact_t;
				delete[] natoms_t;
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		
		//======== write the ewald energies ========
		//==== training systems ====
		if(struc_train.size()>0 && nnPotOpt.charge_ && write_ewald){
			if(rank==0) std::cout<<"writing ewald - training set\n";
			//gather energies
			double* ewald_train_l=new double[ewald_train.size()];
			for(int i=ewald_train.size()-1; i>=0; --i) ewald_train_l[i]=ewald_train[i];
			double* ewald_train_t=NULL;
			if(rank==0) ewald_train_t=new double[nnPotOpt.nTrain_];
			MPI_Gatherv(ewald_train_l,subset_train,MPI_DOUBLE,ewald_train_t,thread_dist_train,thread_offset_train,MPI_DOUBLE,0,MPI_COMM_WORLD);
			//write energies
			if(rank==0){
				const char* file="nn_pot_ewald_train.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					std::vector<std::pair<unsigned int,double> > ewald_train_pair(nnPotOpt.nTrain_);
					for(unsigned int n=0; n<nnPotOpt.nTrain_; ++n){
						ewald_train_pair[n].first=indices_train[n];
						ewald_train_pair[n].second=ewald_train_t[n];
					}
					std::sort(ewald_train_pair.begin(),ewald_train_pair.end(),compare_pair);
					fprintf(writer,"#STRUCTURE ENERGY_EWALD\n");
					for(unsigned int n=0; n<nnPotOpt.nTrain_; ++n){
						fprintf(writer,"%s %f\n",files_train[n].c_str(),ewald_train_pair[n].second);
					}
					fclose(writer);
					writer=NULL;
				}
			}
			delete[] ewald_train_l;
			if(rank==0) delete[] ewald_train_t;
		}
		//==== validation systems ====
		if(struc_val.size()>0 && nnPotOpt.charge_ && write_ewald){
			if(rank==0) std::cout<<"writing ewald - validation set\n";
			//gather energies
			double* ewald_val_l=new double[ewald_val.size()];
			for(int i=ewald_val.size()-1; i>=0; --i) ewald_val_l[i]=ewald_val[i];
			double* ewald_val_t=NULL;
			if(rank==0) ewald_val_t=new double[nnPotOpt.nVal_];
			MPI_Gatherv(ewald_val_l,subset_val,MPI_DOUBLE,ewald_val_t,thread_dist_val,thread_offset_val,MPI_DOUBLE,0,MPI_COMM_WORLD);
			//write energies
			if(rank==0){
				const char* file="nn_pot_ewald_val.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					std::vector<std::pair<unsigned int,double> > ewald_val_pair(nnPotOpt.nVal_);
					for(unsigned int n=0; n<nnPotOpt.nVal_; ++n){
						ewald_val_pair[n].first=indices_val[n];
						ewald_val_pair[n].second=ewald_val_t[n];
					}
					std::sort(ewald_val_pair.begin(),ewald_val_pair.end(),compare_pair);
					fprintf(writer,"#STRUCTURE ENERGY_EWALD\n");
					for(unsigned int n=0; n<nnPotOpt.nVal_; ++n){
						fprintf(writer,"%s %f\n",files_val[n].c_str(),ewald_val_pair[n].second);
					}
					fclose(writer);
					writer=NULL;
				}
			}
			delete[] ewald_val_l;
			if(rank==0) delete[] ewald_val_t;
		}
		//==== test systems ====
		if(struc_test.size()>0 && nnPotOpt.charge_ && write_ewald){
			if(rank==0) std::cout<<"writing ewald - testing set\n";
			//gather energies
			double* ewald_test_l=new double[ewald_test.size()];
			for(int i=ewald_test.size()-1; i>=0; --i) ewald_test_l[i]=ewald_test[i];
			double* ewald_test_t=NULL;
			if(rank==0) ewald_test_t=new double[nnPotOpt.nTest_];
			MPI_Gatherv(ewald_test.data(),subset_test,MPI_DOUBLE,ewald_test_t,thread_dist_test,thread_offset_test,MPI_DOUBLE,0,MPI_COMM_WORLD);
			//write energies
			if(rank==0){
				const char* file="nn_pot_ewald_test.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					std::vector<std::pair<unsigned int,double> > ewald_test_pair(nnPotOpt.nTest_);
					for(unsigned int n=0; n<nnPotOpt.nTest_; ++n){
						ewald_test_pair[n].first=indices_test[n];
						ewald_test_pair[n].second=ewald_test_t[n];
					}
					std::sort(ewald_test_pair.begin(),ewald_test_pair.end(),compare_pair);
					fprintf(writer,"#STRUCTURE ENERGY_EWALD\n");
					for(unsigned int n=0; n<nnPotOpt.nTest_; ++n){
						fprintf(writer,"%s %f\n",files_test[n].c_str(),ewald_test_pair[n].second);
					}
					fclose(writer);
					writer=NULL;
				}
			}
			delete[] ewald_test_l;
			if(rank==0) delete[] ewald_test_t;
		}
		
		//======== compute the final forces ========
		//==== training structures ====
		if(struc_train.size()>0 && nnPotOpt.calcForce_){
			if(rank==0) std::cout<<"computing final forces - training set\n";
			//local variables
			unsigned int count=0,ndata=0,ndata_t=0;
			unsigned int* natoms=new unsigned int[struc_train.size()];
			unsigned int* natoms_t=NULL;
			double* forces_nn=NULL;
			double* forces_exact=NULL;
			double* forces_nn_t=NULL;
			double* forces_exact_t=NULL;
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_exact_v(struc_train.size());
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_nn_v(struc_train.size());
			//compute forces
			for(unsigned int n=0; n<struc_train.size(); ++n){
				forces_exact_v[n].resize(struc_train[n].nAtoms());
				for(unsigned int j=0; j<struc_train[n].nAtoms(); ++j) forces_exact_v[n][j]=struc_train[n].force(j);
				ndata+=forces_exact_v[n].size()*3;
			}
			start=std::clock();
			for(unsigned int n=0; n<struc_train.size(); ++n){
				std::cout<<"structure-train["<<n<<"]\n";
				nnPotOpt.nnpot_.forces(struc_train[n],false);
			}
			stop=std::clock();
			for(unsigned int n=0; n<struc_train.size(); ++n){
				forces_nn_v[n].resize(struc_train[n].nAtoms());
				for(unsigned int j=0; j<struc_train[n].nAtoms(); ++j) forces_nn_v[n][j]=struc_train[n].force(j);
			}
			time_force_train=((double)(stop-start))/CLOCKS_PER_SEC;
			MPI_Barrier(MPI_COMM_WORLD);
			//gather forces
			int* thread_dist=new int[nprocs];
			int* thread_offset=new int[nprocs];
			MPI_Reduce(&ndata,&ndata_t,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Gather(&ndata,1,MPI_INT,thread_dist,1,MPI_INT,0,MPI_COMM_WORLD);
			thread_offset[0]=0;
			for(int i=1; i<nprocs; ++i) thread_offset[i]=thread_offset[i-1]+thread_dist[i-1];
			forces_nn=new double[ndata];
			forces_exact=new double[ndata];
			if(rank==0){
				forces_nn_t=new double[ndata_t];
				forces_exact_t=new double[ndata_t];
				natoms_t=new unsigned int[nnPotOpt.nTrain_];
			}
			count=0;
			for(unsigned int n=0; n<forces_exact_v.size(); ++n){
				for(unsigned int j=0; j<forces_exact_v[n].size(); ++j){
					forces_exact[count++]=forces_exact_v[n][j][0];
					forces_exact[count++]=forces_exact_v[n][j][1];
					forces_exact[count++]=forces_exact_v[n][j][2];
				}
			}
			count=0;
			for(unsigned int n=0; n<forces_nn_v.size(); ++n){
				for(unsigned int j=0; j<forces_nn_v[n].size(); ++j){
					forces_nn[count++]=forces_nn_v[n][j][0];
					forces_nn[count++]=forces_nn_v[n][j][1];
					forces_nn[count++]=forces_nn_v[n][j][2];
				}
			}
			MPI_Gatherv(forces_nn,ndata,MPI_DOUBLE,forces_nn_t,thread_dist,thread_offset,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(forces_exact,ndata,MPI_DOUBLE,forces_exact_t,thread_dist,thread_offset,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(natoms,subset_train,MPI_INT,natoms_t,thread_dist_train,thread_offset_train,MPI_INT,0,MPI_COMM_WORLD);
			//accumulate statistics
			if(rank==0){
				for(unsigned int i=0; i<ndata_t; i+=3){
					Eigen::Vector3d f_exact,f_nn;
					f_exact[0]=forces_exact_t[i+0];
					f_exact[1]=forces_exact_t[i+1];
					f_exact[2]=forces_exact_t[i+2];
					f_nn[0]=forces_nn_t[i+0];
					f_nn[1]=forces_nn_t[i+1];
					f_nn[2]=forces_nn_t[i+2];
					acc1d_force_train_a.push((f_exact-f_nn).norm());
					acc1d_force_train_p.push((f_exact-f_nn).norm()/f_exact.norm()*100.0);
					acc2d_forcex_train.push(f_exact[0],f_nn[0]);
					acc2d_forcey_train.push(f_exact[1],f_nn[1]);
					acc2d_forcez_train.push(f_exact[2],f_nn[2]);
				}
			}
			if(write_force && rank==0){
				const char* file="nn_pot_force_train.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					for(unsigned int i=0; i<ndata_t; i+=3){
						Eigen::Vector3d f_exact,f_nn;
						f_exact[0]=forces_exact_t[i+0];
						f_exact[1]=forces_exact_t[i+1];
						f_exact[2]=forces_exact_t[i+2];
						f_nn[0]=forces_nn_t[i+0];
						f_nn[1]=forces_nn_t[i+1];
						f_nn[2]=forces_nn_t[i+2];
						fprintf(writer,"%f %f %f %f %f %f\n",
							f_exact[0],f_exact[1],f_exact[2],
							f_nn[0],f_nn[1],f_nn[2]
						);
					}
					fclose(writer); writer=NULL;
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			delete[] thread_dist;
			delete[] thread_offset;
			delete[] forces_exact;
			delete[] forces_nn;
			delete[] natoms;
			if(rank==0){
				delete[] forces_nn_t;
				delete[] forces_exact_t;
				delete[] natoms_t;
			}
		}
		//==== validation structures ====
		if(struc_val.size()>0 && nnPotOpt.calcForce_){
			if(rank==0) std::cout<<"computing final forces - validation set\n";
			//local variables
			unsigned int count=0,ndata=0,ndata_t=0;
			unsigned int* natoms=new unsigned int[struc_val.size()];
			unsigned int* natoms_t=NULL;
			double* forces_nn=NULL;
			double* forces_exact=NULL;
			double* forces_nn_t=NULL;
			double* forces_exact_t=NULL;
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_exact_v(struc_val.size());
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_nn_v(struc_val.size());
			//compute forces
			for(unsigned int n=0; n<struc_val.size(); ++n){
				forces_exact_v[n].resize(struc_val[n].nAtoms());
				for(unsigned int j=0; j<struc_val[n].nAtoms(); ++j) forces_exact_v[n][j]=struc_val[n].force(j);
				ndata+=forces_exact_v[n].size()*3;
			}
			start=std::clock();
			for(unsigned int n=0; n<struc_val.size(); ++n){
				std::cout<<"structure-val["<<n<<"]\n";
				nnPotOpt.nnpot_.forces(struc_val[n],false);
			}
			stop=std::clock();
			for(unsigned int n=0; n<struc_val.size(); ++n){
				forces_nn_v[n].resize(struc_val[n].nAtoms());
				for(unsigned int j=0; j<struc_val[n].nAtoms(); ++j) forces_nn_v[n][j]=struc_val[n].force(j);
			}
			time_force_train=((double)(stop-start))/CLOCKS_PER_SEC;
			//gather forces
			int* thread_dist=new int[nprocs];
			int* thread_offset=new int[nprocs];
			MPI_Reduce(&ndata,&ndata_t,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Gather(&ndata,1,MPI_INT,thread_dist,1,MPI_INT,0,MPI_COMM_WORLD);
			thread_offset[0]=0;
			for(int i=1; i<nprocs; ++i) thread_offset[i]=thread_offset[i-1]+thread_dist[i-1];
			forces_nn=new double[ndata];
			forces_exact=new double[ndata];
			if(rank==0){
				forces_nn_t=new double[ndata_t];
				forces_exact_t=new double[ndata_t];
				natoms_t=new unsigned int[nnPotOpt.nTrain_];
			}
			count=0;
			for(unsigned int n=0; n<forces_exact_v.size(); ++n){
				for(unsigned int j=0; j<forces_exact_v[n].size(); ++j){
					forces_exact[count++]=forces_exact_v[n][j][0];
					forces_exact[count++]=forces_exact_v[n][j][1];
					forces_exact[count++]=forces_exact_v[n][j][2];
				}
			}
			count=0;
			for(unsigned int n=0; n<forces_nn_v.size(); ++n){
				for(unsigned int j=0; j<forces_nn_v[n].size(); ++j){
					forces_nn[count++]=forces_nn_v[n][j][0];
					forces_nn[count++]=forces_nn_v[n][j][1];
					forces_nn[count++]=forces_nn_v[n][j][2];
				}
			}
			MPI_Gatherv(forces_nn,ndata,MPI_DOUBLE,forces_nn_t,thread_dist,thread_offset,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(forces_exact,ndata,MPI_DOUBLE,forces_exact_t,thread_dist,thread_offset,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(natoms,subset_train,MPI_INT,natoms_t,thread_dist_train,thread_offset_train,MPI_INT,0,MPI_COMM_WORLD);
			//accumulate statistics
			if(rank==0){
				for(unsigned int i=0; i<ndata_t; i+=3){
					Eigen::Vector3d f_exact,f_nn;
					f_exact[0]=forces_exact_t[i+0];
					f_exact[1]=forces_exact_t[i+1];
					f_exact[2]=forces_exact_t[i+2];
					f_nn[0]=forces_nn_t[i+0];
					f_nn[1]=forces_nn_t[i+1];
					f_nn[2]=forces_nn_t[i+2];
					acc1d_force_train_a.push((f_exact-f_nn).norm());
					acc1d_force_train_p.push((f_exact-f_nn).norm()/f_exact.norm()*100.0);
					acc2d_forcex_train.push(f_exact[0],f_nn[0]);
					acc2d_forcey_train.push(f_exact[1],f_nn[1]);
					acc2d_forcez_train.push(f_exact[2],f_nn[2]);
				}
			}
			if(write_force && rank==0){
				const char* file="nn_pot_force_val.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					for(unsigned int i=0; i<ndata_t; i+=3){
						Eigen::Vector3d f_exact,f_nn;
						f_exact[0]=forces_exact_t[i+0];
						f_exact[1]=forces_exact_t[i+1];
						f_exact[2]=forces_exact_t[i+2];
						f_nn[0]=forces_nn_t[i+0];
						f_nn[1]=forces_nn_t[i+1];
						f_nn[2]=forces_nn_t[i+2];
						fprintf(writer,"%f %f %f %f %f %f\n",
							f_exact[0],f_exact[1],f_exact[2],
							f_nn[0],f_nn[1],f_nn[2]
						);
					}
					fclose(writer); writer=NULL;
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			delete[] thread_dist;
			delete[] thread_offset;
			delete[] forces_exact;
			delete[] forces_nn;
			delete[] natoms;
			if(rank==0){
				delete[] forces_nn_t;
				delete[] forces_exact_t;
				delete[] natoms_t;
			}
		}
		//==== test structures ====
		if(struc_test.size()>0 && nnPotOpt.calcForce_){
			if(rank==0) std::cout<<"computing final forces - testing set\n";
			//local variables
			unsigned int count=0,ndata=0,ndata_t=0;
			unsigned int* natoms=new unsigned int[struc_test.size()];
			unsigned int* natoms_t=NULL;
			double* forces_nn=NULL;
			double* forces_exact=NULL;
			double* forces_nn_t=NULL;
			double* forces_exact_t=NULL;
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_exact_v(struc_test.size());
			std::vector<std::vector<Eigen::Vector3d>,Eigen::aligned_allocator<Eigen::Vector3d>  > forces_nn_v(struc_test.size());
			//compute forces
			for(unsigned int n=0; n<struc_test.size(); ++n){
				forces_exact_v[n].resize(struc_test[n].nAtoms());
				for(unsigned int j=0; j<struc_test[n].nAtoms(); ++j) forces_exact_v[n][j]=struc_test[n].force(j);
				ndata+=forces_exact_v[n].size()*3;
			}
			start=std::clock();
			for(unsigned int n=0; n<struc_test.size(); ++n){
				std::cout<<"structure-test["<<n<<"]\n";
				nnPotOpt.nnpot_.forces(struc_test[n],false);
			}
			stop=std::clock();
			for(unsigned int n=0; n<struc_test.size(); ++n){
				forces_nn_v[n].resize(struc_test[n].nAtoms());
				for(unsigned int j=0; j<struc_test[n].nAtoms(); ++j) forces_nn_v[n][j]=struc_test[n].force(j);
			}
			time_force_train=((double)(stop-start))/CLOCKS_PER_SEC;
			//gather forces
			int* thread_dist=new int[nprocs];
			int* thread_offset=new int[nprocs];
			MPI_Reduce(&ndata,&ndata_t,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
			MPI_Gather(&ndata,1,MPI_INT,thread_dist,1,MPI_INT,0,MPI_COMM_WORLD);
			thread_offset[0]=0;
			for(int i=1; i<nprocs; ++i) thread_offset[i]=thread_offset[i-1]+thread_dist[i-1];
			forces_nn=new double[ndata];
			forces_exact=new double[ndata];
			if(rank==0){
				forces_nn_t=new double[ndata_t];
				forces_exact_t=new double[ndata_t];
				natoms_t=new unsigned int[nnPotOpt.nTrain_];
			}
			count=0;
			for(unsigned int n=0; n<forces_exact_v.size(); ++n){
				for(unsigned int j=0; j<forces_exact_v[n].size(); ++j){
					forces_exact[count++]=forces_exact_v[n][j][0];
					forces_exact[count++]=forces_exact_v[n][j][1];
					forces_exact[count++]=forces_exact_v[n][j][2];
				}
			}
			count=0;
			for(unsigned int n=0; n<forces_nn_v.size(); ++n){
				for(unsigned int j=0; j<forces_nn_v[n].size(); ++j){
					forces_nn[count++]=forces_nn_v[n][j][0];
					forces_nn[count++]=forces_nn_v[n][j][1];
					forces_nn[count++]=forces_nn_v[n][j][2];
				}
			}
			MPI_Gatherv(forces_nn,ndata,MPI_DOUBLE,forces_nn_t,thread_dist,thread_offset,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(forces_exact,ndata,MPI_DOUBLE,forces_exact_t,thread_dist,thread_offset,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Gatherv(natoms,subset_train,MPI_INT,natoms_t,thread_dist_train,thread_offset_train,MPI_INT,0,MPI_COMM_WORLD);
			//accumulate statistics
			if(rank==0){
				for(unsigned int i=0; i<ndata_t; i+=3){
					Eigen::Vector3d f_exact,f_nn;
					f_exact[0]=forces_exact_t[i+0];
					f_exact[1]=forces_exact_t[i+1];
					f_exact[2]=forces_exact_t[i+2];
					f_nn[0]=forces_nn_t[i+0];
					f_nn[1]=forces_nn_t[i+1];
					f_nn[2]=forces_nn_t[i+2];
					acc1d_force_train_a.push((f_exact-f_nn).norm());
					acc1d_force_train_p.push((f_exact-f_nn).norm()/f_exact.norm()*100.0);
					acc2d_forcex_train.push(f_exact[0],f_nn[0]);
					acc2d_forcey_train.push(f_exact[1],f_nn[1]);
					acc2d_forcez_train.push(f_exact[2],f_nn[2]);
				}
			}
			if(write_force && rank==0){
				const char* file="nn_pot_force_test.dat";
				FILE* writer=fopen(file,"w");
				if(writer==NULL) std::cout<<"WARNING: Could not open file: \""<<file<<"\"\n";
				else{
					for(unsigned int i=0; i<ndata_t; i+=3){
						Eigen::Vector3d f_exact,f_nn;
						f_exact[0]=forces_exact_t[i+0];
						f_exact[1]=forces_exact_t[i+1];
						f_exact[2]=forces_exact_t[i+2];
						f_nn[0]=forces_nn_t[i+0];
						f_nn[1]=forces_nn_t[i+1];
						f_nn[2]=forces_nn_t[i+2];
						fprintf(writer,"%f %f %f %f %f %f\n",
							f_exact[0],f_exact[1],f_exact[2],
							f_nn[0],f_nn[1],f_nn[2]
						);
					}
					fclose(writer); writer=NULL;
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			delete[] thread_dist;
			delete[] thread_offset;
			delete[] forces_exact;
			delete[] forces_nn;
			delete[] natoms;
			if(rank==0){
				delete[] forces_nn_t;
				delete[] forces_exact_t;
				delete[] natoms_t;
			}
		}
		
		//************************************************************************************
		// OUTPUT
		//************************************************************************************
		
		//======== print the timing info ========
		if(rank==0){
			//======== stop wall clock ========
			stop_wall=std::clock();
			time_wall=((double)(stop_wall-start_wall))/CLOCKS_PER_SEC;
		}
		double temp;
		MPI_Reduce(&time_symm_train,&temp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); if(rank==0) time_symm_train=temp;
		MPI_Reduce(&time_energy_train,&temp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); if(rank==0) time_energy_train=temp;
		MPI_Reduce(&time_force_train,&temp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); if(rank==0) time_force_train=temp;
		MPI_Reduce(&time_symm_val,&temp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); if(rank==0) time_symm_val=temp;
		MPI_Reduce(&time_energy_val,&temp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); if(rank==0) time_energy_val=temp;
		MPI_Reduce(&time_force_val,&temp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); if(rank==0) time_force_val=temp;
		MPI_Reduce(&time_symm_test,&temp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); if(rank==0) time_symm_test=temp;
		MPI_Reduce(&time_energy_test,&temp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); if(rank==0) time_energy_test=temp;
		MPI_Reduce(&time_force_test,&temp,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); if(rank==0) time_force_test=temp;
		if(rank==0){
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
		std::cout<<"time - wall           = "<<time_wall<<"\n";
		std::cout<<"******************* TIMING (S) *******************\n";
		std::cout<<"**************************************************\n";
		}
		
		//======== print the error statistics - training ========
		if(rank==0 && mode==MODE::TRAIN){
		std::cout<<"**************************************************\n";
		std::cout<<"********* STATISTICS - ERROR - TRAINING *********\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tAVG  = "<<acc1d_energy_train_a.avg()<<" "<<acc1d_energy_train_n.avg()<<" "<<acc1d_energy_train_p.avg()<<"\n";
		std::cout<<"\tDEV  = "<<std::sqrt(acc1d_energy_train_a.var())<<" "<<std::sqrt(acc1d_energy_train_n.var())<<" "<<std::sqrt(acc1d_energy_train_p.var())<<"\n";
		std::cout<<"\tMAX  = "<<acc1d_energy_train_a.max()<<" "<<acc1d_energy_train_n.max()<<" "<<acc1d_energy_train_p.max()<<"\n";
		std::cout<<"\tM/R2 = "<<acc2d_energy_train.m()<<" "<<acc2d_energy_train.r2()<<"\n";
		std::cout<<"FORCE:\n";
		std::cout<<"\tAVG  = "<<acc1d_force_train_a.avg()<<" "<<acc1d_force_train_p.avg()<<"\n";
		std::cout<<"\tDEV  = "<<std::sqrt(acc1d_force_train_a.var())<<" "<<std::sqrt(acc1d_force_train_p.var())<<"\n";
		std::cout<<"\tMAX  = "<<acc1d_force_train_a.max()<<" "<<acc1d_force_train_p.max()<<"\n";
		std::cout<<"\tM    = "<<acc2d_forcex_train.m() <<" "<<acc2d_forcey_train.m() <<" "<<acc2d_forcez_train.m() <<"\n";
		std::cout<<"\tR2   = "<<acc2d_forcex_train.r2()<<" "<<acc2d_forcey_train.r2()<<" "<<acc2d_forcez_train.r2()<<"\n";
		std::cout<<"********* STATISTICS - ERROR - TRAINING *********\n";
		std::cout<<"**************************************************\n";
		}
		
		//======== print the error statistics - validation ========
		if(rank==0 && struc_val.size()>0 && mode==MODE::TRAIN){
		std::cout<<"**************************************************\n";
		std::cout<<"******** STATISTICS - ERROR - VALIDATION ********\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tAVG  = "<<acc1d_energy_val_a.avg()<<" "<<acc1d_energy_val_n.avg()<<" "<<acc1d_energy_val_p.avg()<<"\n";
		std::cout<<"\tDEV  = "<<std::sqrt(acc1d_energy_val_a.var())<<" "<<std::sqrt(acc1d_energy_val_n.var())<<" "<<std::sqrt(acc1d_energy_val_p.var())<<"\n";
		std::cout<<"\tMAX  = "<<acc1d_energy_val_a.max()<<" "<<acc1d_energy_val_n.max()<<" "<<acc1d_energy_val_p.max()<<"\n";
		std::cout<<"\tM/R2 = "<<acc2d_energy_val.m()<<" "<<acc2d_energy_val.r2()<<"\n";
		std::cout<<"FORCE:\n";
		std::cout<<"\tAVG  = "<<acc1d_force_val_a.avg()<<" "<<acc1d_force_val_p.avg()<<"\n";
		std::cout<<"\tDEV  = "<<std::sqrt(acc1d_force_val_a.var())<<" "<<std::sqrt(acc1d_force_val_p.var())<<"\n";
		std::cout<<"\tMAX  = "<<acc1d_force_val_a.max()<<" "<<acc1d_force_val_p.max()<<"\n";
		std::cout<<"\tM    = "<<acc2d_forcex_val.m() <<" "<<acc2d_forcey_val.m() <<" "<<acc2d_forcez_val.m() <<"\n";
		std::cout<<"\tR2   = "<<acc2d_forcex_val.r2()<<" "<<acc2d_forcey_val.r2()<<" "<<acc2d_forcez_val.r2()<<"\n";
		std::cout<<"******** STATISTICS - ERROR - VALIDATION ********\n";
		std::cout<<"**************************************************\n";
		}
		
		//======== print the error statistics - test ========
		if(rank==0 && struc_test.size()>0){
		std::cout<<"**************************************************\n";
		std::cout<<"*********** STATISTICS - ERROR - TEST ***********\n";
		std::cout<<"ENERGY:\n";
		std::cout<<"\tAVG  = "<<acc1d_energy_test_a.avg()<<" "<<acc1d_energy_test_n.avg()<<" "<<acc1d_energy_test_p.avg()<<"\n";
		std::cout<<"\tDEV  = "<<std::sqrt(acc1d_energy_test_a.var())<<" "<<std::sqrt(acc1d_energy_test_n.var())<<" "<<std::sqrt(acc1d_energy_test_p.var())<<"\n";
		std::cout<<"\tMAX  = "<<acc1d_energy_test_a.max()<<" "<<acc1d_energy_test_n.max()<<" "<<acc1d_energy_test_p.max()<<"\n";
		std::cout<<"\tM/R2 = "<<acc2d_energy_test.m()<<" "<<acc2d_energy_test.r2()<<"\n";
		std::cout<<"FORCE:\n";
		std::cout<<"\tAVG  = "<<acc1d_force_test_a.avg()<<" "<<acc1d_force_test_p.avg()<<"\n";
		std::cout<<"\tDEV  = "<<std::sqrt(acc1d_force_test_a.var())<<" "<<std::sqrt(acc1d_force_test_p.var())<<"\n";
		std::cout<<"\tMAX  = "<<acc1d_force_test_a.max()<<" "<<acc1d_force_test_p.max()<<"\n";
		std::cout<<"\tM    = "<<acc2d_forcex_test.m() <<" "<<acc2d_forcey_test.m() <<" "<<acc2d_forcez_test.m() <<"\n";
		std::cout<<"\tR2   = "<<acc2d_forcex_test.r2()<<" "<<acc2d_forcey_test.r2()<<" "<<acc2d_forcez_test.r2()<<"\n";
		std::cout<<"*********** STATISTICS - ERROR - TEST ***********\n";
		std::cout<<"**************************************************\n";
		}
		
		//======== print the basis functions ========
		if(rank==0 && write_basis){
			unsigned int N=200;
			for(unsigned int n=0; n<nnPotOpt.nnpot_.nAtoms(); ++n){
				for(unsigned int m=0; m<nnPotOpt.nnpot_.nAtoms(); ++m){
					std::string filename="basisR_"+nnPotOpt.nnpot_.atom(n).name()+"_"+nnPotOpt.nnpot_.atom(m).name()+".dat";
					FILE* writer=fopen(filename.c_str(),"w");
					if(writer!=NULL){
						for(unsigned int i=0; i<N; ++i){
							const double dr=nnPotOpt.nnpot_.rc()*i/(N-1.0);
							fprintf(writer,"%f ",dr);
							for(unsigned int j=0; j<nnPotOpt.nnpot_.basisR(n,m).nfR(); ++j){
								fprintf(writer,"%f ",nnPotOpt.nnpot_.basisR(n,m).fR(j).val(dr,CutoffF::funcs[nnPotOpt.nnpot_.basisR(n,m).tcut()](dr,nnPotOpt.nnpot_.basisR(n,m).rc())));
							}
							fprintf(writer,"\n");
						}
						fclose(writer);
						writer=NULL;
					} else std::cout<<"WARNING: Could not open: \""<<filename<<"\"\n";
				}
			}
			for(unsigned int n=0; n<nnPotOpt.nnpot_.nAtoms(); ++n){
				for(unsigned int m=0; m<nnPotOpt.nnpot_.nAtoms(); ++m){
					for(unsigned int l=m; l<nnPotOpt.nnpot_.nAtoms(); ++l){
						std::string filename="basisA_"+nnPotOpt.nnpot_.atom(n).name()+"_"+nnPotOpt.nnpot_.atom(m).name()+"_"+nnPotOpt.nnpot_.atom(l).name()+".dat";
						FILE* writer=fopen(filename.c_str(),"w");
						if(writer!=NULL){
							for(unsigned int i=0; i<N; ++i){
								const double angle=num_const::PI*i/(N-1.0);
								fprintf(writer,"%f ",angle);
								for(unsigned int j=0; j<nnPotOpt.nnpot_.basisA(n,m,l).nfA(); ++j){
									double tvec[3]={0,0,0};
									double cvec[3]={0,0,0};
									fprintf(writer,"%f ",nnPotOpt.nnpot_.basisA(n,m,l).fA(j).val(std::cos(angle),tvec,cvec));
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
		MPI_Barrier(MPI_COMM_WORLD);
		
		//======== write the nn's ========
		if(NN_POT_TRAIN_PRINT_STATUS>-1 && rank==0) std::cout<<"writing the nn's\n";
		if(rank==0){
			nnPotOpt.nnpot_.tail()="";
			nnPotOpt.nnpot_.write();
		}
		//======== write restart file ========
		if(NN_POT_TRAIN_PRINT_STATUS>-1 && rank==0) std::cout<<"writing restart file\n";
		if(rank==0){
			std::string file=nnPotOpt.restart_file_+".restart";
			nnPotOpt.write_restart(file.c_str());
		}
		
		//======== finalize mpi ========
		if(NN_POT_TRAIN_PRINT_STATUS>-1 && rank==0) std::cout<<"finalizing mpi\n"<<std::flush;
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
	}catch(std::exception& e){
		std::cout<<"ERROR in nn_pot_train::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//======== free local variables ========
	if(thread_dist_train!=NULL)   delete[] thread_dist_train;
	if(thread_dist_val!=NULL)     delete[] thread_dist_val;
	if(thread_dist_test!=NULL)    delete[] thread_dist_test;
	if(thread_offset_train!=NULL) delete[] thread_offset_train;
	if(thread_offset_val!=NULL)   delete[] thread_offset_val;
	if(thread_offset_test!=NULL)  delete[] thread_offset_test;
	delete[] paramfile;
	delete[] input;
	
	return 0;
}
