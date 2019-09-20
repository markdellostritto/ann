#include "nn_pot.hpp"

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const NNPot& obj){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"nbytes(const NNPot&):\n";
	unsigned int size=0;
	//species
	if(NN_POT_PRINT_STATUS>1) std::cout<<"sizing species\n";
	size+=sizeof(unsigned int);//natoms
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		size+=nbytes(obj.atom(i));//atoms
	}
	size+=nbytes(obj.atomMap());
	//cutoff
	if(NN_POT_PRINT_STATUS>1) std::cout<<"sizing cutoff\n";
	size+=sizeof(double);//rc_
	//element nn's
	if(NN_POT_PRINT_STATUS>1) std::cout<<"sizing nn's\n";
	for(unsigned int i=0; i<obj.nAtoms(); ++i) size+=nbytes(obj.nn(i));
	//basis for pair/triple interactions
	if(NN_POT_PRINT_STATUS>1) std::cout<<"sizing radial basis\n";
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		for(unsigned int j=0; j<obj.nAtoms(); ++j){
			size+=nbytes(obj.basisR(i,j));
		}
	}
	if(NN_POT_PRINT_STATUS>1) std::cout<<"sizing angular basis\n";
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		for(unsigned int j=0; j<obj.nAtoms(); ++j){
			for(unsigned int k=j; k<obj.nAtoms(); ++k){
				size+=nbytes(obj.basisA(i,j,k));
			}
		}
	}
	//input/output
	if(NN_POT_PRINT_STATUS>1) std::cout<<"sizing input/output\n";
	size+=nbytes(obj.head());
	size+=nbytes(obj.tail());
	//return the size
	return size;
}

//**********************************************
// packing
//**********************************************

template <> void pack(const NNPot& obj, char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"pack(const NNPot&,char*):\n";
	unsigned int pos=0;
	//species
	if(NN_POT_PRINT_STATUS>1) std::cout<<"packing species\n";
	const unsigned int nAtoms=obj.nAtoms();
	std::memcpy(arr+pos,&nAtoms,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		pack(obj.atom(i),arr+pos); pos+=nbytes(obj.atom(i));
	}
	pack(obj.atomMap(),arr+pos); pos+=nbytes(obj.atomMap());
	//cutoff
	if(NN_POT_PRINT_STATUS>1) std::cout<<"packing cutoff\n";
	std::memcpy(arr+pos,&obj.rc(),sizeof(double)); pos+=sizeof(double);
	//element nn's
	if(NN_POT_PRINT_STATUS>1) std::cout<<"packing nn's\n";
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		pack(obj.nn(i),arr+pos); pos+=nbytes(obj.nn(i));
	}
	//basis for pair/triple interactions
	if(NN_POT_PRINT_STATUS>1) std::cout<<"packing radial basis\n";
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		for(unsigned int j=0; j<obj.nAtoms(); ++j){
			pack(obj.basisR(i,j),arr+pos); pos+=nbytes(obj.basisR(i,j));
		}
	}
	if(NN_POT_PRINT_STATUS>1) std::cout<<"packing angular basis\n";
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		for(unsigned int j=0; j<obj.nAtoms(); ++j){
			for(unsigned int k=j; k<obj.nAtoms(); ++k){
				pack(obj.basisA(i,j,k),arr+pos); pos+=nbytes(obj.basisA(i,j,k));
			}
		}
	}
	//input/output
	if(NN_POT_PRINT_STATUS>1) std::cout<<"packing input/output\n";
	pack(obj.head(),arr+pos); pos+=nbytes(obj.head());
	pack(obj.tail(),arr+pos); pos+=nbytes(obj.tail());
}

//**********************************************
// unpacking
//**********************************************

template <> void unpack(NNPot& obj, const char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"unpack(NNPot&,const char*):\n";
	unsigned int pos=0;
	//species
	if(NN_POT_PRINT_STATUS>1) std::cout<<"unpacking species\n";
	unsigned int nAtoms=0;
	std::memcpy(&nAtoms,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::vector<Atom> atoms(nAtoms);
	for(unsigned int i=0; i<nAtoms; ++i){
		unpack(atoms[i],arr+pos); pos+=nbytes(atoms[i]);
	}
	Map<unsigned int,unsigned int> map;
	unpack(map,arr+pos); pos+=nbytes(map);
	obj.resize(atoms);
	obj.atomMap()=map;
	//cutoff
	if(NN_POT_PRINT_STATUS>1) std::cout<<"unpacking cutoff\n";
	double rc=0;
	std::memcpy(&rc,arr+pos,sizeof(double)); pos+=sizeof(double);
	obj.rc()=rc;
	//element nn's
	if(NN_POT_PRINT_STATUS>1) std::cout<<"unpacking nn's\n";
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		unpack(obj.nn(i),arr+pos); pos+=nbytes(obj.nn(i));
	}
	//basis for pair/triple interactions
	if(NN_POT_PRINT_STATUS>1) std::cout<<"unpacking radial basis\n";
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		for(unsigned int j=0; j<obj.nAtoms(); ++j){
			unpack(obj.basisR(i,j),arr+pos); pos+=nbytes(obj.basisR(i,j));
		}
	}
	if(NN_POT_PRINT_STATUS>1) std::cout<<"unpacking angular basis\n";
	for(unsigned int i=0; i<obj.nAtoms(); ++i){
		for(unsigned int j=0; j<obj.nAtoms(); ++j){
			for(unsigned int k=j; k<obj.nAtoms(); ++k){
				unpack(obj.basisA(i,j,k),arr+pos); pos+=nbytes(obj.basisA(i,j,k));
			}
		}
	}
	//input/output
	if(NN_POT_PRINT_STATUS>1) std::cout<<"unpacking input/output\n";
	unpack(obj.head(),arr+pos); pos+=nbytes(obj.head());
	unpack(obj.tail(),arr+pos); pos+=nbytes(obj.tail());
	//intialize the inputs and offsets
	obj.init_inputs();
}

}

//************************************************************
// NNPot::Init - Neural Network Potential - Initialization from Scratch
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPot::Init& init){
	out<<"**************************************************\n";
	out<<"**************** NN - POT - INIT ****************\n";
	out<<"TRANSFER = "<<init.tfType<<"\n";
	out<<"LAMBDA   = "<<init.lambda<<"\n";
	out<<"N_HIDDEN = "; for(unsigned int i=0; i<init.nh.size(); ++i) std::cout<<init.nh[i]<<" "; std::cout<<"\n";
	out<<"**************** NN - POT - INIT ****************\n";
	out<<"**************************************************";
	return out;
}

void NNPot::Init::defaults(){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::Init::defaults():\n";
	//network configuration
		nh.clear();
	//regularization
		lambda=0;
	//transfer function
		tfType=NN::TransferN::TANH;
}

//************************************************************
// NNPot - Neural Network Potential
//************************************************************

//operators

std::ostream& operator<<(std::ostream& out, const NNPot& nnpot){
	out<<"**************************************************\n";
	out<<"******************** NN - POT ********************\n";
	//file i/o
	out<<"HEAD     = "<<nnpot.head_<<"\n";
	out<<"TAIL     = "<<nnpot.tail_<<"\n";
	//rcut
	out<<"R_CUT    = "<<nnpot.rc_<<"\n";
	//species
	out<<"N_ATOMS  = "<<nnpot.atoms_.size()<<"\n";
	out<<"ATOMS    = \n";
	for(unsigned int i=0; i<nnpot.atoms_.size(); ++i) std::cout<<"\t"<<nnpot.atoms_[i]<<"\n";
	//potential parameters
	out<<"N_INPUT  = "; for(unsigned int i=0; i<nnpot.nInput_.size(); ++i) std::cout<<nnpot.nInput_[i]<<" "; std::cout<<"\n";
	out<<"N_INPUTR = "; for(unsigned int i=0; i<nnpot.nInputR_.size(); ++i) std::cout<<nnpot.nInputR_[i]<<" "; std::cout<<"\n";
	out<<"N_INPUTA = "; for(unsigned int i=0; i<nnpot.nInputA_.size(); ++i) std::cout<<nnpot.nInputA_[i]<<" "; std::cout<<"\n";
	//basis
	for(unsigned int i=0; i<nnpot.atoms_.size(); ++i){
		for(unsigned int j=0; j<nnpot.atoms_.size(); ++j){
			std::cout<<nnpot.atoms_[i].name()<<"-"<<nnpot.atoms_[j].name()<<" "<<nnpot.basisR_[i][j]<<"\n";
		}
	}
	for(unsigned int i=0; i<nnpot.atoms_.size(); ++i){
		for(unsigned int j=0; j<nnpot.atoms_.size(); ++j){
			for(unsigned int k=j; k<nnpot.atoms_.size(); ++k){
				std::cout<<nnpot.atoms_[i].name()<<"-"<<nnpot.atoms_[j].name()<<"-"<<nnpot.atoms_[k].name()<<" "<<nnpot.basisA_[i](j,k)<<"\n";
			}
		}
	}
	out<<"******************** NN - POT ********************\n";
	out<<"**************************************************";
	return out;
}

//member functions

//set defaults for the neural network potential
void NNPot::defaults(){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::defaults():\n";
	//species
		atoms_.clear();
		atomMap_.clear();
	//element nn's
		nn_.clear();
	//basis functions
		rc_=0;
		basisR_.clear();
		basisA_.clear();
	//network configuration
		nInput_.clear();
		nInputR_.clear();
		nInputA_.clear();
		offsetR_.clear();
		offsetA_.clear();
	//input/output
		head_="ann_";
		tail_="";
	//resize the lattice vector shifts
		R_.resize(3*3*3,Eigen::Vector3d::Zero());
}

//resizing

//set the number of species and species names to the total number of species in the simulations
void NNPot::resize(const Structure& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::resize(const Structure&):\n";
	//set the species
		unsigned int nAtoms=0;
		atomMap_.clear();
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			if(!atomMap_.find(string::hash(struc.name(n)))){
				atomMap_.add(string::hash(struc.name(n)),nAtoms++);
				atoms_.push_back(Atom());
				atoms_.back().name()=struc.name(n);
				atoms_.back().id()=string::hash(struc.name(n));
				atoms_.back().mass()=struc.mass(n);
			}
		}
		if(NN_POT_PRINT_DATA>0){
			std::cout<<"====================================\n";
			std::cout<<"Atoms = \n";
			for(unsigned i=0; i<atoms_.size(); ++i){
				std::cout<<"\t"<<atoms_[i]<<"\n";
			}
			std::cout<<"AtomMap = \n";
			for(unsigned i=0; i<atomMap_.size(); ++i){
				std::cout<<"\t"<<atomMap_.key(i)<<" "<<atomMap_.val(i)<<"\n";
			}
			std::cout<<"====================================\n";
		}
	//set the radial basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"resizing the radial basis\n";
		basisR_.resize(atoms_.size(),std::vector<BasisR>(atoms_.size()));
	//set the angular basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"resizing the angular basis\n";
		basisA_.resize(atoms_.size(),LMat<BasisA>(atoms_.size()));
	//resize the network
		nn_.resize(nAtoms);
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"set the number of inputs\n";
		nInput_.resize(atoms_.size(),0);
		nInputR_.resize(atoms_.size(),0);
		nInputA_.resize(atoms_.size(),0);
		offsetR_.resize(atoms_.size(),std::vector<unsigned int>(atoms_.size(),0));
		offsetA_.resize(atoms_.size(),LMat<unsigned int>(atoms_.size()));
}

//set the number of species and species names to the total number of species in the simulations
void NNPot::resize(const std::vector<Structure>& strucv){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::resize(const std::vector<Structure>&):\n";
	//set the species
		unsigned int nAtoms=0;
		atomMap_.clear();
		for(unsigned int i=0; i<strucv.size(); ++i){
			for(unsigned int n=0; n<strucv[i].nAtoms(); ++n){
				if(!atomMap_.find(string::hash(strucv[i].name(n)))){
					atomMap_.add(string::hash(strucv[i].name(n)),nAtoms++);
				}
			}
		}
		if(NN_POT_PRINT_DATA>0){
			std::cout<<"====================================\n";
			std::cout<<"Atoms = \n";
			for(unsigned i=0; i<atoms_.size(); ++i){
				std::cout<<"\t"<<atoms_[i]<<"\n";
			}
			std::cout<<"AtomMap = \n";
			for(unsigned i=0; i<atomMap_.size(); ++i){
				std::cout<<"\t"<<atomMap_.key(i)<<" "<<atomMap_.val(i)<<"\n";
			}
			std::cout<<"====================================\n";
		}
	//set the radial basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"resizing the radial basis\n";
		basisR_.resize(atomMap_.size(),std::vector<BasisR>(atomMap_.size()));
	//set the angular basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"resizing the angular basis\n";
		basisA_.resize(atomMap_.size(),LMat<BasisA>(atomMap_.size()));
	//resize the network
		nn_.resize(nAtoms);
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"set the number of inputs\n";
		nInput_.resize(atomMap_.size(),0);
		nInputR_.resize(atomMap_.size(),0);
		nInputA_.resize(atomMap_.size(),0);
		offsetR_.resize(atomMap_.size(),std::vector<unsigned int>(atomMap_.size(),0));
		offsetA_.resize(atomMap_.size(),LMat<unsigned int>(atomMap_.size()));
}

//set the number of species and species names to the total number of species in the simulations
void NNPot::resize(const std::vector<std::string>& speciesNames){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::resize(const std::vector<std::string>&):\n";
	//set the species
		atomMap_.clear();
		for(unsigned int i=0; i<speciesNames.size(); ++i){
			atomMap_.add(string::hash(speciesNames[i]),i);
			atoms_.push_back(Atom());
			atoms_.back().name()=speciesNames[i];
			atoms_.back().id()=string::hash(speciesNames[i]);
		}
		if(NN_POT_PRINT_DATA>0){
			std::cout<<"====================================\n";
			std::cout<<"Atoms = \n";
			for(unsigned i=0; i<atoms_.size(); ++i){
				std::cout<<"\t"<<atoms_[i]<<"\n";
			}
			std::cout<<"AtomMap = \n";
			for(unsigned i=0; i<atomMap_.size(); ++i){
				std::cout<<"\t"<<atomMap_.key(i)<<" "<<atomMap_.val(i)<<"\n";
			}
			std::cout<<"====================================\n";
		}
	//set the radial basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"resizing the radial basis\n";
		basisR_.resize(atomMap_.size(),std::vector<BasisR>(atomMap_.size()));
	//set the angular basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"resizing the angular basis\n";
		basisA_.resize(atomMap_.size(),LMat<BasisA>(atomMap_.size()));
	//resize the network
		nn_.resize(atomMap_.size());
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"set the number of inputs\n";
		nInput_.resize(atomMap_.size(),0);
		nInputR_.resize(atomMap_.size(),0);
		nInputA_.resize(atomMap_.size(),0);
		offsetR_.resize(atomMap_.size(),std::vector<unsigned int>(atomMap_.size(),0));
		offsetA_.resize(atomMap_.size(),LMat<unsigned int>(atomMap_.size()));
}

//set the number of species and species names to the total number of species in the simulations
void NNPot::resize(const std::vector<Atom>& atoms){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::resize(const std::vector<Atom>&):\n";
	//set the species
		atoms_=atoms;
		atomMap_.clear();
		for(unsigned int i=0; i<atoms.size(); ++i){
			atomMap_.add(string::hash(atoms[i].name()),i);
		}
		if(NN_POT_PRINT_DATA>0){
			std::cout<<"====================================\n";
			std::cout<<"Atoms = \n";
			for(unsigned i=0; i<atoms_.size(); ++i){
				std::cout<<"\t"<<atoms_[i]<<"\n";
			}
			std::cout<<"AtomMap = \n";
			for(unsigned i=0; i<atomMap_.size(); ++i){
				std::cout<<"\t"<<atomMap_.key(i)<<" "<<atomMap_.val(i)<<"\n";
			}
			std::cout<<"====================================\n";
		}
	//set the radial basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"resizing the radial basis\n";
		basisR_.resize(atoms_.size(),std::vector<BasisR>(atoms_.size()));
	//set the angular basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"resizing the angular basis\n";
		basisA_.resize(atoms_.size(),LMat<BasisA>(atoms_.size()));
	//resize the network
		nn_.resize(atoms_.size());
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"set the number of inputs\n";
		nInput_.resize(atoms_.size(),0);
		nInputR_.resize(atoms_.size(),0);
		nInputA_.resize(atoms_.size(),0);
		offsetR_.resize(atoms_.size(),std::vector<unsigned int>(atoms_.size(),0));
		offsetA_.resize(atoms_.size(),LMat<unsigned int>(atoms_.size()));
}

//set the number of inputs and offsets, and resize the neural networks
void NNPot::init(const NNPot::Init& init){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::init(const NNPot::Init&):\n";
	//check the parameters
		if(init.tfType==NN::TransferN::UNKNOWN) throw std::invalid_argument("Invalid transfer function.");
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"setting the number of inputs\n";
		init_inputs();
	//resize the number of neural networks
		if(NN_POT_PRINT_STATUS>0) std::cout<<"resizing neural networks\n";
		nn_.resize(atoms_.size());
		for(unsigned int i=0; i<nn_.size(); ++i){
			//set the transfer function
			nn_[i].tfType()=init.tfType;
			//set regularization parameter
			nn_[i].lambda()=init.lambda;
			//resize the network
			nn_[i].resize(nInput_[i],init.nh,nOutput_);
		}
}

//nn-struc

//resize the symmetry function vectors to store the inputs
void NNPot::init_symm(Structure& struc)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::init_symm(Structure&):\n";
	for(unsigned int n=0; n<struc.nAtoms(); ++n){
		struc.symm(n).resize(nInput_[atom_index(struc.name(n))]);
	}
}

//initialize the inputs numbers and offsets
void NNPot::init_inputs(){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::init_inputs():\n";
	for(unsigned int n=0; n<atoms_.size(); ++n){
		nInputR_[n]=0;
		for(unsigned int i=0; i<atoms_.size(); ++i){
			nInputR_[n]+=basisR_[n][i].nfR();
		}
	}
	for(unsigned int n=0; n<atoms_.size(); ++n){
		offsetR_[n][0]=0;
		for(unsigned int i=1; i<atoms_.size(); ++i){
			offsetR_[n][i]=offsetR_[n][i-1]+basisR_[n][i-1].nfR();
		}
	}
	for(unsigned int n=0; n<atoms_.size(); ++n){
		nInputA_[n]=0;
		for(unsigned int i=0; i<atoms_.size(); ++i){
			for(unsigned int j=i; j<atoms_.size(); ++j){
				nInputA_[n]+=basisA_[n](j,i).nfA();
			}
		}
	}
	for(unsigned int n=0; n<atoms_.size(); ++n){
		offsetA_[n][0]=0;
		for(unsigned int i=1; i<basisA_[n].size(); ++i){
			offsetA_[n][i]=offsetA_[n][i-1]+basisA_[n][i-1].nfA();
		}
	}
	for(unsigned int n=0; n<atoms_.size(); ++n) nInput_[n]=nInputR_[n]+nInputA_[n];
}

//compute inputs - symmetry functions 
void NNPot::inputs_symm(Structure& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::inputs_symm(Structure&):\n";
	//lattice vector shifts
	const short shellx=std::ceil(rc_/struc.cell().R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
	const short shelly=std::ceil(rc_/struc.cell().R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
	const short shellz=std::ceil(rc_/struc.cell().R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
	const unsigned short Rsize=(2*shellx+1)*(2*shelly+1)*(2*shellz+1);
	if(NN_POT_PRINT_DATA>0) std::cout<<"R = "<<struc.cell().R()<<"\n";
	if(NN_POT_PRINT_DATA>0) std::cout<<"shell = ("<<shellx<<","<<shelly<<","<<shellz<<") = "<<(2*shellx+1)*(2*shelly+1)*(2*shellz+1)<<"\n";
	if(Rsize>R_.size()) R_.resize(Rsize);
	unsigned short count=0;
	for(short ix=-shellx; ix<=shellx; ++ix){
		for(short iy=-shelly; iy<=shelly; ++iy){
			for(short iz=-shellz; iz<=shellz; ++iz){
				R_[count].setZero();
				R_[count].noalias()+=ix*struc.cell().R().col(0);
				R_[count].noalias()+=iy*struc.cell().R().col(1);
				R_[count].noalias()+=iz*struc.cell().R().col(2);
				++count;
			}
		}
	}
	//loop over all atoms
	if(NN_POT_PRINT_STATUS>0) std::cout<<"computing symmetry functions\n";
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		//find the index of the species of atom j
		const unsigned short II=atomMap_[string::hash(struc.name(i).c_str())];
		//reset the inputs
		if(NN_POT_PRINT_STATUS>2) std::cout<<"resetting inputs\n";
		struc.symm(i).setZero();
		//loop over pairs
		for(unsigned int j=0; j<struc.nAtoms(); ++j){
			//find the index of the species of atom j
			const unsigned short JJ=atomMap_[string::hash(struc.name(j).c_str())];
			//calc rIJ
			if(NN_POT_PRINT_STATUS>2) std::cout<<"symm r("<<i<<","<<j<<")\n";
			//calc radial contribution - loop over all radial functions
			if(NN_POT_PRINT_STATUS>2) std::cout<<"computing radial functions\n";
			Cell::diff(struc.posn(i),struc.posn(j),rIJ_,struc.cell().R(),struc.cell().RInv());
			//loop over lattice vector shifts
			for(unsigned short idIJ=0; idIJ<Rsize; ++idIJ){
				rIJt_.noalias()=rIJ_+R_[idIJ]; const double dIJ=rIJt_.norm();
				if(num_const::ZERO<dIJ && dIJ<rc_){
					if(NN_POT_PRINT_STATUS>2) std::cout<<"setting radial symmetry functions\n";
					//compute the IJ contribution to all radial basis functions
					unsigned short offset_=offsetR_[II][JJ];
					BasisR& basisRij_=basisR_[II][JJ];
					basisRij_.symm(dIJ);
					for(unsigned short nr=0; nr<basisRij_.nfR(); ++nr){
						struc.symm(i)[offset_+nr]+=basisRij_.symm()[nr];
					}
					//loop over all triplets
					for(unsigned int k=0; k<struc.nAtoms(); ++k){
						//find the index of the species of atom i
						const unsigned short KK=atomMap_[string::hash(struc.name(k).c_str())];
						//calculate rIK and rJK
						if(NN_POT_PRINT_STATUS>2) std::cout<<"computing theta("<<i<<","<<j<<","<<k<<")\n";
						Cell::diff(struc.posn(i),struc.posn(k),rIK_,struc.cell().R(),struc.cell().RInv());
						Cell::diff(struc.posn(j),struc.posn(k),rJK_,struc.cell().R(),struc.cell().RInv());
						//loop over all cell shifts
						for(unsigned short idIK=0; idIK<Rsize; ++idIK){
							rIKt_.noalias()=rIK_+R_[idIK]; const double dIK=rIKt_.norm();
							if(num_const::ZERO<dIK && dIK<rc_){
								for(unsigned short idJK=0; idJK<Rsize; ++idJK){
									rJKt_.noalias()=rJK_+R_[idJK]; const double dJK=rJKt_.norm();
									if(num_const::ZERO<dJK && dJK<rc_){
										//compute the IJ,IK,JK contribution to all angular basis functions
										offset_=nInputR_[II]+offsetA_[II](JJ,KK);
										BasisA& basisAijk_=basisA_[II](JJ,KK);
										const double cosIJK=rIJt_.dot(rIKt_)/(dIJ*dIK);
										const double d[3]={dIJ,dIK,dJK};
										basisAijk_.symm(cosIJK,d);
										for(unsigned short na=0; na<basisAijk_.nfA(); ++na){
											struc.symm(i)[offset_+na]+=basisAijk_.symm()[na];
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

//compute forces
void NNPot::forces(Structure& struc, bool calc_symm){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::forces(Structure&,bool):\n";
	//local variables
	Eigen::VectorXd dEdG;
	//reset the force
	for(unsigned int i=0; i<struc.nAtoms(); ++i) struc.force(i).setZero();
	//lattice vector shifts
	const short shellx=std::ceil(rc_/struc.cell().R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
	const short shelly=std::ceil(rc_/struc.cell().R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
	const short shellz=std::ceil(rc_/struc.cell().R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
	if(NN_POT_PRINT_DATA>0) std::cout<<"R = "<<struc.cell().R()<<"\n";
	if(NN_POT_PRINT_DATA>0) std::cout<<"shell = ("<<shellx<<","<<shelly<<","<<shellz<<")\n";
	const unsigned short Rsize=(2*shellx+1)*(2*shelly+1)*(2*shellz+1);
	if(Rsize>R_.size()) R_.resize(Rsize);
	unsigned short count=0;
	for(short ix=-shellx; ix<=shellx; ++ix){
		for(short iy=-shelly; iy<=shelly; ++iy){
			for(short iz=-shellz; iz<=shellz; ++iz){
				R_[count].setZero();
				R_[count].noalias()+=ix*struc.cell().R().col(0);
				R_[count].noalias()+=iy*struc.cell().R().col(1);
				R_[count].noalias()+=iz*struc.cell().R().col(2);
				++count;
			}
		}
	}
	//set the inputs for the atoms
	if(calc_symm) inputs_symm(struc);
	//loop over all atoms
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		//find the index of the species of atom i
		const unsigned short II=atomMap_[string::hash(struc.name(i))];
		//execute the appropriate network
		nn_[II].execute(struc.symm(i));
		//calculate the network gradient
		nn_[II].grad_out();
		//set the gradient
		dEdG=nn_[II].dOut(0).row(0);
		//loop over pairs
		for(unsigned int j=0; j<struc.nAtoms(); ++j){
			//find the index of the species of atom j
			const unsigned short JJ=atomMap_[string::hash(struc.name(j))];
			//calc rIJ
			Cell::diff(struc.posn(i),struc.posn(j),rIJ_,struc.cell().R(),struc.cell().RInv());
			//loop over lattice vector shifts
			for(unsigned short idIJ=0; idIJ<Rsize; ++idIJ){
				rIJt_.noalias()=rIJ_+R_[idIJ]; const double dIJ=rIJt_.norm();
				if(dIJ<rc_ && dIJ>num_const::ZERO){
					rIJt_/=dIJ;
					//compute the IJ contribution to the radial force
					unsigned short offset_=offsetR_[II][JJ];
					const double* dedg=dEdG.data()+offset_;
					const double amp=basisR_[II][JJ].force(dIJ,dedg);
					struc.force(i).noalias()+=amp*rIJt_;
					struc.force(j).noalias()-=amp*rIJt_;
					//loop over all triplets
					for(unsigned int k=0; k<struc.nAtoms(); ++k){
						//find the index of the species of atom k
						const unsigned short KK=atomMap_[string::hash(struc.name(k))];
						//calculate rIK and rJK
						if(NN_POT_PRINT_STATUS>2) std::cout<<"computing theta("<<i<<","<<j<<","<<k<<")\n";
						Cell::diff(struc.posn(i),struc.posn(k),rIK_,struc.cell().R(),struc.cell().RInv());
						Cell::diff(struc.posn(j),struc.posn(k),rJK_,struc.cell().R(),struc.cell().RInv());
						//loop over all cell shifts
						for(unsigned short idIK=0; idIK<Rsize; ++idIK){
							rIKt_.noalias()=rIK_+R_[idIK]; const double dIK=rIKt_.norm();
							if(dIK<rc_ && dIK>num_const::ZERO){
								for(unsigned short idJK=0; idJK<Rsize; ++idJK){
									rJKt_.noalias()=rJK_+R_[idJK]; const double dJK=rJKt_.norm();
									if(dJK<rc_ && dJK>num_const::ZERO){
										//compute the IJ,IK,JK contribution to the angular force
										rIKt_/=dIK;
										offset_=nInputR_[II]+offsetA_[II][JJ];
										const double cosIJK=rIJt_.dot(rIKt_);
										const double d[3]={dIJ,dIK,dJK};
										double fij[2]={0,0},fik[2]={0,0};
										basisA_[II](JJ,KK).force(fij,fik,cosIJK,d,dEdG.data()+offset_);
										struc.force(i).noalias()+=(fij[0]+fij[1])*rIJt_+(fik[0]+fik[1])*rIKt_;
										struc.force(j).noalias()-=fij[0]*rIJt_+fik[1]*rIKt_;
										struc.force(k).noalias()-=fik[0]*rIKt_+fij[1]*rIJt_;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

//execute all atomic networks and return energy
double NNPot::energy(Structure& struc, bool calc_symm){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::energy(Structure&,bool):\n";
	double energy=0;
	//set the inputs for the atoms
	if(calc_symm) inputs_symm(struc);
	//loop over atoms
	for(unsigned int n=0; n<struc.nAtoms(); ++n){
		//set the index
		const unsigned int index=atomMap_[string::hash(struc.name(n))];
		//execute the network
		nn_[index].execute(struc.symm(n));
		//add the energy
		energy+=nn_[index].output()[0]+atoms_[index].energy();
	}
	return energy;
}

//static functions

//write all neural network potentials to file
void NNPot::write()const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::write():\n";
	//==== local function variables ====
	std::string filename;
	//==== write all networks ====
	for(unsigned int n=0; n<atoms_.size(); ++n){
		//create the file name
		filename=head_+atoms_[n].name()+tail_;
		if(NN_POT_PRINT_DATA>0) std::cout<<"filename = "<<filename<<"\n";
		//write the network
		NNPot::write(n,filename);
	}
}

//read all neural network potentials from file
void NNPot::read(){
	if(NN_POT_PRINT_FUNC) std::cout<<"NNPot::read():\n";
	//==== local function variables ====
	std::string filename;
	//==== read all networks ====
	for(unsigned int n=0; n<atoms_.size(); ++n){
		//create the file name
		filename=head_+atoms_[n].name()+tail_;
		if(NN_POT_PRINT_DATA>0) std::cout<<"filename = "<<filename<<"\n";
		//read the network
		NNPot::read(n,filename);
	}
}

//write neural network potential to file
void NNPot::write(unsigned int index, const std::string& filename)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::write(unsigned int,const std::string&):\n";
	FILE* writer=NULL;
	writer=fopen(filename.c_str(),"w");
	if(writer!=NULL){
		write(index,writer);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("I/O ERROR: Could not write to nnpot file: \"")+filename+std::string("\""));
}

//write neural network potential to file
void NNPot::write(unsigned int index, FILE* writer)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::write(unsigned int,FILE*):\n";
	//==== write the header ====
	fprintf(writer,"ann\n");
	//==== write the global cutoff ====
	fprintf(writer,"cut %f\n",rc_);
	//==== write the central species ====
	fprintf(writer,"%s %f %f\n",atoms_[index].name().c_str(),atoms_[index].mass(),atoms_[index].energy());
	//==== write the number of species ====
	fprintf(writer,"nspecies %i\n",atoms_.size());
	//==== write all species ====
	for(unsigned int i=0; i<atoms_.size(); ++i){
		fprintf(writer,"%s %f %f\n",atoms_[i].name().c_str(),atoms_[i].mass(),atoms_[i].energy());
	}
	//==== write the radial basis ====
	for(unsigned int j=0; j<atoms_.size(); ++j){
		fprintf(writer,"basis_radial %s\n",atoms_[j].name().c_str());
		BasisR::write(writer,basisR_[index][j]);
	}
	//==== write the angular basis ====
	for(unsigned int j=0; j<atoms_.size(); ++j){
		for(unsigned int k=j; k<atoms_.size(); ++k){
			fprintf(writer,"basis_angular %s %s\n",atoms_[j].name().c_str(),atoms_[k].name().c_str());
			BasisA::write(writer,basisA_[index](j,k));
		}
	}
	//==== write the neural network ====
	NN::Network::write(writer,nn_[index]);
}

//read neural network potential from file
void NNPot::read(unsigned int index, const std::string& filename){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::read(unsigned int,const std::string&):\n";
	FILE* reader=NULL;
	reader=fopen(filename.c_str(),"r");
	if(reader!=NULL){
		read(index,reader);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("I/O ERROR: Could not open nnpot file: \"")+filename+std::string("\""));
}

//read neural network potential from file
void NNPot::read(unsigned int index, FILE* reader){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::read(unsigned int,FILE*):\n";
	//==== local function variables ====
	Atom atom;
	std::vector<std::string> strlist;
	char* input=new char[string::M];
	//==== reader in header ====
	fgets(input,string::M,reader);
	//==== read in global cutoff ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const double rc=std::atof(std::strtok(NULL,string::WS));
	if(rc>rc_) rc_=rc;
	//==== read the central atom ====
	Atom::read(fgets(input,string::M,reader),atom);
	if(atom!=atoms_[index]) throw std::invalid_argument("Mismatch between atom in nnpot and atom in file.");
	//==== read the number of species ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const unsigned int nspecies=std::atoi(std::strtok(NULL,string::WS));
	if(nspecies!=atoms_.size()) throw std::invalid_argument("Mismatch in number of species.");
	//==== read all species ====
	for(unsigned int i=0; i<nspecies; ++i){
		fgets(input,string::M,reader);
		Atom::read(input,atom);
		if(atom!=atoms_[i]) throw std::invalid_argument(std::string("Mismatch in species ")+std::to_string(i));
	}
	//==== read the radial basis ====
	for(unsigned int i=0; i<nspecies; ++i){
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const unsigned int ii=atom_index(strlist[1]);
		const unsigned int jj=atom_index(strlist[2]);
		if(ii!=index) throw std::invalid_argument("Invalid central atom.");
		BasisR::read(reader,basisR_[index][jj]);
	}
	//==== read the angular basis ====
	for(unsigned int i=0; i<nspecies; ++i){
		for(unsigned int j=i; j<nspecies; ++j){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const unsigned int ii=atom_index(strlist[1]);
			const unsigned int jj=atom_index(strlist[2]);
			const unsigned int kk=atom_index(strlist[3]);
			if(ii!=index) throw std::invalid_argument("Invalid central atom.");
			BasisA::read(reader,basisA_[index](jj,kk));
		}
	}
	//==== read the neural network ====
	NN::Network::read(reader,nn_[index]);
	//==== set the number of inputs and offsets ====
	nInputR_[index]=0;
	for(unsigned int i=0; i<atoms_.size(); ++i){
		nInputR_[index]+=basisR_[index][i].nfR();
	}
	offsetR_[index][0]=0;
	for(unsigned int i=1; i<atoms_.size(); ++i){
		offsetR_[index][i]=offsetR_[index][i-1]+basisR_[index][i-1].nfR();
	}
	nInputA_[index]=0;
	for(unsigned int i=0; i<atoms_.size(); ++i){
		for(unsigned int j=i; j<atoms_.size(); ++j){
			nInputA_[index]+=basisA_[index](j,i).nfA();
		}
	}
	offsetA_[index][0]=0;
	for(unsigned int i=1; i<basisA_[index].size(); ++i){
		offsetA_[index][i]=offsetA_[index][i-1]+basisA_[index][i-1].nfA();
	}
	nInput_[index]=nInputR_[index]+nInputA_[index];
	//======== free local variables ========
	delete[] input;
}

//operators
	
bool operator==(const NNPot& nnPot1, const NNPot& nnPot2){
	if(nnPot1.rc()!=nnPot2.rc()) return false;
	else if(nnPot1.nAtoms()!=nnPot2.nAtoms()) return false;
	else if(nnPot1.atomMap()!=nnPot2.atomMap()) return false;
	else if(nnPot1.head()!=nnPot2.head()) return false;
	else if(nnPot1.tail()!=nnPot2.tail()) return false;
	else{
		if(nnPot1.basisR()!=nnPot2.basisR()) return false;
		if(nnPot1.basisA()!=nnPot2.basisA()) return false;
		if(nnPot1.atoms()!=nnPot2.atoms()) return false;
		if(nnPot1.nn()!=nnPot2.nn()) return false;
		return true;
	}
}
