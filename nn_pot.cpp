#include "nn_pot.hpp"

//************************************************************
// NNPot::Init - Neural Network Potential - Initialization from Scratch
//************************************************************

std::ostream& operator<<(std::ostream& out, const NNPot::Init& init){
	out<<"**************************************************\n";
	out<<"**************** NN - POT - INIT ****************\n";
	out<<"BASIS:\n";
	out<<"\tNR         = "<<init.nR<<"\n";
	out<<"\tNA         = "<<init.nA<<"\n";
	out<<"\tPHIRN      = "<<init.phiRN<<"\n";
	out<<"\tPHIAN      = "<<init.phiAN<<"\n";
	out<<"CUTOFF:\n";
	out<<"\tRM         = "<<init.rm<<"\n";
	out<<"\tRC         = "<<init.rc<<"\n";
	out<<"\tCUTOFF     = "<<init.tcut<<"\n";
	out<<"ELEMENT NN:\n";
	out<<"\tTRANSFER   = "<<init.tfType<<"\n";
	out<<"\tLAMBDA     = "<<init.lambda<<"\n";
	out<<"**************** NN - POT - INIT ****************\n";
	out<<"**************************************************";
	return out;
}

void NNPot::Init::defaults(){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::Init::defaults():\n";
	//basis functions
		phiRN=PhiRN::G2;
		phiAN=PhiAN::G4;
		nR=0;
		nA=0;
	//cutoff 
		rm=0.0;
		rc=0.0;
		tcut=CutoffN::COS;
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
	//species
	out<<"SPECIES    = ";
	for(unsigned int i=0; i<nnpot.speciesMap_.size(); ++i) std::cout<<nnpot.speciesMap_.key(i)<<" ";
	std::cout<<"\n";
	//potential parameters
	out<<"HEADER     = "<<nnpot.header_<<"\n";
	out<<"N_ELEMENTS = "<<nnpot.speciesMap_.size()<<"\n";
	out<<"N_INPUT    = "; for(unsigned int i=0; i<nnpot.nInput_.size(); ++i) std::cout<<nnpot.nInput_[i]<<" "; std::cout<<"\n";
	out<<"N_INPUTR   = "; for(unsigned int i=0; i<nnpot.nInputR_.size(); ++i) std::cout<<nnpot.nInputR_[i]<<" "; std::cout<<"\n";
	out<<"N_INPUTA   = "; for(unsigned int i=0; i<nnpot.nInputA_.size(); ++i) std::cout<<nnpot.nInputA_[i]<<" "; std::cout<<"\n";
	//basis
	for(unsigned int i=0; i<nnpot.speciesMap_.size(); ++i){
		for(unsigned int j=0; j<nnpot.speciesMap_.size(); ++j){
			std::cout<<nnpot.speciesMap_.key(i)<<"-"<<nnpot.speciesMap_.key(j)<<" "<<nnpot.basisR_[i][j]<<"\n";
		}
	}
	for(unsigned int i=0; i<nnpot.speciesMap_.size(); ++i){
		for(unsigned int j=0; j<nnpot.speciesMap_.size(); ++j){
			for(unsigned int k=j; k<nnpot.speciesMap_.size(); ++k){
				std::cout<<nnpot.speciesMap_.key(i)<<"-"<<nnpot.speciesMap_.key(j)<<"-"<<nnpot.speciesMap_.key(k)<<" "<<nnpot.basisA_[i](j,k)<<"\n";
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
	//basis functions
		rc_=0;
		basisR_.clear();
		basisA_.clear();
	//element nn's
		speciesMap_.clear();
		nInput_.clear();
		nInputR_.clear();
		nInputA_.clear();
		offsetR_.clear();
		offsetA_.clear();
		nn_.clear();
	//pre-/post-conditioning
		energyAtom_.clear();
	//input/output
		header_="ann_";
	//resize the lattice vector shifts
		R_.resize(3*3*3,Eigen::Vector3d::Zero());
}

//initialize the basis functions with reasonable first guess, resize neural networks
void NNPot::init(const NNPot::Init& init_){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::init():\n";
	//check the parameters
		if(init_.tfType==NN::TransferN::UNKNOWN) throw std::invalid_argument("Invalid transfer function.");
		if(init_.rm<0) throw std::invalid_argument("Invalid minimum radius.");
		if(init_.rc<=0) throw std::invalid_argument("Invalid cutoff radius.");
		if(std::fabs(init_.rm-init_.rc)<num_const::ZERO) throw std::invalid_argument("Min and cutoff radius are equivalent.");
		if(init_.rm<num_const::ZERO){
			std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
			std::cout<<"!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!\n";
			std::cout<<"!! The minimum radius is close to zero.         !!\n";
			std::cout<<"!! This will lead to nodes with vanishing       !!\n";
			std::cout<<"!! values, complicating training and            !!\n";
			std::cout<<"!! introducing errors into force calculations.  !!\n";
			std::cout<<"!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!\n";
			std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
		}
		if(init_.nR==0) throw std::invalid_argument("Invalid number of radial basis functions.");
		if(init_.nA==0) throw std::invalid_argument("Invalid number of angular basis functions.");
		if(speciesMap_.size()==0) throw std::invalid_argument("Invalid number of species.");
	//set the global cutoff
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Setting global cutoff...\n";
		rc_=init_.rc;
	//set the radial basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Setting the radial basis...\n";
		basisR_.resize(speciesMap_.size(),std::vector<BasisR>(speciesMap_.size()));
		if(init_.phiRN==PhiRN::G1){
			for(unsigned int n=0; n<speciesMap_.size(); ++n){
				for(unsigned int i=0; i<speciesMap_.size(); ++i){
					BasisR::init_G1(basisR_[n][i],init_.tcut,init_.rm,init_.rc);
				}
			}
		} else if(init_.phiRN==PhiRN::G2){
			for(unsigned int n=0; n<speciesMap_.size(); ++n){
				for(unsigned int i=0; i<speciesMap_.size(); ++i){
					BasisR::init_G2(basisR_[n][i],init_.nR,init_.tcut,init_.rm,init_.rc);
				}
			}
		} else throw std::invalid_argument("Invalid radial basis type");
	//set the angular basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Setting the angular basis...\n";
		basisA_.resize(speciesMap_.size(),LMat<BasisA>(speciesMap_.size()));
		if(init_.phiAN==PhiAN::G3){
			for(unsigned int n=0; n<speciesMap_.size(); ++n){
				for(unsigned int i=0; i<basisA_[n].n(); ++i){
					for(unsigned int j=i; j<basisA_[n].n(); ++j){
						BasisA::init_G3(basisA_[n](j,i),init_.nA,init_.tcut,init_.rc);
					}
				}
			}
		} else if(init_.phiAN==PhiAN::G4){
			for(unsigned int n=0; n<speciesMap_.size(); ++n){
				for(unsigned int i=0; i<speciesMap_.size(); ++i){
					for(unsigned int j=i; j<speciesMap_.size(); ++j){
						BasisA::init_G4(basisA_[n](j,i),init_.nA,init_.tcut,init_.rc);
					}
				}
			}
		} else throw std::invalid_argument("Invalid angular basis type");
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Set the number of inputs...\n";
		nInput_.resize(speciesMap_.size(),0);
		nInputR_.resize(speciesMap_.size(),0);
		nInputA_.resize(speciesMap_.size(),0);
		offsetR_.resize(speciesMap_.size(),std::vector<unsigned int>(speciesMap_.size(),0));
		offsetA_.resize(speciesMap_.size(),LMat<unsigned int>(speciesMap_.size()));
		for(unsigned int n=0; n<speciesMap_.size(); ++n){
			for(unsigned int i=0; i<speciesMap_.size(); ++i){
				nInputR_[n]+=basisR_[n][i].fR.size();
			}
		}
		for(unsigned int n=0; n<speciesMap_.size(); ++n){
			for(unsigned int i=1; i<speciesMap_.size(); ++i){
				offsetR_[n][i]=offsetR_[n][i-1]+basisR_[n][i-1].fR.size();
			}
		}
		for(unsigned int n=0; n<speciesMap_.size(); ++n){
			for(unsigned int i=0; i<speciesMap_.size(); ++i){
				for(unsigned int j=i; j<speciesMap_.size(); ++j){
					nInputA_[n]+=basisA_[n](j,i).fA.size();
				}
			}
		}
		for(unsigned int n=0; n<speciesMap_.size(); ++n){
			for(unsigned int i=1; i<basisA_.size(); ++i){
				offsetA_[n][i]=offsetA_[n][i-1]+basisA_[n][i-1].fA.size();
			}
		}
		for(unsigned int n=0; n<speciesMap_.size(); ++n) nInput_[n]=nInputR_[n]+nInputA_[n];
	//resize the number of neural networks
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Resizing neural networks...\n";
		nn_.resize(speciesMap_.size());
		for(unsigned int i=0; i<nn_.size(); ++i){
			//set the transfer function
			nn_[i].tfType()=init_.tfType;
			//set regularization parameter
			nn_[i].lambda()=init_.lambda;
			//resize the network
			nn_[i].resize(nInput_[i],init_.nh,nOutput_);
		}
}

//resizing

//set the number of species and species names to the total number of species in the simulations
void NNPot::resize(const Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::resize(const Structure<AtomT>&):\n";
	//set the species
		unsigned int nSpecies=0;
		speciesMap_.clear();
		for(unsigned int n=0; n<struc.nSpecies(); ++n){
			if(!speciesMap_.find(struc.atomNames(n))){
				speciesMap_.add(struc.atomNames(n),nSpecies++);
			}
		}
		if(NN_POT_PRINT_DATA>0){
			std::cout<<"====================================\n";
			std::cout<<"SpeciesMap = \n";
			for(unsigned i=0; i<speciesMap_.size(); ++i){
				std::cout<<"("<<speciesMap_.key(i)<<","<<speciesMap_.val(i)<<") ";
			}
			std::cout<<"\n";
			std::cout<<"====================================\n";
		}
		energyAtom_.resize(nSpecies,0);
	//set the radial basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Resizing the radial basis...\n";
		basisR_.resize(speciesMap_.size(),std::vector<BasisR>(speciesMap_.size()));
	//set the angular basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Resizing the angular basis...\n";
		basisA_.resize(speciesMap_.size(),LMat<BasisA>(speciesMap_.size()));
	//resize the network
		nn_.resize(nSpecies);
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Set the number of inputs...\n";
		nInput_.resize(speciesMap_.size(),0);
		nInputR_.resize(speciesMap_.size(),0);
		nInputA_.resize(speciesMap_.size(),0);
		offsetR_.resize(speciesMap_.size(),std::vector<unsigned int>(speciesMap_.size(),0));
		offsetA_.resize(speciesMap_.size(),LMat<unsigned int>(speciesMap_.size()));
}

//set the number of species and species names to the total number of species in the simulations
void NNPot::resize(const std::vector<Structure<AtomT> >& strucv){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::resize(const std::vector<Structure<AtomT> >&):\n";
	//set the species
		unsigned int nSpecies=0;
		speciesMap_.clear();
		for(unsigned int i=0; i<strucv.size(); ++i){
			for(unsigned int n=0; n<strucv[i].nSpecies(); ++n){
				if(!speciesMap_.find(strucv[i].atomNames(n))){
					speciesMap_.add(strucv[i].atomNames(n),nSpecies++);
				}
			}
		}
		if(NN_POT_PRINT_DATA>0){
			std::cout<<"====================================\n";
			std::cout<<"SpeciesMap = \n";
			for(unsigned i=0; i<speciesMap_.size(); ++i){
				std::cout<<"("<<speciesMap_.key(i)<<","<<speciesMap_.val(i)<<") ";
			}
			std::cout<<"\n";
			std::cout<<"====================================\n";
		}
		energyAtom_.resize(nSpecies,0);
	//set the radial basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Resizing the radial basis...\n";
		basisR_.resize(speciesMap_.size(),std::vector<BasisR>(speciesMap_.size()));
	//set the angular basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Resizing the angular basis...\n";
		basisA_.resize(speciesMap_.size(),LMat<BasisA>(speciesMap_.size()));
	//resize the network
		nn_.resize(nSpecies);
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Set the number of inputs...\n";
		nInput_.resize(speciesMap_.size(),0);
		nInputR_.resize(speciesMap_.size(),0);
		nInputA_.resize(speciesMap_.size(),0);
		offsetR_.resize(speciesMap_.size(),std::vector<unsigned int>(speciesMap_.size(),0));
		offsetA_.resize(speciesMap_.size(),LMat<unsigned int>(speciesMap_.size()));
}

//set the number of species and species names to the total number of species in the simulations
void NNPot::resize(const std::vector<std::string>& speciesNames){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::resize(const std::vector<std::string>&):\n";
	//set the species
		speciesMap_.clear();
		for(unsigned int i=0; i<speciesNames.size(); ++i){
			speciesMap_.add(speciesNames[i],i);
		}
		energyAtom_.resize(speciesMap_.size(),0);
	//set the radial basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Resizing the radial basis...\n";
		basisR_.resize(speciesMap_.size(),std::vector<BasisR>(speciesMap_.size()));
	//set the angular basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Resizing the angular basis...\n";
		basisA_.resize(speciesMap_.size(),LMat<BasisA>(speciesMap_.size()));
	//resize the network
		nn_.resize(speciesMap_.size());
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Set the number of inputs...\n";
		nInput_.resize(speciesMap_.size(),0);
		nInputR_.resize(speciesMap_.size(),0);
		nInputA_.resize(speciesMap_.size(),0);
		offsetR_.resize(speciesMap_.size(),std::vector<unsigned int>(speciesMap_.size(),0));
		offsetA_.resize(speciesMap_.size(),LMat<unsigned int>(speciesMap_.size()));
}

//nn-struc

//resize the symmetry function vectors to store the inputs
void NNPot::initSymm(Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::initSymm(const Structure<AtomT>&):\n";
	for(unsigned int n=0; n<struc.nSpecies(); ++n){
		for(unsigned int m=0; m<struc.nAtoms(n); ++m){
			struc.atom(n,m).symm().resize(nInput_[n]);
		}
	}
}

//resize the symmetry function vectors to store the inputs
void NNPot::initSymm(std::vector<Structure<AtomT> >& strucv){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::initSymm(const std::vector<Structure<AtomT> >&):\n";
	for(unsigned int i=0; i<strucv.size(); ++i){
		for(unsigned int n=0; n<strucv[i].nSpecies(); ++n){
			for(unsigned int m=0; m<strucv[i].nAtoms(n); ++m){
				strucv[i].atom(n,m).symm().resize(nInput_[n]);
			}
		}
	}
}

//calculate inputs - symmetry functions 
void NNPot::inputs_symm(Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::inputs_symm(Structure<AtomT>&,unsigned int):\n";
	//lattice vector shifts
	unsigned short count=0;
	for(short ix=-1; ix<=1; ++ix){
		for(short iy=-1; iy<=1; ++iy){
			for(short iz=-1; iz<=1; ++iz){
				R_[count].setZero();
				R_[count].noalias()+=ix*struc.cell().R().col(0);
				R_[count].noalias()+=iy*struc.cell().R().col(1);
				R_[count].noalias()+=iz*struc.cell().R().col(2);
				++count;
			}
		}
	}
	//loop over all atoms
	if(NN_POT_PRINT_STATUS>0) std::cout<<"calculating symmetry functions...\n";
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		double dIJ,dIK,dJK;
		//find the index of the species of atom j
		unsigned int II=speciesMap_[struc.atom(i).name()];
		//reset the inputs
		if(NN_POT_PRINT_STATUS>2) std::cout<<"resetting inputs...\n";
		struc.atom(i).symm().setZero();
		//loop over pairs
		for(unsigned int j=0; j<struc.nAtoms(); ++j){
			if(i==j) continue;
			//find the index of the species of atom j
			unsigned int JJ=speciesMap_[struc.atom(j).name()];
			//calc rIJ
			if(NN_POT_PRINT_STATUS>2) std::cout<<"symm r("<<i<<","<<j<<")\n";
			//calc radial contribution - loop over all radial functions
			if(NN_POT_PRINT_STATUS>2) std::cout<<"Computing radial functions...\n";
			Cell::diff(struc.atom(i).posn(),struc.atom(j).posn(),rIJ_,struc.cell().R(),struc.cell().RInv());
			//loop over lattice vector shifts
			for(unsigned short idIJ=0; idIJ<R_.size(); ++idIJ){
				rIJt_.noalias()=rIJ_; rIJt_.noalias()+=R_[idIJ]; dIJ=rIJt_.norm();
				if(dIJ<rc_){
					if(NN_POT_PRINT_STATUS>2) std::cout<<"Setting radial symmetry functions...\n";
					//compute the IJ contribution to all radial basis functions
					for(unsigned int nr=0; nr<basisR_[II][JJ].fR.size(); ++nr){
						struc.atom(i).symm()[offsetR_[II][JJ]+nr]+=basisR_[II][JJ].fR[nr]->val(dIJ);
					}
					//loop over all triplets
					for(unsigned int k=0; k<struc.nAtoms(); ++k){
						if(k==i || k==j) continue;
						//find the index of the species of atom i
						unsigned int KK=speciesMap_[struc.atom(k).name()];
						//calculate rIK and rJK
						if(NN_POT_PRINT_STATUS>2) std::cout<<"calculating theta("<<i<<","<<j<<","<<k<<")\n";
						Cell::diff(struc.atom(i).posn(),struc.atom(k).posn(),rIK_,struc.cell().R(),struc.cell().RInv());
						Cell::diff(struc.atom(j).posn(),struc.atom(k).posn(),rJK_,struc.cell().R(),struc.cell().RInv());
						//loop over all cell shifts
						for(unsigned short idIK=0; idIK<R_.size(); ++idIK){
							rIKt_=rIK_; rIKt_.noalias()+=R_[idIK]; dIK=rIKt_.norm();
							for(unsigned short idJK=0; idJK<R_.size(); ++idJK){
								rJKt_=rJK_; rJKt_.noalias()+=R_[idJK]; dJK=rJKt_.norm();
								if(dIK<rc_ && dJK<rc_){
									//compute the IJ,IK,JK contribution to all angular basis functions
									double cosIJK=rIJt_.dot(rIKt_)/(dIJ*dIK);
									for(unsigned int na=0; na<basisA_[II](JJ,KK).fA.size(); ++na){
										struc.atom(i).symm()[nInputR_[II]+offsetA_[II](JJ,KK)+na]+=basisA_[II](JJ,KK).fA[na]->val(cosIJK,dIJ,dIK,dJK);
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

//calculate forces
void NNPot::forces(Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::forces(Structure<AtomT>&):\n";
	//reset the force
	for(unsigned int i=0; i<struc.nAtoms(); ++i) struc.atom(i).force().setZero();
	//lattice vector shifts
	unsigned short count=0;
	for(short ix=-1; ix<=1; ++ix){
		for(short iy=-1; iy<=1; ++iy){
			for(short iz=-1; iz<=1; ++iz){
				R_[count].setZero();
				R_[count].noalias()+=ix*struc.cell().R().col(0);
				R_[count].noalias()+=iy*struc.cell().R().col(1);
				R_[count].noalias()+=iz*struc.cell().R().col(2);
				++count;
			}
		}
	}
	//set the inputs for the atoms
	inputs_symm(struc);
	//calculate the network gradients
	dEdG_.resize(struc.nAtoms());
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		//resize the gradient
		dEdG_[i]=Eigen::VectorXd::Zero(nInput_[speciesMap_[struc.atom(i).name()]]);
		//set the index for the species
		unsigned int index=speciesMap_[struc.atom(i).name()];
		//execute the appropriate network
		nn_[index].execute(struc.atom(i).symm());
		//calculate the network gradients
		nn_[index].grad_out();
		//set the gradient
		dEdG_[i]=nn_[index].dOut(0).row(0);
	}
	//loop over all atoms
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		unsigned int II=speciesMap_[struc.atom(i).name()];
		double dIJ,dIK,dJK;
		Eigen::Vector3d force=Eigen::Vector3d::Zero();
		//loop over pairs
		for(unsigned int j=0; j<struc.nAtoms(); ++j){
			if(i==j) continue;
			//find the index of the species of atom j
			unsigned int JJ=speciesMap_[struc.atom(j).name()];
			//calc rIJ
			Cell::diff(struc.atom(i).posn(),struc.atom(j).posn(),rIJ_,struc.cell().R(),struc.cell().RInv());
			//loop over lattice vector shifts
			for(unsigned short idIJ=0; idIJ<R_.size(); ++idIJ){
				rIJt_=rIJ_; rIJt_.noalias()+=R_[idIJ]; dIJ=rIJt_.norm();
				if(dIJ<rc_){
					//compute the IJ contribution to the radial force
					Eigen::Vector3d ftemp=Eigen::Vector3d::Zero();
					for(unsigned int nr=0; nr<basisR_[II][JJ].fR.size(); ++nr){
						ftemp+=-1*rIJt_/dIJ*dEdG_[i][offsetR_[II][JJ]+nr]*basisR_[II][JJ].fR[nr]->grad(dIJ);
					}
					struc.atom(i).force().noalias()+=ftemp;
					struc.atom(j).force().noalias()-=ftemp;
					//loop over all triplets
					for(unsigned int k=0; k<struc.nAtoms(); ++k){
						if(k==i || k==j) continue;
						//find the index of the species of atom k
						unsigned int KK=speciesMap_[struc.atom(k).name()];
						//calculate rIK and rJK
						if(NN_POT_PRINT_STATUS>2) std::cout<<"calculating theta("<<i<<","<<j<<","<<k<<")\n";
						Cell::diff(struc.atom(i).posn(),struc.atom(k).posn(),rIK_,struc.cell().R(),struc.cell().RInv());
						Cell::diff(struc.atom(j).posn(),struc.atom(k).posn(),rJK_,struc.cell().R(),struc.cell().RInv());
						//loop over all cell shifts
						for(unsigned short idIK=0; idIK<R_.size(); ++idIK){
							rIKt_=rIK_; rIKt_.noalias()+=R_[idIK]; dIK=rIKt_.norm();
							for(unsigned short idJK=0; idJK<R_.size(); ++idJK){
								rJKt_=rJK_; rJKt_.noalias()+=R_[idJK]; dJK=rJKt_.norm();
								if(dIK<rc_ && dJK<rc_){
									//compute the IJ,IK,JK contribution to the angular force
									double cosIJK,amp;
									// first version - mathematically more transparent, but less efficient, have to skip around memory a lot
									/*
									for(unsigned int na=0; na<basisA_[II](JJ,KK).fA.size(); ++na){
										//gradient - cosine - central atom
										cosIJK=rIJt_.dot(rIKt_)/(dIJ*dIK);
										amp=-0.5*basisA_[II](JJ,KK).fA[na]->grad_angle(cosIJK)*basisA_[II](JJ,KK).fA[na]->dist(dIJ,dIK,dJK)*dEdG_[i][nInputR_[II]+offsetA_[II][JJ]+na];
										struc.atom(i).force().noalias()+=amp*(1.0/dIK-cosIJK/dIJ)*rIJt_/dIJ;
										struc.atom(i).force().noalias()+=amp*(1.0/dIJ-cosIJK/dIK)*rIKt_/dIK;
										//gradient distance - central atom
										amp=-0.5*basisA_[II](JJ,KK).fA[na]->angle(cosIJK)*dEdG_[i][nInputR_[II]+offsetA_[II][JJ]+na];
										struc.atom(i).force().noalias()+=amp*basisA_[II](JJ,KK).fA[na]->grad_dist(dIJ,dIK,dJK,0)*rIJt_/dIJ;
										struc.atom(i).force().noalias()+=amp*basisA_[II](JJ,KK).fA[na]->grad_dist(dIJ,dIK,dJK,1)*rIKt_/dIK;
									}
									for(unsigned int na=0; na<basisA_[JJ](II,KK).fA.size(); ++na){
										//gradient - cosine - first neighbor
										cosIJK=-rIJt_.dot(rJKt_)/(dIJ*dJK);
										amp=-0.5*basisA_[JJ](II,KK).fA[na]->grad_angle(cosIJK)*basisA_[JJ](II,KK).fA[na]->dist(dIJ,dJK,dIK)*dEdG_[j][nInputR_[JJ]+offsetA_[JJ][II]+na];
										struc.atom(i).force().noalias()-=amp*cosIJK/dIJ*rIJt_/dIJ;
										struc.atom(i).force().noalias()-=amp*1.0/(dIJ)*rJKt_/dJK;
										//gradient - cosine - first neighbor
										amp=-0.5*basisA_[JJ](II,KK).fA[na]->angle(cosIJK)*dEdG_[j][nInputR_[JJ]+offsetA_[JJ][II]+na];
										struc.atom(i).force().noalias()+=amp*basisA_[JJ](II,KK).fA[na]->grad_dist(dIJ,dJK,dIK,0)*rIJt_/dIJ;
										struc.atom(i).force().noalias()+=amp*basisA_[JJ](II,KK).fA[na]->grad_dist(dIJ,dJK,dIK,2)*rIKt_/dIK;
									}
									for(unsigned int na=0; na<basisA_[JJ](KK,II).fA.size(); ++na){
										//gradient - cosine - second neighbor
										cosIJK=-rJKt_.dot(rIJt_)/(dJK*dIJ);
										amp=-0.5*basisA_[JJ](KK,II).fA[na]->grad_angle(cosIJK)*basisA_[JJ](KK,II).fA[na]->dist(dJK,dIJ,dIK)*dEdG_[j][nInputR_[JJ]+offsetA_[JJ][KK]+na];
										struc.atom(i).force().noalias()-=amp*cosIJK/dIJ*rIJt_/dIJ;
										struc.atom(i).force().noalias()-=amp*1.0/(dIJ)*rJKt_/dJK;
										//gradient - cosine - second neighbor
										amp=-0.5*basisA_[JJ](KK,II).fA[na]->angle(cosIJK)*dEdG_[j][nInputR_[JJ]+offsetA_[JJ][KK]+na];
										struc.atom(i).force().noalias()+=amp*basisA_[JJ](KK,II).fA[na]->grad_dist(dJK,dIJ,dIK,1)*rIJt_/dIJ;
										struc.atom(i).force().noalias()+=amp*basisA_[JJ](KK,II).fA[na]->grad_dist(dJK,dIJ,dIK,2)*rIKt_/dIK;
									}
									*/
									// second version - mathematically more obtuses, but more efficient, better memory locality
									///*
									Eigen::Vector3d fij1=Eigen::Vector3d::Zero();
									Eigen::Vector3d fij2=Eigen::Vector3d::Zero();
									Eigen::Vector3d fik1=Eigen::Vector3d::Zero();
									Eigen::Vector3d fik2=Eigen::Vector3d::Zero();
									Eigen::Vector3d fij=Eigen::Vector3d::Zero();
									Eigen::Vector3d fik=Eigen::Vector3d::Zero();
									for(unsigned int na=0; na<basisA_[II](JJ,KK).fA.size(); ++na){
										//gradient - cosine - central atom
										cosIJK=rIJt_.dot(rIKt_)/(dIJ*dIK);
										amp=-0.5*basisA_[II](JJ,KK).fA[na]->grad_angle(cosIJK)*basisA_[II](JJ,KK).fA[na]->dist(dIJ,dIK,dJK)*dEdG_[i][nInputR_[II]+offsetA_[II][JJ]+na];
										fij1.noalias()+=amp*(-cosIJK/dIJ)*rIJt_/dIJ;
										fij2.noalias()+=amp*(1.0/dIK)*rIJt_/dIJ;
										fik1.noalias()+=amp*(-cosIJK/dIK)*rIKt_/dIK;
										fik2.noalias()+=amp*(1.0/dIJ)*rIKt_/dIK;
										//gradient distance - central atom
										amp=-0.5*basisA_[II](JJ,KK).fA[na]->angle(cosIJK)*dEdG_[i][nInputR_[II]+offsetA_[II][JJ]+na];
										fij.noalias()+=amp*basisA_[II](JJ,KK).fA[na]->grad_dist(dIJ,dIK,dJK,0)*rIJt_/dIJ;
										fik.noalias()+=amp*basisA_[II](JJ,KK).fA[na]->grad_dist(dIJ,dIK,dJK,1)*rIKt_/dIK;
									}
									struc.atom(i).force().noalias()+=fij1+fij2+fij;
									struc.atom(i).force().noalias()+=fik1+fik2+fik;
									struc.atom(j).force().noalias()-=fij1+fik2+fij;
									struc.atom(k).force().noalias()-=fik1+fij2+fik;
									//*/
								}
							}
						}
					}
				}
			}
		}
	}
}

//calculate forces
void NNPot::forces_radial(Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::force_radial(Structure<AtomT>&,unsigned int):\n";
	//reset the force
	for(unsigned int i=0; i<struc.nAtoms(); ++i) struc.atom(i).force().setZero();
	//lattice vector shifts
	unsigned short count=0;
	for(short ix=-1; ix<=1; ++ix){
		for(short iy=-1; iy<=1; ++iy){
			for(short iz=-1; iz<=1; ++iz){
				R_[count].setZero();
				R_[count].noalias()+=ix*struc.cell().R().col(0);
				R_[count].noalias()+=iy*struc.cell().R().col(1);
				R_[count].noalias()+=iz*struc.cell().R().col(2);
				++count;
			}
		}
	}
	//set the inputs for the atoms
	inputs_symm(struc);
	//calculate the network gradients
	dEdG_.resize(struc.nAtoms());
	if(NN_POT_PRINT_STATUS>0) std::cout<<"Calculating network gradients...\n";
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		//resize the gradient
		dEdG_[i]=Eigen::VectorXd::Zero(nInput_[speciesMap_[struc.atom(i).name()]]);
		//set the index for the species
		unsigned int index=speciesMap_[struc.atom(i).name()];
		//execute the appropriate network
		nn_[index].execute(struc.atom(i).symm());
		//calculate the network gradients
		nn_[index].grad_out();
		//set the gradient
		dEdG_[i]=nn_[index].dOut(0).row(0);
	}
	//loop over all atoms
	Eigen::Vector3d force=Eigen::Vector3d::Zero();
	//loop over pairs
	if(NN_POT_PRINT_STATUS>0) std::cout<<"Calculating radial force...\n";
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		unsigned int II=speciesMap_[struc.atom(i).name()];
		for(unsigned int j=0; j<struc.nAtoms(); ++j){
			if(i==j) continue;
			//find the index of the species of atom j
			unsigned int JJ=speciesMap_[struc.atom(j).name()];
			//calc rIJ
			Cell::diff(struc.atom(i).posn(),struc.atom(j).posn(),rIJ_,struc.cell().R(),struc.cell().RInv());
			//loop over lattice vector shifts
			for(unsigned short idIJ=0; idIJ<R_.size(); ++idIJ){
				rIJt_=rIJ_; rIJt_.noalias()+=R_[idIJ]; double dIJ=rIJt_.norm();
				if(dIJ<rc_){
					//compute the IJ contribution to all radial basis functions
					/*for(unsigned int nr=0; nr<basisR_[II][JJ].fR.size(); ++nr){
						struc.atom(i).force().noalias()+=-1*rIJt_/dIJ*(
							dEdG_[i][offsetR_[II][JJ]+nr]*basisR_[II][JJ].fR[nr]->grad(dIJ)
							+dEdG_[j][offsetR_[JJ][II]+nr]*basisR_[JJ][II].fR[nr]->grad(dIJ)
						);
					}*/
					Eigen::Vector3d ftemp=Eigen::Vector3d::Zero();
					for(unsigned int nr=0; nr<basisR_[II][JJ].fR.size(); ++nr){
						ftemp+=-1*rIJt_/dIJ*dEdG_[i][offsetR_[II][JJ]+nr]*basisR_[II][JJ].fR[nr]->grad(dIJ);
					}
					struc.atom(i).force().noalias()+=ftemp;
					struc.atom(j).force().noalias()-=ftemp;
				}
			}
		}
	}
}

//calculate forces
void NNPot::forces_angular(Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::forces_angular(Structure<AtomT>&):\n";
	//reset the forces
	for(unsigned int i=0; i<struc.nAtoms(); ++i) struc.atom(i).force().setZero();
	//lattice vector shifts
	unsigned short count=0;
	for(short ix=-1; ix<=1; ++ix){
		for(short iy=-1; iy<=1; ++iy){
			for(short iz=-1; iz<=1; ++iz){
				R_[count].setZero();
				R_[count].noalias()+=ix*struc.cell().R().col(0);
				R_[count].noalias()+=iy*struc.cell().R().col(1);
				R_[count].noalias()+=iz*struc.cell().R().col(2);
				++count;
			}
		}
	}
	//set the inputs for the atoms
	inputs_symm(struc);
	//calculate the network gradients
	dEdG_.resize(struc.nAtoms());
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		//resize the gradient
		dEdG_[i]=Eigen::VectorXd::Zero(nInput_[speciesMap_[struc.atom(i).name()]]);
		//set the index for the species
		unsigned int index=speciesMap_[struc.atom(i).name()];
		//execute the appropriate network
		nn_[index].execute(struc.atom(i).symm());
		//calculate the network gradients
		nn_[index].grad_out();
		//set the gradient
		dEdG_[i]=nn_[index].dOut(0).row(0);
	}
	//loop over all atoms
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		unsigned int II=speciesMap_[struc.atom(i).name()];
		double dIJ,dIK,dJK;
		//loop over pairs
		for(unsigned int j=0; j<struc.nAtoms(); ++j){
			if(i==j) continue;
			//find the index of the species of atom j
			unsigned int JJ=speciesMap_[struc.atom(j).name()];
			//calc rIJ
			Cell::diff(struc.atom(i).posn(),struc.atom(j).posn(),rIJ_,struc.cell().R(),struc.cell().RInv());
			//loop over lattice vector shifts
			for(unsigned short idIJ=0; idIJ<R_.size(); ++idIJ){
				rIJt_=rIJ_; rIJt_.noalias()+=R_[idIJ]; dIJ=rIJt_.norm();
				if(dIJ<rc_){
					//loop over all (unique) triplets
					for(unsigned int k=0; k<struc.nAtoms(); ++k){
						if(k==i || k==j) continue;
						//find the index of the species of atom k
						unsigned int KK=speciesMap_[struc.atom(k).name()];
						//calculate rIK and rJK
						if(NN_POT_PRINT_STATUS>2) std::cout<<"calculating theta("<<i<<","<<j<<","<<k<<")\n";
						Cell::diff(struc.atom(i).posn(),struc.atom(k).posn(),rIK_,struc.cell().R(),struc.cell().RInv());
						Cell::diff(struc.atom(j).posn(),struc.atom(k).posn(),rJK_,struc.cell().R(),struc.cell().RInv());
						//loop over all cell shifts
						for(unsigned short idIK=0; idIK<R_.size(); ++idIK){
							rIKt_=rIK_; rIKt_.noalias()+=R_[idIK]; dIK=rIKt_.norm();
							for(unsigned short idJK=0; idJK<R_.size(); ++idJK){
								rJKt_=rJK_; rJKt_.noalias()+=R_[idJK]; dJK=rJKt_.norm();
								if(dIK<rc_ && dJK<rc_){
									//compute the IJ,IK,JK contribution to all angular basis functions
									double cosIJK,amp,c=1;
									Eigen::Vector3d fij=Eigen::Vector3d::Zero();
									Eigen::Vector3d fik=Eigen::Vector3d::Zero();
									Eigen::Vector3d fij1=Eigen::Vector3d::Zero();
									Eigen::Vector3d fik1=Eigen::Vector3d::Zero();
									Eigen::Vector3d fij2=Eigen::Vector3d::Zero();
									Eigen::Vector3d fik2=Eigen::Vector3d::Zero();
									// test - cosine only
									/*
									cosIJK=rIJt_.dot(rIKt_)/(dIJ*dIK);
									//struc.atom(i).force().noalias()+=-1*(-cosIJK/dIJ)*rIJt_/dIJ;
									//struc.atom(i).force().noalias()+=-1*(1.0/dIK)*rIJt_/dIJ;
									//struc.atom(i).force().noalias()+=-1*(-cosIJK/dIK)*rIKt_/dIK;
									//struc.atom(i).force().noalias()+=-1*(1.0/dIJ)*rIKt_/dIK;
									fij1.noalias()+=-1*(-cosIJK/dIJ)*rIJt_/dIJ;
									fij2.noalias()+=-1*(1.0/dIK)*rIJt_/dIJ;
									fik1.noalias()+=-1*(-cosIJK/dIK)*rIKt_/dIK;
									fik2.noalias()+=-1*(1.0/dIJ)*rIKt_/dIK;
									//cosIJK=-rIJt_.dot(rJKt_)/(dIJ*dJK);
									//struc.atom(i).force().noalias()-=-1*cosIJK/dIJ*rIJt_/dIJ;
									//struc.atom(i).force().noalias()-=-1*1.0/(dIJ)*rJKt_/dJK;
									//cosIJK=-rJKt_.dot(rIJt_)/(dJK*dIJ);
									//struc.atom(i).force().noalias()-=-1*cosIJK/dIJ*rIJt_/dIJ;
									//struc.atom(i).force().noalias()-=-1*1.0/(dIJ)*rJKt_/dJK;
									struc.atom(i).force().noalias()+=fij1+fij2;
									struc.atom(i).force().noalias()+=fik1+fik2;
									struc.atom(j).force().noalias()-=fij1+fik2;
									struc.atom(k).force().noalias()-=fik1+fij2;
									*/
									
									for(unsigned int na=0; na<basisA_[II](JJ,KK).fA.size(); ++na){
										//gradient - cosine - central atom
										cosIJK=rIJt_.dot(rIKt_)/(dIJ*dIK);
										amp=c*basisA_[II](JJ,KK).fA[na]->grad_angle(cosIJK)*basisA_[II](JJ,KK).fA[na]->dist(dIJ,dIK,dJK)*dEdG_[i][nInputR_[II]+offsetA_[II][JJ]+na];
										//struc.atom(i).force().noalias()+=-1*amp*(1.0/dIK-cosIJK/dIJ)*rIJt_/dIJ;
										//struc.atom(i).force().noalias()+=-1*amp*(1.0/dIJ-cosIJK/dIK)*rIKt_/dIK;
										fij1.noalias()+=-1*amp*(-cosIJK/dIJ)*rIJt_/dIJ;
										fij2.noalias()+=-1*amp*(1.0/dIK)*rIJt_/dIJ;
										fik1.noalias()+=-1*amp*(-cosIJK/dIK)*rIKt_/dIK;
										fik2.noalias()+=-1*amp*(1.0/dIJ)*rIKt_/dIK;
										//gradient distance - central atom
										amp=c*basisA_[II](JJ,KK).fA[na]->angle(cosIJK)*dEdG_[i][nInputR_[II]+offsetA_[II][JJ]+na];
										//struc.atom(i).force().noalias()+=amp*basisA_[II](JJ,KK).fA[na]->grad_dist(dIJ,dIK,dJK,0)*rIJt_/dIJ;
										//struc.atom(i).force().noalias()+=amp*basisA_[II](JJ,KK).fA[na]->grad_dist(dIJ,dIK,dJK,1)*rIKt_/dIK;
										fij.noalias()+=amp*basisA_[II](JJ,KK).fA[na]->grad_dist(dIJ,dIK,dJK,0)*rIJt_/dIJ;
										fik.noalias()+=amp*basisA_[II](JJ,KK).fA[na]->grad_dist(dIJ,dIK,dJK,1)*rIKt_/dIK;
										//gradient - cosine - first neighbor
										cosIJK=-rIJt_.dot(rJKt_)/(dIJ*dJK);
										amp=c*basisA_[JJ](II,KK).fA[na]->grad_angle(cosIJK)*basisA_[JJ](II,KK).fA[na]->dist(dIJ,dJK,dIK)*dEdG_[j][nInputR_[JJ]+offsetA_[JJ][II]+na];
										//struc.atom(i).force().noalias()-=-1*amp*cosIJK/dIJ*rIJt_/dIJ;
										//struc.atom(i).force().noalias()-=-1*amp*1.0/(dIJ)*rJKt_/dJK;
										//gradient - distance - first neighbor
										amp=c*basisA_[JJ](II,KK).fA[na]->angle(cosIJK)*dEdG_[j][nInputR_[JJ]+offsetA_[JJ][II]+na];
										//struc.atom(i).force().noalias()+=amp*basisA_[JJ](II,KK).fA[na]->grad_dist(dIJ,dJK,dIK,0)*rIJt_/dIJ;
										//struc.atom(i).force().noalias()+=amp*basisA_[JJ](II,KK).fA[na]->grad_dist(dIJ,dJK,dIK,2)*rIKt_/dIK;
										//gradient - cosine - second neighbor
										cosIJK=-rJKt_.dot(rIJt_)/(dJK*dIJ);
										amp=c*basisA_[JJ](KK,II).fA[na]->grad_angle(cosIJK)*basisA_[JJ](KK,II).fA[na]->dist(dJK,dIJ,dIK)*dEdG_[j][nInputR_[JJ]+offsetA_[JJ][KK]+na];
										//struc.atom(i).force().noalias()-=-1*amp*cosIJK/dIJ*rIJt_/dIJ;
										//struc.atom(i).force().noalias()-=-1*amp*1.0/(dIJ)*rJKt_/dJK;
										//gradient - distance - second neighbor
										amp=c*basisA_[JJ](KK,II).fA[na]->angle(cosIJK)*dEdG_[j][nInputR_[JJ]+offsetA_[JJ][KK]+na];
										//struc.atom(i).force().noalias()+=amp*basisA_[JJ](KK,II).fA[na]->grad_dist(dJK,dIJ,dIK,1)*rIJt_/dIJ;
										//struc.atom(i).force().noalias()+=amp*basisA_[JJ](KK,II).fA[na]->grad_dist(dJK,dIJ,dIK,2)*rIKt_/dIK;
									}
									
									struc.atom(i).force().noalias()+=fij1+fij2+fij;
									struc.atom(i).force().noalias()+=fik1+fik2+fik;
									struc.atom(j).force().noalias()-=fij1+fik2+fij;
									struc.atom(k).force().noalias()-=fik1+fij2+fik;
									
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
double NNPot::energy(Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::energy(Structure<AtomT>&):\n";
	double energy=0;
	//set the inputs for the atoms
	inputs_symm(struc);
	//loop over species
	for(unsigned int n=0; n<struc.nSpecies(); ++n){
		//loop over atoms
		for(unsigned int m=0; m<struc.nAtoms(n); ++m){
			//set the index
			unsigned int index=speciesMap_[struc.atom(n,m).name()];
			//execute the network
			nn_[index].execute(struc.atom(n,m).symm());
			//add the energy
			energy+=nn_[index].output()[0]+energyAtom_[n];
		}
	}
	//return energy*postScale_+postBias_;
	return energy;
}

//static functions

//write all neural network potentials to file
void NNPot::write()const{
	if(NN_POT_PRINT_FUNC) std::cout<<"NNPot::write(const NNPot&):\n";
	//======== local function variables ========
	std::string filename;
	//======== write all networks ========
	for(unsigned int n=0; n<nSpecies(); ++n){
		//create the file name
		filename=header_+speciesMap_.key(n);
		if(NN_POT_PRINT_DATA>0) std::cout<<"filename = "<<filename<<"\n";
		//write the network
		NNPot::write(n,filename);
	}
}

//read all neural network potentials from file
void NNPot::read(){
	if(NN_POT_PRINT_FUNC) std::cout<<"NNPot::read(const NNPot&):\n";
	//======== local function variables ========
	std::string filename;
	//======== read all networks ========
	for(unsigned int n=0; n<nSpecies(); ++n){
		//create the file name
		filename=header_+speciesMap_.key(n);
		if(NN_POT_PRINT_DATA>0) std::cout<<"filename = "<<filename<<"\n";
		//read the network
		NNPot::read(n,filename);
	}
	//======== set the number of inputs and offsets ========
	for(unsigned int n=0; n<nSpecies(); ++n){
		for(unsigned int i=0; i<nSpecies(); ++i){
			nInputR_[n]+=basisR_[n][i].fR.size();
		}
	}
	for(unsigned int n=0; n<nSpecies(); ++n){
		for(unsigned int i=1; i<nSpecies(); ++i){
			offsetR_[n][i]=offsetR_[n][i-1]+basisR_[n][i-1].fR.size();
		}
	}
	for(unsigned int n=0; n<nSpecies(); ++n){
		for(unsigned int i=0; i<nSpecies(); ++i){
			for(unsigned int j=i; j<nSpecies(); ++j){
				nInputA_[n]+=basisA_[n](j,i).fA.size();
			}
		}
	}
	for(unsigned int n=0; n<nSpecies(); ++n){
		for(unsigned int i=1; i<basisA_[n].size(); ++i){
			offsetA_[n][i]=offsetA_[n][i-1]+basisA_[n][i-1].fA.size();
		}
	}
	for(unsigned int n=0; n<speciesMap_.size(); ++n) nInput_[n]=nInputR_[n]+nInputA_[n];
}

//write neural network potential to file
void NNPot::write(unsigned int index, const std::string& filename)const{
	if(NN_POT_PRINT_FUNC) std::cout<<"NNPot::write(const NNPot&,unsigned int,const std::string&):\n";
	//======== local function variables ========
	FILE* writer=NULL;
	//======== write the basis ========
	writer=fopen(filename.c_str(),"w");
	if(writer!=NULL){
		//write the header
		fprintf(writer,"ann\n");
		//write the global cutoff
		fprintf(writer,"cut %f\n",rc_);
		//write the atomic name
		fprintf(writer,"species %s\n",speciesName(index).c_str());
		//write the atomic mass
		fprintf(writer,"mass %f\n",PTable::mass(PTable::an(speciesName(index).c_str())));
		//write the atomic energy
		fprintf(writer,"energy %.5f\n",energyAtom_[index]);
		//print the number of species
		fprintf(writer,"nspecies %i\n",nSpecies());
		//write the radial basis
		for(unsigned int i=0; i<nSpecies(); ++i){
			fprintf(writer,"basis_radial %s\n",speciesName(i).c_str());
			BasisR::write(writer,basisR_[index][i]);
		}
		//write the angular basis
		for(unsigned int i=0; i<nSpecies(); ++i){
			for(unsigned int j=i; j<nSpecies(); ++j){
				fprintf(writer,"basis_angular %s %s\n",speciesName(index).c_str(),speciesName(i).c_str(),speciesName(j).c_str());
				BasisA::write(writer,basisA_[index](i,j));
			}
		}
		//write the neural network
		NN::Network::write(writer,nn_[index]);
		//close the file
		fclose(writer);
		writer=NULL;
	}
}

//write neural network potential to file
void NNPot::read(unsigned int index, const std::string& filename){
	if(NN_POT_PRINT_FUNC) std::cout<<"NNPot::read(NNPot&,unsigned int,const std::string&):\n";
	//======== local function variables ========
	FILE* reader=NULL;
	double rc=0;
	std::string name;
	double mass=0,energy=0;
	unsigned int nspecies=0;
	char* input=(char*)malloc(sizeof(char)*string::M);
	char* temp=(char*)malloc(sizeof(char)*string::M);
	//======== write the basis ========
	reader=fopen(filename.c_str(),"r");
	if(reader!=NULL){
		//reader in header
		fgets(input,string::M,reader);
		//read in global cutoff
		std::strtok(fgets(input,string::M,reader),string::WS);
		rc=std::atof(std::strtok(NULL,string::WS));
		if(rc>rc_) rc_=rc;
		//read in species name
		std::strtok(fgets(input,string::M,reader),string::WS);
		name=std::string(std::strtok(NULL,string::WS));
		if(name!=speciesMap_.key(index)) throw std::invalid_argument("Invalid index for species in nn file.");
		//read in species mass
		std::strtok(fgets(input,string::M,reader),string::WS);
		mass=std::atof(std::strtok(NULL,string::WS));
		//read in species energy
		std::strtok(fgets(input,string::M,reader),string::WS);
		energy=std::atof(std::strtok(NULL,string::WS));
		energyAtom_[index]=energy;
		//read in the number of species
		std::strtok(fgets(input,string::M,reader),string::WS);
		nspecies=std::atoi(std::strtok(NULL,string::WS));
		//read the radial basis
		for(unsigned int i=0; i<nspecies; ++i){
			std::strtok(fgets(input,string::M,reader),string::WS);
			name=std::string(std::strtok(NULL,string::WS));
			unsigned int jj=speciesIndex(name);
			BasisR::read(reader,basisR_[index][jj]);
		}
		//read the angular basis
		for(unsigned int i=0; i<nspecies; ++i){
			for(unsigned int j=i; j<nspecies; ++j){
				std::strtok(fgets(input,string::M,reader),string::WS);
				name=std::string(std::strtok(NULL,string::WS));
				unsigned int jj=speciesIndex(name);
				name=std::string(std::strtok(NULL,string::WS));
				unsigned int kk=speciesIndex(name);
				BasisA::read(reader,basisA_[index](jj,kk));
			}
		}
		//read the neural network
		NN::Network::read(reader,nn_[index]);
		//close the file
		fclose(reader);
		reader=NULL;
	}
	//======== free local variables ========
	free(temp);
	free(input);
}
