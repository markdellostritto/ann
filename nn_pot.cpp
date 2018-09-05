#include "nn_pot.hpp"

//************************************************************
// NNPot - Neural Network Potential
//************************************************************

//constants

const char* NNPot::header="nn_pot_";

//operators

std::ostream& operator<<(std::ostream& out, const NNPot& nnpot){
	out<<"**************************************************\n";
	out<<"******************** NN - POT ********************\n";
	out<<"SPECIES = ";
	for(unsigned int i=0; i<nnpot.speciesMap_.size(); ++i) std::cout<<nnpot.speciesMap_.key(i)<<" ";
	std::cout<<"\n";
	out<<"BASIS:\n";
	out<<"\tNR         = "<<nnpot.nR_<<"\n";
	out<<"\tNA         = "<<nnpot.nA_<<"\n";
	out<<"\tPHIRN      = "<<nnpot.phiRN_<<"\n";
	out<<"\tPHIAN      = "<<nnpot.phiAN_<<"\n";
	out<<"CUTOFF:\n";
	out<<"\tRM         = "<<nnpot.rm_<<"\n";
	out<<"\tRC         = "<<nnpot.rc_<<"\n";
	out<<"\tCUTOFF     = "<<nnpot.tcut_<<"\n";
	out<<"ELEMENT NN:\n";
	out<<"\tN_ELEMENTS = "<<nnpot.speciesMap_.size()<<"\n";
	out<<"\tNN_CONFIG  = "<<nnpot.nInput_<<" ";
	for(unsigned int i=0; i<nnpot.nh_.size(); ++i) out<<nnpot.nh_[i]<<" ";
	out<<nnpot.nOutput_<<"\n";
	out<<"\tN_PARAMS   = "<<nnpot.nParams_<<"\n";
	out<<"\tTRANSFER   = "<<nnpot.tfType_<<"\n";
	out<<"\tLAMBDA     = "<<nnpot.lambda_<<"\n";
	out<<"\tPRE-COND   = "<<nnpot.preCond_<<"\n";
	out<<"\tPOST-COND  = "<<nnpot.postCond_<<"\n";
	for(unsigned int i=0; i<nnpot.speciesMap_.size(); ++i){
		std::cout<<"X-"<<nnpot.speciesMap_.key(i)<<" "<<nnpot.basisR_[i]<<"\n";
	}
	for(unsigned int i=0; i<nnpot.speciesMap_.size(); ++i){
		for(unsigned int j=i; j<nnpot.speciesMap_.size(); ++j){
			std::cout<<"X-"<<nnpot.speciesMap_.key(i)<<"-"<<nnpot.speciesMap_.key(j)<<" "<<nnpot.basisA_(i,j)<<"\n";
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
		nR_=0; nA_=0;
		basisR_.clear();
		basisA_.clear();
		phiRN_=PhiRN::G2;
		phiAN_=PhiAN::G4;
	//cutoff
		rm_=0.0;
		rc_=0.0;
		tcut_=CutoffN::COS;
	//element nn's
		nParams_=0;
		speciesMap_.clear();
		nn_.clear();
		nInput_=0;
		nh_.clear();
	//pre-/post-conditioning
		preCond_=true;
		postCond_=true;
		preBias_.clear();
		preScale_.clear();
		postBias_=0.0;
		postScale_=1.0;
	//transfer function
		tfType_=NN::TransferN::TANH;
	//regularization
		lambda_=0.0;//no regularization
	//input/output
		nPrint_=1000;
		nSave_=1000;
}

//initialize the basis functions and element networks
void NNPot::init(){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::init():\n";
	//check the parameters
		if(tfType_==NN::TransferN::UNKNOWN) throw std::invalid_argument("Invalid transfer function.");
		if(rm_<0) throw std::invalid_argument("Invalid minimum radius.");
		if(rc_<=0) throw std::invalid_argument("Invalid cutoff radius.");
		if(std::fabs(rm_-rc_)<num_const::ZERO) throw std::invalid_argument("Min and cutoff radius are equivalent.");
		if(rm_<num_const::ZERO){
			std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
			std::cout<<"!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!\n";
			std::cout<<"!! The minimum radius is close to zero.         !!\n";
			std::cout<<"!! This will lead to nodes with vanishing       !!\n";
			std::cout<<"!! values, complicating training and            !!\n";
			std::cout<<"!! introducing errors into force calculations.  !!\n";
			std::cout<<"!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!\n";
			std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
		}
		if(nR_==0) throw std::invalid_argument("Invalid number of radial basis functions.");
		if(nA_==0) throw std::invalid_argument("Invalid number of angular basis functions.");
		if(speciesMap_.size()==0) throw std::invalid_argument("Invalid number of species.");
	//set the radial basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Setting the radial basis...\n";
		basisR_.resize(speciesMap_.size());
		if(phiRN_==PhiRN::G1){
			for(unsigned int i=0; i<basisR_.size(); ++i) BasisR::init_G1(basisR_[i],tcut_,rm_,rc_);
		} else if(phiRN_==PhiRN::G2){
			for(unsigned int i=0; i<basisR_.size(); ++i) BasisR::init_G2(basisR_[i],nR_,tcut_,rm_,rc_);
		} else throw std::invalid_argument("Invalid radial basis type");
	//set the angular basis
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Setting the angular basis...\n";
		basisA_.resize(speciesMap_.size());
		if(phiAN_==PhiAN::G3){
			for(unsigned int i=0; i<basisA_.size(); ++i){
				for(unsigned int j=i; j<basisA_.size(); ++j){
					BasisA::init_G3(basisA_(i,j),nA_,tcut_,rc_);
				}
			}
		} else if(phiAN_==PhiAN::G4){
			for(unsigned int i=0; i<basisA_.size(); ++i){
				for(unsigned int j=i; j<basisA_.size(); ++j){
					BasisA::init_G4(basisA_(i,j),nA_,tcut_,rc_);
				}
			}
		} else throw std::invalid_argument("Invalid angular basis type");
	//set the number of inputs (number of radial + angular basis functions)
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Set the number of inputs...\n";
		nInput_=nR_+nA_*basisA_.size();
	//resize the bias and scale
		preBias_.clear();
		postBias_=0;
		preScale_.clear();
		postScale_=0;
	//resize the number of neural networks
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Resizing neural networks...\n";
		nn_.resize(speciesMap_.size());
		for(unsigned int i=0; i<nn_.size(); ++i){
			//set the transfer function
			nn_[i].tfType()=tfType_;
			//set the pre-/post-conditioning
			nn_[i].preCond()=preCond_;
			nn_[i].postCond()=postCond_;
			//set regularization parameter
			nn_[i].lambda()=lambda_;
			//resize the network
			nn_[i].resize(nInput_,nh_,nOutput_);
		}
	//set the number of parameters
		if(NN_POT_PRINT_STATUS>0) std::cout<<"Setting the number of parameters...\n";
		nParams_=nn_.front().size();
		for(unsigned int i=1; i<nn_.size(); ++i){
			if(nn_[i].size()!=nParams_) throw std::runtime_error("Mismatch in number of network parameters...\n");
		}
	//resize the lvShifts
		lvShifts_.resize(3*3*3,Eigen::Vector3d::Zero());
}

//nn-struc

//set the number of species and species names to the total number of species in the simulations
void NNPot::initSpecies(const Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::initSpecies(const Structure<AtomT>&):\n";
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
}

//set the number of species and species names to the total number of species in the simulations
void NNPot::initSpecies(const std::vector<Structure<AtomT> >& strucv){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::initSpecies(const std::vector<Structure<AtomT> >&):\n";
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
}

//resize the symmetry function vectors to store the of inputs
void NNPot::initSymm(Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::initSymm(const Structure<AtomT>&):\n";
	for(unsigned int n=0; n<struc.nSpecies(); ++n){
		for(unsigned int m=0; m<struc.nAtoms(n); ++m){
			struc.atom(n,m).symm().resize(nInput_);
		}
	}
}

//resize the symmetry function vectors to store the of inputs
void NNPot::initSymm(std::vector<Structure<AtomT> >& strucv){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::initSymm(const std::vector<Structure<AtomT> >&):\n";
	for(unsigned int i=0; i<strucv.size(); ++i){
		for(unsigned int n=0; n<strucv[i].nSpecies(); ++n){
			for(unsigned int m=0; m<strucv[i].nAtoms(n); ++m){
				strucv[i].atom(n,m).symm().resize(nInput_);
			}
		}
	}
}

//set the number of species and species names to the total number of species in the simulations
void NNPot::setSpecies(const std::vector<std::string>& speciesNames){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::setSpecies(const std::vector<std::string>&):\n";
	unsigned int nSpecies=0;
	speciesMap_.clear();
	for(unsigned int i=0; i<speciesNames.size(); ++i){
		speciesMap_.add(speciesNames[i],i);
	}
}

//calculate inputs - symmetry functions - warning: no periodic self-interactions included
void NNPot::inputs_symm(Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::inputs_symm(Structure<AtomT>&,unsigned int):\n";
	//lattice vector shifts
	unsigned short count=0;
	for(short ix=-1; ix<=1; ++ix){
		for(short iy=-1; iy<=1; ++iy){
			for(short iz=-1; iz<=1; ++iz){
				lvShifts_[count].setZero();
				lvShifts_[count].noalias()+=ix*struc.cell().R().col(0);
				lvShifts_[count].noalias()+=iy*struc.cell().R().col(1);
				lvShifts_[count].noalias()+=iz*struc.cell().R().col(2);
				++count;
			}
		}
	}
	//loop over all atoms
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		double dIJ,dIK,dJK;
		//reset the inputs
		if(NN_POT_PRINT_STATUS>2) std::cout<<"resetting inputs...\n";
		struc.atom(i).symm().setZero();
		//loop over pairs
		for(unsigned int j=0; j<struc.nAtoms(); ++j){
			if(i==j) continue;
			//find the index of the species of atom i
			unsigned int JJ=speciesMap_[struc.atom(j).name()];
			//calc rIJ
			if(NN_POT_PRINT_STATUS>2) std::cout<<"symm r("<<i<<","<<j<<")\n";
			//calc radial contribution - loop over all radial functions
			if(NN_POT_PRINT_STATUS>2) std::cout<<"Computing radial functions...\n";
			Cell::diff(struc.atom(i).posn(),struc.atom(j).posn(),rIJ_,struc.cell().R(),struc.cell().RInv());
			//loop over lattice vector shifts
			for(unsigned short idIJ=0; idIJ<lvShifts_.size(); ++idIJ){
				rIJt_.noalias()=rIJ_; rIJt_.noalias()+=lvShifts_[idIJ]; dIJ=rIJt_.norm();
				if(dIJ<rc_){
					if(NN_POT_PRINT_STATUS>2) std::cout<<"Setting radial symmetry functions...\n";
					//compute the IJ contribution to all radial basis functions
					double cut_r=basisR_[JJ].fR[0]->cut(dIJ);
					for(unsigned int nr=0; nr<nR_; ++nr){
						struc.atom(i).symm()[nr]+=cut_r*basisR_[JJ].fR[nr]->amp(dIJ);
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
						for(unsigned short idIK=0; idIK<lvShifts_.size(); ++idIK){
							rIKt_=rIK_; rIKt_.noalias()+=lvShifts_[idIK]; dIK=rIKt_.norm();
							for(unsigned short idJK=0; idJK<lvShifts_.size(); ++idJK){
								rJKt_=rJK_; rJKt_.noalias()+=lvShifts_[idJK]; dJK=rJKt_.norm();
								if(dIK<rc_ && dJK<rc_){
									//compute the IJ,IK,JK contribution to all angular basis functions
									if(NN_POT_PRINT_STATUS>2) std::cout<<"calculating angular functions...\n";
									double cosIJK=rIJt_.dot(rIKt_)/(dIJ*dIK);
									double cut_a=basisA_(JJ,KK).fA[0]->cut(dIJ,dIK,dJK);
									for(unsigned int na=0; na<nA_; ++na){
										struc.atom(i).symm()[nR_+na]+=cut_a*basisA_(JJ,KK).fA[na]->amp(cosIJK,dIJ,dIK,dJK);
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
	//lattice vector shifts
	unsigned short count=0;
	for(short ix=-1; ix<=1; ++ix){
		for(short iy=-1; iy<=1; ++iy){
			for(short iz=-1; iz<=1; ++iz){
				lvShifts_[count].setZero();
				lvShifts_[count].noalias()+=ix*struc.cell().R().col(0);
				lvShifts_[count].noalias()+=iy*struc.cell().R().col(1);
				lvShifts_[count].noalias()+=iz*struc.cell().R().col(2);
				++count;
			}
		}
	}
	//set the inputs for the atoms
	inputs_symm(struc);
	//calculate the network gradients
	dOutdIn_.resize(struc.nAtoms(),Eigen::VectorXd::Zero(nInput_));
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
		//set the index for the species
		unsigned int index=speciesMap_[struc.atom(i).name()];
		//execute the appropriate network
		nn_[index].execute(struc.atom(i).symm());
		//calculate the network gradients
		nn_[index].grad_out();
		//set the gradient
		dOutdIn_[i]=nn_[index].dOut(0).row(0);
		dOutdIn_[i]*=postScale_;
	}
	//loop over all atoms
	for(unsigned int i=0; i<struc.nAtoms(); ++i){
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
			for(unsigned short idIJ=0; idIJ<lvShifts_.size(); ++idIJ){
				rIJt_=rIJ_; rIJt_.noalias()+=lvShifts_[idIJ]; dIJ=rIJt_.norm();
				if(dIJ<rc_){
					//compute the IJ contribution to all radial basis functions
					for(unsigned int nr=0; nr<nR_; ++nr){
						force.noalias()+=(dOutdIn_[i][nr]+dOutdIn_[j][nr])*basisR_[JJ].fR[nr]->grad(dIJ)*rIJt_/dIJ;
					}
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
						for(unsigned short idIK=0; idIK<lvShifts_.size(); ++idIK){
							rIKt_=rIK_; rIKt_.noalias()+=lvShifts_[idIK]; dIK=rIKt_.norm();
							for(unsigned short idJK=0; idJK<lvShifts_.size(); ++idJK){
								rJKt_=rJK_; rJKt_.noalias()+=lvShifts_[idJK]; dJK=rJKt_.norm();
								if(dIK<rc_ && dJK<rc_){
									//compute the IJ,IK,JK contribution to all angular basis functions
									double cosIJK=0.0;
									for(unsigned int na=0; na<nA_; ++na){
										cosIJK=rIJt_.dot(rIKt_)/(dIJ*dIK);
										force.noalias()+=dOutdIn_[i][nR_+na]*basisA_(JJ,KK).fA[na]->grad(cosIJK,dIJ,dIK,dJK,0)*rIJt_/dIJ;
										force.noalias()+=dOutdIn_[i][nR_+na]*basisA_(JJ,KK).fA[na]->grad(cosIJK,dIJ,dIK,dJK,1)*rIKt_/dIJ;
										cosIJK=rIJt_.dot(rJKt_)/(dIJ*dJK);
										force.noalias()+=dOutdIn_[j][nR_+na]*basisA_(JJ,KK).fA[na]->grad(cosIJK,dIJ,dJK,dIK,0)*rIJt_/dIJ;
										force.noalias()+=dOutdIn_[j][nR_+na]*basisA_(JJ,KK).fA[na]->grad(cosIJK,dJK,dIJ,dIK,1)*rIJt_/dIJ;
										force.noalias()+=dOutdIn_[j][nR_+na]*basisA_(JJ,KK).fA[na]->grad(cosIJK,dIJ,dJK,dIK,2)*rIKt_/dIK;
										force.noalias()+=dOutdIn_[j][nR_+na]*basisA_(JJ,KK).fA[na]->grad(cosIJK,dJK,dIJ,dIK,2)*rIKt_/dIK;
									}
								}
							}
						}
					}
				}
			}
		}
		struc.atom(i).force().noalias()=-1*force;
	}
}

//execute all atomic networks and return energy
double NNPot::energy(Structure<AtomT>& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::energy(Structure<AtomT>&):\n";
	double energy=0;
	//loop over species
	for(unsigned int n=0; n<struc.nSpecies(); ++n){
		//loop over atoms
		for(unsigned int m=0; m<struc.nAtoms(n); ++m){
			//set the index
			unsigned int index=speciesIndex(struc.atom(n,m).name());
			//execute the network
			nn_[index].execute(struc.atom(n,m).symm());
			//add the energy
			energy+=nn_[index].output()[0];
		}
	}
	return energy*postScale_+postBias_;
}

//static functions

//read potential parameters from potential files
void NNPot::write(const char* file, const NNPot& nnpot){
	if(NN_POT_PRINT_FUNC) std::cout<<"NNPot::write(const char*,const NNPot&):\n";
}

//read potential parameters from potential files
void NNPot::read(const char* file, NNPot& nnpot){
	if(NN_POT_PRINT_FUNC) std::cout<<"NNPot::read(FILE*,const NNPot&):\n";
}
