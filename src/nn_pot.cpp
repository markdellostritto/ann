// c libraries
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
// c++ libraries
#include <iostream>
// ann - structure
#include "structure.hpp"
#include "cell_list.hpp"
// ann - math
#include "math_const.hpp"
// ann - string
#include "string.hpp"
// ann - print
#include "print.hpp"
// ann - nn_pot
#include "nn_pot.hpp"

//************************************************************
// NEURAL NETWORK HAMILTONIAN
//************************************************************

//==== operators ====

/**
* print neural network hamiltonian
* @param out - output stream
* @param nnh - neural network hamiltonian
*/
std::ostream& operator<<(std::ostream& out, const NNH& nnh){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	//hamiltonian
	out<<"ATOM     = "<<nnh.atom_<<"\n";
	//species
	out<<"NSPECIES = "<<nnh.nspecies_<<"\n";
	//potential parameters
	out<<"N_INPUT  = "; std::cout<<nnh.nInput_<<" "; std::cout<<"\n";
	out<<"N_INPUTR = "; std::cout<<nnh.nInputR_<<" "; std::cout<<"\n";
	out<<"N_INPUTA = "; std::cout<<nnh.nInputA_<<" "; std::cout<<"\n";
	out<<nnh.nn_<<"\n";
	out<<print::title("NN - HAMILTONIAN",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

/**
* set NNH defaults
*/
void NNH::defaults(){
	//hamiltonian
		nspecies_=0;
		atom_.clear();
		nn_.clear();
	//basis for pair/triple interactions
		basisR_.clear();
		basisA_.clear();
	//network configuration
		nInput_=0;
		nInputR_=0;
		nInputA_=0;
		offsetR_.clear();
		offsetA_.clear();
}

/**
* resize the number of species
* @param nspecies - the total number of species
*/
void NNH::resize(int nspecies){
	if(nspecies<=0) throw std::invalid_argument("NNH::resize(int): invalid number of species.");
	nspecies_=nspecies;
	basisR_.resize(nspecies_);
	basisA_.resize(nspecies_);
	offsetR_.resize(nspecies_);
	offsetA_.resize(nspecies_);
}

/**
* Initialize the number of inputs and offsets associated with the basis functions.
* Must be done after the basis has been defined, otherwise the values will make no sense.
* Different from resizing: resizing sets the number of species, this sets the number of inputs
* associated with the basis associated with each species.
*/
void NNH::init_input(){
	//radial inputs
	nInputR_=0;
	for(int i=0; i<nspecies_; ++i){
		nInputR_+=basisR_[i].nfR();
	}
	//radial offsets
	offsetR_[0]=0;
	for(int i=1; i<nspecies_; ++i){
		offsetR_[i]=offsetR_[i-1]+basisR_[i-1].nfR();
	}
	//angular inputs
	nInputA_=0;
	for(int i=0; i<nspecies_; ++i){
		for(int j=i; j<nspecies_; ++j){
			nInputA_+=basisA_(j,i).nfA();
		}
	}
	//angular offsets
	offsetA_[0]=0;
	for(int i=1; i<basisA_.size(); ++i){
		offsetA_[i]=offsetA_[i-1]+basisA_[i-1].nfA();
	}
	//total number of inputs
	nInput_=nInputR_+nInputA_;
}

/**
* compute energy of atom with symmetry function "symm"
* @param symm - the symmetry function
*/
double NNH::energy(const Eigen::VectorXd& symm){
	return nn_.execute(symm)[0]+atom_.energy();
}

//************************************************************
// NNPot - Neural Network Potential
//************************************************************

//==== operators ====

/**
* print the nnpot to screen
* @param out - output stream
* @nnpot - the neural network potential
*/
std::ostream& operator<<(std::ostream& out, const NNPot& nnpot){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("NN - POT",str)<<"\n";
	out<<"R_CUT    = "<<nnpot.rc_<<"\n";
	out<<"NSPECIES = "<<nnpot.nspecies_<<"\n";
	for(int i=0; i<nnpot.nspecies_; ++i) std::cout<<nnpot.nnh_[i]<<"\n";
	out<<print::title("NN - POT",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

/**
* set defaults for the neural network potential
*/
void NNPot::defaults(){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::defaults():\n";
	//species
		nspecies_=0;
		map_.clear();
		nnh_.clear();
	//cutoff
		rc_=0;
	//resize the lattice vector shifts
		R_.clear();
}

//==== resizing ====

/**
* resize the number of species and each NNH
* @param species - the species of the neural network potential
*/
void NNPot::resize(const std::vector<Atom>& species){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::resize(const std::vector<Atom>&)\n";
	if(species.size()==0) throw std::invalid_argument("NNPot::resize(int): invalid number of species");
	nspecies_=species.size();
	nnh_.resize(nspecies_);
	for(int i=0; i<nspecies_; ++i){
		nnh_[i].resize(nspecies_);
		nnh_[i].atom()=species[i];
		map_.add(string::hash(species[i].name()),i);
	}
}

//==== nn-struc ====

/**
* resize the symmetry function vectors to store the inputs
* @param struc - the structure for which we will be resizing the symmetry functions
*/
void NNPot::init_symm(Structure& struc)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::init_symm(Structure&):\n";
	for(int n=0; n<struc.nAtoms(); ++n){
		struc.symm(n).resize(nnh_[index(struc.name(n))].nInput());
	}
}

/**
* compute the symmetry functions for a given structure
* @param struc - the structure for which we will compute the symmetry functions
*/
void NNPot::calc_symm(Structure& struc){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::calc_symm(Structure&):\n";
	if(struc.R().norm()>math::constant::ZERO){
		//lattice vector shifts - factor of two: max distance = 1/2 lattice vector
		const int ratiox=floor(2.0*rc_/struc.R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
		const int ratioy=floor(2.0*rc_/struc.R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
		const int ratioz=floor(2.0*rc_/struc.R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
		if(ratiox>0 || ratioy>0 || ratioz>0){
			Eigen::Vector3d tmp;
			const int shellx=floor(1.0*rc_/struc.R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
			const int shelly=floor(1.0*rc_/struc.R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
			const int shellz=floor(1.0*rc_/struc.R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
			const int Rmax=(2*shellx+1)*(2*shelly+1)*(2*shellz+1);
			if(NN_POT_PRINT_DATA>0) std::cout<<"Rmax = "<<Rmax<<"\n";
			if(NN_POT_PRINT_DATA>0) std::cout<<"shell = ("<<shellx<<","<<shelly<<","<<shellz<<") = "<<(2*shellx+1)*(2*shelly+1)*(2*shellz+1)<<"\n";
			R_.resize(Rmax);
			int Rsize=0;
			for(int ix=-shellx; ix<=shellx; ++ix){
				for(int iy=-shelly; iy<=shelly; ++iy){
					for(int iz=-shellz; iz<=shellz; ++iz){
						R_[Rsize++].noalias()=ix*struc.R().col(0)+iy*struc.R().col(1)+iz*struc.R().col(2);
					}
				}
			}
			if(NN_POT_PRINT_DATA>0) std::cout<<"Rsize = "<<Rsize<<"\n";
			//loop over all atoms
			if(NN_POT_PRINT_STATUS>0) std::cout<<"computing symmetry functions\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				//reset the inputs
				if(NN_POT_PRINT_STATUS>2) std::cout<<"resetting inputs\n";
				struc.symm(i).setZero();
				//find the index of the species of atom i
				const int II=index(struc.name(i));
				//loop over all pairs
				for(int j=0; j<struc.nAtoms(); ++j){
					//find the index of the species of atom j
					const int JJ=index(struc.name(j));
					//find rIJ_:=rI_-rJ_ with respect to the unit cell
					const Eigen::Vector3d rIJ_=struc.diff(struc.posn(i),struc.posn(j),tmp);
					//loop over lattice vector shifts - atom j
					for(int iJ=0; iJ<Rsize; ++iJ){
						//alter the rIJ_ distance by a lattice vector shift
						const Eigen::Vector3d rIJt_=rIJ_+R_[iJ];
						const double dIJ=rIJt_.norm();
						if(math::constant::ZERO<dIJ && dIJ<rc_){
							if(NN_POT_PRINT_STATUS>2) std::cout<<"computing phir("<<i<<","<<j<<")\n";
							//compute the IJ contribution to all radial basis functions
							const int offsetR_=nnh_[II].offsetR(JJ);
							BasisR& basisRij_=nnh_[II].basisR(JJ);
							basisRij_.symm(dIJ);
							for(int nr=0; nr<basisRij_.nfR(); ++nr){
								struc.symm(i)[offsetR_+nr]+=basisRij_.symm()[nr];
							}
							//loop over all triplets
							for(int k=0; k<struc.nAtoms(); ++k){
								//find the index of the species of atom k
								const int KK=index(struc.name(k));
								//find rIJ_:=rI_-rK_ with respect to the unit cell
								const Eigen::Vector3d rIK_=struc.diff(struc.posn(i),struc.posn(k),tmp);
								//loop over all cell shifts  - atom k
								for(int iK=0; iK<Rsize; ++iK){
									//alter the rIK_ distance by a lattice vector shift
									const Eigen::Vector3d rIKt_=rIK_+R_[iK];
									const double dIK=rIKt_.norm();
									if(math::constant::ZERO<dIK && dIK<rc_){
										//calc rJK
										const Eigen::Vector3d rJKt_=rIKt_-rIJt_;
										const double dJK=rJKt_.norm();
										if(math::constant::ZERO<dJK){
											//compute the IJ,IK,JK contribution to all angular basis functions
											const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
											BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
											const double cosIJK=rIJt_.dot(rIKt_)/(dIJ*dIK);
											const double d[3]={dIJ,dIK,dJK};
											basisAijk_.symm(cosIJK,d);
											for(int na=0; na<basisAijk_.nfA(); ++na){
												struc.symm(i)[offsetA_+na]+=0.5*basisAijk_.symm()[na];//0.5 for double counting
											}
										}
									}
								}
							}
						}
					}
				}
			}
		} else {
			Eigen::Vector3d rIJ_,rIK_,rJK_;
			//generate cell list
			CellList cellList(rc_,struc);
			std::vector<Eigen::Vector3i> nnc(27);
			int count=0;
			for(int i=-1; i<=1; i++){
				for(int j=-1; j<=1; j++){
					for(int k=-1; k<=1; k++){
						nnc[count++]<<i,j,k;
					}
				}
			}
			//loop over all atoms
			if(NN_POT_PRINT_STATUS>0) std::cout<<"computing symmetry functions\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				//find the index of the species of atom j
				const int II=index(struc.name(i));
				const Eigen::Vector3i& cell=cellList.cell(i);
				//reset the inputs
				if(NN_POT_PRINT_STATUS>2) std::cout<<"resetting inputs\n";
				struc.symm(i).setZero();
				//loop over all neighboring cells
				for(int jc=0; jc<nnc.size(); ++jc){
					const Eigen::Vector3i jcell=cell+nnc[jc];
					//loop over atoms in cell
					for(int j=0; j<cellList.atoms(jcell).size(); ++j){
						const int nj=cellList.atoms(jcell)[j];
						//find the index of the species of atom j
						const int JJ=index(struc.name(nj));
						//calc rIJ
						if(NN_POT_PRINT_STATUS>2) std::cout<<"symm r("<<i<<","<<nj<<")\n";
						//calc radial contribution - loop over all radial functions
						if(NN_POT_PRINT_STATUS>2) std::cout<<"computing radial functions\n";
						struc.diff(struc.posn(i),struc.posn(nj),rIJ_);
						const double dIJ=rIJ_.norm();
						if(math::constant::ZERO<dIJ && dIJ<rc_){
							if(NN_POT_PRINT_STATUS>2) std::cout<<"computing phir("<<i<<","<<nj<<")\n";
							//compute the IJ contribution to all radial basis functions
							const int offsetR_=nnh_[II].offsetR(JJ);
							BasisR& basisRij_=nnh_[II].basisR(JJ);
							basisRij_.symm(dIJ);
							for(int nr=0; nr<basisRij_.nfR(); ++nr){
								struc.symm(i)[offsetR_+nr]+=basisRij_.symm()[nr];
							}
							//loop over all neighboring cells
							for(int kc=0; kc<nnc.size(); ++kc){
								const Eigen::Vector3i kcell=cell+nnc[kc];
								//loop over all triplets
								for(int k=0; k<cellList.atoms(kcell).size(); ++k){
									const int nk=cellList.atoms(kcell)[k];
									//find the index of the species of atom i
									const int KK=index(struc.name(nk));
									//calculate rIK
									if(NN_POT_PRINT_STATUS>2) std::cout<<"computing phia("<<i<<","<<nj<<","<<nk<<")\n";
									struc.diff(struc.posn(i),struc.posn(nk),rIK_);
									const double dIK=rIK_.norm();
									if(math::constant::ZERO<dIK && dIK<rc_){
										//calculate rJK
										struc.diff(struc.posn(nj),struc.posn(nk),rJK_);
										const double dJK=rJK_.norm();
										if(math::constant::ZERO<dJK){
											//compute the IJ,IK,JK contribution to all angular basis functions
											const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
											BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
											const double cosIJK=rIJ_.dot(rIK_)/(dIJ*dIK);
											const double d[3]={dIJ,dIK,dJK};
											basisAijk_.symm(cosIJK,d);
											for(int na=0; na<basisAijk_.nfA(); ++na){
												struc.symm(i)[offsetA_+na]+=0.5*basisAijk_.symm()[na];//0.5 for double counting
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
	} else {
		//loop over all atoms
		if(NN_POT_PRINT_STATUS>0) std::cout<<"computing symmetry functions\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			//find the index of the species of atom i
			const int II=index(struc.name(i));
			//reset the inputs
			if(NN_POT_PRINT_STATUS>2) std::cout<<"resetting inputs\n";
			struc.symm(i).setZero();
			//loop over all pairs
			for(int j=0; j<struc.nAtoms(); ++j){
				if(i==j) continue;
				//find the index of the species of atom j
				const int JJ=index(struc.name(j));
				//calc rIJ
				const Eigen::Vector3d rIJ_=struc.posn(i)-struc.posn(j);
				const double dIJ=rIJ_.norm();
				if(dIJ<rc_){
					if(NN_POT_PRINT_STATUS>2) std::cout<<"computing phir("<<i<<","<<j<<")\n";
					//compute the IJ contribution to all radial basis functions
					const int offsetR_=nnh_[II].offsetR(JJ);
					BasisR& basisRij_=nnh_[II].basisR(JJ);
					basisRij_.symm(dIJ);
					for(int nr=0; nr<basisRij_.nfR(); ++nr){
						struc.symm(i)[offsetR_+nr]+=basisRij_.symm()[nr];
					}
					//loop over all triplets
					for(int k=j+1; k<struc.nAtoms(); ++k){
						if(i==k) continue;
						//find the index of the species of atom k
						const int KK=index(struc.name(k));
						//calc rIK
						const Eigen::Vector3d rIK_=struc.posn(i)-struc.posn(k);
						const double dIK=rIK_.norm();
						if(dIK<rc_){
							//calc rJK
							const Eigen::Vector3d rJK_=struc.posn(j)-struc.posn(k);
							const double dJK=rJK_.norm();
							if(math::constant::ZERO<dJK){
								//compute the IJ,IK,JK contribution to all angular basis functions
								const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
								BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
								const double cosIJK=rIJ_.dot(rIK_)/(dIJ*dIK);
								const double d[3]={dIJ,dIK,dJK};
								basisAijk_.symm(cosIJK,d);
								for(int na=0; na<basisAijk_.nfA(); ++na){
									struc.symm(i)[offsetA_+na]+=basisAijk_.symm()[na];
								}
							}
						}
					}
				}
			}
		}
	}
	/*for(int i=0; i<struc.nAtoms(); ++i){
		std::cout<<struc.name(i)<<" symm["<<i<<"] = "<<struc.symm(i).transpose()<<"\n";
	}*/
}

/**
* compute the forces on the atoms for a given structure
* @param struc - the structure for which we will compute the forces
* @param calc_symm_ - whether we need to compute the symmetry functions
*/
void NNPot::forces(Structure& struc, bool calc_symm_){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::forces(Structure&,bool):\n";
	//local variables
	Eigen::VectorXd dEdG;
	//set the inputs for the atoms
	if(calc_symm_) calc_symm(struc);
	//reset the force
	for(int i=0; i<struc.nAtoms(); ++i) struc.force(i).setZero();
	if(struc.nAtoms()>1){
	//compute the forces
	if(struc.R().norm()>math::constant::ZERO){
		//periodic structure
		Eigen::Vector3d tmp;
		//lattice vector shifts
		const int shellx=floor(1.0*rc_/struc.R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
		const int shelly=floor(1.0*rc_/struc.R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
		const int shellz=floor(1.0*rc_/struc.R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
		const int Rmax=(2*shellx+1)*(2*shelly+1)*(2*shellz+1);
		if(NN_POT_PRINT_DATA>0) std::cout<<"Rmax = "<<Rmax<<"\n";
		if(NN_POT_PRINT_DATA>0) std::cout<<"shell = ("<<shellx<<","<<shelly<<","<<shellz<<") = "<<(2*shellx+1)*(2*shelly+1)*(2*shellz+1)<<"\n";
		R_.resize(Rmax);
		int Rsize=0;
		for(int ix=-shellx; ix<=shellx; ++ix){
			for(int iy=-shelly; iy<=shelly; ++iy){
				for(int iz=-shellz; iz<=shellz; ++iz){
					R_[Rsize++].noalias()=ix*struc.R().col(0)+iy*struc.R().col(1)+iz*struc.R().col(2);
				}
			}
		}
		//loop over all atoms
		for(int i=0; i<struc.nAtoms(); ++i){
			//find the index of the species of atom i
			const int II=index(struc.name(i));
			//execute the appropriate network
			nnh_[II].nn().execute(struc.symm(i));
			//calculate the network gradient
			nnh_[II].dOutDVal().grad(nnh_[II].nn());
			//set the gradient (n.b. dodi - do/di - deriv. of out w.r.t. in)
			dEdG=nnh_[II].dOutDVal().dodi().row(0);
			//loop over pairs
			for(int j=0; j<struc.nAtoms(); ++j){
				//find the index of the species of atom j
				const int JJ=index(struc.name(j));
				//find rIJ_:=rI_-rJ_ with respect to the unit cell
				const Eigen::Vector3d rIJ_=struc.diff(struc.posn(i),struc.posn(j),tmp);
				//loop over lattice vector shifts - atom j
				for(int iJ=0; iJ<Rsize; ++iJ){
					//alter the rIJ_ distance by a lattice vector shift
					const Eigen::Vector3d rIJt_=rIJ_+R_[iJ];
					const double dIJ=rIJt_.norm();
					//check rIJ
					if(math::constant::ZERO<dIJ && dIJ<rc_){
						const double dIJi=1.0/dIJ;
						//compute the IJ contribution to the radial force
						const int offsetR_=nnh_[II].offsetR(JJ);
						const double amp=nnh_[II].basisR(JJ).force(dIJ,dEdG.data()+offsetR_)*dIJi;
						struc.force(i).noalias()+=amp*rIJt_;
						struc.force(j).noalias()-=amp*rIJt_;
						//loop over all triplets
						for(int k=0; k<struc.nAtoms(); ++k){
							if(NN_POT_PRINT_STATUS>2) std::cout<<"computing theta("<<i<<","<<j<<","<<k<<")\n";
							//find the index of the species of atom k
							const int KK=index(struc.name(k));
							//find rIK_:=rI_-rK_ with respect to the unit cell
							const Eigen::Vector3d rIK_=struc.diff(struc.posn(i),struc.posn(k),tmp);
							//loop over all cell shifts - atom k
							for(int iK=0; iK<Rsize; ++iK){
								//alter the rIK_ distance by a lattice vector shift
								const Eigen::Vector3d rIKt_=rIK_+R_[iK];
								const double dIK=rIKt_.norm();
								if(math::constant::ZERO<dIK && dIK<rc_){
									const double dIKi=1.0/dIK;
									//compute rJK
									const Eigen::Vector3d rJKt_=rIKt_-rIJt_;
									const double dJK=rJKt_.norm();
									if(math::constant::ZERO<dJK){
										const double dJKi=1.0/dJK;
										//compute the IJ,IK,JK contribution to the angular force
										const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
										const double cosIJK=rIJt_.dot(rIKt_)*dIJi*dIKi;
										const double d[3]={dIJ,dIK,dJK};
										double phi=0; double eta[3]={0,0,0};
										nnh_[II].basisA(JJ,KK).force(phi,eta,cosIJK,d,dEdG.data()+offsetA_);
										phi*=0.5; eta[0]*=0.5; eta[1]*=0.5; eta[2]*=0.5;//0.5 for double counting
										struc.force(i).noalias()+=(phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJt_*dIJi;
										struc.force(i).noalias()+=(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIKt_*dIKi;
										struc.force(j).noalias()-=(-phi*cosIJK*dIJi+eta[0])*rIJt_*dIJi+phi*dIJi*rIKt_*dIKi;
										struc.force(j).noalias()-=eta[2]*rJKt_*dJKi;
										struc.force(k).noalias()-=(-phi*cosIJK*dIKi+eta[1])*rIKt_*dIKi+phi*dIKi*rIJt_*dIJi;
										struc.force(k).noalias()+=eta[2]*rJKt_*dJKi;
									}
								}
							}
						}
					}
				}
			}
		}
	} else {
		//loop over all atoms
		for(int i=0; i<struc.nAtoms(); ++i){
			//find the index of the species of atom i
			const int II=index(struc.name(i));
			//execute the appropriate network
			nnh_[II].nn().execute(struc.symm(i));
			//calculate the network gradient
			nnh_[II].dOutDVal().grad(nnh_[II].nn());
			//set the gradient (n.b. dodi - do/di - deriv. of out w.r.t. in)
			dEdG=nnh_[II].dOutDVal().dodi().row(0);
			//loop over pairs
			for(int j=0; j<struc.nAtoms(); ++j){
				//find the index of the species of atom j
				const int JJ=index(struc.name(j));
				//compute rIJ
				const Eigen::Vector3d rIJ_=struc.posn(i)-struc.posn(j);
				const double dIJ=rIJ_.norm();
				//check rIJ
				if(math::constant::ZERO<dIJ && dIJ<rc_){
					const double dIJi=1.0/dIJ;
					//compute the IJ contribution to the radial force
					const int offsetR_=nnh_[II].offsetR(JJ);
					const double amp=nnh_[II].basisR(JJ).force(dIJ,dEdG.data()+offsetR_)*dIJi;
					struc.force(i).noalias()+=amp*rIJ_;
					struc.force(j).noalias()-=amp*rIJ_;
					//loop over all triplets
					for(int k=0; k<struc.nAtoms(); ++k){
						if(NN_POT_PRINT_STATUS>2) std::cout<<"computing theta("<<i<<","<<j<<","<<k<<")\n";
						//find the index of the species of atom k
						const int KK=index(struc.name(k));
						//compute rIK
						const Eigen::Vector3d rIK_=struc.posn(i)-struc.posn(k);
						const double dIK=rIK_.norm();
						if(math::constant::ZERO<dIK && dIK<rc_){
							const double dIKi=1.0/dIK;
							//compute rJK
							const Eigen::Vector3d rJK_=struc.posn(j)-struc.posn(k);
							const double dJK=rJK_.norm();
							if(math::constant::ZERO<dJK){
								const double dJKi=1.0/dJK;
								//compute the IJ,IK,JK contribution to the angular force
								const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
								const double cosIJK=rIJ_.dot(rIK_)*dIJi*dIKi;
								const double d[3]={dIJ,dIK,dJK};
								double phi=0; double eta[3]={0,0,0};
								nnh_[II].basisA(JJ,KK).force(phi,eta,cosIJK,d,dEdG.data()+offsetA_);
								phi*=0.5; eta[0]*=0.5; eta[1]*=0.5; eta[2]*=0.5;//0.5 for double counting
								struc.force(i).noalias()+=(phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJ_*dIJi;
								struc.force(i).noalias()+=(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIK_*dIKi;
								struc.force(j).noalias()-=(-phi*cosIJK*dIJi+eta[0])*rIJ_*dIJi+phi*dIJi*rIK_*dIKi;
								struc.force(j).noalias()-=eta[2]*rJK_*dJKi;
								struc.force(k).noalias()-=(-phi*cosIJK*dIKi+eta[1])*rIK_*dIKi+phi*dIKi*rIJ_*dIJi;
								struc.force(k).noalias()+=eta[2]*rJK_*dJKi;
							}
						}
					}
				}
			}
		}
	}
	}
}

/**
* execute all atomic networks and return energy
* @param struc - the structure for which we will compute the energy
* @param calc_symm_ - whether we need to compute the symmetry functions
*/
double NNPot::energy(Structure& struc, bool calc_symm_){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::energy(Structure&,bool):\n";
	double energy=0;
	//set the inputs for the atoms
	if(calc_symm_) calc_symm(struc);
	//loop over atoms
	for(int n=0; n<struc.nAtoms(); ++n){
		//set the index
		const int index=map_[string::hash(struc.name(n))];
		//compute the energy
		energy+=nnh_[index].energy(struc.symm(n));
	}
	return energy;
}

double NNPot::compute(Structure& struc, bool calc_symm_){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::compute(Structure&,bool):\n";
	//local variables
	Eigen::VectorXd dEdG;
	double energyt=0;
	//set the inputs for the atoms
	if(calc_symm_) calc_symm(struc);
	//reset the force
	for(int i=0; i<struc.nAtoms(); ++i) struc.force(i).setZero();
	if(struc.R().norm()>math::constant::ZERO){
		//lattice vector shifts
		const int shellx=floor(2.0*rc_/struc.R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
		const int shelly=floor(2.0*rc_/struc.R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
		const int shellz=floor(2.0*rc_/struc.R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
		const int Rmax=(2*shellx+1)*(2*shelly+1)*(2*shellz+1);
		if(NN_POT_PRINT_DATA>0) std::cout<<"Rmax = "<<Rmax<<"\n";
		if(NN_POT_PRINT_DATA>0) std::cout<<"shell = ("<<shellx<<","<<shelly<<","<<shellz<<") = "<<(2*shellx+1)*(2*shelly+1)*(2*shellz+1)<<"\n";
		R_.resize(Rmax);
		int Rsize=0;
		for(int ix=-shellx; ix<=shellx; ++ix){
			for(int iy=-shelly; iy<=shelly; ++iy){
				for(int iz=-shellz; iz<=shellz; ++iz){
					//const Eigen::Vector3d R=ix*struc.R().col(0)+iy*struc.R().col(1)+iz*struc.R().col(2);
					//if(R.norm()<2.0*rc_) R_[Rsize++]=R;
					R_[Rsize++].noalias()=ix*struc.R().col(0)+iy*struc.R().col(1)+iz*struc.R().col(2);
				}
			}
		}
		//loop over all atoms
		for(int i=0; i<struc.nAtoms(); ++i){
			//find the index of the species of atom i
			const int II=map_[string::hash(struc.name(i))];
			//copy position
			const Eigen::Vector3d rI_=struc.posn(i);
			//execute the appropriate network
			nnh_[II].nn().execute(struc.symm(i));
			energyt+=nnh_[II].nn().out()[0]+nnh_[II].atom().energy();
			//calculate the network gradient
			nnh_[II].dOutDVal().grad(nnh_[II].nn());
			//set the gradient (n.b. dodi - do/di - deriv. of out w.r.t. in)
			dEdG=nnh_[II].dOutDVal().dodi().row(0);
			//loop over pairs
			for(int j=0; j<struc.nAtoms(); ++j){
				//find the index of the species of atom j
				const int JJ=map_[string::hash(struc.name(j))];
				//loop over lattice vector shifts - atom j
				for(int iJ=0; iJ<Rsize; ++iJ){
					//compute rIJ
					const Eigen::Vector3d rJ_=struc.posn(j)+R_[iJ];
					const Eigen::Vector3d rIJ_=rI_-rJ_;
					const double dIJ=rIJ_.norm();
					//check rIJ
					if(math::constant::ZERO<dIJ && dIJ<rc_){
						const double dIJi=1.0/dIJ;
						//compute the IJ contribution to the radial force
						const int offsetR_=nnh_[II].offsetR(JJ);
						const double amp=nnh_[II].basisR(JJ).force(dIJ,dEdG.data()+offsetR_)*dIJi;
						struc.force(i).noalias()+=amp*rIJ_;
						struc.force(j).noalias()-=amp*rIJ_;
						//loop over all triplets
						for(int k=0; k<struc.nAtoms(); ++k){
							if(NN_POT_PRINT_STATUS>2) std::cout<<"computing theta("<<i<<","<<j<<","<<k<<")\n";
							//find the index of the species of atom k
							const int KK=map_[string::hash(struc.name(k))];
							//loop over all cell shifts - atom k
							for(int iK=0; iK<Rsize; ++iK){
								//compute rIK
								const Eigen::Vector3d rK_=struc.posn(k)+R_[iK];
								const Eigen::Vector3d rIK_=rI_-rK_;
								const double dIK=rIK_.norm();
								if(math::constant::ZERO<dIK && dIK<rc_){
									const double dIKi=1.0/dIK;
									//compute rJK
									const Eigen::Vector3d rJK_=rJ_-rK_;
									const double dJK=rJK_.norm();
									if(math::constant::ZERO<dJK){
										const double dJKi=1.0/dJK;
										//compute the IJ,IK,JK contribution to the angular force
										const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
										const double cosIJK=rIJ_.dot(rIK_)*dIJi*dIKi;
										const double d[3]={dIJ,dIK,dJK};
										double phi=0; double eta[3]={0,0,0};
										nnh_[II].basisA(JJ,KK).force(phi,eta,cosIJK,d,dEdG.data()+offsetA_);
										struc.force(i).noalias()+=(phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJ_*dIJi;
										struc.force(i).noalias()+=(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIK_*dIKi;
										struc.force(j).noalias()-=(-phi*cosIJK*dIJi+eta[0])*rIJ_*dIJi+phi*dIJi*rIK_*dIKi;
										struc.force(j).noalias()-=eta[2]*rJK_*dJKi;
										struc.force(k).noalias()-=(-phi*cosIJK*dIKi+eta[1])*rIK_*dIKi+phi*dIKi*rIJ_*dIJi;
										struc.force(k).noalias()+=eta[2]*rJK_*dJKi;
									}
								}
							}
						}
					}
				}
			}
		}
	} else {
		//loop over all atoms
		for(int i=0; i<struc.nAtoms(); ++i){
			//find the index of the species of atom i
			const int II=index(struc.name(i));
			//execute the appropriate network
			nnh_[II].nn().execute(struc.symm(i));
			energyt+=nnh_[II].nn().out()[0]+nnh_[II].atom().energy();
			//calculate the network gradient
			nnh_[II].dOutDVal().grad(nnh_[II].nn());
			//set the gradient (n.b. dodi - do/di - deriv. of out w.r.t. in)
			dEdG=nnh_[II].dOutDVal().dodi().row(0);
			//loop over pairs
			for(int j=0; j<struc.nAtoms(); ++j){
				//find the index of the species of atom j
				const int JJ=index(struc.name(j));
				//compute rIJ
				const Eigen::Vector3d rIJ_=struc.posn(i)-struc.posn(j);
				const double dIJ=rIJ_.norm();
				//check rIJ
				if(math::constant::ZERO<dIJ && dIJ<rc_){
					const double dIJi=1.0/dIJ;
					//compute the IJ contribution to the radial force
					const int offsetR_=nnh_[II].offsetR(JJ);
					const double amp=nnh_[II].basisR(JJ).force(dIJ,dEdG.data()+offsetR_)*dIJi;
					struc.force(i).noalias()+=amp*rIJ_;
					struc.force(j).noalias()-=amp*rIJ_;
					//loop over all triplets
					for(int k=0; k<struc.nAtoms(); ++k){
						if(NN_POT_PRINT_STATUS>2) std::cout<<"computing theta("<<i<<","<<j<<","<<k<<")\n";
						//find the index of the species of atom k
						const int KK=index(struc.name(k));
						//compute rIK
						const Eigen::Vector3d rIK_=struc.posn(i)-struc.posn(k);
						const double dIK=rIK_.norm();
						if(math::constant::ZERO<dIK && dIK<rc_){
							const double dIKi=1.0/dIK;
							//compute rJK
							const Eigen::Vector3d rJK_=struc.posn(j)-struc.posn(k);
							const double dJK=rJK_.norm();
							if(math::constant::ZERO<dJK){
								const double dJKi=1.0/dJK;
								//compute the IJ,IK,JK contribution to the angular force
								const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
								const double cosIJK=rIJ_.dot(rIK_)*dIJi*dIKi;
								const double d[3]={dIJ,dIK,dJK};
								double phi=0; double eta[3]={0,0,0};
								nnh_[II].basisA(JJ,KK).force(phi,eta,cosIJK,d,dEdG.data()+offsetA_);
								struc.force(i).noalias()+=(phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJ_*dIJi;
								struc.force(i).noalias()+=(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIK_*dIKi;
								struc.force(j).noalias()-=(-phi*cosIJK*dIJi+eta[0])*rIJ_*dIJi+phi*dIJi*rIK_*dIKi;
								struc.force(j).noalias()-=eta[2]*rJK_*dJKi;
								struc.force(k).noalias()-=(-phi*cosIJK*dIKi+eta[1])*rIK_*dIKi+phi*dIKi*rIJ_*dIJi;
								struc.force(k).noalias()+=eta[2]*rJK_*dJKi;
							}
						}
					}
				}
			}
		}
	}
	return energyt;
}

//==== static functions ====

//read/write basis

void NNPot::read_basis(const char* file, NNPot& nnpot, const char* atomName){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::read(const char*,NNPot&,const char*):\n";
	FILE* reader=NULL;
	reader=fopen(file,"r");
	if(reader!=NULL){
		NNPot::read_basis(reader,nnpot,atomName);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("NNPot::read(const char*,NNPot&): Could not open nnpot file: \"")+std::string(file)+std::string("\""));
}

void NNPot::read_basis(FILE* reader, NNPot& nnpot, const char* atomName){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::read_basis(FILE*,NNPot&,const char*):\n";
	//==== local function variables ====
	std::vector<std::string> strlist;
	char* input=new char[string::M];
	//==== get atom species ====
	const int atomIndex=nnpot.index(atomName);
	//==== global cutoff ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const double rc=std::atof(strlist.at(1).c_str());
	if(rc!=nnpot.rc()) throw std::invalid_argument("NNPot::read_basis(FILE*,NNPot&,const char*): invalid cutoff.");
	//==== number of species ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const int nspecies=std::atoi(strlist.at(1).c_str());
	if(nspecies!=nnpot.nspecies()) throw std::invalid_argument("NNPot::read_basis(FILE*,NNPot&,const char*): invalid number of species.");
	//==== central species ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const int II=nnpot.index(strlist.at(1));
	//==== check indices ====
	if(atomIndex!=II) throw std::invalid_argument("NNPot::read_basis(FILE*,NNPot&,const char*): invalid central species.\n");
	//==== basis - radial ====
	for(int j=0; j<nspecies; ++j){
		//read species
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int JJ=nnpot.index(strlist.at(1));
		//read basis
		BasisR::read(reader,nnpot.nnh(II).basisR(JJ));
	}
	//==== basis - angular ====
	for(int j=0; j<nspecies; ++j){
		for(int k=j; k<nspecies; ++k){
			//read species
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const int JJ=nnpot.index(strlist.at(1));
			const int KK=nnpot.index(strlist.at(2));
			//read basis
			BasisA::read(reader,nnpot.nnh(II).basisA(JJ,KK));
		}
	}
	//==== initialize the inputs ====
	nnpot.nnh(II).init_input();
	//==== clear local variables ====
	delete[] input;
}

//read/write nnpot

/**
* Write the neural network to file
* @param file - the name of the file to which the object will be written
* @param nnpot - the neural network potential to be written
*/
void NNPot::write(const char* file, const NNPot& nnpot){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::write(const char*,const NNPot&):\n";
	FILE* writer=NULL;
	writer=fopen(file,"w");
	if(writer!=NULL){
		NNPot::write(writer,nnpot);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("NNPot::write(const char*,const NNPot&): Could not write to nnh file: \"")+std::string(file)+std::string("\""));
}

/**
* Read the neural network from file
* @param file - the name of the file fro
* @param nnpot - stores the neural network potential to be read
*/
void NNPot::read(const char* file, NNPot& nnpot){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::read(const char*,NNPot&):\n";
	FILE* reader=NULL;
	reader=fopen(file,"r");
	if(reader!=NULL){
		NNPot::read(reader,nnpot);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("NNPot::read(const char*,NNPot&): Could not open nnpot file: \"")+std::string(file)+std::string("\""));
}

/**
* Write the neural network to file
* @param writer - the file pointer used to write the object to file
* @param nnpot - the neural network potential to be written
*/
void NNPot::write(FILE* writer, const NNPot& nnpot){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::write(FILE*,const NNPot&):\n";
	//==== header ====
	fprintf(writer,"ann\n");
	//==== species ====
	fprintf(writer, "nspecies %i\n",nnpot.nspecies());
	for(int n=0; n<nnpot.nspecies(); ++n){
		const Atom& atom=nnpot.nnh(n).atom();
		fprintf(writer,"%s %f %f %f\n",atom.name().c_str(),atom.mass(),atom.energy(),atom.charge());
	}
	//==== cutoff ====
	fprintf(writer,"rc %f\n",nnpot.rc());
	//==== basis ====
	for(int i=0; i<nnpot.nspecies(); ++i){
		//write central species
		fprintf(writer,"basis %s\n",nnpot.nnh(i).atom().name().c_str());
		//write basis - radial
		for(int j=0; j<nnpot.nspecies(); ++j){
			//write species
			fprintf(writer,"basis_radial %s\n",nnpot.nnh(j).atom().name().c_str());
			//write basis
			BasisR::write(writer,nnpot.nnh(i).basisR(j));
		}
		//write basis - angular
		for(int j=0; j<nnpot.nspecies(); ++j){
			for(int k=j; k<nnpot.nspecies(); ++k){
				//write species
				fprintf(writer,"basis_angular %s %s\n",nnpot.nnh(j).atom().name().c_str(),nnpot.nnh(k).atom().name().c_str());
				//write basis
				BasisA::write(writer,nnpot.nnh(i).basisA(j,k));
			}
		}
	}
	//==== neural network ====
	for(int n=0; n<nnpot.nspecies(); ++n){
		//write central species
		fprintf(writer,"nn %s\n",nnpot.nnh(n).atom().name().c_str());
		//write the network
		NeuralNet::ANN::write(writer,nnpot.nnh(n).nn());
	}
}

/**
* Read the neural network from file
* @param reader - the file pointer used to read the object from file
* @param nnpot - stores the neural network potential to be read
*/
void NNPot::read(FILE* reader, NNPot& nnpot){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::read(FILE*,NNPot&):\n";
	//==== local function variables ====
	std::vector<std::string> strlist;
	char* input=new char[string::M];
	//==== header ====
	fgets(input,string::M,reader);
	//==== number of species ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const int nspecies=std::atoi(strlist.at(1).c_str());
	if(nspecies<=0) throw std::invalid_argument("NNPot::read(FILE*,NNPot&): invalid number of species.");
	//==== species ====
	std::vector<Atom> species(nspecies);
	for(int n=0; n<nspecies; ++n){
		Atom::read(fgets(input,string::M,reader),species[n]);
	}
	//==== resize ====
	nnpot.resize(species);
	//==== global cutoff ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const double rc=std::atof(strlist.at(1).c_str());
	if(rc<=0) throw std::invalid_argument("NNPot::read(FILE*,NNPot&): invalid cutoff.");
	else nnpot.rc()=rc;
	//==== basis ====
	for(int i=0; i<nspecies; ++i){
		//read central species
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int II=nnpot.index(strlist.at(1));
		//read basis - radial
		for(int j=0; j<nspecies; ++j){
			//read species
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const int JJ=nnpot.index(strlist.at(1));
			//read basis
			BasisR::read(reader,nnpot.nnh(II).basisR(JJ));
		}
		//read basis - angular
		for(int j=0; j<nspecies; ++j){
			for(int k=j; k<nspecies; ++k){
				//read species
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				const int JJ=nnpot.index(strlist.at(1));
				const int KK=nnpot.index(strlist.at(2));
				//read basis
				BasisA::read(reader,nnpot.nnh(II).basisA(JJ,KK));
			}
		}
	}
	//==== initialize inputs ====
	for(int i=0; i<nspecies; ++i){
		nnpot.nnh(i).init_input();
	}
	//==== neural network ====
	for(int n=0; n<nspecies; ++n){
		//read species
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int II=nnpot.index(strlist.at(1));
		//read network
		NeuralNet::ANN::read(reader,nnpot.nnh(II).nn());
		//resize gradient object
		nnpot.nnh(II).dOutDVal().resize(nnpot.nnh(II).nn());
	}
	//==== clear local variables ====
	delete[] input;
}

//************************************************************
// serialization
//************************************************************

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const NNH& obj){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"nbytes(const NNH&):\n";
	int size=0;
	//hamiltonian
	size+=nbytes(obj.atom());
	size+=nbytes(obj.nn());
	//species
	size+=nbytes(obj.nspecies());//nspecies_
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		size+=nbytes(obj.basisR(j));
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			size+=nbytes(obj.basisA(j,k));
		}
	}
	//return the size
	return size;
}

template <> int nbytes(const NNPot& obj){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"nbytes(const NNPot&):\n";
	int size=0;
	//species
	size+=nbytes(obj.nspecies());
	size+=nbytes(obj.map());
	for(int i=0; i<obj.nspecies(); ++i){
		size+=nbytes(obj.nnh(i));
	}
	//cutoff
	size+=nbytes(obj.rc());
	//return the size
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const NNH& obj, char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"pack(const NNH&,char*):\n";
	int pos=0;
	//hamiltonian
	pos+=pack(obj.atom(),arr+pos);
	pos+=pack(obj.nn(),arr+pos);
	//species
	pos+=pack(obj.nspecies(),arr+pos);
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		pos+=pack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			pos+=pack(obj.basisA(j,k),arr+pos);
		}
	}
	//return bytes written
	return pos;
}
template <> int pack(const NNPot& obj, char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"pack(const NNPot&,char*):\n";
	int pos=0;
	//species
	pos+=pack(obj.nspecies(),arr+pos);
	pos+=pack(obj.map(),arr+pos);
	for(int i=0; i<obj.nspecies(); ++i){
		pos+=pack(obj.nnh(i),arr+pos);
	}
	//cutoff
	pos+=pack(obj.rc(),arr+pos);
	//return bytes written
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(NNH& obj, const char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"unpack(NNH&,const char*):\n";
	int pos=0;
	//hamiltonian
	pos+=unpack(obj.atom(),arr+pos);
	pos+=unpack(obj.nn(),arr+pos);
	obj.dOutDVal().resize(obj.nn());
	//species
	int nspecies=0;
	pos+=unpack(nspecies,arr+pos);
	obj.resize(nspecies);
	//basis for pair/triple interactions
	for(int j=0; j<obj.nspecies(); ++j){
		pos+=unpack(obj.basisR(j),arr+pos);
	}
	for(int j=0; j<obj.nspecies(); ++j){
		for(int k=j; k<obj.nspecies(); ++k){
			pos+=unpack(obj.basisA(j,k),arr+pos);
		}
	}
	//intialize the inputs and offsets
	obj.init_input();
	//return bytes read
	return pos;
}
template <> int unpack(NNPot& obj, const char* arr){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"unpack(NNPot&,const char*):\n";
	int pos=0;
	//species
	int nspecies=0;
	Map<int,int> map;
	pos+=unpack(nspecies,arr+pos);
	pos+=unpack(map,arr+pos);
	std::vector<NNH> nnh(nspecies);
	std::vector<Atom> species(nspecies);
	for(int i=0; i<nspecies; ++i){
		pos+=unpack(nnh[i],arr+pos);
		species[i]=nnh[i].atom();
	}
	obj.resize(species);
	for(int i=0; i<obj.nspecies(); ++i){
		obj.nnh(i)=nnh[i];
	}
	obj.map()=map;
	//cutoff
	pos+=unpack(obj.rc(),arr+pos);
	//return bytes read
	return pos;
}

}
