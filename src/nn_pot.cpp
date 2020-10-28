// c libraries
#include <cmath>
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
	out<<"R_CUT    = "<<nnh.rc_<<"\n";
	//species
	out<<"NSPECIES = "<<nnh.nspecies_<<"\n";
	out<<"ATOMS    = \n";
	for(int i=0; i<nnh.nspecies_; ++i) std::cout<<"\t"<<nnh.species_[i]<<"\n";
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

//misc

/**
* set NNH defaults
*/
void NNH::defaults(){
	//hamiltonian
		rc_=0;
		atom_.clear();
		nn_.clear();
	//interacting species
		nspecies_=0;
		species_.clear();
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

//resizing

/**
* resize the number of species
* @param species - the set of species in the hamiltonian
*/
void NNH::resize(const std::vector<Atom>& species){
	if(species.size()==0) throw std::invalid_argument("NNH::resize(const std::vector<Atom>&): invalid number of species.");
	nspecies_=species.size();
	basisR_.resize(nspecies_);
	basisA_.resize(nspecies_);
	species_.resize(nspecies_);
	offsetR_.resize(nspecies_);
	offsetA_.resize(nspecies_);
	species_=species;
	for(int i=0; i<nspecies_; ++i){
		map_.add(string::hash(species_[i].name()),i);
	}
}

/**
* initialize the inputs and offsets associated with the basis functions
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

//output

/**
* compute energy of atom with symmetry function "symm"
* @param symm - the symmetry function
*/
double NNH::energy(const Eigen::VectorXd& symm){
	return nn_.execute(symm)[0]+atom_.energy();
}

//reading/writing - all

/**
* write NNH to file with name "filename"
* @param filename - the file the NNH will be written to
*/
void NNH::write(const std::string& filename)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::write(const std::string&):\n";
	FILE* writer=NULL;
	writer=fopen(filename.c_str(),"w");
	if(writer!=NULL){
		write(writer);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("NNH::write(const std::string&): Could not write to nnh file: \"")+filename+std::string("\""));
}

/**
* write NNH to file pointed to by "writer"
* @param writer - file pointer
*/
void NNH::write(FILE* writer)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::write(FILE*):\n";
	//==== write the header ====
	fprintf(writer,"ann\n");
	//==== write the global cutoff ====
	fprintf(writer,"cut %f\n",rc_);
	//==== write the central species ====
	fprintf(writer,"%s %f %f %f\n",atom_.name().c_str(),atom_.mass(),atom_.energy(),atom_.charge());
	//==== write the number of species ====
	fprintf(writer,"nspecies %i\n",nspecies_);
	//==== write all species ====
	for(int i=0; i<nspecies_; ++i){
		fprintf(writer,"%s %f %f %f\n",species_[i].name().c_str(),species_[i].mass(),species_[i].energy(),species_[i].charge());
	}
	//==== write the radial basis ====
	for(int j=0; j<nspecies_; ++j){
		fprintf(writer,"basis_radial %s\n",species_[j].name().c_str());
		BasisR::write(writer,basisR_[j]);
	}
	//==== write the angular basis ====
	for(int j=0; j<nspecies_; ++j){
		for(int k=j; k<nspecies_; ++k){
			fprintf(writer,"basis_angular %s %s\n",species_[j].name().c_str(),species_[k].name().c_str());
			BasisA::write(writer,basisA_(j,k));
		}
	}
	//==== write the neural network ====
	NN::Network::write(writer,nn_);
}

/**
* write NNH to file pointed to by "writer"
* @param filename - the file the NNH will be read from
*/
void NNH::read(const std::string& filename){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::read(const std::string&):\n";
	FILE* reader=NULL;
	reader=fopen(filename.c_str(),"r");
	if(reader!=NULL){
		read(reader);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("NNH::read(const std::string&): Could not open nnpot file: \"")+filename+std::string("\""));
}

/**
* read NNH from file pointed to by "reader"
* @param reader - file pointer
*/
void NNH::read(FILE* reader){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::read(FILE*):\n";
	//==== local function variables ====
	std::vector<std::string> strlist;
	char* input=new char[string::M];
	//==== reader in header ====
	fgets(input,string::M,reader);
	//==== read in global cutoff ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const double rc=std::atof(std::strtok(NULL,string::WS));
	if(rc<=0) throw std::invalid_argument("NNH::read(FILE*): invalid cutoff.");
	else rc_=rc;
	//==== read the central atom ====
	Atom::read(fgets(input,string::M,reader),atom_);
	//==== read the number of species ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const int nspecies=std::atoi(std::strtok(NULL,string::WS));
	//==== read all species ====
	std::vector<Atom> species;
	for(int i=0; i<nspecies; ++i){
		Atom::read(fgets(input,string::M,reader),species[i]);
		if(species[i].id()==atom_.id() && species[i]!=atom_) throw std::invalid_argument("Central atom and neighbor atom do not match.");
	}
	//==== resize the hamiltonian ====
	resize(species);
	//==== read the radial basis ====
	for(int i=0; i<nspecies_; ++i){
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int jj=index(strlist[1]);
		BasisR::read(reader,basisR_[jj]);
	}
	//==== read the angular basis ====
	for(int i=0; i<nspecies_; ++i){
		for(int j=i; j<nspecies_; ++j){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const int jj=index(strlist[1]);
			const int kk=index(strlist[2]);
			BasisA::read(reader,basisA_(jj,kk));
		}
	}
	//==== read the neural network ====
	NN::Network::read(reader,nn_);
	//==== set the number of inputs and offsets ====
	init_input();
	//==== free local variables ====
	delete[] input;
}

//reading/writing - basis

/**
* write the basis associated with the NNH to file with name "filename"
* @param filename - the file the basis will be written to
*/
void NNH::write_basis(const std::string& filename)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::write_basis(const std::string&):\n";
	FILE* writer=NULL;
	writer=fopen(filename.c_str(),"w");
	if(writer!=NULL){
		write_basis(writer);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("NNH::write_basis(const std::string&): Could not write to nnh file: \"")+filename+std::string("\""));
}

/**
* write the basis associated with the NNH to file pointed to by "writer"
* @param writer - file pointer
*/
void NNH::write_basis(FILE* writer)const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::write_basis(FILE*):\n";
	//==== write the header ====
	fprintf(writer,"ann\n");
	//==== write the global cutoff ====
	fprintf(writer,"cut %f\n",rc_);
	//==== write the central species ====
	fprintf(writer,"%s %f %f %f\n",atom_.name().c_str(),atom_.mass(),atom_.energy(),atom_.charge());
	//==== write the number of species ====
	fprintf(writer,"nspecies %i\n",nspecies_);
	//==== write all species ====
	for(int i=0; i<nspecies_; ++i){
		fprintf(writer,"%s %f %f %f\n",species_[i].name().c_str(),species_[i].mass(),species_[i].energy(),species_[i].charge());
	}
	//==== write the radial basis ====
	for(int j=0; j<nspecies_; ++j){
		fprintf(writer,"basis_radial %s\n",species_[j].name().c_str());
		BasisR::write(writer,basisR_[j]);
	}
	//==== write the angular basis ====
	for(int j=0; j<nspecies_; ++j){
		for(int k=j; k<nspecies_; ++k){
			fprintf(writer,"basis_angular %s %s\n",species_[j].name().c_str(),species_[k].name().c_str());
			BasisA::write(writer,basisA_(j,k));
		}
	}
}

/**
* read the basis associated with the NNH from file with name "filename"
* @param filename - the file which the NNH will be read from
*/
void NNH::read_basis(const std::string& filename){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::read_basis(const std::string&):\n";
	FILE* reader=NULL;
	reader=fopen(filename.c_str(),"r");
	if(reader!=NULL){
		read_basis(reader);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("NNH::read_basis(const std::string&): Could not open nnpot file: \"")+filename+std::string("\""));
}

/**
* read the basis associated with the NNH from file pointed to by "reader"
* @param reader - file pointer
*/
void NNH::read_basis(FILE* reader){
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNH::read_basis(FILE*):\n";
	//==== local function variables ====
	std::vector<std::string> strlist;
	char* input=new char[string::M];
	try{
	//==== reader in header ====
	fgets(input,string::M,reader);
	//==== read in global cutoff ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const double rc=std::atof(std::strtok(NULL,string::WS));
	if(rc<=0) throw std::invalid_argument("NNH::read(FILE*): invalid cutoff.");
	else rc_=rc;
	//==== read the central atom ====
	Atom::read(fgets(input,string::M,reader),atom_);
	//==== read the number of species ====
	std::strtok(fgets(input,string::M,reader),string::WS);
	const int nspecies=std::atoi(std::strtok(NULL,string::WS));
	//==== read all species ====
	std::vector<Atom> species(nspecies);
	for(int i=0; i<nspecies; ++i){
		Atom::read(fgets(input,string::M,reader),species[i]);
		if(species[i].id()==atom_.id() && species[i]!=atom_) throw std::invalid_argument("Central atom and neighbor atom do not match.");
	}
	//==== resize the hamiltonian ====
	resize(species);
	//==== read the radial basis ====
	for(int i=0; i<nspecies_; ++i){
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int jj=index(strlist[1]);
		BasisR::read(reader,basisR_[jj]);
	}
	//==== read the angular basis ====
	for(int i=0; i<nspecies_; ++i){
		for(int j=i; j<nspecies_; ++j){
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const int jj=index(strlist[1]);
			const int kk=index(strlist[2]);
			BasisA::read(reader,basisA_(jj,kk));
		}
	}
	//==== set the number of inputs and offsets ====
	init_input();
	} catch(std::exception& e){
		std::cout<<"ERROR in NNH::read_basis(FILE*):\n";
		std::cout<<e.what()<<"\n";
	}
	//==== free local variables ====
	delete[] input;
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
	//input/output
		head_="ann_";
		tail_="";
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
	Eigen::Vector3d rI_,rJ_,rK_;
	Eigen::Vector3d rIJ_,rIK_,rJK_;
	if(struc.R().norm()>math::constant::ZERO){
		//lattice vector shifts - factor of two: max distance = 1/2 lattice vector
		const int shellx=std::floor(2.0*rc_/struc.R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
		const int shelly=std::floor(2.0*rc_/struc.R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
		const int shellz=std::floor(2.0*rc_/struc.R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
		const int Rmax=(2*shellx+1)*(2*shelly+1)*(2*shellz+1);
		if(NN_POT_PRINT_DATA>0) std::cout<<"Rmax = "<<Rmax<<"\n";
		if(NN_POT_PRINT_DATA>0) std::cout<<"shell = ("<<shellx<<","<<shelly<<","<<shellz<<") = "<<(2*shellx+1)*(2*shelly+1)*(2*shellz+1)<<"\n";
		if(Rmax>1){
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
			if(NN_POT_PRINT_DATA>0) std::cout<<"Rsize = "<<Rsize<<"\n";
			//loop over all atoms
			if(NN_POT_PRINT_STATUS>0) std::cout<<"computing symmetry functions\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				//find the index of the species of atom i
				const int II=map_[string::hash(struc.name(i))];
				rI_=struc.posn(i);
				//reset the inputs
				if(NN_POT_PRINT_STATUS>2) std::cout<<"resetting inputs\n";
				struc.symm(i).setZero();
				//loop over all pairs
				for(int j=0; j<struc.nAtoms(); ++j){
					//find the index of the species of atom j
					const int JJ=nnh_[II].index(struc.name(j));
					//loop over lattice vector shifts - atom j
					for(int iJ=0; iJ<Rsize; ++iJ){
						rJ_.noalias()=struc.posn(j)+R_[iJ];
						//calc rIJ
						rIJ_.noalias()=rI_-rJ_;
						const double dIJ=rIJ_.norm();
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
								const int KK=nnh_[II].index(struc.name(k));
								//loop over all cell shifts  - atom k
								for(int iK=0; iK<Rsize; ++iK){
									rK_.noalias()=struc.posn(k)+R_[iK];
									//calc rIK
									rIK_.noalias()=rI_-rK_;
									const double dIK=rIK_.norm();
									if(math::constant::ZERO<dIK && dIK<rc_){
										//calc rJK
										rJK_.noalias()=rJ_-rK_;
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
			}
		} else {
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
				const int II=map_[string::hash(struc.name(i).c_str())];
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
						const int JJ=map_[string::hash(struc.name(nj).c_str())];
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
									const int KK=map_[string::hash(struc.name(nk).c_str())];
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
												struc.symm(i)[offsetA_+na]+=basisAijk_.symm()[na];
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
			const int II=map_[string::hash(struc.name(i))];
			//reset the inputs
			if(NN_POT_PRINT_STATUS>2) std::cout<<"resetting inputs\n";
			struc.symm(i).setZero();
			//loop over all pairs
			for(int j=0; j<struc.nAtoms(); ++j){
				//find the index of the species of atom j
				const int JJ=nnh_[II].index(struc.name(j));
				//calc rIJ
				rIJ_.noalias()=struc.posn(i)-struc.posn(j);
				const double dIJ=rIJ_.norm();
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
						const int KK=nnh_[II].index(struc.name(k));
						//calc rIK
						rIK_.noalias()=struc.posn(i)-struc.posn(k);
						const double dIK=rIK_.norm();
						if(math::constant::ZERO<dIK && dIK<rc_){
							//calc rJK
							rJK_.noalias()=struc.posn(j)-struc.posn(k);
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
	Eigen::Vector3d rI_,rJ_,rK_;
	Eigen::Vector3d rIJ_,rIK_,rJK_;
	//set the inputs for the atoms
	if(calc_symm_) calc_symm(struc);
	//reset the force
	for(int i=0; i<struc.nAtoms(); ++i) struc.force(i).setZero();
	if(struc.R().norm()>math::constant::ZERO){
		//lattice vector shifts
		const int shellx=std::floor(2.0*rc_/struc.R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
		const int shelly=std::floor(2.0*rc_/struc.R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
		const int shellz=std::floor(2.0*rc_/struc.R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
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
			rI_=struc.posn(i);
			//execute the appropriate network
			nnh_[II].nn().execute(struc.symm(i));
			//calculate the network gradient
			nnh_[II].nn().grad_out();
			//set the gradient (n.b. dodi - do/di - deriv. of out w.r.t. in)
			dEdG=nnh_[II].nn().dodi().row(0);
			//loop over pairs
			for(int j=0; j<struc.nAtoms(); ++j){
				//find the index of the species of atom j
				const int JJ=map_[string::hash(struc.name(j))];
				//loop over lattice vector shifts - atom j
				for(int iJ=0; iJ<Rsize; ++iJ){
					//compute rIJ
					rJ_=struc.posn(j)+R_[iJ];
					rIJ_.noalias()=rI_-rJ_;
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
								rK_=struc.posn(k)+R_[iK];
								rIK_.noalias()=rI_-rK_;
								const double dIK=rIK_.norm();
								if(math::constant::ZERO<dIK && dIK<rc_){
									const double dIKi=1.0/dIK;
									//compute rJK
									rJK_.noalias()=rJ_-rK_;
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
			const int II=map_[string::hash(struc.name(i))];
			//execute the appropriate network
			nnh_[II].nn().execute(struc.symm(i));
			//calculate the network gradient
			nnh_[II].nn().grad_out();
			//set the gradient (n.b. dodi - do/di - deriv. of out w.r.t. in)
			dEdG=nnh_[II].nn().dodi().row(0);
			//loop over pairs
			for(int j=0; j<struc.nAtoms(); ++j){
				//find the index of the species of atom j
				const int JJ=map_[string::hash(struc.name(j))];
				//compute rIJ
				rIJ_.noalias()=struc.posn(i)-struc.posn(j);
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
						//compute rIK
						rIK_.noalias()=struc.posn(i)-struc.posn(k);
						const double dIK=rIK_.norm();
						if(math::constant::ZERO<dIK && dIK<rc_){
							const double dIKi=1.0/dIK;
							//compute rJK
							rJK_.noalias()=struc.posn(j)-struc.posn(k);
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

/**
* write all neural network potentials to file
*/
void NNPot::write()const{
	if(NN_POT_PRINT_FUNC>0) std::cout<<"NNPot::write():\n";
	//==== write all networks ====
	for(int n=0; n<nspecies_; ++n){
		//create the file name
		const std::string filename=head_+nnh_[n].atom().name()+tail_;
		//write the network
		nnh_[n].write(filename);
	}
}

/**
* read all neural network potentials from file
*/
void NNPot::read(){
	if(NN_POT_PRINT_FUNC) std::cout<<"NNPot::read():\n";
	//==== read all networks ====
	for(int n=0; n<nspecies_; ++n){
		//create the file name
		const std::string filename=head_+nnh_[n].atom().name()+tail_;
		if(NN_POT_PRINT_DATA>0) std::cout<<"filename = "<<filename<<"\n";
		//read the network
		nnh_[n].read(filename);
	}
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
	size+=nbytes(obj.rc());
	//species
	size+=nbytes(obj.nspecies());//nspecies_
	for(int i=0; i<obj.nspecies(); ++i){
		size+=nbytes(obj.species(i));//species_
	}
	size+=nbytes(obj.map());
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
	pos+=pack(obj.rc(),arr+pos);
	//species
	pos+=pack(obj.nspecies(),arr+pos);
	for(int i=0; i<obj.nspecies(); ++i){
		pos+=pack(obj.species(i),arr+pos);
	}
	pos+=pack(obj.map(),arr+pos);
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
	pos+=unpack(obj.rc(),arr+pos);
	//species
	int nspecies=0;
	pos+=unpack(nspecies,arr+pos);
	std::vector<Atom> species(nspecies);
	for(int i=0; i<nspecies; ++i){
		pos+=unpack(species[i],arr+pos);
	}
	obj.resize(species);
	pos+=unpack(obj.map(),arr+pos);
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
