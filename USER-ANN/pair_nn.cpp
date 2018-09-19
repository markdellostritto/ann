/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//c libraries
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// lammps libraries
#include "pair_nn.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

//==========================================================================

PairNN::PairNN(LAMMPS *lmp):Pair(lmp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::PairNN(LAMMPS):\n";
	writedata=1;//write coefficients to data file
}

//==========================================================================

PairNN::~PairNN(){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::~PairNN():\n";
	if(allocated){
		//global pair data
		memory->destroy(setflag);
		memory->destroy(cutsq);
		//local pair data
		basisR_.clear();
		basisA_.clear();
		nn_.clear();
		energyAtom_.clear();
		nInput_.clear();
		nInputR_.clear();
		nInputA_.clear();
		offsetR_.clear();
		offsetA_.clear();
	}
}

//==========================================================================

void PairNN::compute(int eflag, int vflag){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::compute(int,int):\n";
	//vector utilities
	Eigen::Vector3d rIJ,rIK,rJK;
	Eigen::VectorXd dEdG;
	//distances
	double dIJ,dIK,dJK;
	//loop indices
	int ii,jj,kk,nr,na;
	//ids
	int i,j,k;
	//types
	int itype,jtype,ktype;
	//atom properties - global
	double **x = atom->x;
	double **f = atom->f;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	//neighbors - I
	int inum = list->inum;// # of I atoms neighbors are stored for
	int* index_list = list->ilist;// local indices of I atoms
	int* numneigh = list->numneigh;// # of J neighbors for each I atom
	int** firstneigh = list->firstneigh;// ptr to 1st J int value of each I atom
	//neighbors - J,K
	int* nn_list;// local variable for list of nearest neighbors of I atom
	int num_nn;// local variable for number of nearest neighbors of I atom
	//triple
	double cosIJK,amp;
	
	if (eflag || vflag) ev_setup(eflag,vflag);
	else evflag = vflag_fdotr = 0;
	
	std::vector<Eigen::VectorXd,Eigen::aligned_allocator<Eigen::VectorXd> > symm(nlocal);//symmetry functions
	
	//calculate symmetry functions
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Calculating symmetry functions...\n";
	for(ii=0; ii<inum; ++ii){
		//get the index of type of i
		i=index_list[ii];//get the index
		itype=type[i];//get the type
		//get the nearest neighbors of i (full list)
		nn_list=firstneigh[i];//get the list of nearest neighbors
		num_nn=numneigh[i];//get the number of neighbors
		if(PAIR_NN_PRINT_DATA>2) std::cout<<"atomi "<<itype<<" "<<i<<"\n";
		//resize the symmetry function
		symm[i]=Eigen::VectorXd::Zero(nn_[itype-1].nInput());
		//loop over all pairs
		for(jj=0; jj<num_nn; ++jj){
			j=nn_list[jj];//get the index
			j&=NEIGHMASK;//clear the two highest bits
			jtype=type[j];//get the type
			//skip if the same
			if(PAIR_NN_PRINT_DATA>2) std::cout<<"\tatomj "<<jtype<<" "<<j<<"\n";
			//compute rIJ
			rIJ<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
			//compute distance
			dIJ=rIJ.norm();
			if(dIJ<rc_){
				//compute the IJ contribution to all radial basis functions
				for(nr=0; nr<basisR_[itype-1][jtype-1].nfR(); ++nr){
					symm[i][offsetR_[itype-1][jtype-1]+nr]+=basisR_[itype-1][jtype-1].fR(nr).val(dIJ);
				}
				//loop over all triplets
				for(kk=0; kk<num_nn; ++kk){
					//skip the local pair if they're the same
					k=nn_list[kk];//get the index
					k&=NEIGHMASK;//clear the two highest bits
					ktype=type[k];//get the type
					//skip if the same
					if(k==j) continue;
					if(PAIR_NN_PRINT_DATA>2) std::cout<<"\t\tatomk "<<ktype<<" "<<k<<"\n";
					//compute dIK and dJK
					rIK<<x[i][0]-x[k][0],x[i][1]-x[k][1],x[i][2]-x[k][2];
					rJK<<x[j][0]-x[k][0],x[j][1]-x[k][1],x[j][2]-x[k][2];
					dIK=rIK.norm(); dJK=rJK.norm();
					if(dIK<rc_ && dJK<rc_){
						//compute the IJ,IK,JK contribution to all angular basis functions
						cosIJK=rIJ.dot(rIK)/(dIJ*dIK);
						if(PAIR_NN_PRINT_DATA>3) std::cout<<"i j k "<<index_list[ii]<<" "<<index_list[jj]<<" "<<index_list[kk]<<"\n";
						if(PAIR_NN_PRINT_DATA>3) std::cout<<"\t\tdIK "<<dIK<<" dJK "<<dJK<<" cosIJK "<<cosIJK<<"\n";
						for(na=0; na<basisA_[itype-1](jtype-1,ktype-1).nfA(); ++na){
							symm[i][nInputR_[itype-1]+offsetA_[itype-1](jtype-1,ktype-1)+na]+=
								basisA_[itype-1](jtype-1,ktype-1).fA(na).val(cosIJK,dIJ,dIK,dJK);
						}
					}
				}
			}
		}
	}
	if(PAIR_NN_PRINT_DATA>1){
		for(unsigned int i=0; i<nlocal; ++i) std::cout<<"symm["<<i<<"] "<<symm[i].transpose()<<"\n";
	}
	
	//loop over all local atoms
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Calculating forces...\n";
	for(ii=0; ii<inum; ++ii){
		//==== get the index of type of i ====
		i=index_list[ii];//get the index
		itype=type[i];//get the type
		//==== get the nearest neighbors of i (full list) ====
		nn_list=firstneigh[i];//get the list of nearest neighbors
		num_nn=numneigh[i];//get the number of neighbors
		if(PAIR_NN_PRINT_DATA>2) std::cout<<"atom "<<itype<<" "<<i<<"\n";
		//==== execute the appropriate network ====
		nn_[itype-1].execute(symm[i]);
		if(eflag) eng_vdwl+=nn_[itype-1].output(0)+energyAtom_[itype-1];
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"energy["<<i<<"] "<<nn_[itype-1].output(0)+energyAtom_[itype-1]<<"\n";
		//==== compute the network gradients ====
		nn_[itype-1].grad_out();
		//==== set the gradient ====
		//dOut(0) - derivative of output w.r.t. zeroth layer (i.e. input)
		//row(0) - derivative of zeroth output node (note: only one output node by definition)
		dEdG=nn_[itype-1].dOut(0).row(0);//zero-indexed
		//==== loop over all pairs ====
		for(jj=0; jj<num_nn; ++jj){
			//==== get the index of type of j ====
			j=nn_list[jj];//get the index
			j&=NEIGHMASK;//clear the two highest bits
			jtype=type[j];//get the type
			if(PAIR_NN_PRINT_DATA>2) std::cout<<"\tatom "<<jtype<<" "<<j<<"\n";
			//==== compute rIJ ====
			rIJ<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
			//==== compute distance ====
			dIJ=rIJ.norm();
			if(dIJ<rc_){
				rIJ/=dIJ;
				//==== compute the IJ contribution to all radial basis functions ====
				double ftemp=0;
				for(nr=0; nr<basisR_[itype-1][jtype-1].nfR(); ++nr){
					ftemp-=dEdG[offsetR_[itype-1][jtype-1]+nr]*basisR_[itype-1][jtype-1].fR(nr).grad(dIJ);
				}
				f[i][0]+=ftemp*rIJ[0]; f[i][1]+=ftemp*rIJ[1]; f[i][2]+=ftemp*rIJ[2];
				f[j][0]-=ftemp*rIJ[0]; f[j][1]-=ftemp*rIJ[1]; f[j][2]-=ftemp*rIJ[2];
				//==== loop over all triplets ====
				for(kk=0; kk<num_nn; ++kk){
					k=nn_list[kk];//get the index
					k&=NEIGHMASK;//clear the two highest bits
					ktype=type[k];//get the type
					if(PAIR_NN_PRINT_DATA>2) std::cout<<"\t\tatom "<<ktype<<" "<<k<<"\n";
					//==== skip if the same ====
					if(k==j) continue;
					//==== compute dIK and dJK ====
					rIK<<x[i][0]-x[k][0],x[i][1]-x[k][1],x[i][2]-x[k][2];
					rJK<<x[j][0]-x[k][0],x[j][1]-x[k][1],x[j][2]-x[k][2];
					dIK=rIK.norm(); dJK=rJK.norm();
					if(dIK<rc_ && dJK<rc_){
						rIK/=dIK;
						//==== compute forces acting on atom i ONLY, reverse forces added to atoms j,k below ====
						double fij1=0,fij2=0,fik1=0,fik2=0;
						cosIJK=rIJ.dot(rIK);
						for(na=0; na<basisA_[itype-1](jtype-1,ktype-1).nfA(); ++na){
							//==== gradient - cosine - central atom ====
							amp=-0.5*basisA_[itype-1](jtype-1,ktype-1).fA(na).grad_angle(cosIJK)
								*basisA_[itype-1](jtype-1,ktype-1).fA(na).dist(dIJ,dIK,dJK)
								*dEdG[nInputR_[itype-1]+offsetA_[itype-1][jtype-1]+na];
							fij1+=amp*(-cosIJK/dIJ);
							fij2+=amp*(1.0/dIK);
							fik1+=amp*(-cosIJK/dIK);
							fik2+=amp*(1.0/dIJ);
							//==== gradient distance - central atom ====
							amp=-0.5*basisA_[itype-1](jtype-1,ktype-1).fA(na).angle(cosIJK)
								*dEdG[nInputR_[itype-1]+offsetA_[itype-1][jtype-1]+na];
							fij1+=amp*basisA_[itype-1](jtype-1,ktype-1).fA(na).grad_dist_0(dIJ,dIK,dJK);
							fik1+=amp*basisA_[itype-1](jtype-1,ktype-1).fA(na).grad_dist_1(dIJ,dIK,dJK);
						}
						//==== add force to atoms i,j,k ====
						Eigen::Vector3d ff=(fij1+fij2)*rIJ+(fik1+fik2)*rIK;
						f[i][0]+=ff[0]; f[i][1]+=ff[1]; f[i][2]+=ff[2];
						ff=fij1*rIJ+fik2*rIK;
						f[j][0]-=ff[0]; f[j][1]-=ff[1]; f[j][2]-=ff[2];
						ff=fik1*rIK+fij2*rIJ;
						f[k][0]-=ff[0]; f[k][1]-=ff[1]; f[k][2]-=ff[2];
					}
				}
			}
		}
	}
	if(PAIR_NN_PRINT_DATA>1){
		for(unsigned int i=0; i<nlocal; ++i) std::cout<<"force["<<i<<"] "<<f[i][0]<<" "<<f[i][1]<<" "<<f[i][2]<<"\n";
	}
	
	// CHECK - not sure if I should include this
	if(vflag_fdotr) virial_fdotr_compute();
}

//----------------------------------------------------------------------
// allocate all arrays
//----------------------------------------------------------------------

void PairNN::allocate(){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::allocate():\n";
	//==== set variables ====
	allocated=1;
	int ntypes=atom->ntypes;
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"ntypes = "<<ntypes<<"\n";
	//==== global pair data ====
	memory->create(cutsq,ntypes+1,ntypes+1,"pair:cutsq");
	memory->create(setflag,ntypes+1,ntypes+1,"pair:setflag");
	for (int i=1; i<=ntypes; ++i){
		for (int j=1; j<=ntypes; ++j){
			setflag[i][j]=0;
		}
	}
	//==== element nn's ====
	nn_.resize(ntypes);
	energyAtom_.resize(ntypes,0);
	//==== basis ====
	basisR_.resize(ntypes,std::vector<BasisR>(ntypes));
	basisA_.resize(ntypes,LMat<BasisA>(ntypes));
	//==== inputs/offsets ====
	nInput_.resize(ntypes,0);
	nInputR_.resize(ntypes,0);
	nInputA_.resize(ntypes,0);
	offsetR_.resize(ntypes,std::vector<unsigned int>(ntypes,0));
	offsetA_.resize(ntypes,LMat<unsigned int>(ntypes,0));
}

//----------------------------------------------------------------------
// global settings
//----------------------------------------------------------------------

void PairNN::settings(int narg, char **arg){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::settings(int,char**):\n";
	if(narg!=1) error->all(FLERR,"Illegal pair_style command");
	//==== set the global cutoff ====
	rc_=force->numeric(FLERR,arg[0]);
}

//----------------------------------------------------------------------
// set coeffs for one or more type pairs
//----------------------------------------------------------------------

void PairNN::coeff(int narg, char **arg){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::coeff(int,char**):\n";
	//==== local variables ====
	int ntypes=atom->ntypes;
	//==== pair_coeff atom_type file ====
	if(narg!=2) error->all(FLERR,"Incorrect args for pair coefficients");
	if(!allocated) allocate();
	//==== read in the atom type ====
	int atom_type=force->numeric(FLERR,arg[0]);
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"atom_type = "<<atom_type<<"\n";
	//==== read potential parameters from file ====
	read_pot(atom_type,arg[1]);
	if(PAIR_NN_PRINT_DATA>1) std::cout<<"rc_ = "<<rc_<<"\n";
	//==== check flags (need at least radial basis) ====
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Setting flags...\n";
	for (int i=1; i<=ntypes; ++i){
		for (int j=1; j<=ntypes; ++j){
			if(basisR_[i-1][j-1].nfR()>0) setflag[i][j]=1;
			if(PAIR_NN_PRINT_DATA>0) std::cout<<"\tnfR = "<<basisR_[i-1][j-1].nfR()<<" setflag "<<setflag[i][j]<<"\n";
		}
	}
	// CHECK - is broadcasting appropriate here? 
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Broadcasting data...\n";
	//==== bcast data ====
	int me=comm->me;
	char* arr=NULL;
	unsigned int nBytes=0;
	//==== bcast cutoff ====
	MPI_Bcast(&rc_,sizeof(double),MPI_DOUBLE,0,world);//zero-indexed
	//==== bcast vacuum energy ====
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Broadcasting vacuum energy...\n";
	MPI_Bcast(&energyAtom_[atom_type-1],sizeof(double),MPI_DOUBLE,0,world);//zero-indexed
	//==== bcast radial functions ====
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Broadcasting radial functions...\n";
	for(unsigned int i=0; i<ntypes; ++i){
		nBytes=serialize::nbytes(basisR_[atom_type-1][i]);
		arr=new char[nBytes];
		if(me==0) serialize::pack(basisR_[atom_type-1][i],arr);
		MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
		if(me!=0) serialize::unpack(basisR_[atom_type-1][i],arr);
		delete[] arr;
	}
	//==== bcast angular functions ====
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Broadcasting angular functions...\n";
	for(unsigned int i=0; i<ntypes; ++i){
		for(unsigned int j=i; j<ntypes; ++j){
			nBytes=serialize::nbytes(basisA_[atom_type-1](i,j));
			arr=new char[nBytes];
			if(me==0) serialize::pack(basisA_[atom_type-1](i,j),arr);
			MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
			if(me!=0) serialize::unpack(basisA_[atom_type-1](i,j),arr);
			delete[] arr;
		}
	}
	//==== bcast neural networks ====
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Broadcasting neural networks...\n";
	nBytes=serialize::nbytes(nn_[atom_type-1]);
	arr=new char[nBytes];
	if(me==0) serialize::pack(nn_[atom_type-1],arr);
	MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
	if(me!=0) serialize::unpack(nn_[atom_type-1],arr);
	delete[] arr;
}

//----------------------------------------------------------------------
// init specific to this pair style
//----------------------------------------------------------------------

void PairNN::init_style(){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::init_style():\n";
	//flags
	if(atom->tag_enable==0) error->all(FLERR,"Pair style NN requires atom IDs");
	if(force->newton_pair==0) error->all(FLERR,"Pair style NN requires newton pair on");
	/*
		Note: calculating forces is relatively expensive for ann's.  Thus, it is best to have newton_pair
		turned on.  As newton_pair on or off require completely different algorithms and code, and as
		we have chosen to have newton_pair on, we enforce that newton_pair is on.
	*/
	// need a full neighbor list
	int irequest=neighbor->request(this,instance_me);
	//neighbor->cutneighmax=rc_;
	neighbor->requests[irequest]->half=0;//disable half-neighbor list
	neighbor->requests[irequest]->full=1;//enable full-neighbor list
}

//----------------------------------------------------------------------
// init for one type pair i,j and corresponding j,i
//----------------------------------------------------------------------

double PairNN::init_one(int i, int j){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::init_one(int,int):\n";
	return rc_;
}

//----------------------------------------------------------------------
// proc 0 writes to restart file
//----------------------------------------------------------------------

void PairNN::write_restart(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_restart(FILE*):\n";
	write_restart_settings(fp);
	int ntypes=atom->ntypes;
	//==== loop over all types ====
	for(unsigned int n=0; n<ntypes; ++n){
		//==== vaccuum energy ====
		fwrite(&energyAtom_[n],sizeof(double),1,fp);
		//==== radial basis ====
		for(unsigned int i=0; i<ntypes; ++i){
			unsigned int nsymm=basisR_[n][i].nfR();
			fwrite(&nsymm,sizeof(int),1,fp);//number of symmetry functions
			fwrite(&basisR_[n][i].phiRN(),sizeof(int),1,fp);//type of symmetry function
			if(basisR_[n][i].phiRN()==PhiRN::G1){
				for(unsigned int ii=0; ii<nsymm; ++ii){
					const PhiR_G1& phirG1=static_cast<const PhiR_G1&>(basisR_[n][i].fR(ii));
					fwrite(&phirG1.rc,sizeof(double),1,fp);//cutoff length
					fwrite(&phirG1.tcut,sizeof(int),1,fp);//transfer function type
				}
			} else if(basisR_[n][i].phiRN()==PhiRN::G2){
				for(unsigned int ii=0; ii<nsymm; ++ii){
					const PhiR_G2& phirG2=static_cast<const PhiR_G2&>(basisR_[n][i].fR(ii));
					fwrite(&phirG2.rc,sizeof(double),1,fp);//cutoff length
					fwrite(&phirG2.tcut,sizeof(int),1,fp);//transfer function type
					fwrite(&phirG2.eta,sizeof(double),1,fp);//exponent
					fwrite(&phirG2.rs,sizeof(double),1,fp);//position
				}
			} else error->all(FLERR,"Invalid radial function.");
		}
		//==== angular basis ====
		for(unsigned int i=0; i<ntypes; ++i){
			for(unsigned int j=i; j<ntypes; ++j){
				unsigned int nsymm=basisA_[n](i,j).nfA();//number of symmetry functions
				fwrite(&basisA_[n](i,j).phiAN(),sizeof(int),1,fp);//type of symmetry function
				if(basisA_[n](i,j).phiAN()==PhiAN::G3){
					for(unsigned int ii=0; ii<nsymm; ++ii){
						const PhiA_G3& phirG3=static_cast<const PhiA_G3&>(basisA_[n](i,j).fA(ii));
						fwrite(&phirG3.rc,sizeof(double),1,fp);//cutoff length
						fwrite(&phirG3.tcut,sizeof(int),1,fp);//transfer function type
						fwrite(&phirG3.eta,sizeof(double),1,fp);//exponent
						fwrite(&phirG3.zeta,sizeof(double),1,fp);//power
						fwrite(&phirG3.lambda,sizeof(int),1,fp);//sign
					}
				} else if(basisA_[n](i,j).phiAN()==PhiAN::G4){
					for(unsigned int ii=0; ii<nsymm; ++ii){
						const PhiA_G4& phirG4=static_cast<const PhiA_G4&>(basisA_[n](i,j).fA(ii));
						fwrite(&phirG4.rc,sizeof(double),1,fp);//cutoff length
						fwrite(&phirG4.tcut,sizeof(int),1,fp);//transfer function type
						fwrite(&phirG4.eta,sizeof(double),1,fp);//exponent
						fwrite(&phirG4.zeta,sizeof(double),1,fp);//power
						fwrite(&phirG4.lambda,sizeof(int),1,fp);//sign
					}
				} else error->all(FLERR,"Invalid angular function.");
			}
		}
		//==== neural network ====
		//write network configuration
		unsigned int nlayer=nn_[n].nlayer();
		unsigned int ninput=nn_[n].nInput();
		fwrite(&nlayer,sizeof(int),1,fp);//number of layers
		fwrite(&ninput,sizeof(int),1,fp);//number of input nodes
		for(unsigned int m=0; m<nn_[n].nlayer(); ++m){
			nlayer=nn_[n].nlayer(m);
			fwrite(&nlayer,sizeof(int),1,fp);//number of hidden/ouput nodes
		}
		fwrite(&nn_[n].tfType(),sizeof(int),1,fp);//transfer function
		//write the scaling layers
		for(unsigned int m=0; m<nn_[n].nInput(); ++m){
			fwrite(&nn_[n].preScale(m),sizeof(double),1,fp);
		}
		for(unsigned int m=0; m<nn_[n].nInput(); ++m){
			fwrite(&nn_[n].preBias(m),sizeof(double),1,fp);
		}
		//write the biases
		for(unsigned int l=0; l<nn_[n].nlayer(); ++l){
			for(unsigned int i=0; i<nn_[n].nlayer(l); ++i){
				fwrite(&nn_[n].bias(l)[i],sizeof(double),1,fp);
			}
		}
		//write the edges
		for(unsigned int l=0; l<nn_[n].nlayer(); ++l){
			for(unsigned int i=0; i<nn_[n].edge(l).cols(); ++i){
				for(unsigned int j=0; j<nn_[n].edge(l).rows(); ++j){
					fwrite(&nn_[n].edge(l)(j,i),sizeof(double),1,fp);
				}
			}
		}
	}
}

//----------------------------------------------------------------------
// proc 0 reads from restart file, bcasts
//----------------------------------------------------------------------

void PairNN::read_restart(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_restart(FILE*):\n";
	read_restart_settings(fp);
	allocate();
	int ntypes=atom->ntypes;
	int me = comm->me;
	//**** proc 0 reads from restart file ****
	if(me==0){
		//==== loop over all types ====
		for(unsigned int n=0; n<ntypes; ++n){
			//==== vacuum energy ====
			fread(&energyAtom_[n],sizeof(int),1,fp);
			//==== radial basis ====
			for(unsigned int i=0; i<ntypes; ++i){
				unsigned int nsymm;
				fread(&nsymm,sizeof(int),1,fp);//number of symmetry functions
				fread(&basisR_[n][i].phiRN(),sizeof(int),1,fp);//type of symmetry function
				if(basisR_[n][i].phiRN()==PhiRN::G1){
					basisR_[n][i].init_G1(CutoffN::COS,1,0.5);//values don't matter, will be overwritten
					for(unsigned int ii=0; ii<nsymm; ++ii){
						PhiR_G1& phirG1=static_cast<PhiR_G1&>(basisR_[n][i].fR(ii));
						fread(&phirG1.rc,sizeof(double),1,fp);//cutoff length
						fread(&phirG1.tcut,sizeof(int),1,fp);//transfer function type
					}
				} else if(basisR_[n][i].phiRN()==PhiRN::G2){
					basisR_[n][i].init_G2(nsymm,CutoffN::COS,1,0.5);//values don't matter, will be overwritten
					for(unsigned int ii=0; ii<nsymm; ++ii){
						PhiR_G2& phirG2=static_cast<PhiR_G2&>(basisR_[n][i].fR(ii));
						fread(&phirG2.rc,sizeof(double),1,fp);//cutoff length
						fread(&phirG2.tcut,sizeof(int),1,fp);//transfer function type
						fread(&phirG2.eta,sizeof(double),1,fp);//exponent
						fread(&phirG2.rs,sizeof(double),1,fp);//position
					}
				} else error->all(FLERR,"Invalid radial function.");
			}
			//==== angular basis ====
			for(unsigned int i=0; i<ntypes; ++i){
				for(unsigned int j=i; j<ntypes; ++j){
					unsigned int nsymm;
					fread(&nsymm,sizeof(int),1,fp);//number of symmetry functions
					fread(&basisA_[n](i,j).phiAN(),sizeof(int),1,fp);//type of symmetry function
					if(basisA_[n](i,j).phiAN()==PhiAN::G3){
						basisA_[n](i,j).init_G3(nsymm,CutoffN::COS,6);
						for(unsigned int ii=0; ii<nsymm; ++ii){
							PhiA_G3& phirG3=static_cast<PhiA_G3&>(basisA_[n](i,j).fA(ii));
							fread(&phirG3.rc,sizeof(double),1,fp);//cutoff length
							fread(&phirG3.tcut,sizeof(int),1,fp);//transfer function type
							fread(&phirG3.eta,sizeof(double),1,fp);//exponent
							fread(&phirG3.zeta,sizeof(double),1,fp);//power
							fread(&phirG3.lambda,sizeof(int),1,fp);//sign
						}
					} else if(basisA_[n](i,j).phiAN()==PhiAN::G4){
						basisA_[n](i,j).init_G4(nsymm,CutoffN::COS,6);
						for(unsigned int ii=0; ii<nsymm; ++ii){
							PhiA_G4& phirG4=static_cast<PhiA_G4&>(basisA_[n](i,j).fA(ii));
							fread(&phirG4.rc,sizeof(double),1,fp);//cutoff length
							fread(&phirG4.tcut,sizeof(int),1,fp);//transfer function type
							fread(&phirG4.eta,sizeof(double),1,fp);//exponent
							fread(&phirG4.zeta,sizeof(double),1,fp);//power
							fread(&phirG4.lambda,sizeof(int),1,fp);//sign
						}
					} else error->all(FLERR,"Invalid angular function.");
				}
			}
			//==== neural networks ====
			nn_.resize(ntypes);
			unsigned int nlayer=0,nInput=0;
			std::vector<unsigned int> nNodes;
			//read network configuration
			fread(&nlayer,sizeof(int),1,fp);//number of layers
			fread(&nInput,sizeof(int),1,fp);//number of input nodes
			nNodes.resize(nlayer);
			for(unsigned int m=0; m<nlayer; ++m){
				fread(&nNodes[m],sizeof(int),1,fp);//number of hidden/ouput nodes
			}
			fread(&nn_[n].tfType(),sizeof(int),1,fp);//transfer function
			//resize the network
			nn_[n].resize(nInput,nNodes);
			//read the scaling layers
			for(unsigned int m=0; m<nn_[n].nInput(); ++m){
				fread(&nn_[n].preScale(m),sizeof(double),1,fp);
			}
			for(unsigned int m=0; m<nn_[n].nInput(); ++m){
				fread(&nn_[n].preBias(m),sizeof(double),1,fp);
			}
			//print the biases
			for(unsigned int l=0; l<nn_[n].nlayer(); ++l){
				for(unsigned int i=0; i<nn_[n].nlayer(l); ++i){
					fread(&nn_[n].bias(l,i),sizeof(double),1,fp);
				}
			}
			//print the edges
			for(unsigned int l=0; l<nn_[n].nlayer(); ++l){
				for(unsigned int i=0; i<nn_[n].edge(l).cols(); ++i){
					for(unsigned int j=0; j<nn_[n].edge(l).rows(); ++j){
						fread(&nn_[n].edge(l,j,i),sizeof(double),1,fp);
					}
				}
			}
		}
	}
	//**** broadcast data ****
	char* arr=NULL;
	unsigned int nBytes=0;
	//==== vacuum energy ====
	for(unsigned int n=0; n<ntypes; ++n){
		MPI_Bcast(&energyAtom_[n],sizeof(double),MPI_DOUBLE,0,world);
	}
	//==== radial functions ====
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=0; i<ntypes; ++i){
			nBytes=serialize::nbytes(basisR_[n][i]);
			arr=new char[nBytes];
			if(me==0) serialize::pack(basisR_[n][i],arr);
			MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
			if(me!=0) serialize::unpack(basisR_[n][i],arr);
			delete[] arr;
		}
	}
	//==== angular functions ====
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=0; i<ntypes; ++i){
			for(unsigned int j=i; j<ntypes; ++j){
				nBytes=serialize::nbytes(basisA_[n](i,j));
				arr=new char[nBytes];
				if(me==0) serialize::pack(basisA_[n](i,j),arr);
				MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
				if(me!=0) serialize::unpack(basisA_[n](i,j),arr);
				delete[] arr;
			}
		}
	}
	//==== neural networks ====
	for(unsigned int n=0; n<ntypes; ++n){
		nBytes=serialize::nbytes(nn_[n]);
		arr=new char[nBytes];
		if(me==0) serialize::pack(nn_[n],arr);
		MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
		if(me!=0) serialize::unpack(nn_[n],arr);
		delete[] arr;
	}
}

//----------------------------------------------------------------------
// proc 0 writes to restart file
//----------------------------------------------------------------------

void PairNN::write_restart_settings(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_restart_settings(FILE*):\n";
	fwrite(&rc_,sizeof(double),1,fp);
}

//----------------------------------------------------------------------
// proc 0 reads from restart file, bcasts
//----------------------------------------------------------------------

void PairNN::read_restart_settings(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_restart_settings(FILE*):\n";
	int me=comm->me;
	if(me==0) fwrite(&rc_,sizeof(double),1,fp);
	MPI_Bcast(&rc_,1,MPI_DOUBLE,0,world);
}

//----------------------------------------------------------------------
// proc 0 writes to data file
//----------------------------------------------------------------------

void PairNN::write_data(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_data(FILE*):\n";
}

//----------------------------------------------------------------------
// proc 0 writes all pairs to data file
//----------------------------------------------------------------------

void PairNN::write_data_all(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_data_all(FILE*):\n";
}

//==========================================================================

//read all neural network potentials from file
void PairNN::read_pot(int type, const char* file){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_pot(int,const char*):\n";
	if(PAIR_NN_PRINT_DATA>0){
		std::cout<<"type = "<<type<<"\n";
		std::cout<<"file = "<<file<<"\n";
	}
	//======== local function variables ========
	const double mass_tol=1e-2;//mass tolerance, not very strict
	const char* WS=" \r\t\n";//whitespace for parsing
	const unsigned int M=500;//max line size
	FILE* reader=NULL;//file pointer for reading
	double rc=0;//cutoff specified in file, resets global cutoff if greater than global cutoff
	double mass=0,energy=0;//mass and energy of species
	unsigned int nspecies=0;//the number of species interactions specified in the file
	int ntypes=atom->ntypes;//number of types in the simulation
	//string utilities
	char* input=(char*)malloc(sizeof(char)*M);
	char* temp=(char*)malloc(sizeof(char)*M);
	char* name=(char*)malloc(sizeof(char)*M);
	//======== write the basis ========
	reader=fopen(file,"r");
	if(reader!=NULL){
		//reader in header
		fgets(input,M,reader);
		//read in global cutoff
		std::strtok(fgets(input,M,reader),WS);
		rc=std::atof(std::strtok(NULL,WS));
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"rc "<<rc<<" rc_ "<<rc_<<"\n";
		if(rc>rc_) rc_=rc;
		//read in species name
		std::strtok(fgets(input,M,reader),WS);
		std::strcpy(name,std::strtok(NULL,WS));
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"name "<<name<<"\n";
		//read in species mass
		std::strtok(fgets(input,M,reader),WS);
		mass=std::atof(std::strtok(NULL,WS));
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"mass "<<mass<<" mass-type "<<atom->mass[type]<<" err "<<std::fabs((mass-atom->mass[type])/atom->mass[type])<<"\n";
		if(std::fabs((mass-atom->mass[type])/atom->mass[type])>mass_tol) error->all(FLERR,"Mismatch in type mass and nn mass.");
		//read in species energy
		std::strtok(fgets(input,M,reader),WS);
		energy=std::atof(std::strtok(NULL,WS));
		energyAtom_[type-1]=energy;
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"energy "<<energy<<"\n";
		//read in the number of species
		std::strtok(fgets(input,M,reader),WS);
		nspecies=std::atoi(std::strtok(NULL,WS));
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"nspecies "<<nspecies<<"\n";
		if(nspecies!=ntypes) error->all(FLERR,"Mismatch in ntypes in file and in simulation.");
		//read the radial basis
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Reading in radial basis...\n";
		for(unsigned int i=0; i<nspecies; ++i){
			//read in the name of the species associated with the ith radial basis function
			std::strtok(fgets(input,M,reader),WS);
			std::strcpy(name,std::strtok(NULL,WS));
			//find the mass of the ith species
			double m=PTable::mass(PTable::an(name));
			if(PAIR_NN_PRINT_DATA>1) std::cout<<"\tm "<<m<<"\n";
			//find the index of the ith species
			int ii=-1;
			for(unsigned int nn=0; nn<atom->ntypes; ++nn){
				double err=std::fabs((mass-atom->mass[type])/atom->mass[type]);
				if(err<mass_tol){ii=nn; break;}
			}
			if(ii<0) error->all(FLERR,"Could not find element in radial basis.");
			//read in the type-ii radial basis
			BasisR::read(reader,basisR_[type-1][ii]);//zero-indexed
		}
		//read the angular basis
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Reading in angular basis...\n";
		for(unsigned int i=0; i<nspecies; ++i){
			for(unsigned int j=i; j<nspecies; ++j){
				//read in the name for the first species of the (i,j) angular basis
				std::strtok(fgets(input,M,reader),WS);
				std::strcpy(name,std::strtok(NULL,WS));
				//find the mass of the ith species
				double mii=PTable::mass(PTable::an(name));
				//find the index of the ith species
				int ii=-1;
				for(unsigned int nn=0; nn<atom->ntypes; ++nn){
					double err=std::fabs((mii-atom->mass[type])/atom->mass[type]);
					if(err<mass_tol){ii=nn; break;}
				}
				if(ii<0) error->all(FLERR,"Could not find element in angular basis.");
				//read in the name for the second species of the (i,j) angular basis
				std::strcpy(name,std::strtok(NULL,WS));
				//find the mass of the jth species
				double mjj=PTable::mass(PTable::an(name));
				//find the index of the jth species
				int jj=-1;
				for(unsigned int nn=0; nn<atom->ntypes; ++nn){
					double err=std::fabs((mjj-atom->mass[type])/atom->mass[type]);
					if(err<mass_tol){jj=nn; break;}
				}
				if(jj<0) error->all(FLERR,"Could not find element in angular basis.");
				if(PAIR_NN_PRINT_DATA>1) std::cout<<"\tmii "<<mii<<" mjj "<<mjj<<"\n";
				//read in the type-ii-jj angular basis
				BasisA::read(reader,basisA_[type-1](ii,jj));//zero-indexed
			}
		}
		//read the neural network
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Reading in neural network...\n";
		NN::Network::read(reader,nn_[type-1]);//zero-indexed
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Closing the file...\n";
		//close the file
		fclose(reader);
		reader=NULL;
		if(PAIR_NN_PRINT_DATA>0){
			for(unsigned int i=0; i<nspecies; ++i) std::cout<<"BasisR["<<type-1<<"]["<<i<<"] = "<<basisR_[type-1][i]<<"\n";
			for(unsigned int i=0; i<nspecies; ++i){
				for(unsigned int j=i; j<nspecies; ++j){
					std::cout<<"BasisA["<<type-1<<"]("<<i<<","<<j<<") = "<<basisA_[type-1](i,j)<<"\n";
				}
			}
			std::cout<<"nn["<<type-1<<"] = "<<nn_[type-1]<<"\n";
			std::cout<<"prescale["<<type-1<<"] = "<<nn_[type-1].preScale().transpose()<<"\n";
			std::cout<<"postscale["<<type-1<<"] = "<<nn_[type-1].postScale().transpose()<<"\n";
			std::cout<<"prebias["<<type-1<<"] = "<<nn_[type-1].preBias().transpose()<<"\n";
			std::cout<<"postbias["<<type-1<<"] = "<<nn_[type-1].postBias().transpose()<<"\n";
			for(unsigned int i=0; i<nn_[type-1].nlayer(); ++i) std::cout<<"bias["<<type-1<<"]("<<i<<") = "<<nn_[type-1].bias(i).transpose()<<"\n";
			for(unsigned int i=0; i<nn_[type-1].nlayer(); ++i) std::cout<<"edge["<<type-1<<"]("<<i<<") = "<<nn_[type-1].edge(i)<<"\n";
		}
	} else error->all(FLERR,"Could not open potential file.");
	//======== free local variables ========
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Freeing local variables...\n";
	free(temp);
	free(input);
	free(name);
	//======== set the number of inputs and offsets ========
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Setting number of inputs and the offsets...\n";
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Setting nInputR_...\n";
	for(unsigned int i=0; i<basisR_[type-1].size(); ++i){
		nInputR_[type-1]+=basisR_[type-1][i].nfR();
	}
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"nInputR_ = "<<nInputR_[type-1]<<"\n";
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Setting offsetR_...\n";
	for(unsigned int i=1; i<basisR_[type-1].size(); ++i){
		offsetR_[type-1][i]=offsetR_[type-1][i-1]+basisR_[type-1][i-1].nfR();
	}
	if(PAIR_NN_PRINT_DATA>0){ std::cout<<"offsetR_ = "; for(unsigned int i=0; i<offsetR_[type-1].size(); ++i) std::cout<<offsetR_[type-1][i]<<" "; std::cout<<"\n";}
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Setting nInputA_...\n";
	for(unsigned int i=0; i<basisA_[type-1].n(); ++i){
		for(unsigned int j=i; j<basisA_[type-1].n(); ++j){
			nInputA_[type-1]+=basisA_[type-1](j,i).nfA();
		}
	}
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"nInputA_ = "<<nInputA_[type-1]<<"\n";
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Setting offsetA_...\n";
	for(unsigned int i=1; i<basisA_[type-1].size(); ++i){
		offsetA_[type-1][i]=offsetA_[type-1][i-1]+basisA_[type-1][i-1].nfA();
	}
	if(PAIR_NN_PRINT_DATA>0){ std::cout<<"offsetA_ = "; for(unsigned int i=0; i<offsetA_[type-1].size(); ++i) std::cout<<offsetA_[type-1][i]<<" "; std::cout<<"\n";}
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"Setting nInput_...\n";
	nInput_[type-1]=nInputR_[type-1]+nInputA_[type-1];
}

//==========================================================================

double PairNN::single(int i, int j, int itype, int jtype, double rsq, double factor_coul, double factor_lj, double &fforce){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::single(int,int,int,int,double,double,double,double&):\n";
	return 0;//single force/energy between two atoms is not well-defined for neural network potentials
}

//==========================================================================

void *PairNN::extract(const char *str, int &dim){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::extract(const char*,int&):\n";
	dim=0;//global cutoff is the only "simple" parameter
	return NULL;
}
