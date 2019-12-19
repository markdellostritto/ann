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
#include "domain.h"
#include "output.h"

using namespace LAMMPS_NS;
using namespace MathConst;

//==========================================================================

PairNN::PairNN(LAMMPS *lmp):Pair(lmp){
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::PairNN(LAMMPS):\n";
	writedata=1;//write coefficients to data file
}

//==========================================================================

PairNN::~PairNN(){
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::~PairNN():\n";
	if(allocated){
		//global pair data
		memory->destroy(setflag);
		memory->destroy(cutsq);
		//local pair data
		basisR_.clear();
		basisA_.clear();
		nn_.clear();
		atoms_.clear();
		nInput_.clear();
		nInputR_.clear();
		nInputA_.clear();
		offsetR_.clear();
		offsetA_.clear();
	}
}

//==========================================================================

void PairNN::compute(int eflag, int vflag){
	if(PAIR_NN_PRINT_FUNC>0 && comm->me==0) std::cout<<"PairNN::compute(int,int):\n";
	//======== local variables ========
	//atom properties - global
	double** __restrict__ x = atom->x;
	double** __restrict__ f = atom->f;
	const int* __restrict__ type = atom->type;
	const int nlocal = atom->nlocal;
	//neighbors - I
	const int inum = list->inum;// # of I atoms neighbors are stored for
	const int* __restrict__ index_list = list->ilist;// local indices of I atoms
	const int* __restrict__ numneigh = list->numneigh;// # of J neighbors for each I atom
	int** __restrict__ firstneigh = list->firstneigh;// ptr to 1st J int value of each I atom
	
	if (eflag || vflag) ev_setup(eflag,vflag);
	else evflag = vflag_fdotr = 0;
	
	//======== compute forces ========
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"computing forces\n";
	for(unsigned int ii=0; ii<inum; ++ii){
		//==== get the index of type of i ====
		const int i=index_list[ii];//get the index
		const int itype=type[i];//get the type
		//==== get the nearest neighbors of i (full list) ====
		const int* nn_list=firstneigh[i];//get the list of nearest neighbors
		const int num_nn=numneigh[i];//get the number of neighbors
		if(PAIR_NN_PRINT_DATA>2) std::cout<<"atomi "<<itype<<" "<<i<<"\n";
		//====reset the symmetry function ====
		symm_[itype-1].setZero();
		//==== compute the symmetry function ====
		//loop over all pairs
		for(unsigned int jj=0; jj<num_nn; ++jj){
			const int j=nn_list[jj]&NEIGHMASK;//get the index, clear two highest bits
			const int jtype=type[j];//get the type
			if(PAIR_NN_PRINT_DATA>2) std::cout<<"\tatomj "<<jtype<<" "<<j<<"\n";
			//compute rIJ
			rIJ<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
			const double dIJ=rIJ.norm();//compute norm
			if(dIJ<rc_){
				//compute the IJ contribution to all radial basis functions
				unsigned int offset_=offsetR_[itype-1][jtype-1];//input vector offset
				BasisR& basisRij_=basisR_[itype-1][jtype-1];//radial basis alias
				basisRij_.symm(dIJ);//compute the symmetery functions
				for(unsigned short nr=0; nr<basisRij_.nfR(); ++nr){
					symm_[itype-1][offset_+nr]+=basisRij_.symm()[nr];//add to total input (in proper location on array)
				}
				//loop over all triples
				for(unsigned int kk=0; kk<num_nn; ++kk){
					const int k=nn_list[kk]&NEIGHMASK;//get the index, clear two highest bits
					const int ktype=type[k];//get the type
					//skip if the same
					if(k!=j){
						if(PAIR_NN_PRINT_DATA>2) std::cout<<"\t\tatomk "<<ktype<<" "<<k<<"\n";
						//compute dIK and dJK
						rIK<<x[i][0]-x[k][0],x[i][1]-x[k][1],x[i][2]-x[k][2];
						rJK<<x[j][0]-x[k][0],x[j][1]-x[k][1],x[j][2]-x[k][2];
						const double dIK=rIK.norm();//compute norm
						const double dJK=rJK.norm();//compute norm
						if(dIK<rc_ && dJK<rc_){
							//compute the IJ,IK,JK contribution to all angular basis functions
							if(PAIR_NN_PRINT_DATA>3) std::cout<<"i j k "<<index_list[ii]<<" "<<index_list[jj]<<" "<<index_list[kk]<<"\n";
							offset_=nInputR_[itype-1]+offsetA_[itype-1](jtype-1,ktype-1);//input vector offset
							BasisA& basisAijk_=basisA_[itype-1](jtype-1,ktype-1);//angular basis alias
							const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);//cosine of ijk interior angle
							const double d[3]={dIJ,dIK,dJK};//utility vector (reduces number of function arguments)
							basisAijk_.symm(cosIJK,d);//compute the symmetry functions
							for(unsigned short na=0; na<basisAijk_.nfA(); ++na){
								symm_[itype-1][offset_+na]+=basisAijk_.symm()[na];//add to total input (in proper location on array)
							}
						}
					}
				}
			}
		}
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"symm["<<i<<"] = "<<symm_[itype-1].transpose()<<"\n";
		//==== collect input statistics ====
		#ifdef PAIR_NN_PRINT_INPUT
		Eigen::VectorXd dx=(symm_[itype-1]-avg_[itype-1]);
		avg_[itype-1].noalias()+=dx/(update->ntimestep+1);
		m2_[itype-1].noalias()+=dx.cwiseProduct(symm_[itype-1]-avg_[itype-1]);
		var_[itype-1].noalias()=m2_[itype-1]/(update->ntimestep+1);
		#endif
		//==== execute the appropriate network ====
		nn_[itype-1].execute(symm_[itype-1]);
		if(eflag) eng_vdwl+=nn_[itype-1].output(0)+atoms_[itype-1].energy();
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"energy["<<i<<"] "<<nn_[itype-1].output(0)+atoms_[itype-1].energy()<<"\n";
		//==== compute the network gradients ====
		nn_[itype-1].grad_out();
		//==== set the gradient ====
		//dOut(0) - derivative of output w.r.t. zeroth layer (i.e. input)
		//row(0) - derivative of zeroth output node (note: only one output node by definition)
		dEdG_[itype-1]=nn_[itype-1].dOut(0).row(0);//zero-indexed
		//==== copmpute the force ====
		for(unsigned int jj=0; jj<num_nn; ++jj){
			//==== get the index of type of j ====
			const int j=nn_list[jj]&NEIGHMASK;//get the index, clear two highest bits
			const int jtype=type[j];//get the type
			if(PAIR_NN_PRINT_DATA>2) std::cout<<"\tatom "<<jtype<<" "<<j<<"\n";
			//==== compute rIJ ====
			rIJ<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
			const double dIJ=rIJ.norm();//compute norm
			if(dIJ<rc_){
				rIJ/=dIJ;//normalize difference vector
				//==== compute the IJ contribution to the pair force ====
				const double ftemp=basisR_[itype-1][jtype-1].force(
					dIJ,dEdG_[itype-1].data()+offsetR_[itype-1][jtype-1]
				);
				f[i][0]+=ftemp*rIJ[0]; f[i][1]+=ftemp*rIJ[1]; f[i][2]+=ftemp*rIJ[2];
				f[j][0]-=ftemp*rIJ[0]; f[j][1]-=ftemp*rIJ[1]; f[j][2]-=ftemp*rIJ[2];
				//==== loop over all triplets ====
				for(unsigned int kk=0; kk<num_nn; ++kk){
					//==== get the index of type of k ====
					const int k=nn_list[kk]&NEIGHMASK;//get the index, clear two highest bits
					const int ktype=type[k];//get the type
					if(PAIR_NN_PRINT_DATA>2) std::cout<<"\t\tatom "<<ktype<<" "<<k<<"\n";
					//==== skip if the same ====
					if(k!=j){
						//==== compute dIK and dJK ====
						rIK<<x[i][0]-x[k][0],x[i][1]-x[k][1],x[i][2]-x[k][2];
						rJK<<x[j][0]-x[k][0],x[j][1]-x[k][1],x[j][2]-x[k][2];
						const double dIK=rIK.norm();//compute norm
						const double dJK=rJK.norm();//compute norm
						if(dIK<rc_ && dJK<rc_){
							rIK/=dIK;//normalize difference vector
							//==== compute forces acting on atom i ONLY, reverse forces added to atoms j,k below ====
							const double cosIJK=rIJ.dot(rIK);//normalization is implicit
							const double d[3]={dIJ,dIK,dJK};//utility array to reduce number of function arguments
							double fij[2],fik[2];//utility arrays to reduce number of function arguments
							basisA_[itype-1](jtype-1,ktype-1).force(
								fij,fik,cosIJK,d,dEdG_[itype-1].data()+nInputR_[itype-1]+offsetA_[itype-1](jtype-1,ktype-1)
							);
							//==== add force to atoms i,j,k ====
							ffj.noalias()=fij[0]*rIJ+fik[1]*rIK;
							ffk.noalias()=fij[1]*rIJ+fik[0]*rIK;
							f[j][0]-=ffj[0]; f[j][1]-=ffj[1]; f[j][2]-=ffj[2];
							f[k][0]-=ffk[0]; f[k][1]-=ffk[1]; f[k][2]-=ffk[2];
							f[i][0]+=ffj[0]+ffk[0];
							f[i][1]+=ffj[1]+ffk[1];
							f[i][2]+=ffj[2]+ffk[2];
						}
					}
				}
			}
		}
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"force["<<i<<"] "<<f[i][0]<<" "<<f[i][1]<<" "<<f[i][2]<<"\n";
	}
	
	//======== print input statistics ========
	#ifdef PAIR_NN_PRINT_INPUT
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"printing input statistics\n";
	if(update->ntimestep%(output->thermo_every)==0){
		std::vector<Eigen::VectorXd,Eigen::aligned_allocator<Eigen::VectorXd> > avgT(atom->ntypes);
		std::vector<Eigen::VectorXd,Eigen::aligned_allocator<Eigen::VectorXd> > varT(atom->ntypes);
		for(unsigned int i=0; i<atom->ntypes; ++i){
			avgT[i]=Eigen::VectorXd::Zero(nn_[i].nInput());
			varT[i]=Eigen::VectorXd::Zero(nn_[i].nInput());
			for(unsigned int j=0; j<nn_[i].nInput(); ++j){
				MPI_Reduce(&(avg_[i][j]),&(avgT[i][j]),1,MPI_DOUBLE,MPI_SUM,0,world);
				MPI_Reduce(&(var_[i][j]),&(varT[i][j]),1,MPI_DOUBLE,MPI_SUM,0,world);
				avgT[i][j]/=comm->nprocs;
				varT[i][j]/=comm->nprocs;
			}
			if(comm->me==0) std::cout<<"avg["<<update->ntimestep<<"]["<<i<<"] = "<<avgT[i].transpose()<<"\n";
			if(comm->me==0) std::cout<<"var["<<update->ntimestep<<"]["<<i<<"] = "<<varT[i].transpose()<<"\n";
		}
	}
	#endif
	
	//compute virial
	if(vflag_fdotr) virial_fdotr_compute();
}

//----------------------------------------------------------------------
// allocate all arrays
//----------------------------------------------------------------------

void PairNN::allocate(){
	if(PAIR_NN_PRINT_FUNC>0 && comm->me==0) std::cout<<"PairNN::allocate():\n";
	//==== set variables ====
	allocated=1;
	int ntypes=atom->ntypes;
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"ntypes = "<<ntypes<<"\n";
	//==== global pair data ====
	memory->create(cutsq,ntypes+1,ntypes+1,"pair:cutsq");
	memory->create(setflag,ntypes+1,ntypes+1,"pair:setflag");
	for (int i=1; i<=ntypes; ++i){
		for (int j=1; j<=ntypes; ++j){
			setflag[i][j]=1;
		}
	}
	//==== element nn's ====
	nn_.resize(ntypes);
	atoms_.resize(ntypes);
	//==== basis ====
	basisR_.resize(ntypes,std::vector<BasisR>(ntypes));
	basisA_.resize(ntypes,LMat<BasisA>(ntypes));
	//==== inputs/offsets ====
	nInput_.resize(ntypes,0);
	nInputR_.resize(ntypes,0);
	nInputA_.resize(ntypes,0);
	offsetR_.resize(ntypes,std::vector<unsigned int>(ntypes,0));
	offsetA_.resize(ntypes,LMat<unsigned int>(ntypes,0));
	symm_.resize(ntypes,Eigen::VectorXd::Zero(1));
	dEdG_.resize(ntypes,Eigen::VectorXd::Zero(1));
}

//----------------------------------------------------------------------
// global settings
//----------------------------------------------------------------------

void PairNN::settings(int narg, char **arg){
	if(PAIR_NN_PRINT_FUNC>0 && comm->me==0) std::cout<<"PairNN::settings(int,char**):\n";
	const int me=comm->me;
	//==== local variables ====
	const int ntypes=atom->ntypes;
	unsigned int index=0;
	char* name=new char[100];//reasonable upper limit - if you're using names larger than 100 characters, that's your problem
	//==== check arguments ====
	if(narg<=2) error->all(FLERR,"Illegal pair_style command");//must have at least one name/pair combo
	if(narg%2==0) error->all(FLERR,"Illegal pair_style command");//must have name/pair combo + rc -> odd number of parameters
	//==== set the global cutoff ====
	rc_=force->numeric(FLERR,arg[0]);
	//==== read atom/index combos ====
	atoms_.resize((narg-1)/2);
	for(unsigned int i=1; i<narg; i+=2){
		std::strcpy(name,arg[i]);//read name
		index=force->numeric(FLERR,arg[i+1]);//read index
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"pair-style atom: "<<name<<" "<<index<<"\n";
		if(index>ntypes) error->all(FLERR,"Illegal atom index");
		else{
			atoms_[index-1].name()=name;
			atoms_[index-1].id()=index;
			atoms_[index-1].mass()=atom->mass[index];
		}
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"atoms_["<<index<<"] = "<<atoms_[index-1]<<"\n";
	}
	delete[] name;
}

//----------------------------------------------------------------------
// set coeffs for one or more type pairs
//----------------------------------------------------------------------

void PairNN::coeff(int narg, char **arg){
	if(PAIR_NN_PRINT_FUNC>0 && comm->me==0) std::cout<<"PairNN::coeff(int,char**):\n";
	//==== local variables ====
	int ntypes=atom->ntypes;
	const int me = comm->me;
	//==== pair_coeff atom_type file ====
	if(narg!=2) error->all(FLERR,"Incorrect args for pair coefficients");
	if(!allocated) allocate();
	//==== read in the atom type ====
	int atom_type=force->numeric(FLERR,arg[0]);
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"atom_type = "<<atom_type<<"\n";
	//==== read potential parameters from file ====
	if(me==0) read_pot(atom_type,arg[1]);
}

//----------------------------------------------------------------------
// init specific to this pair style
//----------------------------------------------------------------------

void PairNN::init_style(){
	if(PAIR_NN_PRINT_FUNC>0 && comm->me==0) std::cout<<"PairNN::init_style():\n";
	//======== local variables ========
	const int ntypes=atom->ntypes;
	const int me=comm->me;
	//======== flags ========
	if(atom->tag_enable==0) error->all(FLERR,"Pair style NN requires atom IDs");
	if(force->newton_pair==0) error->all(FLERR,"Pair style NN requires newton pair on");
	/*
		Note: calculating forces is relatively expensive for ann's.  Thus, it is best to have newton_pair
		turned on.  As newton_pair on/off requires completely different algorithms and code, and as
		we have chosen to have newton_pair on, we enforce that newton_pair is on.
	*/
	// need a full neighbor list
	int irequest=neighbor->request(this,instance_me);
	neighbor->requests[irequest]->half=0;//disable half-neighbor list
	neighbor->requests[irequest]->full=1;//enable full-neighbor list
	//======== broadcast data ========
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"b_casting data\n";
	//==== atom data ====
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"b_casting atom data\n";
	MPI_Barrier(world);
	for(unsigned int n=0; n<ntypes; ++n){
		unsigned int nBytes=0;
		if(me==0) nBytes=serialize::nbytes(atoms_[n]);
		MPI_Bcast(&nBytes,1,MPI_INT,0,world);
		char* arr=new char[nBytes];
		if(me==0) serialize::pack(atoms_[n],arr);
		MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
		if(me!=0) serialize::unpack(atoms_[n],arr);
		delete[] arr;
		MPI_Barrier(world);
	}
	//==== radial functions ====
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"b_casting radial functions\n";
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=0; i<ntypes; ++i){
			//b-cast number of bytes
			unsigned int nBytes=0;
			if(me==0) nBytes=serialize::nbytes(basisR_[n][i]);
			MPI_Bcast(&nBytes,1,MPI_INT,0,world);
			//b-cast basis
			char* arr=new char[nBytes];
			if(me==0) serialize::pack(basisR_[n][i],arr);
			MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
			if(me!=0) serialize::unpack(basisR_[n][i],arr);
			//free memory
			delete[] arr;
			MPI_Barrier(world);
		}
	}
	//==== angular functions ====
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"b_casting angular functions\n";
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=0; i<ntypes; ++i){
			for(unsigned int j=i; j<ntypes; ++j){
				//b-cast number of bytes
				unsigned int nBytes=0;
				if(me==0) nBytes=serialize::nbytes(basisA_[n](i,j));
				MPI_Bcast(&nBytes,1,MPI_INT,0,world);
				//b-cast basis
				char* arr=new char[nBytes];
				if(me==0) serialize::pack(basisA_[n](i,j),arr);
				MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
				if(me!=0) serialize::unpack(basisA_[n](i,j),arr);
				//free memory
				delete[] arr;
				MPI_Barrier(world);
			}
		}
	}
	//==== neural networks ====
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"b_casting atomic nns\n";
	for(unsigned int n=0; n<ntypes; ++n){
		//b-cast number of bytes
		unsigned int nBytes=0;
		if(me==0) nBytes=serialize::nbytes(nn_[n]);
		MPI_Bcast(&nBytes,1,MPI_INT,0,world);
		//b-cast nn
		char* arr=new char[nBytes];
		if(me==0) serialize::pack(nn_[n],arr);
		MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
		if(me!=0) serialize::unpack(nn_[n],arr);
		//free memory
		delete[] arr;
		MPI_Barrier(world);
	}
	//======== set the number of inputs and offsets ========
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"setting number of inputs and the offsets\n";
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"setting nInputR_\n";
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=0; i<basisR_[n].size(); ++i){
			nInputR_[n]+=basisR_[n][i].nfR();
		}
		if(me==0 && PAIR_NN_PRINT_DATA>0) std::cout<<"nInputR_["<<n<<"] = "<<nInputR_[n]<<"\n";
	}
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"setting offsetR_\n";
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=1; i<basisR_[n].size(); ++i){
			offsetR_[n][i]=offsetR_[n][i-1]+basisR_[n][i-1].nfR();
		}
		if(me==0 && PAIR_NN_PRINT_DATA>0){ std::cout<<"offsetR_["<<n<<"] = "; for(unsigned int i=0; i<offsetR_[n].size(); ++i) std::cout<<offsetR_[n][i]<<" "; std::cout<<"\n";}
	}
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"setting nInputA_\n";
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=0; i<basisA_[n].n(); ++i){
			for(unsigned int j=i; j<basisA_[n].n(); ++j){
				nInputA_[n]+=basisA_[n](j,i).nfA();
			}
		}
		if(me==0 && PAIR_NN_PRINT_DATA>0) std::cout<<"nInputA_["<<n<<"] = "<<nInputA_[n]<<"\n";
	}
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"setting offsetA_\n";
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=1; i<basisA_[n].size(); ++i){
			offsetA_[n][i]=offsetA_[n][i-1]+basisA_[n][i-1].nfA();
		}
		if(me==0 && PAIR_NN_PRINT_DATA>0){ std::cout<<"offsetA_ = "; for(unsigned int i=0; i<offsetA_[n].size(); ++i) std::cout<<offsetA_[n][i]<<" "; std::cout<<"\n";}
	}
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"setting nInput_\n";
	for(unsigned int n=0; n<ntypes; ++n){
		nInput_[n]=nInputR_[n]+nInputA_[n];
	}
	nInputMax_=0;
	for(unsigned int n=0; n<ntypes; ++n){
		if(nInput_[n]>nInputMax_) nInputMax_=nInput_[n];
	}
	symm_.resize(ntypes);
	for(unsigned int n=0; n<ntypes; ++n){
		symm_[n]=Eigen::VectorXd::Zero(nInput_[n]);
	}
	dEdG_.resize(ntypes);
	for(unsigned int n=0; n<ntypes; ++n){
		dEdG_[n]=Eigen::VectorXd::Zero(nInput_[n]);
	}
	//======== input statistics ========
	#ifdef PAIR_NN_PRINT_INPUT
	avg_=std::vector<Eigen::VectorXd>(ntypes);
	var_=std::vector<Eigen::VectorXd>(ntypes);
	m2_=std::vector<Eigen::VectorXd>(ntypes);
	for(unsigned int i=0; i<ntypes; ++i){
		avg_[i]=Eigen::VectorXd::Zero(nn_[i].nInput());
		var_[i]=Eigen::VectorXd::Zero(nn_[i].nInput());
		m2_[i]=Eigen::VectorXd::Zero(nn_[i].nInput());
	}
	#endif
	//======== mpi barrier ========
	if(me==0 && PAIR_NN_PRINT_DATA>1) std::cout<<"barrier\n";
	MPI_Barrier(world);
}

//----------------------------------------------------------------------
// init for one type pair i,j and corresponding j,i
//----------------------------------------------------------------------

double PairNN::init_one(int i, int j){
	if(PAIR_NN_PRINT_FUNC>0 && comm->me==0) std::cout<<"PairNN::init_one(int,int):\n";
	return rc_;
}

//----------------------------------------------------------------------
// proc 0 writes to restart file
//----------------------------------------------------------------------

void PairNN::write_restart(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0 && comm->me==0) std::cout<<"PairNN::write_restart(FILE*):\n";
	write_restart_settings(fp);
	const int ntypes=atom->ntypes;
	//==== loop over all types ====
	for(unsigned int n=0; n<ntypes; ++n){
		//==== atom info ====
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"writing atom "<<atoms_[n]<<"\n";
		fwrite(&atoms_[n].id(),sizeof(unsigned int),1,fp);
		fwrite(&atoms_[n].mass(),sizeof(double),1,fp);
		fwrite(&atoms_[n].energy(),sizeof(double),1,fp);
		const unsigned int len=atoms_[n].name().size();
		fwrite(&len,sizeof(unsigned int),1,fp);
		fwrite(atoms_[n].name().c_str(),sizeof(char),len,fp);
		//==== radial basis ====
		for(unsigned int i=0; i<ntypes; ++i){
			if(PAIR_NN_PRINT_STATUS>0) std::cout<<"basis - radial - "<<i<<"\n";
			const unsigned int nsymm=basisR_[n][i].nfR();
			if(PAIR_NN_PRINT_DATA>0) std::cout<<"nymmr["<<n<<"]["<<i<<"] = "<<nsymm<<"\n";
			fwrite(&nsymm,sizeof(unsigned int),1,fp);//number of symmetry functions
			if(PAIR_NN_PRINT_DATA>0) std::cout<<"phiRN["<<n<<"]["<<i<<"] = "<<basisR_[n][i].phiRN()<<"\n";
			fwrite(&basisR_[n][i].phiRN(),sizeof(int),1,fp);//type of symmetry function
			fwrite(&basisR_[n][i].tcut(),sizeof(int),1,fp);//cutoff function type
			fwrite(&basisR_[n][i].rc(),sizeof(double),1,fp);//cutoff length
			if(basisR_[n][i].phiRN()==PhiRN::G1){
				for(unsigned int ii=0; ii<nsymm; ++ii){
					if(PAIR_NN_PRINT_DATA>1) std::cout<<"G1\n";
				}
			} else if(basisR_[n][i].phiRN()==PhiRN::G2){
				for(unsigned int ii=0; ii<nsymm; ++ii){
					const PhiR_G2& phirG2=static_cast<const PhiR_G2&>(basisR_[n][i].fR(ii));
					fwrite(&phirG2.eta,sizeof(double),1,fp);//exponent
					fwrite(&phirG2.rs,sizeof(double),1,fp);//position
					if(PAIR_NN_PRINT_DATA>1) std::cout<<"G2"<<phirG2.eta<<" "<<phirG2.rs<<"\n";
				}
			} else error->all(FLERR,"Invalid radial function.");
		}
		//==== angular basis ====
		for(unsigned int i=0; i<ntypes; ++i){
			for(unsigned int j=i; j<ntypes; ++j){
				if(PAIR_NN_PRINT_STATUS>0) std::cout<<"basis - angular - "<<i<<" "<<j<<"\n";
				const unsigned int nsymm=basisA_[n](i,j).nfA();
				fwrite(&nsymm,sizeof(unsigned int),1,fp);//number of symmetry functions
				if(PAIR_NN_PRINT_DATA>0) std::cout<<"nsymma = "<<nsymm<<"\n";
				fwrite(&basisA_[n](i,j).phiAN(),sizeof(int),1,fp);//type of symmetry function
				fwrite(&basisA_[n](i,j).tcut(),sizeof(int),1,fp);//transfer function type
				fwrite(&basisA_[n](i,j).rc(),sizeof(double),1,fp);//cutoff length
				if(PAIR_NN_PRINT_DATA>0) std::cout<<"phiAN["<<n<<"]["<<i<<"]["<<j<<"] = "<<basisA_[n](i,j).phiAN()<<"\n";
				if(basisA_[n](i,j).phiAN()==PhiAN::G3){
					for(unsigned int ii=0; ii<nsymm; ++ii){
						const PhiA_G3& phirG3=static_cast<const PhiA_G3&>(basisA_[n](i,j).fA(ii));
						fwrite(&phirG3.eta,sizeof(double),1,fp);//exponent
						fwrite(&phirG3.zeta,sizeof(double),1,fp);//power
						fwrite(&phirG3.lambda,sizeof(int),1,fp);//sign
						if(PAIR_NN_PRINT_DATA>1) std::cout<<"G4 "<<phirG3.eta<<" "<<phirG3.zeta<<" "<<phirG3.lambda<<"\n";
					}
				} else if(basisA_[n](i,j).phiAN()==PhiAN::G4){
					for(unsigned int ii=0; ii<nsymm; ++ii){
						const PhiA_G4& phirG4=static_cast<const PhiA_G4&>(basisA_[n](i,j).fA(ii));
						fwrite(&phirG4.eta,sizeof(double),1,fp);//exponent
						fwrite(&phirG4.zeta,sizeof(double),1,fp);//power
						fwrite(&phirG4.lambda,sizeof(int),1,fp);//sign
						if(PAIR_NN_PRINT_DATA>1) std::cout<<"G4 "<<phirG4.eta<<" "<<phirG4.zeta<<" "<<phirG4.lambda<<"\n";
					}
				} else error->all(FLERR,"Invalid angular function.");
			}
		}
		//==== neural network ====
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"writing atom nn\n";
		//write network configuration
		unsigned int nlayer=nn_[n].nlayer();
		unsigned int ninput=nn_[n].nInput();
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"nlayer["<<n<<"] = "<<nlayer<<"\n";
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"ninput["<<n<<"] = "<<ninput<<"\n";
		fwrite(&nlayer,sizeof(unsigned int),1,fp);//number of layers
		fwrite(&ninput,sizeof(unsigned int),1,fp);//number of input nodes
		for(unsigned int m=0; m<nn_[n].nlayer(); ++m){
			nlayer=nn_[n].nlayer(m);
			fwrite(&nlayer,sizeof(unsigned int),1,fp);//number of hidden/ouput nodes
			if(PAIR_NN_PRINT_DATA>0) std::cout<<"nh["<<n<<"]["<<m<<"] = "<<nlayer<<"\n";
		}
		fwrite(&nn_[n].tfType(),sizeof(int),1,fp);//transfer function
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"tf["<<n<<"] = "<<nn_[n].tfType()<<"\n";
		//write the scaling layer
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"writing scaling layer\n";
		for(unsigned int m=0; m<nn_[n].nInput(); ++m){
			fwrite(&nn_[n].preScale(m),sizeof(double),1,fp);
		}
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"writing biasing layer\n";
		for(unsigned int m=0; m<nn_[n].nInput(); ++m){
			fwrite(&nn_[n].preBias(m),sizeof(double),1,fp);
		}
		//write the biases
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"writing biases\n";
		for(unsigned int l=0; l<nn_[n].nlayer(); ++l){
			for(unsigned int i=0; i<nn_[n].nlayer(l); ++i){
				fwrite(&nn_[n].bias(l)[i],sizeof(double),1,fp);
			}
		}
		//write the edges
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"writing weights\n";
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
	if(PAIR_NN_PRINT_FUNC>0 && comm->me==0) std::cout<<"PairNN::read_restart(FILE*):\n";
	read_restart_settings(fp);
	allocate();
	const int ntypes=atom->ntypes;
	const int me = comm->me;
	//======== proc 0 reads from restart file ========
	if(me==0){
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"ntypes = "<<ntypes<<"\n";
		//==== allocate local variables ====
		char* str=new char[100];
		//==== loop over all types ====
		for(unsigned int n=0; n<ntypes; ++n){
			//==== atom info ====
			fread(&atoms_[n].id(),sizeof(unsigned int),1,fp);
			fread(&atoms_[n].mass(),sizeof(double),1,fp);
			fread(&atoms_[n].energy(),sizeof(double),1,fp);
			unsigned int len=0;
			fread(&len,sizeof(unsigned int),1,fp);
			fread(str,sizeof(char),len,fp); str[len]='\0';
			atoms_[n].name()=str;
			if(PAIR_NN_PRINT_DATA>0) std::cout<<"reading atom "<<atoms_[n]<<"\n";
			//==== radial basis ====
			for(unsigned int i=0; i<ntypes; ++i){
				if(PAIR_NN_PRINT_STATUS>0) std::cout<<"basis - radial - "<<i<<"\n";
				unsigned int nsymm=0;
				CutoffN::type tcut=CutoffN::UNKNOWN;
				double rc=0;
				fread(&nsymm,sizeof(unsigned int),1,fp);//number of symmetry functions
				fread(&basisR_[n][i].phiRN(),sizeof(int),1,fp);//type of symmetry function
				fread(&tcut,sizeof(int),1,fp);//type of cutoff function
				fread(&rc,sizeof(double),1,fp);//cutoff value
				if(PAIR_NN_PRINT_DATA>0) std::cout<<"phiRN["<<n<<"]["<<i<<"] "<<tcut<<" "<<rc<<" "<<basisR_[n][i].phiRN()<<" "<<nsymm<<"\n";
				if(basisR_[n][i].phiRN()==PhiRN::G1){
					basisR_[n][i].init_G1(nsymm,tcut,rc);
					for(unsigned int ii=0; ii<nsymm; ++ii){
						if(PAIR_NN_PRINT_DATA>1) std::cout<<"G1\n";
					}
				} else if(basisR_[n][i].phiRN()==PhiRN::G2){
					basisR_[n][i].init_G2(nsymm,tcut,rc);
					for(unsigned int ii=0; ii<nsymm; ++ii){
						PhiR_G2& phirG2=static_cast<PhiR_G2&>(basisR_[n][i].fR(ii));
						fread(&phirG2.eta,sizeof(double),1,fp);//exponent
						fread(&phirG2.rs,sizeof(double),1,fp);//position
						if(PAIR_NN_PRINT_DATA>1) std::cout<<"G2 "<<phirG2.eta<<" "<<phirG2.rs<<"\n";
					}
				} else error->all(FLERR,"Invalid radial function.");
			}
			//==== angular basis ====
			for(unsigned int i=0; i<ntypes; ++i){
				for(unsigned int j=i; j<ntypes; ++j){
					if(PAIR_NN_PRINT_DATA>0) std::cout<<"basis - angular - "<<i<<" "<<j<<"\n";
					unsigned int nsymm=0;
					CutoffN::type tcut=CutoffN::UNKNOWN;
					double rc=0;
					fread(&nsymm,sizeof(unsigned int),1,fp);//number of symmetry functions
					fread(&basisA_[n](i,j).phiAN(),sizeof(int),1,fp);//type of symmetry function
					fread(&tcut,sizeof(int),1,fp);//type of cutoff function
					fread(&rc,sizeof(double),1,fp);//cutoff value
					if(PAIR_NN_PRINT_DATA>0) std::cout<<"phiAN["<<n<<"]["<<i<<"]["<<j<<"] "<<tcut<<" "<<rc<<" "<<basisA_[n](i,j).phiAN()<<" "<<nsymm<<"\n";
					if(basisA_[n](i,j).phiAN()==PhiAN::G3){
						basisA_[n](i,j).init_G3(nsymm,tcut,rc);
						for(unsigned int ii=0; ii<nsymm; ++ii){
							PhiA_G3& phirG3=static_cast<PhiA_G3&>(basisA_[n](i,j).fA(ii));
							fread(&phirG3.eta,sizeof(double),1,fp);//exponent
							fread(&phirG3.zeta,sizeof(double),1,fp);//power
							fread(&phirG3.lambda,sizeof(int),1,fp);//sign
							if(PAIR_NN_PRINT_DATA>1) std::cout<<"G3 "<<phirG3.eta<<" "<<phirG3.zeta<<" "<<phirG3.lambda<<"\n";
						}
					} else if(basisA_[n](i,j).phiAN()==PhiAN::G4){
						basisA_[n](i,j).init_G4(nsymm,tcut,rc);
						for(unsigned int ii=0; ii<nsymm; ++ii){
							PhiA_G4& phirG4=static_cast<PhiA_G4&>(basisA_[n](i,j).fA(ii));
							fread(&phirG4.eta,sizeof(double),1,fp);//exponent
							fread(&phirG4.zeta,sizeof(double),1,fp);//power
							fread(&phirG4.lambda,sizeof(int),1,fp);//sign
							if(PAIR_NN_PRINT_DATA>1) std::cout<<"G4 "<<phirG4.eta<<" "<<phirG4.zeta<<" "<<phirG4.lambda<<"\n";
						}
					} else error->all(FLERR,"Invalid angular function.");
				}
			}
			//==== neural networks ====
			if(PAIR_NN_PRINT_STATUS>0) std::cout<<"reading atom nn\n";
			unsigned int nlayer=0,nInput=0;
			std::vector<unsigned int> nNodes;
			//read network configuration
			fread(&nlayer,sizeof(unsigned int),1,fp);//number of layers
			fread(&nInput,sizeof(unsigned int),1,fp);//number of input nodes
			if(PAIR_NN_PRINT_DATA>0) std::cout<<"nlayer["<<n<<"] = "<<nlayer<<"\n";
			if(PAIR_NN_PRINT_DATA>0) std::cout<<"ninput["<<n<<"] = "<<nInput<<"\n";
			nNodes.resize(nlayer);
			for(unsigned int m=0; m<nlayer; ++m){
				fread(&nNodes[m],sizeof(unsigned int),1,fp);//number of hidden/ouput nodes
				if(PAIR_NN_PRINT_DATA>0) std::cout<<"nh["<<n<<"]["<<m<<"] = "<<nNodes[m]<<"\n";
			}
			fread(&nn_[n].tfType(),sizeof(int),1,fp);//transfer function
			if(PAIR_NN_PRINT_DATA>0) std::cout<<"tf["<<n<<"] = "<<nn_[n].tfType()<<"\n";
			//resize the network
			if(PAIR_NN_PRINT_STATUS>0) std::cout<<"resizing atom nn\n";
			nn_[n].resize(nInput,nNodes);
			//read the scaling layer
			if(PAIR_NN_PRINT_STATUS>0) std::cout<<"reading scaling layer\n";
			for(unsigned int m=0; m<nn_[n].nInput(); ++m){
				fread(&nn_[n].preScale(m),sizeof(double),1,fp);
			}
			if(PAIR_NN_PRINT_STATUS>0) std::cout<<"reading biasing layer\n";
			for(unsigned int m=0; m<nn_[n].nInput(); ++m){
				fread(&nn_[n].preBias(m),sizeof(double),1,fp);
			}
			//read the biases
			if(PAIR_NN_PRINT_STATUS>0) std::cout<<"reading biases\n";
			for(unsigned int l=0; l<nn_[n].nlayer(); ++l){
				for(unsigned int i=0; i<nn_[n].nlayer(l); ++i){
					fread(&nn_[n].bias(l,i),sizeof(double),1,fp);
				}
			}
			//read the edges
			if(PAIR_NN_PRINT_STATUS>0) std::cout<<"reading edges\n";
			for(unsigned int l=0; l<nn_[n].nlayer(); ++l){
				for(unsigned int i=0; i<nn_[n].edge(l).cols(); ++i){
					for(unsigned int j=0; j<nn_[n].edge(l).rows(); ++j){
						fread(&nn_[n].edge(l,j,i),sizeof(double),1,fp);
					}
				}
			}
		}
		//==== free memory ====
		delete[] str;
	}
	//======== broadcast data ========
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"b-casting data\n";
	//==== atom data ====
	for(unsigned int n=0; n<ntypes; ++n){
		unsigned int nBytes=0;
		if(me==0) nBytes=serialize::nbytes(atoms_[n]);
		MPI_Bcast(&nBytes,1,MPI_INT,0,world);
		char* arr=new char[nBytes];
		if(me==0) serialize::pack(atoms_[n],arr);
		MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
		if(me!=0) serialize::unpack(atoms_[n],arr);
		delete[] arr;
		MPI_Barrier(world);
	}
	//==== radial functions ====
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=0; i<ntypes; ++i){
			unsigned int nBytes=0;
			if(me==0) nBytes=serialize::nbytes(basisR_[n][i]);
			MPI_Bcast(&nBytes,1,MPI_INT,0,world);
			char* arr=new char[nBytes];
			if(me==0) serialize::pack(basisR_[n][i],arr);
			MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
			if(me!=0) serialize::unpack(basisR_[n][i],arr);
			delete[] arr;
			MPI_Barrier(world);
		}
	}
	//==== angular functions ====
	for(unsigned int n=0; n<ntypes; ++n){
		for(unsigned int i=0; i<ntypes; ++i){
			for(unsigned int j=i; j<ntypes; ++j){
				unsigned int nBytes=0;
				if(me==0) nBytes=serialize::nbytes(basisA_[n](i,j));
				MPI_Bcast(&nBytes,1,MPI_INT,0,world);
				char* arr=new char[nBytes];
				if(me==0) serialize::pack(basisA_[n](i,j),arr);
				MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
				if(me!=0) serialize::unpack(basisA_[n](i,j),arr);
				delete[] arr;
				MPI_Barrier(world);
			}
		}
	}
	//==== neural networks ====
	for(unsigned int n=0; n<ntypes; ++n){
		unsigned int nBytes=0;
		if(me==0) nBytes=serialize::nbytes(nn_[n]);
		MPI_Bcast(&nBytes,1,MPI_INT,0,world);
		char* arr=new char[nBytes];
		if(me==0) serialize::pack(nn_[n],arr);
		MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
		if(me!=0) serialize::unpack(nn_[n],arr);
		delete[] arr;
		MPI_Barrier(world);
	}
	//==== mpi barrier ====
	MPI_Barrier(world);
}

//----------------------------------------------------------------------
// proc 0 writes to restart file
//----------------------------------------------------------------------

void PairNN::write_restart_settings(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_restart_settings(FILE*):\n";
	int me=comm->me;
	//==== write cutoff ====
	if(me==0) fwrite(&rc_,sizeof(double),1,fp);
}

//----------------------------------------------------------------------
// proc 0 reads from restart file, bcasts
//----------------------------------------------------------------------

void PairNN::read_restart_settings(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_restart_settings(FILE*):\n";
	int me=comm->me;
	//==== read cutoff ====
	if(me==0) fread(&rc_,sizeof(double),1,fp);
	//==== bcast cutoff ====
	MPI_Bcast(&rc_,1,MPI_DOUBLE,0,world);
	//==== mpi barrier ====
	MPI_Barrier(world);
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
	const int ntypes=atom->ntypes;//number of types in the simulation
	//======== string utilities ========
	char* input=new char[M];
	char* temp=new char[M];
	char* name=new char[M];
	char* name2=new char[M];
	//======== read the potential ========
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"opening potential file\n";
	reader=fopen(file,"r");
	if(reader!=NULL){
		//==== read header ====
		fgets(input,M,reader);
		//==== read global cutoff ====
		std::sscanf(fgets(input,M,reader),"%*s %lf",&rc);
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"rc "<<rc<<" rc_ "<<rc_<<"\n";
		if(rc>rc_) rc_=rc;
		//==== read central atom ====
		//read in name, mass, energy
		std::sscanf(fgets(input,M,reader),"%s %lf %lf",name,&mass,&energy);
		//check consistency with input file - mass
		if(std::fabs((mass-atoms_[type-1].mass())/atoms_[type-1].mass())>mass_tol) error->all(FLERR,"Mismatch in mass-input and mass-potential.");
		//check consistency with input file - name
		if(atoms_[type-1].name()!=name) error->all(FLERR,"Mismatch in name-input and name-potential.");
		//set atom energy
		atoms_[type-1].energy()=energy;
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"atom "<<atoms_[type-1]<<"\n";
		//==== read number of species ====
		std::sscanf(fgets(input,M,reader),"%*s %i",&nspecies);
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"nspecies "<<nspecies<<"\n";
		if(nspecies!=ntypes) error->all(FLERR,"Mismatch in ntypes-input and ntypes-potential.");
		//==== read in species - check name/mass ====
		for(unsigned int i=0; i<nspecies; ++i){
			std::sscanf(fgets(input,M,reader),"%s %lf %lf",name,&mass,&energy);
			unsigned int index=name_index(name);
			if(index<0) error->all(FLERR,"Mismatch in name-input and name-potential.");
			if(std::fabs((mass-atoms_[index].mass())/atoms_[index].mass())>mass_tol) error->all(FLERR,"Mismatch in type mass and nn mass.");
		}
		//==== read the radial basis ====
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"reading radial basis\n";
		for(unsigned int i=0; i<nspecies; ++i){
			//read in the name of the species associated with the ith radial basis function
			std::sscanf(fgets(input,M,reader),"%*s %s",name);
			if(PAIR_NN_PRINT_DATA>1) std::cout<<"\tname \""<<name<<"\"\n";
			int ii=name_index(name);
			if(ii<0) error->all(FLERR,"Could not find element in radial basis.");
			//read in the type-ii radial basis
			BasisR::read(reader,basisR_[type-1][ii]);//zero-indexed
		}
		//==== read the angular basis ====
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"reading angular basis\n";
		for(unsigned int i=0; i<nspecies; ++i){
			for(unsigned int j=i; j<nspecies; ++j){
				//read in the names
				std::sscanf(fgets(input,M,reader),"%*s %s %s",name,name2);
				int ii=name_index(name);
				int jj=name_index(name2);
				if(ii<0) error->all(FLERR,"Could not find element in angular basis.");
				if(jj<0) error->all(FLERR,"Could not find element in angular basis.");
				//read in the type-ii-jj angular basis
				BasisA::read(reader,basisA_[type-1](ii,jj));//zero-indexed
			}
		}
		//==== read the neural network ====
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"reading neural network\n";
		NN::Network::read(reader,nn_[type-1]);//zero-indexed
		if(PAIR_NN_PRINT_STATUS>0) std::cout<<"closing potential file\n";
		//==== close the file ====
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
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"freeing local variables\n";
	delete[] input;
	delete[] temp;
	delete[] name;
	delete[] name2;
}

int PairNN::name_index(const char* name){
	int index=-1;
	for(unsigned int i=0; i<atoms_.size(); ++i){
		if(atoms_[i].name()==name){index=i;break;}
	}
	return index;
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
