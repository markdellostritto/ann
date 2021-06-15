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
//c++ libraries
#include <iostream>
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
//ann - cutoff
#include "cutoff.h"
// ann - chemical info
#include "ptable.h"
// ann - symmetry functions
#include "symm_radial_g1.h"
#include "symm_radial_g2.h"
#include "symm_angular_g3.h"
#include "symm_angular_g4.h"
// ann - serialization
#include "serialize.h"
// ann - serialization
#include "string_ann.h"

using namespace LAMMPS_NS;
using namespace MathConst;

//==========================================================================

PairNN::PairNN(LAMMPS *lmp):Pair(lmp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::PairNN(LAMMPS):\n";
	writedata=1;//write coefficients to data file
	//set defaults
	rc_=0;
	nspecies_=0;
}

//==========================================================================

PairNN::~PairNN(){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::~PairNN():\n";
	if(allocated){
		//global pair data
		memory->destroy(setflag);
		memory->destroy(cutsq);
		//==== neural network hamiltonians ====
		nspecies_=0;
		map_type_nnp_.clear();
		nnh_.clear();
		dOutDVal_.clear();
		//==== symmetry functions ====
		symm_.clear();
		dEdG_.clear();
	}
}

//==========================================================================

void PairNN::compute(int eflag, int vflag){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::compute(int,int):\n";
	//======== local variables ========
	//atom properties - global
	double** x = atom->x;//positions
	double** f = atom->f;//forces
	const int* type = atom->type;//types
	const int nlocal = atom->nlocal;//number of local atoms
	const bool newton_pair=force->newton_pair;//must be true (see init_style)
	double etotal_=0;
	//neighbors - ith atom
	const int inum = list->inum;// # of ith atom's neighbors
	const int* index_list = list->ilist;// local indices of I atoms
	const int* numneigh = list->numneigh;// # of J neighbors for each I atom
	int** firstneigh = list->firstneigh;// ptr to 1st J int value of each I atom
	
	if (eflag || vflag) ev_setup(eflag,vflag);
	else evflag = vflag_fdotr = 0;
	
	//======== compute symmetry functions ========
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"computing symmetry functions\n";
	for(int ii=0; ii<inum; ++ii){
		//==== get the index of type of i ====
		const int i=index_list[ii];//get the index
		const int II=map_type_nnp_[type[i]-1];//get the index in the NNP
		if(II<0) continue; //skip if current type is not included in the NNP
		//==== get the nearest neighbors of i (full list) ====
		const int* nn_list=firstneigh[i];//get the list of nearest neighbors
		const int num_nn=numneigh[i];//get the number of neighbors
		if(PAIR_NN_PRINT_DATA>2) std::cout<<"atomi "<<type[i]<<" "<<i<<"\n";
		//==== compute the symmetry function ====
		//reset the symmetry function
		symm_[II].setZero();
		//loop over all pairs
		for(int jj=0; jj<num_nn; ++jj){
			const int j=nn_list[jj]&NEIGHMASK;//get the index, clear two highest bits
			const int JJ=map_type_nnp_[type[j]-1];//get the index in the NNP
			if(JJ<0) continue; //skip if current type is not included in the NNP
			if(PAIR_NN_PRINT_DATA>3) std::cout<<"\tatomj "<<type[j]<<" "<<j<<"\n";
			//compute rIJ
			rIJ<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
			const double dIJ=rIJ.norm();//compute norm
			if(dIJ<rc_){
				//compute the IJ contribution to all radial basis functions
				const int offsetR_=nnh_[II].offsetR(JJ);//input vector offset
				BasisR& basisRij_=nnh_[II].basisR(JJ);//radial basis alias
				basisRij_.symm(dIJ);//compute the symmetry functions
				for(int nr=0; nr<basisRij_.nfR(); ++nr){
					symm_[II][offsetR_+nr]+=basisRij_.symm()[nr];//add to total input (in proper location on array)
				}
				//loop over all triples
				for(int kk=jj+1; kk<num_nn; ++kk){
					const int k=nn_list[kk]&NEIGHMASK;//get the index, clear two highest bits
					const int KK=map_type_nnp_[type[k]-1];//get the index in the NNP
					if(KK<0) continue; //skip if current type is not included in the NNP
					//skip if the same
					if(k!=j){
						if(PAIR_NN_PRINT_DATA>4) std::cout<<"\t\tatomk "<<type[k]<<" "<<k<<"\n";
						//compute dIK
						rIK<<x[i][0]-x[k][0],x[i][1]-x[k][1],x[i][2]-x[k][2];
						const double dIK=rIK.norm();//compute norm
						if(dIK<rc_){
							//compute dJK
							rJK<<x[j][0]-x[k][0],x[j][1]-x[k][1],x[j][2]-x[k][2];
							const double dJK=rJK.norm();//compute norm
							//compute the IJ,IK,JK contribution to all angular basis functions
							if(PAIR_NN_PRINT_DATA>5) std::cout<<"\t\t\ti j k "<<index_list[ii]<<" "<<index_list[jj]<<" "<<index_list[kk]<<"\n";
							const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);//input vector offset
							BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);//angular basis alias
							const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);//cosine of ijk interior angle - i at vertex
							const double d[3]={dIJ,dIK,dJK};//utility vector (reduces number of function arguments)
							basisAijk_.symm(cosIJK,d);//compute the symmetry functions
							for(int na=0; na<basisAijk_.nfA(); ++na){
								symm_[II][offsetA_+na]+=basisAijk_.symm()[na];//add to total input (in proper location on array)
							}
						}
					}
				}
			}
		}
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"symm["<<i<<"] = "<<symm_[II].transpose()<<"\n";
		
		//==== collect input statistics ====
		#ifdef PAIR_NN_PRINT_INPUT
		const Eigen::VectorXd dx=(symm_[II]-avg_[II]);
		avg_[II].noalias()+=dx/(update->ntimestep+1);
		m2_[II].noalias()+=dx.cwiseProduct(symm_[II]-avg_[II]);
		var_[II].noalias()=m2_[II]/(update->ntimestep+1);
		#endif
		
		//======== compute the force ========
		//==== execute the network ====
		nnh_[II].nn().execute(symm_[II]);
		//==== accumulate the energy ====
		const double eatom_=nnh_[II].nn().out()[0]+nnh_[II].atom().energy();//local energy + intrinsic energy
		//if(eflag_global) eng_vdwl+=eatom_;
		//ev_tally(0,0,atom->nlocal,1,interface.getEnergy(),0.0,0.0,0.0,0.0,0.0);
		etotal_+=eatom_;
		if(eflag_atom) eatom[i]+=eatom_;
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"energy-atom["<<i<<"] "<<eatom_<<"\n";
		//==== compute the network gradients ====
		dOutDVal_[II].grad(nnh_[II].nn());
		//==== set the gradient ====
		//dodi() - do/di - derivative of output w.r.t. input
		//row(0) - derivative of zeroth output node (note: only one output node by definition)
		dEdG_[II]=dOutDVal_[II].dodi().row(0);//dEdG_ - dE/dG - gradient of energy w.r.t. nn inputs (G)
		//==== compute the forces ====
		for(int jj=0; jj<num_nn; ++jj){
			//==== get the index of type of j ====
			const int j=nn_list[jj]&NEIGHMASK;//get the index, clear two highest bits
			const int JJ=map_type_nnp_[type[j]-1];//get the index in the NNP
			if(JJ<0) continue; //skip if current type is not included in the NNP
			if(PAIR_NN_PRINT_DATA>2) std::cout<<"\tatom "<<type[j]<<" "<<j<<"\n";
			//==== compute rIJ ====
			rIJ<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
			const double dIJ=rIJ.norm();//compute norm
			if(dIJ<rc_){
				const double dIJi=1.0/dIJ;//compute inverse
				//==== compute the IJ contribution to the pair force ====
				const double fpair=nnh_[II].basisR(JJ).force(
					dIJ,dEdG_[II].data()+nnh_[II].offsetR(JJ)
				)*dIJi;
				f[i][0]+=fpair*rIJ[0]; f[i][1]+=fpair*rIJ[1]; f[i][2]+=fpair*rIJ[2];
				f[j][0]-=fpair*rIJ[0]; f[j][1]-=fpair*rIJ[1]; f[j][2]-=fpair*rIJ[2];
				//==== loop over all triplets ====
				for(int kk=jj+1; kk<num_nn; ++kk){
					//==== get the index and type of k ====
					const int k=nn_list[kk]&NEIGHMASK;//get the index, clear two highest bits
					const int KK=map_type_nnp_[type[k]-1];//get the index in the NNP
					if(KK<0) continue; //skip if current type is not included in the NNP
					if(PAIR_NN_PRINT_DATA>3) std::cout<<"\t\tatom "<<type[k]<<" "<<k<<"\n";
					//==== skip if the same ====
					if(k!=j){
						//==== compute dIK ====
						rIK<<x[i][0]-x[k][0],x[i][1]-x[k][1],x[i][2]-x[k][2];//compute diff
						const double dIK=rIK.norm();//compute norm
						if(dIK<rc_){
							const double dIKi=1.0/dIK;//compute inverse
							//==== compute dJK ====
							rJK<<x[j][0]-x[k][0],x[j][1]-x[k][1],x[j][2]-x[k][2];//compute diff
							const double dJK=rJK.norm();//compute norm
							const double dJKi=1.0/dJK;//compute inverse
							//==== compute forces acting on atom i ONLY, reverse forces added to atoms j,k ====
							const double cosIJK=rIJ.dot(rIK)*dIJi*dIKi;//cosine of (i,j,k) angle (i at vertex)
							const double d[3]={dIJ,dIK,dJK};//utility array to reduce number of function arguments
							double phi=0; double eta[3]={0,0,0};//force constants
							nnh_[II].basisA(JJ,KK).force(
								phi,eta,cosIJK,d,dEdG_[II].data()+nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK)
							);
							Eigen::Vector3d ffk,ffj,ffi=Eigen::Vector3d::Zero();
							ffi.noalias()+=(phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJ*dIJi;//compute force on i due to j
							ffi.noalias()+=(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIK*dIKi;//compute force on i due to k
							ffj.noalias()=-(-phi*cosIJK*dIJi+eta[0])*rIJ*dIJi-phi*dIJi*rIK*dIKi;//compute force on j
							ffk.noalias()=-(-phi*cosIJK*dIKi+eta[1])*rIK*dIKi-phi*dIKi*rIJ*dIJi;//compute force on k
							f[i][0]+=ffi[0]; f[i][1]+=ffi[1]; f[i][2]+=ffi[2];//store force on i
							f[j][0]+=ffj[0]; f[j][1]+=ffj[1]; f[j][2]+=ffj[2];//store force on j
							f[k][0]+=ffk[0]; f[k][1]+=ffk[1]; f[k][2]+=ffk[2];//store force on k
						}
					}
				}
			}
		}
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"force["<<i<<"] "<<f[i][0]<<" "<<f[i][1]<<" "<<f[i][2]<<"\n";
	}
	
	//Pair::ev_tally(int i,int j,int nlocal,int newton_pair,double evdwl,double ecoul,double fpair,double delx,double dely,double delz)
	if(eflag_global) ev_tally(nlocal,nlocal,nlocal,newton_pair,etotal_,0.0,0.0,0.0,0.0,0.0);
	
	//======== compute virial ========
	if(vflag_fdotr) virial_fdotr_compute();
	
	//======== print input statistics ========
	#ifdef PAIR_NN_PRINT_INPUT
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"printing input statistics\n";
	if(update->ntimestep%(output->thermo_every)==0){
		std::vector<Eigen::VectorXd> avgT(nspecies_);
		std::vector<Eigen::VectorXd> varT(nspecies_);
		for(int i=0; i<nspecies_; ++i){
			avgT[i]=Eigen::VectorXd::Zero(nnh_[i].nn().nIn());
			varT[i]=Eigen::VectorXd::Zero(nnh_[i].nn().nIn());
			for(int j=0; j<nnh_[i].nn().nIn(); ++j){
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
}

//----------------------------------------------------------------------
// allocate all arrays
//----------------------------------------------------------------------

void PairNN::allocate(){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::allocate():\n";
	//==== set variables ====
	allocated=1;//flag as allocated
	const int ntypes=atom->ntypes;//the number of types in the simulation
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"ntypes = "<<ntypes<<"\n";
	//==== global pair data ====
	memory->create(cutsq,ntypes+1,ntypes+1,"pair:cutsq");
	memory->create(setflag,ntypes+1,ntypes+1,"pair:setflag");
	//==== neural network hamiltonians ====
	map_type_nnp_.resize(ntypes);
	
	for(int i=1; i<=ntypes; ++i){
		for(int j=1; j<ntypes; ++j){
			setflag[i][j]=0;
		}
	}
}

//----------------------------------------------------------------------
// global settings
//----------------------------------------------------------------------

void PairNN::settings(int narg, char **arg){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::settings(int,char**):\n";
	//==== local variables ====
	const int me=comm->me;
	//==== check arguments ====
	if(narg!=1) error->all(FLERR,"Illegal pair_style command");//cutoff
	//==== set the global cutoff ====
	rc_=force->numeric(FLERR,arg[0]);
}

//----------------------------------------------------------------------
// set coeffs for one or more type pairs
//----------------------------------------------------------------------

void PairNN::coeff(int narg, char **arg){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::coeff(int,char**):\n";
	//pair_coeff * * nn_pot X Y Z
	//==== local variables ====
	const int me = comm->me;
	const int ntypes = atom->ntypes;
	//==== read pair coeffs ====
	//check nargs
	if(narg!=3+atom->ntypes) error->all(FLERR,"Incorrect args for pair coefficients");
	if(!allocated) allocate();
	//ensure I,J args are both *
	if(strcmp(arg[0],"*")!=0 || strcmp(arg[1],"*")!=0) error->all(FLERR,"Incorrect args for pair coefficients");
	if(me==0){
		//==== read the potential ====
		read_pot(arg[2]);
		//==== read atom names/ids ====
		std::vector<std::string> names(ntypes);//names provided in the input file
		std::vector<int> ids(ntypes);//unique hash id's of the atom names
		for(int i=0; i<ntypes; ++i){
			names[i]=std::string(arg[i+3]);
			ids[i]=string::hash(names[i]);
		}
		const int idNULL=string::hash("NULL");
		//==== check atom names and build the map ====
		map_type_nnp_.resize(ntypes);
		for(int i=0; i<ntypes; ++i){
			map_type_nnp_[i]=-1;
			for(int j=0; j<nspecies_; ++j){
				if(ids[i]==nnh_[j].atom().id()){
					map_type_nnp_[i]=j; break;
				}
			}
			if(ids[i]!=idNULL && map_type_nnp_[i]<0) error->all(FLERR,"Could not find atom name in NNP");
		}
		if(PAIR_NN_PRINT_DATA>-1){
			std::cout<<"*************** SPECIES MAP ***************\n";
			std::cout<<"type nnp name\n";
			for(int i=0; i<map_type_nnp_.size(); ++i){
				if(map_type_nnp_[i]>=0){
					std::cout<<i+1<<" "<<map_type_nnp_[i]<<" "<<nnh_[map_type_nnp_[i]].atom().name()<<"\n";
				} else {
					std::cout<<i+1<<" "<<map_type_nnp_[i]<<" NULL\n";
				}
			}
			std::cout<<"*************** SPECIES MAP ***************\n";
		}
	}
	//all flags are set since "coeff" only set once
	for (int i=1; i<=ntypes; ++i){
		for (int j=1; j<=ntypes; ++j){
			setflag[i][j]=1;
		}
	}
}

//----------------------------------------------------------------------
// init specific to this pair style
//----------------------------------------------------------------------

void PairNN::init_style(){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::init_style():\n";
	//==== local variables ====
	const int ntypes=atom->ntypes;
	const int me=comm->me;
	//==== flags ====
	if(atom->tag_enable==0) error->all(FLERR,"Pair style NN requires atom IDs");
	if(force->newton_pair==0) error->all(FLERR,"Pair style NN requires newton pair on");
	/*
		Note: calculating forces is relatively expensive for ann's.  Thus, it is best to have newton_pair
		turned on.  As newton_pair on/off requires completely different algorithms and code, and as
		we have chosen to have newton_pair on, we enforce that newton_pair is on.
	*/
	//==== need a full neighbor list ====
	const int irequest=neighbor->request(this,instance_me);
	neighbor->requests[irequest]->half=0;//disable half-neighbor list
	neighbor->requests[irequest]->full=1;//enable full-neighbor list
	//==== broadcast data ====
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"b_casting data\n";
	//==== broadcast species/types ====
	if(me==0) std::cout<<"bcasting species/types\n";
	MPI_Bcast(&nspecies_,1,MPI_INT,0,world);
	MPI_Bcast(map_type_nnp_.data(),ntypes,MPI_INT,0,world);
	MPI_Barrier(world);
	//==== resize/broadcast nnh ====
	if(me==0) std::cout<<"bcasting nnh\n";
	MPI_Barrier(world);
	nnh_.resize(nspecies_);
	for(int n=0; n<nspecies_; ++n){
		int nBytes=0;
		if(me==0) nBytes=serialize::nbytes(nnh_[n]);
		MPI_Bcast(&nBytes,1,MPI_INT,0,world);
		char* arr=new char[nBytes];
		if(me==0) serialize::pack(nnh_[n],arr);
		MPI_Bcast(arr,nBytes,MPI_CHAR,0,world);
		if(me!=0) serialize::unpack(nnh_[n],arr);
		delete[] arr;
		MPI_Barrier(world);
	}
	//==== resize utility arrays ====
	if(me==0) std::cout<<"resizing utilities\n";
	symm_.resize(nspecies_);
	for(int n=0; n<nspecies_; ++n){
		symm_[n]=Eigen::VectorXd::Zero(nnh_[n].nn().nIn());
	}
	dEdG_.resize(nspecies_);
	for(int n=0; n<nspecies_; ++n){
		dEdG_[n]=Eigen::VectorXd::Zero(nnh_[n].nn().nIn());
	}
	dOutDVal_.resize(nspecies_);
	for(int n=0; n<nspecies_; ++n){
		dOutDVal_[n].resize(nnh_[n].nn());
	}
	//==== input statistics ====
	#ifdef PAIR_NN_PRINT_INPUT
	avg_=std::vector<Eigen::VectorXd>(nspecies_);
	var_=std::vector<Eigen::VectorXd>(nspecies_);
	m2_=std::vector<Eigen::VectorXd>(nspecies_);
	for(int i=0; i<nspecies_; ++i){
		avg_[i]=Eigen::VectorXd::Zero(nnh_[i].nn().nIn());
		var_[i]=Eigen::VectorXd::Zero(nnh_[i].nn().nIn());
		m2_[i]=Eigen::VectorXd::Zero(nnh_[i].nn().nIn());
	}
	#endif
	//==== mpi barrier ====
	MPI_Barrier(world);
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
	const int ntypes=atom->ntypes;
	//==== write flags ====
	for(int i=1; i<=ntypes;++i){
		for(int j=1; j<=ntypes; ++j){
			fwrite(&setflag[i][j],sizeof(int),1,fp);
		}
	}
	//==== write species ====
	fwrite(&nspecies_,sizeof(int),1,fp);
	fwrite(map_type_nnp_.data(),sizeof(int),ntypes,fp);
	//==== loop over all types ====
	for(int n=0; n<nspecies_; ++n){
		const int size=serialize::nbytes(nnh_[n]);
		char* arr=new char[size];
		serialize::pack(nnh_[n],arr);
		//write size (bytes)
		fwrite(&size,sizeof(int),1,fp);
		//write object
		fwrite(arr,sizeof(char),size,fp);
		//free memory
		delete[] arr;
	}
}

//----------------------------------------------------------------------
// proc 0 reads from restart file, bcasts
//----------------------------------------------------------------------

void PairNN::read_restart(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_restart(FILE*):\n";
	read_restart_settings(fp);
	allocate();
	const int ntypes=atom->ntypes;
	const int me = comm->me;
	map_type_nnp_.resize(ntypes);
	//======== proc 0 reads from restart file ========
	if(me==0){
		for(int i=1; i<=ntypes;++i){
			for(int j=1; j<=ntypes; ++j){
				fread(&setflag[i][j],sizeof(int),1,fp);
			}
		}
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"ntypes = "<<ntypes<<"\n";
		//==== species ====
		fread(&nspecies_,sizeof(int),1,fp);
		fread(map_type_nnp_.data(),sizeof(int),ntypes,fp);
		//==== loop over all species ====
		nnh_.resize(nspecies_);
		for(int n=0; n<nspecies_; ++n){
			//read size
			int size=0;
			fread(&size,sizeof(int),1,fp);
			//read data
			char* arr=new char[size];
			fread(arr,sizeof(char),size,fp);
			//unpack object
			serialize::unpack(nnh_[n],arr);
			//free memory
			delete[] arr;
		}
	}
	for(int i=1; i<=ntypes; ++i){
		for(int j=1; j<=ntypes; ++j){
			MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
		}
	}
}

//----------------------------------------------------------------------
// proc 0 writes to restart file
//----------------------------------------------------------------------

void PairNN::write_restart_settings(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_restart_settings(FILE*):\n";
	const int me=comm->me;
	//==== write cutoff ====
	if(me==0) fwrite(&rc_,sizeof(double),1,fp);
}

//----------------------------------------------------------------------
// proc 0 reads from restart file, bcasts
//----------------------------------------------------------------------

void PairNN::read_restart_settings(FILE *fp){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_restart_settings(FILE*):\n";
	const int me=comm->me;
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

//read neural network potential file
void PairNN::read_pot(const char* file){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_pot(const char*):\n";
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"file_pot = "<<file<<"\n";
	//==== local function variables ====
	char* input=new char[string::M];
	FILE* reader=NULL;//file pointer for reading
	std::vector<std::string> strlist;
	//==== open the potential file ====
	reader=fopen(file,"r");
	if(reader==NULL) error->all(FLERR,"PairNN::read_pot(const char*): Could not open neural network potential file.");
	//==== header ====
	fgets(input,string::M,reader);
	//==== number of species ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	nspecies_=std::atoi(strlist.at(1).c_str());
	if(nspecies_<=0) error->all(FLERR,"PairNN::read_pot(const char*): invalid number of species.");
	//==== species ====
	std::vector<AtomANN> species(nspecies_);
	Map<int,int> map_;
	nnh_.resize(nspecies_);
	for(int n=0; n<nspecies_; ++n){
		AtomANN::read(fgets(input,string::M,reader),species[n]);
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"species["<<n<<"] = "<<species[n]<<"\n";
		nnh_[n].resize(nspecies_);
		nnh_[n].atom()=species[n];
		map_.add(string::hash(nnh_[n].atom().name()),n);
	}
	//==== global cutoff ====
	string::split(fgets(input,string::M,reader),string::WS,strlist);
	const double rc=std::atof(strlist.at(1).c_str());
	if(rc!=rc_) error->all(FLERR,"PairNN::read_pot(const char*): invalid cutoff.");
	//==== basis ====
	for(int i=0; i<nspecies_; ++i){
		//read central species
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int II=map_[string::hash(strlist.at(1))];
		//read basis - radial
		for(int j=0; j<nspecies_; ++j){
			//read species
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			const int JJ=map_[string::hash(strlist.at(1))];
			//read basis
			BasisR::read(reader,nnh_[II].basisR(JJ));
			if(PAIR_NN_PRINT_DATA>1) std::cout<<"BasisR("<<nnh_[II].atom().name()<<","<<nnh_[JJ].atom().name()<<") = "<<nnh_[II].basisR(JJ)<<"\n";
		}
		//read basis - angular
		for(int j=0; j<nspecies_; ++j){
			for(int k=j; k<nspecies_; ++k){
				//read species
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				const int JJ=map_[string::hash(strlist.at(1))];
				const int KK=map_[string::hash(strlist.at(2))];
				//read basis
				BasisA::read(reader,nnh_[II].basisA(JJ,KK));
				if(PAIR_NN_PRINT_DATA>1) std::cout<<"BasisA("<<nnh_[II].atom().name()<<","<<nnh_[JJ].atom().name()<<","<<nnh_[KK].atom().name()<<") = "<<nnh_[II].basisA(JJ,KK)<<"\n";
			}
		}
		//initialize the inputs
		nnh_[II].init_input();
	}
	//==== neural network ====
	for(int n=0; n<nspecies_; ++n){
		//read species
		string::split(fgets(input,string::M,reader),string::WS,strlist);
		const int II=map_[string::hash(strlist.at(1))];
		//read network
		NeuralNet::ANN::read(reader,nnh_[II].nn());
	}
	//==== free local variables ====
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"freeing local variables\n";
	delete[] input;
	if(reader!=NULL) fclose(reader);
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
