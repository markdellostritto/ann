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
		nnh_.clear();
	}
}

//==========================================================================

void PairNN::compute(int eflag, int vflag){
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::compute(int,int):\n";
	//======== local variables ========
	//atom properties - global
	double** __restrict__ x = atom->x;//positions
	double** __restrict__ f = atom->f;//forces
	const int* __restrict__ type = atom->type;//types
	const int nlocal = atom->nlocal;//number of local atoms
	const bool newton_pair=force->newton_pair;//must be true (see init_style)
	//neighbors - ith atom
	const int inum = list->inum;// # of ith atom's neighbors
	const int* __restrict__ index_list = list->ilist;// local indices of I atoms
	const int* __restrict__ numneigh = list->numneigh;// # of J neighbors for each I atom
	int** __restrict__ firstneigh = list->firstneigh;// ptr to 1st J int value of each I atom
	
	if (eflag || vflag) ev_setup(eflag,vflag);
	else evflag = vflag_fdotr = 0;
	
	//======== compute forces ========
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"computing forces\n";
	for(int ii=0; ii<inum; ++ii){
		//==== get the index of type of i ====
		const int i=index_list[ii];//get the index
		const int itype=type[i];//get the type
		//==== get the nearest neighbors of i (full list) ====
		const int* __restrict__ nn_list=firstneigh[i];//get the list of nearest neighbors
		const int num_nn=numneigh[i];//get the number of neighbors
		if(PAIR_NN_PRINT_DATA>1) std::cout<<"atomi "<<itype<<" "<<i<<"\n";
		//==== compute the symmetry function ====
		//reset the symmetry function
		symm_[itype-1].setZero();
		//loop over all pairs
		for(int jj=0; jj<num_nn; ++jj){
			const int j=nn_list[jj]&NEIGHMASK;//get the index, clear two highest bits
			const int jtype=type[j];//get the type
			if(PAIR_NN_PRINT_DATA>2) std::cout<<"\tatomj "<<jtype<<" "<<j<<"\n";
			//compute rIJ
			rIJ<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
			const double dIJ=rIJ.norm();//compute norm
			if(dIJ<rc_){
				//compute the IJ contribution to all radial basis functions
				const int offsetR_=nnh_[itype-1].offsetR(jtype-1);//input vector offset
				BasisR& basisRij_=nnh_[itype-1].basisR(jtype-1);//radial basis alias
				basisRij_.symm(dIJ);//compute the symmetry functions
				for(int nr=0; nr<basisRij_.nfR(); ++nr){
					symm_[itype-1][offsetR_+nr]+=basisRij_.symm()[nr];//add to total input (in proper location on array)
				}
				//loop over all triples
				for(int kk=0; kk<num_nn; ++kk){
					const int k=nn_list[kk]&NEIGHMASK;//get the index, clear two highest bits
					const int ktype=type[k];//get the type
					//skip if the same
					if(k!=j){
						if(PAIR_NN_PRINT_DATA>3) std::cout<<"\t\tatomk "<<ktype<<" "<<k<<"\n";
						//compute dIK
						rIK<<x[i][0]-x[k][0],x[i][1]-x[k][1],x[i][2]-x[k][2];
						const double dIK=rIK.norm();//compute norm
						if(dIK<rc_){
							//compute dJK
							rJK<<x[j][0]-x[k][0],x[j][1]-x[k][1],x[j][2]-x[k][2];
							const double dJK=rJK.norm();//compute norm
							//compute the IJ,IK,JK contribution to all angular basis functions
							if(PAIR_NN_PRINT_DATA>4) std::cout<<"\t\t\ti j k "<<index_list[ii]<<" "<<index_list[jj]<<" "<<index_list[kk]<<"\n";
							const int offsetA_=nnh_[itype-1].nInputR()+nnh_[itype-1].offsetA(jtype-1,ktype-1);//input vector offset
							BasisA& basisAijk_=nnh_[itype-1].basisA(jtype-1,ktype-1);//angular basis alias
							const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);//cosine of ijk interior angle - i at vertex
							const double d[3]={dIJ,dIK,dJK};//utility vector (reduces number of function arguments)
							basisAijk_.symm(cosIJK,d);//compute the symmetry functions
							for(int na=0; na<basisAijk_.nfA(); ++na){
								symm_[itype-1][offsetA_+na]+=basisAijk_.symm()[na];//add to total input (in proper location on array)
							}
						}
					}
				}
			}
		}
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"symm["<<i<<"] = "<<symm_[itype-1].transpose()<<"\n";
		
		//==== collect input statistics ====
		#ifdef PAIR_NN_PRINT_INPUT
		const Eigen::VectorXd dx=(symm_[itype-1]-avg_[itype-1]);
		avg_[itype-1].noalias()+=dx/(update->ntimestep+1);
		m2_[itype-1].noalias()+=dx.cwiseProduct(symm_[itype-1]-avg_[itype-1]);
		var_[itype-1].noalias()=m2_[itype-1]/(update->ntimestep+1);
		#endif
		
		//======== compute the force ========
		//==== execute the network ====
		nnh_[itype-1].nn().execute(symm_[itype-1]);
		//==== accumulate the energy ====
		const double eatom_=nnh_[itype-1].nn().out()[0]+nnh_[itype-1].atom().energy();//local energy + intrinsic energy
		if(eflag_global) eng_vdwl+=eatom_;
		if(eflag_atom) eatom[i]+=eatom_;
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"energy-atom["<<i<<"] "<<eatom_<<"\n";
		//==== compute the network gradients ====
		nnh_[itype-1].nn().grad_out();
		//==== set the gradient ====
		//dodi() - do/di - derivative of output w.r.t. input
		//row(0) - derivative of zeroth output node (note: only one output node by definition)
		dEdG_[itype-1]=nnh_[itype-1].nn().dodi().row(0);//dEdG_ - dE/dG - gradient of energy w.r.t. nn inputs (G)
		//==== compute the forces ====
		for(int jj=0; jj<num_nn; ++jj){
			//==== get the index of type of j ====
			const int j=nn_list[jj]&NEIGHMASK;//get the index, clear two highest bits
			const int jtype=type[j];//get the type
			if(PAIR_NN_PRINT_DATA>1) std::cout<<"\tatom "<<jtype<<" "<<j<<"\n";
			//==== compute rIJ ====
			rIJ<<x[i][0]-x[j][0],x[i][1]-x[j][1],x[i][2]-x[j][2];
			const double dIJ=rIJ.norm();//compute norm
			if(dIJ<rc_){
				const double dIJi=1.0/dIJ;//compute inverse
				//==== compute the IJ contribution to the pair force ====
				const double fpair=nnh_[itype-1].basisR(jtype-1).force(
					dIJ,dEdG_[itype-1].data()+nnh_[itype-1].offsetR(jtype-1)
				)*dIJi;
				f[i][0]+=fpair*rIJ[0]; f[i][1]+=fpair*rIJ[1]; f[i][2]+=fpair*rIJ[2];
				f[j][0]-=fpair*rIJ[0]; f[j][1]-=fpair*rIJ[1]; f[j][2]-=fpair*rIJ[2];
				//==== loop over all triplets ====
				for(int kk=0; kk<num_nn; ++kk){
					//==== get the index and type of k ====
					const int k=nn_list[kk]&NEIGHMASK;//get the index, clear two highest bits
					const int ktype=type[k];//get the type
					if(PAIR_NN_PRINT_DATA>2) std::cout<<"\t\tatom "<<ktype<<" "<<k<<"\n";
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
							nnh_[itype-1].basisA(jtype-1,ktype-1).force(
								phi,eta,cosIJK,d,dEdG_[itype-1].data()+nnh_[itype-1].nInputR()+nnh_[itype-1].offsetA(jtype-1,ktype-1)
							);
							Eigen::Vector3d ffk,ffj,ffi=Eigen::Vector3d::Zero();
							ffi.noalias()+=(phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJ*dIJi;//compute force on i due to j
							ffi.noalias()+=(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIK*dIKi;//compute force on i due to k
							ffj.noalias()=-(-phi*cosIJK*dIJi+eta[0])*rIJ*dIJi-phi*dIJi*rIK*dIKi-eta[2]*rJK*dJKi;//compute force on j
							ffk.noalias()=-(-phi*cosIJK*dIKi+eta[1])*rIK*dIKi-phi*dIKi*rIJ*dIJi+eta[2]*rJK*dJKi;//compute force on k
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
	
	//======== compute virial ========
	if(vflag_fdotr) virial_fdotr_compute();
	
	//======== print input statistics ========
	#ifdef PAIR_NN_PRINT_INPUT
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"printing input statistics\n";
	if(update->ntimestep%(output->thermo_every)==0){
		std::vector<Eigen::VectorXd> avgT(atom->ntypes);
		std::vector<Eigen::VectorXd> varT(atom->ntypes);
		for(int i=0; i<atom->ntypes; ++i){
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
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::allocate():\n";
	//==== set variables ====
	allocated=1;
	const int ntypes=atom->ntypes;
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
	nnh_.resize(ntypes);
	//==== inputs/offsets ====
	symm_.resize(ntypes);
	dEdG_.resize(ntypes);
}

//----------------------------------------------------------------------
// global settings
//----------------------------------------------------------------------

void PairNN::settings(int narg, char **arg){
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::settings(int,char**):\n";
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
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::coeff(int,char**):\n";
	//==== local variables ====
	const int me = comm->me;
	//==== pair_coeff atom_type file ====
	if(!(narg==2 || narg==3)) error->all(FLERR,"Incorrect args for pair coefficients");
	if(!allocated) allocate();
	//==== read in the atom type ====
	const int atom_type=force->numeric(FLERR,arg[0]);
	if(PAIR_NN_PRINT_DATA>0) std::cout<<"atom_type = "<<atom_type<<"\n";
	//==== read potential parameters from file ====
	if(narg==2){
		//non-hybrid - read for single atom type
		if(me==0) read_pot(atom_type,arg[1]);
	} else if(narg==3){
		//hybrid or reading from data file
		if(me==0) read_pot(atom_type,arg[2]);
	}
}

//----------------------------------------------------------------------
// init specific to this pair style
//----------------------------------------------------------------------

void PairNN::init_style(){
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::init_style():\n";
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
	MPI_Barrier(world);
	for(int n=0; n<ntypes; ++n){
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
	//==== set the number of inputs and offsets ====
	for(int n=0; n<ntypes; ++n){
		nnh_[n].init_input();
	}
	//==== resize utility arrays ====
	symm_.resize(ntypes);
	for(int n=0; n<ntypes; ++n){
		symm_[n]=Eigen::VectorXd::Zero(nnh_[n].nn().nIn());
	}
	dEdG_.resize(ntypes);
	for(int n=0; n<ntypes; ++n){
		dEdG_[n]=Eigen::VectorXd::Zero(nnh_[n].nn().nIn());
	}
	//==== input statistics ====
	#ifdef PAIR_NN_PRINT_INPUT
	avg_=std::vector<Eigen::VectorXd>(ntypes);
	var_=std::vector<Eigen::VectorXd>(ntypes);
	m2_=std::vector<Eigen::VectorXd>(ntypes);
	for(int i=0; i<ntypes; ++i){
		avg_[i]=Eigen::VectorXd::Zero(nnh_[i].nn().nIn());
		var_[i]=Eigen::VectorXd::Zero(nnh_[i].nn().nIn());
		m2_[i]=Eigen::VectorXd::Zero(nnh_[i].nn().nIn());
	}
	#endif
	//==== mpi barrier ====
	if(me==0 && PAIR_NN_PRINT_DATA>1) std::cout<<"barrier\n";
	MPI_Barrier(world);
}

//----------------------------------------------------------------------
// init for one type pair i,j and corresponding j,i
//----------------------------------------------------------------------

double PairNN::init_one(int i, int j){
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::init_one(int,int):\n";
	return rc_;
}

//----------------------------------------------------------------------
// proc 0 writes to restart file
//----------------------------------------------------------------------

void PairNN::write_restart(FILE *fp){
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::write_restart(FILE*):\n";
	write_restart_settings(fp);
	const int ntypes=atom->ntypes;
	//==== loop over all types ====
	for(int n=0; n<ntypes; ++n){
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
	if(comm->me==0 && PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_restart(FILE*):\n";
	read_restart_settings(fp);
	allocate();
	const int ntypes=atom->ntypes;
	const int me = comm->me;
	//======== proc 0 reads from restart file ========
	if(me==0){
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"ntypes = "<<ntypes<<"\n";
		//==== loop over all types ====
		for(int n=0; n<ntypes; ++n){
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
	//======== broadcast data ========
	if(me==0 && PAIR_NN_PRINT_STATUS>0) std::cout<<"b-casting data\n";
	//==== atom data ====
	for(int n=0; n<ntypes; ++n){
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
	//==== mpi barrier ====
	MPI_Barrier(world);
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

//read all neural network potentials from file
void PairNN::read_pot(int type, const char* file){
	if(PAIR_NN_PRINT_FUNC>0) std::cout<<"PairNN::read_pot(int,const char*):\n";
	if(PAIR_NN_PRINT_DATA>0){
		std::cout<<"type = "<<type<<"\n";
		std::cout<<"file = "<<file<<"\n";
	}
	//======== local function variables ========
	const double mass_tol=1e-2;//mass tolerance, not very strict
	FILE* reader=NULL;//file pointer for reading
	//======== read the potential ========
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"opening potential file\n";
	reader=fopen(file,"r");
	if(reader!=NULL){
		//==== read nn Hamiltonian ====
		nnh_[type-1].read(reader);
		//check consistency with input file
		if(nnh_[type-1].nspecies()!=atom->ntypes) error->all(FLERR,"Mismatch in ntypes-input and ntypes-potential.");
		//==== close the file ====
		fclose(reader);
		reader=NULL;
		//==== print data ====
		if(PAIR_NN_PRINT_DATA>0) std::cout<<"nnh["<<type-1<<"] = \n"<<nnh_[type-1]<<"\n";
	} else error->all(FLERR,"Could not open potential file.");
	//======== free local variables ========
	if(PAIR_NN_PRINT_STATUS>0) std::cout<<"freeing local variables\n";
}

int PairNN::name_index(const char* name){
	int index=-1;
	for(int i=0; i<nnh_.size(); ++i){
		if(nnh_[i].atom().name()==name){index=i;break;}
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
