//c libraries
#include <cstdlib>
#include <ctime>
// math
#include "math/const.hpp"
#include "math/func.hpp"
#include "math/eigen.hpp"
// structure
#include "struc/sim.hpp"
// format
#include "format/vasp_struc.hpp"
#include "format/vasp_sim.hpp"
#include "format/lammps_sim.hpp"
// str
#include "str/string.hpp"
#include "str/print.hpp"
#include "str/token.hpp"
// mem
#include "mem/tensor.hpp"
// chem
#include "chem/units.hpp"
// thread
#include "thread/comm.hpp"
#include "thread/mpif.hpp"
// torch
#include "torch/phonon.hpp"
//fftw3
#include <fftw3.h>
//mpi
#include <mpi.h>

using math::constant::PI;

//******************************************************************
// KPath
//******************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const KPath& kpath){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("KPATH",str)<<"\n";
	for(int i=0; i<kpath.nkpts_; ++i){
		out<<kpath.kpts_[i].transpose()<<" "<<kpath.npts_[i]<<"\n";
	}
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

void KPath::resize(int nkpts){
	if(nkpts<=0) throw std::invalid_argument("Invalid number of k-points.");
	nkpts_=nkpts;
	npts_.resize(nkpts_);
	kpts_.resize(nkpts_);
}

void KPath::init(){
	for(int i=0; i<nkpts_-1; ++i){
		const Eigen::Vector3d diff=(kpts_[i+1]-kpts_[i])/npts_[i];
		for(int j=0; j<npts_[i]; ++j){
			const Eigen::Vector3d tmp=kpts_[i]+j*diff;
			kvecs_.push_back(tmp);
		}
	}
	kvecs_.push_back(kpts_.back());
	kval_.resize(kvecs_.size());
	nkvecs_=kvecs_.size();
}

//******************************************************************
// DOS
//******************************************************************

//******************************************************************
// MAIN
//******************************************************************

thread::Comm WORLD;//all processors

int main(int argc, char* argv[]){
	//==== global variables ====
	//file i/o
		FILE* reader=NULL;
		FILE* writer=NULL;
		FILE_FORMAT::type fileFormat;
		char* paramfile=new char[string::M];
		char* input=new char[string::M];
		char* simstr=new char[string::M];
	//simulation
		int nAtoms=0;
		int nprint=-1;
		std::string file_sim_;
		Simulation sim;
		Interval interval;
		int tstot=0;
		std::vector<std::string> names;
		bool restart=false;
	//atom type
		sim.atomT().name	=true;
		sim.atomT().type	=true;
		sim.atomT().posn	=true;
		sim.atomT().force	=true;
	//units
		units::System unitsys;
	//phonon - file i/o
		std::string file_pcell;
		std::string file_dynmat="dynmat.dat";
		std::string file_disp="disp.dat";
		std::string file_map="map.in";
		std::string file_restart="phonon.restart";
		std::string file_mass="mass.dat";
		bool read_mass=false;
		bool read_map=false;
	//phonon - calculation parameters
		double temp=0;
		int nasr=0;
	//phonon - primitive cell
		int N=0;//total number of lattice points
		double sqrtNinv=0;
		Eigen::Vector3i nlat=Eigen::Vector3i::Zero();
		Structure pcell;//primitive cell
		AtomType atomT;
		atomT.type=true; atomT.posn=true; atomT.name=true;
		std::vector<double> mass;//masses of the atoms in the primitive cell
		std::vector<double> mass_prim;
		std::vector<Tensor<int> > atoms;//the atoms of the primitive cell and their repeated positions (nprim_ x (n1*n2*n3))
	//phonon - fourier transforms
		Tensor<Eigen::VectorXcd> uk;//fourier transform of deviations (N x (nprim*3))
		Tensor<Eigen::VectorXcd> Fk;//fourier transform of forces (N x (nprim*3))
		Tensor<Eigen::MatrixXcd> ukuk;//fourier transform of product of deviations (N x (nprim*3 x nprim*3))
		Tensor<Eigen::MatrixXcd> Fkuk;//fourier transform of product of force and deviation (N x (nprim*3 x nprim*3))
		Tensor<Eigen::MatrixXcd> dynmat;//dynamical matrix (N x (nprim*3 x nprim*3))
		Tensor<Eigen::VectorXcd> evalues;//dynamical matrix (N x (nprim*3))
		Tensor<Eigen::VectorXd> omega;//dynamical matrix (N x (nprim*3))
		std::vector<Eigen::Vector3d> ravg;//average position (natoms)
		std::vector<Eigen::Vector3d> rzero;//starting positions (natoms)
		fftw_plan fftpx,fftpy,fftpz;//fftw plans
		fftw_complex *inx,*iny,*inz;//fftw input arrays
		fftw_complex *outx,*outy,*outz;//fftw output arrays
	//phonon - dispersion
		std::string file_kpath;
		KPath kpath;
	//phonon - density of states
		std::string file_dos="dos.dat";
		DOS dos;
	//mpi
		int nprocs=1;
		int rank=0;
	//misc
		double fbuf;
		Eigen::Vector3d rtmp;
		clock_t start,stop;//starting/stopping time
		char* str=new char[print::len_buf];
		
	//======== initialize mpi ========
	MPI_Init(&argc,&argv);
	WORLD.mpic()=MPI_COMM_WORLD;
	MPI_Comm_size(WORLD.mpic(),&WORLD.size());
	MPI_Comm_rank(WORLD.mpic(),&WORLD.rank());
		
	try{
		
		//======== start wall clock ========
		if(WORLD.rank()==0) start=std::clock();
		
		//==== rank 0 reads from file ====
		if(WORLD.rank()==0){
			
			//==== check the arguments ====
			if(argc!=2) throw std::invalid_argument("Invalid number of command-line arguments.");
			
			//==== read parameters ====
			if(PHONON_PRINT_STATUS>0) std::cout<<"reading parameters\n";
			std::strcpy(paramfile,argv[1]);
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error("I/O Error: could not open parameter file.");
			while(fgets(input,string::M,reader)!=NULL){
				Token token;
				token.read(string::trim_right(input,string::COMMENT),string::WS);
				if(token.end()) continue;
				const std::string tag=string::to_upper(token.next());
				//simulation
				if(tag=="SIM"){
					file_sim_=token.next();
				} else if(tag=="INTERVAL"){
					Interval::read(token.next().c_str(),interval);
				} else if(tag=="FORMAT"){
					fileFormat=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
				} else if(tag=="UNITS"){
					unitsys=units::System::read(string::to_upper(token.next()).c_str());
				} else if(tag=="NAMES"){
					while(!token.end()) names.push_back(token.next());
				}
				//primitive cell
				if(tag=="PCELL"){
					file_pcell=token.next();
				} else if(tag=="MASS"){
					while(!token.end()) mass.push_back(std::atof(token.next().c_str()));
				} else if(tag=="NLAT"){
					nlat[0]=std::atoi(token.next().c_str());
					nlat[1]=std::atoi(token.next().c_str());
					nlat[2]=std::atoi(token.next().c_str());
				} 
				//dispersion
				if(tag=="KPATH"){
					file_kpath=token.next();
				} else if(tag=="NPRINT"){
					nprint=std::atoi(token.next().c_str());
				} else if(tag=="T"){
					temp=std::atof(token.next().c_str());
				} else if(tag=="NASR"){
					nasr=std::atoi(token.next().c_str());
				} 
				//density of states
				if(tag=="WMIN"){
					dos.wlmin()=std::atof(token.next().c_str());
				} else if(tag=="WMAX"){
					dos.wlmax()=std::atof(token.next().c_str());
				} else if(tag=="SIGMA"){
					dos.sigma()=std::atof(token.next().c_str());
				} else if(tag=="DW"){
					dos.dw()=std::atof(token.next().c_str());
				} 
				//files
				if(tag=="DISP"){
					file_disp=token.next();
				} else if(tag=="DOS"){
					file_dos=token.next();
				} else if(tag=="DYNMAT"){
					file_dynmat=token.next();
				} else if(tag=="MAP"){
					file_map=token.next();
					read_map=true;
				} else if(tag=="FILE_RESTART"){
					file_restart=token.next();
				} else if(tag=="FILE_MASS"){
					file_mass=token.next();
					read_mass=true;
				} else if(tag=="RESTART"){
					restart=string::boolean(token.next().c_str());
				} 
			}
			//close the file
			fclose(reader);
			reader=NULL;
			
			//==== print the parameters ====
			std::cout<<print::buf(str)<<"\n";
			std::cout<<print::title("PHONON - PARAMETERS",str)<<"\n";
			//phonon - file i/o
				std::cout<<"FILES:\n";
				std::cout<<"FILE_SIM     = \""<<file_sim_<<"\"\n";
				std::cout<<"FILE_PCELL   = \""<<file_pcell<<"\"\n";
				std::cout<<"FILE_KPATH   = \""<<file_kpath<<"\"\n";
				std::cout<<"FILE_DMAT    = \""<<file_dynmat<<"\"\n";
				std::cout<<"FILE_DISP    = \""<<file_disp<<"\"\n";
				std::cout<<"FILE_DOS     = \""<<file_dos<<"\"\n";
				std::cout<<"FILE_MAP     = \""<<file_map<<"\"\n";
				std::cout<<"FILE_RESTART = \""<<file_restart<<"\"\n";
				std::cout<<"FILE_MASS    = \""<<file_mass<<"\"\n";
			//simuation
				std::cout<<"SIMULATION:\n";
				std::cout<<"UNITS      = "<<unitsys<<"\n";
				std::cout<<"FORMAT     = "<<fileFormat<<"\n";
				std::cout<<"INTERVAL   = "<<interval<<"\n";
				std::cout<<"ATOM_T     = "<<sim.atomT()<<"\n";
				std::cout<<"N_PRINT    = "<<nprint<<"\n";
				std::cout<<"NAMES      = "; for(int i=0; i<names.size(); ++i) std::cout<<names[i]<<" "; std::cout<<"\n";
				std::cout<<"READ_MAP   = "<<read_map<<"\n";
				std::cout<<"RESTART    = "<<restart<<"\n";
			//phonon - primitive cell
				std::cout<<"PCELL:\n";
				std::cout<<"NLAT       = "<<nlat.transpose()<<"\n";
				std::cout<<"MASS       = "; for(int i=0; i<mass.size(); ++i) std::cout<<mass[i]<<" "; std::cout<<"\n";
			//phonon - dispersion
				std::cout<<"DISPERSION:\n";
				std::cout<<"T          = "<<temp<<"\n";
				std::cout<<"KB         = "<<units::consts::kb()<<"\n";
			//phonon - dos
				std::cout<<"DOS:\n";
				std::cout<<"NASR       = "<<nasr<<"\n";
				std::cout<<"WMIN       = "<<dos.wlmin()<<"\n";
				std::cout<<"WMAX       = "<<dos.wlmax()<<"\n";
				std::cout<<"DW         = "<<dos.dw()<<"\n";
				std::cout<<"SIGMA      = "<<dos.sigma()<<"\n";
			//thread
				std::cout<<"THREAD:\n";
				std::cout<<"NPROCS     = "<<WORLD.size()<<"\n";
			std::cout<<print::buf(str)<<"\n";
			
			//==== check the parameters ====
			if(interval.end==0 || interval.beg==0 || (interval.end<interval.beg && interval.end>0)) throw std::invalid_argument("Invalid timestep interval.");
			if(interval.end<0) throw std::invalid_argument("Parallel execution requires a preset ending timestep.");
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			if(fileFormat==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			if(nlat[0]<=0 || nlat[1]<=0 || nlat[2]<=0) throw std::invalid_argument("Invalid lattice.");
			if(mass.size()==0) throw std::invalid_argument("Invalid mass vector.");
			for(int i=0; i<mass.size(); ++i){
				if(mass[i]<=0) throw std::invalid_argument("Invalid mass.");
			}
			if(temp<0) throw std::invalid_argument("Invalid temperature.");
			if(nasr<0) throw std::invalid_argument("Invalid number of asr iterations.");
			N=nlat.prod();
			sqrtNinv=1.0/sqrt(1.0*N);
			
			//======== set the kpath ========
			std::cout<<"set the kpath\n";
			//open kpath file
			reader=fopen(file_kpath.c_str(),"r");
			if(reader==NULL) throw std::runtime_error("Could not open kpath file");
			//read the kpath
			kpath.resize(std::atoi(fgets(input,string::M,reader)));
			for(int i=0; i<kpath.nkpts(); ++i){
				Token token(fgets(input,string::M,reader),string::WSC);
				kpath.kpts(i)[0]=std::atof(token.next().c_str());
				kpath.kpts(i)[1]=std::atof(token.next().c_str());
				kpath.kpts(i)[2]=std::atof(token.next().c_str());
				kpath.npts(i)=std::atoi(token.next().c_str());
			}
			//init kpath
			kpath.init();
			if(PHONON_PRINT_DATA>1) std::cout<<kpath<<"\n";
			
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== broadcast the prameters ========
		if(rank==0) std::cout<<"broadcasting data\n";
		//simulation
			MPI_Bcast(&nprint,1,MPI_INT,0,WORLD.mpic());
			MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.mpic());
			thread::bcast(WORLD.mpic(),0,file_sim_);
			thread::bcast(WORLD.mpic(),0,interval);
			thread::bcast(WORLD.mpic(),0,names);
		//phonon - primitive cell
			MPI_Bcast(&N,1,MPI_INT,0,WORLD.mpic());
			thread::bcast(WORLD.mpic(),0,nlat);
			thread::bcast(WORLD.mpic(),0,file_pcell);
			thread::bcast(WORLD.mpic(),0,mass);
		//set local variables
			tstot=(interval.end-interval.beg+1)/interval.stride;
		//map
			MPI_Bcast(&read_map,1,MPI_C_BOOL,0,WORLD.mpic());
			thread::bcast(WORLD.mpic(),0,file_map);
			
		//======== initialize the unit system ========
		if(rank==0) std::cout<<"initializing the unit system\n";
		units::consts::init(unitsys);
		
		//======== split the interval ========
		if(rank==0) std::cout<<"tstot = "<<tstot<<"\n";
		if(rank==0) std::cout<<"interval - tot = "<<interval<<"\n"<<std::flush;
		MPI_Barrier(WORLD.mpic());
		Interval interval_loc=Interval::split(interval,rank,nprocs);
		for(int i=0; i<nprocs; ++i){
			if(i==rank) std::cout<<"interval "<<i<<" = "<<interval_loc<<"\n";
			MPI_Barrier(WORLD.mpic());
		}
		std::cout<<std::flush;
		MPI_Barrier(WORLD.mpic());
		
		//======== read the simulation ========
		if(rank==0) std::cout<<"reading simulation\n";
		LAMMPS::DUMP::read(file_sim_.c_str(),interval_loc,sim.atomT(),sim);
		nAtoms=sim.frame(0).nAtoms();
		if(rank==0){
			std::cout<<sim<<"\n";
			std::cout<<print::buf(str)<<"\n";
			std::cout<<print::title("SIMULATION CELL",str)<<"\n";
			std::cout<<sim.frame(0)<<"\n";
			std::cout<<print::buf(str)<<"\n";
		}
		const int nbytes=serialize::nbytes(sim);
		if(rank==0) std::cout<<"nAtoms = "<<nAtoms<<"\n";
		if(rank==0) std::cout<<"simulation occupying "<<nbytes/1000000.0<<" MB per proc\n";
		
		//======== read the pcell ========
		if(rank==0) std::cout<<"reading primitive cell\n";
		VASP::POSCAR::read(file_pcell.c_str(),atomT,pcell);
		if(rank==0){
			std::cout<<print::buf(str)<<"\n";
			std::cout<<print::title("PRIMITIVE CELL",str)<<"\n";
			std::cout<<pcell<<"\n";
			std::cout<<print::buf(str)<<"\n";
		}
		int ntypes=0;
		for(int i=0; i<pcell.nAtoms(); ++i){
			if(pcell.type(i)>ntypes) ntypes=pcell.type(i);
		}
		if(ntypes+1!=mass.size()) throw std::invalid_argument("Invalid number of masses.");
		mass_prim.resize(pcell.nAtoms());
		for(int i=0; i<pcell.nAtoms(); ++i){
			mass_prim[i]=mass[pcell.type(i)];
		}
		
		//======== compute the lattice error ========
		if(rank==0) std::cout<<"computing the lattice error\n";
		double error_lat=0;
		{
			Eigen::Matrix3d Rtmp;
			Rtmp.col(0)=pcell.R().col(0)*nlat[0];
			Rtmp.col(1)=pcell.R().col(1)*nlat[1];
			Rtmp.col(2)=pcell.R().col(2)*nlat[2];
			for(int t=0; t<sim.timesteps(); ++t){
				error_lat+=(Rtmp-sim.frame(t).R()).norm();
			}
		}
		MPI_Allreduce(&error_lat,&fbuf,1,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
		error_lat=fbuf/tstot;
		if(rank==0) std::cout<<"error_lat = "<<error_lat<<"\n";
		
		//======== compute the average position ========
		if(rank==0) std::cout<<"computing average position\n";
		ravg.resize(nAtoms,Eigen::Vector3d::Zero());
		rzero.resize(nAtoms,Eigen::Vector3d::Zero());
		//bcast the starting position
		if(rank==0) for(int n=0; n<nAtoms; ++n) rzero[n]=sim.frame(0).posn(n);
		for(int n=0; n<nAtoms; ++n) MPI_Bcast(rzero[n].data(),3,MPI_DOUBLE,0,WORLD.mpic());
		//sum the positions for the local positions in each rank
		for(int t=0; t<sim.timesteps(); ++t){
			for(int n=0; n<nAtoms; ++n){
				ravg[n].noalias()+=sim.frame(t).diff(sim.frame(t).posn(n),rzero[n],rtmp);
			}
		}
		//sum and normalize the positions to find the average
		for(int n=0; n<nAtoms; ++n){
			MPI_Allreduce(ravg[n].data(),rtmp.data(),3,MPI_DOUBLE,MPI_SUM,WORLD.mpic());
			ravg[n].noalias()=rtmp/tstot+rzero[n];
			Cell::returnToCell(ravg[n],ravg[n],sim.frame(0).R(),sim.frame(0).RInv());
		}
		//print average positions
		if(PHONON_PRINT_DATA>1 && WORLD.rank()==0){
			std::cout<<"R-Avg = \n"; for(int n=0; n<nAtoms; ++n) std::cout<<ravg[n].transpose()<<"\n";
		}
		//write avg
		if(WORLD.rank()==0){
			writer=fopen("ravg.dat","w");
			if(writer==NULL) throw std::runtime_error("Could not open average position file.");
			fprintf(writer,"N X Y Z\n");
			for(int n=0; n<nAtoms; ++n){
				fprintf(writer,"%i %f %f %f\n",n,ravg[n][0],ravg[n][1],ravg[n][2]);
			}
			fclose(writer); writer=NULL;
		}
		//write rzero
		if(WORLD.rank()==0){
			writer=fopen("rzero.dat","w");
			if(writer==NULL) throw std::runtime_error("Could not open average position file.");
			fprintf(writer,"N X Y Z\n");
			for(int n=0; n<nAtoms; ++n){
				fprintf(writer,"%i %f %f %f\n",n,rzero[n][0],rzero[n][1],rzero[n][2]);
			}
			fclose(writer); writer=NULL;
		}
		
		//======== set the map ========
		atoms.resize(pcell.nAtoms(),Tensor<int>(3,nlat));
		if(read_map==false){
			
			//==== generate the map ====
			//generate supercell
			Structure scell;
			Structure::super(pcell,scell,nlat);
			std::cout<<"scell = \n"<<scell<<"\n";
			for(int i=0; i<scell.nAtoms(); ++i){
				//std::cout<<"posn["<<i<<"] = "<<scell.posn(i).transpose()<<"\n";
			}
			std::cout<<"nAtoms = "<<nAtoms<<"\n";
			std::cout<<"scell.nAtoms() = "<<scell.nAtoms()<<"\n";
			if(nAtoms!=scell.nAtoms()) throw std::invalid_argument("Invalid number of atoms in supercell.");
			//generate supercell map and indices
			std::vector<Eigen::Vector3i> smap(scell.nAtoms());
			std::vector<int> pindex(scell.nAtoms());
			int c=0;
			for(int i=0; i<nlat[0]; ++i){
				for(int j=0; j<nlat[1]; ++j){
					for(int k=0; k<nlat[2]; ++k){
						for(int n=0; n<pcell.nAtoms(); ++n){
							Eigen::Vector3i index; index<<i,j,k;
							smap[c]=index;
							pindex[c]=n;
							c++;
						}
					}
				}
			}
			//find closest equivalent in average positions
			for(int n=0; n<nAtoms; ++n){
				int ii=0;
				double min=scell.dist(scell.posn(0),ravg[n],rtmp);
				for(int i=1; i<scell.nAtoms(); ++i){
					const double dist=scell.dist(scell.posn(i),ravg[n],rtmp);
					if(dist<min){
						ii=i;
						min=dist;
					}
				}
				atoms[pindex[ii]](smap[ii])=n;
			}
			
			/*
			//==== generate the map ====
			if(WORLD.rank()==0) std::cout<<"generating map\n";
			for(int n=0; n<nAtoms; ++n){
				const std::string name=names[sim.frame(0).type(n)-1];
				if(PHONON_PRINT_DATA>1) std::cout<<"name = "<<name<<"\n";
				//get the fractional coordinate in terms of the primitive cell
				//note: ravg guaranteed to be within the total unit cell
				//thus, vRfrac will not be negative
				if(PHONON_PRINT_DATA>1) std::cout<<"ravg  = "<<ravg[n].transpose()<<"\n";
				Eigen::Vector3d vRfrac=pcell.RInv()*ravg[n];
				if(PHONON_PRINT_DATA>1) std::cout<<"vRfrac = "<<vRfrac.transpose()<<"\n";
				//get the decimal part of the fractional coordinate
				Eigen::Vector3d vPfrac;
				vPfrac[0]=math::func::mod(vRfrac[0],0.0,1.0);
				vPfrac[1]=math::func::mod(vRfrac[1],0.0,1.0);
				vPfrac[2]=math::func::mod(vRfrac[2],0.0,1.0);
				if(PHONON_PRINT_DATA>1) std::cout<<"vPfrac = "<<vPfrac.transpose()<<"\n";
				//get the index in the primitive cell, which most closely matches vpFrac
				const Eigen::Vector3d vPcart=pcell.fracToCart(vPfrac,rtmp,pcell.R());
				int pindex=-1;
				Eigen::Vector3d drmin; drmin<<pcell.R().col(0).norm(),pcell.R().col(1).norm(),pcell.R().col(2).norm();
				for(int i=0; i<pcell.nAtoms(); ++i){
					if(name==pcell.name(i)){
						pcell.diff(vPcart,pcell.posn(i),rtmp);
						if(rtmp.norm()<drmin.norm()){drmin=rtmp;pindex=i;}
					}
				}
				if(pindex<0) throw std::invalid_argument("Could not map atom to primitive cell.\n");
				if(PHONON_PRINT_DATA>1) std::cout<<"pindex = "<<pindex<<"\n";
				Eigen::Vector3d drfrac=pcell.cartToFrac(drmin,rtmp,pcell.RInv());
				if(PHONON_PRINT_DATA>1) std::cout<<"drfrac = "<<drfrac.transpose()<<"\n";
				//subtract drmin from vRfrac to get the crystal position in the copy of the primitive cell
				vRfrac.noalias()-=drfrac;
				if(PHONON_PRINT_DATA>1) std::cout<<"vRfrac = "<<vRfrac.transpose()<<"\n";
				//subtract the fractional position in the primitive cell to get the origin of the copy of the primitive cell
				Eigen::Vector3d pfrac=pcell.cartToFrac(pcell.posn(pindex),rtmp,pcell.RInv());
				vRfrac.noalias()-=pfrac;
				if(PHONON_PRINT_DATA>1) std::cout<<"vRfrac = "<<vRfrac.transpose()<<"\n";
				//set the atom index in the correct position in the tensor storing the copies of the atoms of the primitive cell
				Eigen::VectorXi index=Eigen::VectorXi::Zero(3);
				index[0]=std::round(vRfrac[0]);
				index[1]=std::round(vRfrac[1]);
				index[2]=std::round(vRfrac[2]);
				index[0]=math::func::mod(index[0],nlat[0]);
				index[1]=math::func::mod(index[1],nlat[1]);
				index[2]=math::func::mod(index[2],nlat[2]);
				if(PHONON_PRINT_DATA>1) std::cout<<"index = "<<index.transpose()<<"\n";
				atoms[pindex](index)=n;
			}
			*/
			
			//write the map
			if(WORLD.rank()==0){
				writer=fopen("map.dat","w");
				if(writer==NULL) throw std::runtime_error("Could not open average position file.");
				fprintf(writer,"%i\n",atoms.size());
				fprintf(writer,"%i %i %i\n",nlat[0],nlat[1],nlat[2]);
				for(int n=0; n<atoms.size(); ++n){
					for(int i=0; i<nlat[0]; ++i){
						for(int j=0; j<nlat[1]; ++j){
							for(int k=0; k<nlat[2]; ++k){
								Eigen::VectorXi index(3); index<<i,j,k;
								fprintf(writer,"%i %i %i %i %i\n",i,j,k,n,atoms[n](index));
							}
						}
					}
				}
				fclose(writer);
				writer=NULL;
			}
		} else if(read_map==true){
			//read the map file
			reader=fopen(file_map.c_str(),"r");
			if(reader==NULL) throw std::runtime_error("Could not open map file.");
			//read the number of atoms
			const int nAtomsMap=std::atoi(fgets(input,string::M,reader));
			if(nAtomsMap!=pcell.nAtoms()) throw std::invalid_argument("Invalid number of atoms in the map file.");
			//read the nlat
			std::vector<std::string> strlist;
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			Eigen::Vector3i nlatMap=Eigen::Vector3i::Constant(-1.0);
			nlatMap[0]=std::atoi(strlist[0].c_str());
			nlatMap[1]=std::atoi(strlist[1].c_str());
			nlatMap[2]=std::atoi(strlist[2].c_str());
			if(nlatMap[0]!=nlat[0]) throw std::invalid_argument("Invalid nlat x");
			if(nlatMap[1]!=nlat[1]) throw std::invalid_argument("Invalid nlat y");
			if(nlatMap[2]!=nlat[2]) throw std::invalid_argument("Invalid nlat z");
			for(int i=0; i<nlat[0]; ++i){
				for(int j=0; j<nlat[1]; ++j){
					for(int k=0; k<nlat[2]; ++k){
						for(int n=0; n<atoms.size(); ++n){
							string::split(fgets(input,string::M,reader),string::WS,strlist);
							Eigen::VectorXi index(3); index<<i,j,k;
							atoms[n](index)=std::atoi(strlist.back().c_str())-1;
						}
					}
				}
			}
		}
		//print the map
		if(rank==0) std::cout<<"printing map\n";
		if(PHONON_PRINT_DATA>1){
			std::cout<<"map = \n";
			for(int n=0; n<atoms.size(); ++n){
				for(int i=0; i<nlat[0]; ++i){
					for(int j=0; j<nlat[1]; ++j){
						for(int k=0; k<nlat[2]; ++k){
							Eigen::VectorXi index(3); index<<i,j,k;
							std::cout<<i<<" "<<j<<" "<<k<<" "<<n<<" "<<atoms[n](index)<<"\n";
						}
					}
				}
			}
		}
		//check the map
		if(rank==0) std::cout<<"checking map\n";
		for(int n=0; n<atoms.size(); ++n){
			for(int i=0; i<atoms[n].size(); ++i){
				for(int j=0; j<atoms[n].size(); ++j){
					if(i==j) continue;
					else if(atoms[n][i]==atoms[n][j]) throw std::invalid_argument("Invalid map.");
				}
			}
		}
		
		//======== resize the utility vectors ========
		if(rank==0) std::cout<<"resizing utility vectors\n";
		const int nModes=pcell.nAtoms()*3;
		if(rank==0) std::cout<<"nModes = "<<nModes<<"\n";
		uk.resize(3,nlat,Eigen::VectorXcd::Zero(nModes));
		Fk.resize(3,nlat,Eigen::VectorXcd::Zero(nModes));
		ukuk.resize(3,nlat,Eigen::MatrixXcd::Zero(nModes,nModes));
		Fkuk.resize(3,nlat,Eigen::MatrixXcd::Zero(nModes,nModes));
		dynmat.resize(3,nlat,Eigen::MatrixXcd::Zero(nModes,nModes));
		evalues.resize(3,nlat,Eigen::VectorXcd::Zero(nModes));
		omega.resize(3,nlat,Eigen::VectorXd::Zero(nModes));
		
		//======== compute the fourier transforms ========
		//initialize fft
		if(WORLD.rank()==0) std::cout<<"initializing FFT\n";
		inx=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
		iny=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
		inz=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
		outx=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
		outy=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
		outz=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
		fftpx=fftw_plan_dft_3d(nlat[0],nlat[1],nlat[2],inx,outx,FFTW_BACKWARD,FFTW_ESTIMATE);
		fftpy=fftw_plan_dft_3d(nlat[0],nlat[1],nlat[2],iny,outy,FFTW_BACKWARD,FFTW_ESTIMATE);
		fftpz=fftw_plan_dft_3d(nlat[0],nlat[1],nlat[2],inz,outz,FFTW_BACKWARD,FFTW_ESTIMATE);
		//compute fourier transforms
		if(WORLD.rank()==0) std::cout<<"computing the Fourier transforms\n";
		if(nprint<0) nprint=sim.timesteps()/10;
		//loop over all timesteps
		for(int t=0; t<sim.timesteps(); ++t){
			if(WORLD.rank()==0 && t%nprint==0) std::cout<<"t = "<<t<<"\n";
			//loop over all atoms in the primitive cell
			for(int n=0; n<pcell.nAtoms(); ++n){
				//== fourier transform of deviations ==
				//record the deviation from the average in the fftw input
				for(int i=0; i<N; ++i){
					sim.frame(t).diff(sim.frame(t).posn(atoms[n][i]),ravg[atoms[n][i]],rtmp);
					inx[i][0]=rtmp[0]; inx[i][1]=0.0;
					iny[i][0]=rtmp[1]; iny[i][1]=0.0;
					inz[i][0]=rtmp[2]; inz[i][1]=0.0;
				}
				//perform the fourier transform
				fftw_execute(fftpx);
				fftw_execute(fftpy);
				fftw_execute(fftpz);
				//record the fourier transforms
				for(int i=0; i<N; ++i){
					uk[i][n*3+0]=std::complex<double>(outx[i][0]*sqrtNinv,outx[i][1]*sqrtNinv);
					uk[i][n*3+1]=std::complex<double>(outy[i][0]*sqrtNinv,outy[i][1]*sqrtNinv);
					uk[i][n*3+2]=std::complex<double>(outz[i][0]*sqrtNinv,outz[i][1]*sqrtNinv);
				}
				//== fourier transform of forces ==
				//record the force
				for(int i=0; i<N; ++i){
					const Eigen::Vector3d& f=sim.frame(t).force(atoms[n][i]);
					inx[i][0]=f[0]; inx[i][1]=0.0;
					iny[i][0]=f[1]; iny[i][1]=0.0;
					inz[i][0]=f[2]; inz[i][1]=0.0;
				}
				//perform the fourier transform
				fftw_execute(fftpx);
				fftw_execute(fftpy);
				fftw_execute(fftpz);
				//record the fourier transforms
				for(int i=0; i<N; ++i){
					Fk[i][n*3+0]=std::complex<double>(outx[i][0]*sqrtNinv,outx[i][1]*sqrtNinv);
					Fk[i][n*3+1]=std::complex<double>(outy[i][0]*sqrtNinv,outy[i][1]*sqrtNinv);
					Fk[i][n*3+2]=std::complex<double>(outz[i][0]*sqrtNinv,outz[i][1]*sqrtNinv);
				}
			}
			//== record products ==
			for(int i=0; i<N; ++i){
				for(int n=0; n<nModes; ++n){
					for(int m=0; m<nModes; ++m){
						ukuk[i](n,m)+=std::conj(uk[i][n])*uk[i][m];
						Fkuk[i](n,m)+=std::conj(Fk[i][n])*uk[i][m];
					}
				}
			}
		}
		//free memory
		fftw_free(inx);
		fftw_free(iny);
		fftw_free(inz);
		fftw_free(outx);
		fftw_free(outy);
		fftw_free(outz);
		fftw_destroy_plan(fftpx);
		fftw_destroy_plan(fftpy);
		fftw_destroy_plan(fftpz);
		
		//======== reduce the data across processors ========
		//reduce ukuk
		for(int i=0; i<N; ++i){
			const std::complex<double> I(0.0,1.0);
			const Eigen::MatrixXd rtmp=ukuk[i].real();
			const Eigen::MatrixXd itmp=ukuk[i].imag();
			Eigen::MatrixXd rsum=Eigen::MatrixXd::Zero(nModes,nModes);
			Eigen::MatrixXd isum=Eigen::MatrixXd::Zero(nModes,nModes);
			MPI_Reduce(rtmp.data(),rsum.data(),nModes*nModes,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
			MPI_Reduce(itmp.data(),isum.data(),nModes*nModes,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
			if(rank==0) ukuk[i]=rsum+I*isum;
		}
		//reduce Fkuk
		for(int i=0; i<N; ++i){
			const std::complex<double> I(0.0,1.0);
			const Eigen::MatrixXd rtmp=Fkuk[i].real();
			const Eigen::MatrixXd itmp=Fkuk[i].imag();
			Eigen::MatrixXd rsum=Eigen::MatrixXd::Zero(nModes,nModes);
			Eigen::MatrixXd isum=Eigen::MatrixXd::Zero(nModes,nModes);
			MPI_Reduce(rtmp.data(),rsum.data(),nModes*nModes,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
			MPI_Reduce(itmp.data(),isum.data(),nModes*nModes,MPI_DOUBLE,MPI_SUM,0,WORLD.mpic());
			if(rank==0) Fkuk[i]=rsum+I*isum;
		}
		
		//print data
		/*
		for(int i=0; i<N; ++i){
			std::cout<<"ukuk["<<i<<"] = ";
			for(int n=0; n<nModes; ++n){
				for(int m=0; m<nModes; ++m){
					std::cout<<ukuk[i](n,m)<<" ";
				}
			}
			std::cout<<"\n";
		}
		*/
		/*
		for(int i=0; i<N; ++i){
			std::cout<<"Fkuk["<<i<<"] = ";
			for(int n=0; n<nModes; ++n){
				for(int m=0; m<nModes; ++m){
					std::cout<<Fkuk[i](n,m)<<" ";
				}
			}
			std::cout<<"\n";
		}
		*/
		
		if(WORLD.rank()==0){
			
			//======== read the restart file ========
			if(restart){
				std::cout<<"reading restart file\n";
				int tstotOld=0;
				const int nModes2=nModes*nModes;
				reader=fopen(file_restart.c_str(),"rb");
				if(reader==NULL) throw std::runtime_error("Could not open restart file");
				//read total timesteps
				fread(&tstotOld,sizeof(int),1,reader);
				tstot+=tstotOld;
				//read ukuk
				for(int i=0; i<N; ++i){
					const std::complex<double> I(0.0,1.0);
					Eigen::MatrixXd rtmp=Eigen::MatrixXd::Zero(nModes,nModes);
					Eigen::MatrixXd itmp=Eigen::MatrixXd::Zero(nModes,nModes);
					fread(rtmp.data(),sizeof(double),nModes2,reader);
					fread(itmp.data(),sizeof(double),nModes2,reader);
					ukuk[i]+=rtmp+itmp*I;
				}
				//read Fkuk
				for(int i=0; i<N; ++i){
					const std::complex<double> I(0.0,1.0);
					Eigen::MatrixXd rtmp=Eigen::MatrixXd::Zero(nModes,nModes);
					Eigen::MatrixXd itmp=Eigen::MatrixXd::Zero(nModes,nModes);
					fread(rtmp.data(),sizeof(double),nModes2,reader);
					fread(itmp.data(),sizeof(double),nModes2,reader);
					Fkuk[i]+=rtmp+itmp*I;
				}
				//close the file
				fclose(reader);
				reader=NULL;
			}
			
			//======== write the restart file ========
			{
				std::cout<<"writing restart file\n";
				const int nModes2=nModes*nModes;
				writer=fopen(file_restart.c_str(),"wb");
				if(writer==NULL) throw std::runtime_error("Could not open restart file");
				//write total timesteps
				fwrite(&tstot,sizeof(int),1,writer);
				//read ukuk
				for(int i=0; i<N; ++i){
					const Eigen::MatrixXd rtmp=ukuk[i].real();
					const Eigen::MatrixXd itmp=ukuk[i].imag();
					fwrite(rtmp.data(),sizeof(double),nModes2,writer);
					fwrite(itmp.data(),sizeof(double),nModes2,writer);
				}
				//read Fkuk
				for(int i=0; i<N; ++i){
					const Eigen::MatrixXd rtmp=Fkuk[i].real();
					const Eigen::MatrixXd itmp=Fkuk[i].imag();
					fwrite(rtmp.data(),sizeof(double),nModes2,writer);
					fwrite(itmp.data(),sizeof(double),nModes2,writer);
				}
				//close the file
				fclose(writer);
				reader=NULL;
			}
			
			//==== compute the dynamical matrix ====
			std::cout<<"computing the dynamical matrix\n";
			if(read_mass){
				std::cout<<"reading masses\n";
				reader=fopen(file_mass.c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Could not open mass file");
				for(int i=0; i<pcell.nAtoms(); ++i){
					mass_prim[i]=std::atof(fgets(input,string::M,reader));
				}
				fclose(reader);
				reader=NULL;
			}
			std::vector<double> mvec(nModes);
			for(int i=0; i<nModes; i+=3){
				mvec[i+0]=mass_prim[i/3];
				mvec[i+1]=mass_prim[i/3];
				mvec[i+2]=mass_prim[i/3];
			}
			for(int i=0; i<N; ++i){
				dynmat[i]=Fkuk[i]*ukuk[i].inverse();
				for(int n=0; n<nModes; ++n){
					for(int m=0; m<nModes; ++m){
						dynmat[i](n,m)/=std::sqrt(mvec[n]*mvec[m]);
					}
				}
			}
			mvec.clear();
			
			//==== compute the eigenvalues ====
			std::cout<<"computing the eigenvalues\n";
			double error_im=0;
			//compute eigenvalues for all k points
			double wmin=0,wmax=0;
			for(int i=0; i<N; ++i){
				Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(dynmat[i]);
				evalues[i]=solver.eigenvalues();
				for(int j=0; j<nModes; ++j) omega[i][j]=std::sqrt(std::fabs(evalues[i][j].real()))*-1.0*math::func::sgn(evalues[i][j].real());
				for(int j=0; j<nModes; ++j) error_im+=std::fabs(evalues[i][j].imag()/evalues[i][j].real());
				if(omega[i].minCoeff()<wmin) wmin=omega[i].minCoeff();
				if(omega[i].maxCoeff()>wmax) wmax=omega[i].maxCoeff();
			}
			std::cout<<"w-lim = "<<wmin<<" "<<wmax<<"\n";
			error_im/=(N*nModes);
			std::cout<<"error_im = "<<error_im<<"\n";
			//enforce asr
			double error_asr=0;
			std::vector<Eigen::Matrix3cd> asr(pcell.nAtoms(),Eigen::Matrix3cd::Zero());
			for(int nn=0; nn<nasr; ++nn){
				//reset the asr vector
				for(int n=0; n<pcell.nAtoms(); ++n) asr[n]=Eigen::Matrix3cd::Zero();
				//compute the asr vector
				for(int i=0; i<3; ++i){
					for(int j=0; j<3; ++j){
						for(int n=0; n<pcell.nAtoms(); ++n){
							for(int m=0; m<pcell.nAtoms(); ++m){
								asr[n](i,j)+=dynmat[0](n*3+i,m*3+j);
							}
						}
					}
				}
				//subtract the asr vector from the matrix
				for(int i=0; i<3; ++i){
					for(int j=0; j<3; ++j){
						for(int n=0; n<pcell.nAtoms(); ++n){
							for(int m=0; m<pcell.nAtoms(); ++m){
								std::complex<double> denom=std::complex<double>(pcell.nAtoms()*1.0,0.0);
								dynmat[0](n*3+i,m*3+j)-=asr[n](i,j)/denom;
							}
						}
					}
				}
				Eigen::MatrixXcd dynmat0=Eigen::MatrixXcd::Zero(nModes,nModes);
				for(int i=0; i<nModes; ++i){
					for(int j=0; j<nModes; ++j){
						dynmat0(i,j)=0.5*(dynmat[0](i,j)+dynmat[0](j,i));
					}
				}
				dynmat[0]=dynmat0;
				//reset the asr vector
				for(int n=0; n<pcell.nAtoms(); ++n) asr[n]=Eigen::Matrix3cd::Zero();
				//compute the asr error
				error_asr=0;
				for(int i=0; i<3; ++i){
					for(int j=0; j<3; ++j){
						for(int n=0; n<pcell.nAtoms(); ++n){
							for(int m=0; m<pcell.nAtoms(); ++m){
								asr[n](i,j)+=dynmat[0](n*3+i,m*3+j);
							}
						}
					}
				}
				for(int n=0; n<pcell.nAtoms(); ++n){
					error_asr+=asr[n].norm();
				}
				error_asr/=pcell.nAtoms();
				Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(dynmat[0]);
				evalues[0]=solver.eigenvalues();
				for(int j=0; j<omega[0].size(); ++j) omega[0][j]=std::sqrt(std::fabs(evalues[0][j].real()))*-1.0*math::func::sgn(evalues[0][j].real());
			}
			std::cout<<"error_asr = "<<error_asr<<"\n";
			
			//======== diagonalize and print the dynamical matrix ========
			std::cout<<"diagonalizing the dynamical matrix\n";
			writer=fopen(file_dynmat.c_str(),"w");
			if(writer==NULL) throw std::runtime_error("Could not open dynmat file");
			fprintf(writer,"#kx ky kz ");
			for(int n=0; n<pcell.nAtoms(); ++n){
				const std::string labelx=std::to_string(n)+std::string("x")+std::string("r");
				const std::string labely=std::to_string(n)+std::string("y")+std::string("r");
				const std::string labelz=std::to_string(n)+std::string("z")+std::string("r");
				fprintf(writer,"%s %s %s ",labelx.c_str(),labely.c_str(),labelz.c_str());
			}
			for(int n=0; n<pcell.nAtoms(); ++n){
				const std::string labelx=std::to_string(n)+std::string("x")+std::string("i");
				const std::string labely=std::to_string(n)+std::string("y")+std::string("i");
				const std::string labelz=std::to_string(n)+std::string("z")+std::string("i");
				fprintf(writer,"%s %s %s ",labelx.c_str(),labely.c_str(),labelz.c_str());
			}
			fprintf(writer,"\n");
			for(int i=0; i<nlat[0]; ++i){
				for(int j=0; j<nlat[1]; ++j){
					for(int k=0; k<nlat[2]; ++k){
						//generate index
						Eigen::Vector3i index; index<<i,j,k;
						//generate k-vector
						Eigen::Vector3d kfrac;
						kfrac<<(1.0*i)/nlat[0],(1.0*j)/nlat[1],(1.0*k)/nlat[2];
						Eigen::Vector3d kvec=pcell.K()*kfrac;
						kvec[0]=math::func::mod(kvec[0],-PI,PI);
						kvec[1]=math::func::mod(kvec[1],-PI,PI);
						kvec[2]=math::func::mod(kvec[2],-PI,PI);
						//diagonalize the dynamical matrix
						fprintf(writer,"%f %f %f ",kvec[0],kvec[1],kvec[2]);
						for(int n=0; n<evalues(index).size(); ++n){
							fprintf(writer,"%f ",evalues(index)[n].real());
						}
						for(int n=0; n<evalues(index).size(); ++n){
							fprintf(writer,"%f ",evalues(index)[n].imag());
						}
						fprintf(writer,"\n");
					}
				}
			}
			fclose(writer);
			writer=NULL;
			
			//======== generate the phonon spectrum ========
			std::cout<<"generating the phonon spectrum\n";
			for(int i=0; i<kpath.nkvecs(); ++i){
				//set the origin
				Eigen::Vector3i origin; 
				origin[0]=math::func::mod((int)(kpath.kvec(i)[0]*nlat[0]),0,nlat[0]);
				origin[1]=math::func::mod((int)(kpath.kvec(i)[1]*nlat[1]),0,nlat[1]);
				origin[2]=math::func::mod((int)(kpath.kvec(i)[2]*nlat[2]),0,nlat[2]);
				//std::cout<<"origin["<<i<<"] = "<<origin.transpose()<<"\n";
				//set the points of the cube
				std::vector<Eigen::Vector3i> p(8);
				p[0]<<origin[0],origin[1],origin[2];//c000
				p[1]<<math::func::mod(origin[0]+1,0,nlat[0]),origin[1],origin[2];//c100
				p[2]<<origin[0],math::func::mod(origin[1]+1,0,nlat[1]),origin[2];//c010
				p[3]<<math::func::mod(origin[0]+1,0,nlat[0]),math::func::mod(origin[1]+1,0,nlat[1]),origin[2];//c110
				p[4]<<origin[0],origin[1],math::func::mod(origin[2]+1,0,nlat[2]);//c001
				p[5]<<math::func::mod(origin[0]+1,0,nlat[0]),origin[1],math::func::mod(origin[2]+1,0,nlat[2]);//c101
				p[6]<<origin[0],math::func::mod(origin[1]+1,0,nlat[1]),math::func::mod(origin[2]+1,0,nlat[2]);//c011
				p[7]<<math::func::mod(origin[0]+1,0,nlat[0]),math::func::mod(origin[1]+1,0,nlat[1]),math::func::mod(origin[2]+1,0,nlat[2]);//c111
				//interpolate
				const double xd=math::func::mod(kpath.kvec(i)[0]*nlat[0]-origin[0],0.0,1.0);
				const double yd=math::func::mod(kpath.kvec(i)[1]*nlat[1]-origin[1],0.0,1.0);
				const double zd=math::func::mod(kpath.kvec(i)[2]*nlat[2]-origin[2],0.0,1.0);
				kpath.kval(i).resize(nModes);
				for(int j=0; j<nModes; ++j){
					//points on a cube
					const double c000=omega(p[0])[j];
					const double c100=omega(p[1])[j];
					const double c010=omega(p[2])[j];
					const double c110=omega(p[3])[j];
					const double c001=omega(p[4])[j];
					const double c101=omega(p[5])[j];
					const double c011=omega(p[6])[j];
					const double c111=omega(p[7])[j];
					//points on a square
					const double c00=c000*(1.0-xd)+c100*xd;
					const double c01=c001*(1.0-xd)+c101*xd;
					const double c10=c010*(1.0-xd)+c110*xd;
					const double c11=c011*(1.0-xd)+c111*xd;
					//points on a line
					const double c0=c00*(1.0-yd)+c10*yd;
					const double c1=c01*(1.0-yd)+c11*yd;
					//final value
					kpath.kval(i)[j]=c0*(1.0-zd)+c1*zd;
				}
			}
			
			//======== compute the phonon density of states ========
			std::cout<<"computing the density of states\n";
			if(dos.wlmax()<0) dos.wlmax()=wmax;
			if(dos.wlmin()<0) dos.wlmin()=wmin;
			if(dos.sigma()<0) dos.sigma()=0.01;//default value
			if(dos.dw()<0) dos.dw()=(dos.wlmax()-dos.wlmin())/100.0;
			const int ndos=static_cast<int>((dos.wlmax()-dos.wlmin())/dos.dw())+1;
			dos.resize(ndos);
			for(int i=0; i<dos.size(); ++i){
				const double norm=1.0/(dos.sigma()*sqrt(2.0*PI));
				const double si2=1.0/(2.0*dos.sigma()*dos.sigma());
				const double w=dos.w(i);
				dos.dos(i)[0]=w;
				for(int n=0; n<N; ++n){
					for(int m=0; m<nModes; ++m){
						dos.dos(i)[1]+=norm*std::exp(-(w-omega[n][m])*(w-omega[n][m])*si2);
					}
				}
			}
			for(int i=0; i<dos.size(); ++i){
				dos.dos(i)[1]/=(N*nModes);
			}
			
			//======== print the phonon spectrum ========
			std::cout<<"printing the phonon spectrum\n";
			writer=fopen(file_disp.c_str(),"w");
			if(writer==NULL) throw std::runtime_error("Could not open disp file");
			fprintf(writer,"#kx ky kz kd ");
			for(int n=0; n<nModes; ++n) fprintf(writer,"n%i ",n+1);
			fprintf(writer,"\n");
			const Eigen::Vector3d korigin=pcell.K()*kpath.kvec(0);
			Eigen::Vector3d kposn=korigin;
			double kdist=0;
			fprintf(writer,"%f %f %f %f ",korigin[0],korigin[1],korigin[2],0.0);
			for(int n=0; n<nModes; ++n) fprintf(writer,"%f ",kpath.kval(0)[n]);
			fprintf(writer,"\n");
			for(int i=1; i<kpath.nkvecs(); ++i){
				const Eigen::Vector3d kvec=kpath.kvec(i);
				kdist+=(kvec-kposn).norm();
				kposn=kvec;
				fprintf(writer,"%f %f %f %f ",kvec[0],kvec[1],kvec[2],kdist);
				for(int n=0; n<nModes; ++n) fprintf(writer,"%f ",kpath.kval(i)[n]);
				fprintf(writer,"\n");
			}
			fclose(writer);
			writer=NULL;
			
			//======== print the phonon density of states ========
			std::cout<<"printing the phonon density of states\n";
			writer=fopen(file_dos.c_str(),"w");
			if(writer==NULL) throw std::runtime_error("Could not open dos file");
			fprintf(writer,"#omega dos\n");
			for(int i=0; i<dos.size(); ++i){
				fprintf(writer,"%f %f\n",dos.dos(i)[0],dos.dos(i)[1]);
			}
			fclose(writer);
			writer=NULL;
			
		}
		MPI_Barrier(WORLD.mpic());
		
		//======== stop wall clock ========
		if(rank==0) stop=std::clock();
		
		if(rank==0){
			const double time=((double)(stop-start))/CLOCKS_PER_SEC;
			std::cout<<"time = "<<time<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in Phonon::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//==== finalize mpi ====
	if(rank==0) std::cout<<"finalizing mpi\n";
	MPI_Finalize();
	
	//==== free memory ====
	delete[] paramfile;
	delete[] input;
	delete[] simstr;
	delete[] str;
}