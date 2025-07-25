// mpi
#include <mpi.h>
// c libraries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// c++ libraries
#include <iostream>
#include <exception>
#include <algorithm>
#include <random>
#include <chrono>
// ann - structure
#include "struc/structure.hpp"
#include "struc/neighbor.hpp"
// ann - format
#include "format/file.hpp"
#include "format/format.hpp"
#include "format/vasp_struc.hpp"
#include "format/qe_struc.hpp"
#include "format/xyz_struc.hpp"
#include "format/cp2k_struc.hpp"
// ann - math
#include "math/func.hpp"
// ann - string
#include "str/string.hpp"
#include "str/print.hpp"
// ann - chem
#include "chem/units.hpp"
// ann - thread
#include "thread/parallel.hpp"
// ann - util
#include "util/compiler.hpp"
#include "util/time.hpp"
// ann - opt
#include "opt/optimize.hpp"
// ann - ml
#include "ml/nn.hpp"
#include "ml/pca.hpp"
// ann - nnp
#include "nnp/nnp.hpp"

//************************************************************
// MPI Communicators
//************************************************************

parallel::Comm WORLD;//all processors

//************************************************************
// MAIN
//************************************************************

int main(int argc, char* argv[]){
	//======== global variables ========
	//units
		units::System unitsys=units::System::UNKNOWN;
	//structures - format
		AtomType atomT;
		atomT.name=true; atomT.an=false; atomT.type=true; atomT.index=false;
		atomT.posn=true; atomT.force=false; atomT.symm=true; atomT.charge=false;
		FILE_FORMAT::type format;//format of training data
	//structures - data
		std::vector<std::string> data;  //data files
		std::vector<std::string> files; //structure files - training
		std::vector<Structure> struc;   //structures - training
	//loss
		Opt::Loss loss;
		double error_scale=1.0;
		int ngrid=0;
		double plen=0;
	//nnp
		std::string file_nnp;
		NNP nnp;
	//mpi data distribution
		parallel::Dist dist; //data distribution - batch
	//timing
		Clock clock,clock_wall; //time objects
		double time_wall=0;     //total wall time
		double time_symm=0;     //compute time - symmetry functions
	//file i/o
		FILE* reader=NULL;
		std::vector<std::string> strlist;
		char* paramfile=new char[string::M];
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		char* paramstr =new char[string::M];
		bool read_pot=false;
		std::string file_pot;
		std::vector<std::string> files_basis;//file - stores basis
		
	try{
		//************************************************************************************
		// LOADING/INITIALIZATION
		//************************************************************************************
		
		//======== initialize mpi ========
		MPI_Init(&argc,&argv);
		WORLD.label()=MPI_COMM_WORLD;
		MPI_Comm_size(WORLD.label(),&WORLD.size());
		MPI_Comm_rank(WORLD.label(),&WORLD.rank());
		
		//======== start wall clock ========
		if(WORLD.rank()==0) clock_wall.begin();
		
		//======== print title ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::title("PLOT LOSS",strbuf,' ')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
		}
		
		//======== print compiler information ========
		if(WORLD.rank()==0){
			std::cout<<"date     = "<<compiler::date()<<"\n";
			std::cout<<"time     = "<<compiler::time()<<"\n";
			std::cout<<"compiler = "<<compiler::name()<<"\n";
			std::cout<<"version  = "<<compiler::version()<<"\n";
			std::cout<<"standard = "<<compiler::standard()<<"\n";
			std::cout<<"arch     = "<<compiler::arch()<<"\n";
			std::cout<<"instr    = "<<compiler::instr()<<"\n";
			std::cout<<"os       = "<<compiler::os()<<"\n";
			std::cout<<"omp      = "<<compiler::omp()<<"\n";
		}
		
		//======== print mathematical constants ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::printf("PI    = %.15f\n",math::constant::PI);
			std::printf("RadPI = %.15f\n",math::constant::RadPI);
			std::printf("Rad2  = %.15f\n",math::constant::Rad2);
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== print physical constants ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			std::printf("bohr-r  (A)  = %.12f\n",units::BOHR);
			std::printf("hartree (eV) = %.12f\n",units::HARTREE);
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== set mpi data ========
		{
			int* ranks=new int[WORLD.size()];
			MPI_Gather(&WORLD.rank(),1,MPI_INT,ranks,1,MPI_INT,0,WORLD.label());
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MPI",strbuf)<<"\n";
				std::cout<<"world - size = "<<WORLD.size()<<"\n"<<std::flush;
				//for(int i=0; i<WORLD.size(); ++i) std::cout<<"reporting from process "<<ranks[i]<<" out of "<<WORLD.size()-1<<"\n"<<std::flush;
				std::cout<<print::title("MPI",strbuf)<<"\n";
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<std::flush;
			}
			delete[] ranks;
		}
		
		//======== rank 0 reads from file ========
		if(WORLD.rank()==0){
			
			//======== check the arguments ========
			if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
			
			//======== load the parameter file ========
			std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//======== open the parameter file ========
			std::cout<<"opening parameter file\n";
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+paramfile);
			
			//======== read in the parameters ========
			std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL){
				string::trim_right(input,string::COMMENT);//trim comments
				if(string::split(input,string::WS,strlist)==0) continue;//skip if empty
				string::to_upper(strlist.at(0));//convert tag to upper case
				if(strlist.size()<2) throw std::runtime_error("Parameter tag without corresponding value.");
				//general
				if(strlist.at(0)=="UNITS"){//units
					unitsys=units::System::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="FORMAT"){//simulation format
					format=FILE_FORMAT::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="DATA"){//data - training
					data.push_back(strlist.at(1));
				} else if(strlist.at(0)=="FILE_NNP"){
					file_nnp=strlist.at(1);//file storing the ann
				} else if(strlist.at(0)=="LOSS"){
					loss=Opt::Loss::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="PARAMS"){
					std::strcpy(paramstr,strlist.at(1).c_str());
				} else if(strlist.at(0)=="ERROR_SCALE"){
					error_scale=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="NGRID"){
					ngrid=std::atoi(strlist.at(1).c_str());
				} else if(strlist.at(0)=="PLEN"){
					plen=std::atof(strlist.at(1).c_str());
				} 
			}
			
			//======== close parameter file ========
			std::cout<<"closing parameter file\n";
			fclose(reader);
			reader=NULL;
		}
		
		//======== bcast the paramters ========
		if(WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.label());
		MPI_Bcast(&loss,1,MPI_INT,0,WORLD.label());
		MPI_Bcast(&ngrid,1,MPI_INT,0,WORLD.label());
		MPI_Bcast(&plen,1,MPI_DOUBLE,0,WORLD.label());
		MPI_Bcast(&error_scale,1,MPI_DOUBLE,0,WORLD.label());
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.label());
		MPI_Bcast(paramstr,string::M,MPI_CHAR,0,WORLD.label());
		
		//======== print parameters ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"ATOM_T  = "<<atomT<<"\n";
			std::cout<<"FORMAT  = "<<format<<"\n";
			std::cout<<"UNITS   = "<<unitsys<<"\n";
			std::cout<<"LOSS    = "<<loss<<"\n";
			std::cout<<"NGRID   = "<<ngrid<<"\n";
			std::cout<<"PLEN    = "<<plen<<"\n";
			std::cout<<"ERROR_S = "<<error_scale<<"\n";
			std::cout<<"PARAMS  = "<<paramstr<<"\n";
			std::cout<<"DATA    = \n"; for(int i=0; i<data.size(); ++i) std::cout<<"\t\t"<<data[i]<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		MPI_Barrier(WORLD.label());
		
		//======== check the parameters ========
		if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		if(loss==Opt::Loss::UNKNOWN) throw std::invalid_argument("Invalid loss function.");
		if(error_scale<=0) throw std::invalid_argument("Invalid error scaling.");
		
		//======== set the unit system ========
		if(WORLD.rank()==0) std::cout<<"setting the unit system\n";
		units::consts::init(unitsys);
		
		//************************************************************************************
		// READ DATA
		//************************************************************************************
		
		//======== rank 0 reads the data files (lists of structure files) ========
		if(WORLD.rank()==0){
			//==== read the training data ====
			std::cout<<"reading data - training\n";
			for(int i=0; i<data.size(); ++i){
				//open the data file
				reader=fopen(data[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data[i]);
				//read in the data
				while(fgets(input,string::M,reader)!=NULL){
					if(!string::empty(input)) files.push_back(std::string(string::trim(input)));
				}
				//close the file
				fclose(reader);
				reader=NULL;
			}
		}
		
		//======== bcast the file names =======
		if(WORLD.rank()==0) std::cout<<"bcasting file names\n";
		//bcast names
		parallel::bcast(WORLD.label(),0,files);
		//set number of structures
		const int nStruc=files.size();
		if(WORLD.rank()==0) std::cout<<"nstruc = "<<nStruc<<"\n";
		
		//======== gen thread dist + offset ========
		//thread dist - divide structures equally among the process
		dist.init(WORLD.size(),WORLD.rank(),nStruc);
		//print
		{
			//allocate arrays
			int* thread_dist=new int[WORLD.size()];
			int* thread_offset=new int[WORLD.size()];
			//assign arrays
			parallel::Dist::size(WORLD.size(),nStruc,thread_dist);
			parallel::Dist::offset(WORLD.size(),nStruc,thread_offset);
			//print
			if(WORLD.rank()==0){
				std::cout<<"thread_dist   = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_dist[i]<<" "; std::cout<<"\n";
				std::cout<<"thread_offset = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_offset[i]<<" "; std::cout<<"\n";
				std::cout<<std::flush;
			}
			//free arrays
			delete[] thread_dist;
			delete[] thread_offset;
		}
		
		//======== gen indices (random shuffle) ========
		std::vector<int> indices(nStruc,0);
		if(WORLD.rank()==0){
			for(int i=0; i<indices.size(); ++i) indices[i]=i;
			//std::random_shuffle(indices.begin(),indices.end());
		}
		//======== bcast randomized indices ========
		parallel::bcast(WORLD.label(),0,indices);
		
		//======== read the structures ========
		if(WORLD.rank()==0) std::cout<<"reading structures - training\n";
		struc.resize(dist.size());
		for(int i=0; i<dist.size(); ++i){
			const std::string& file=files[indices[dist.index(i)]];
			read_struc(file.c_str(),format,atomT,struc[i]);
		}
		MPI_Barrier(WORLD.label());
		
		//======== check the structures ========
		if(WORLD.rank()==0) std::cout<<"checking the structures\n";
		for(int i=0; i<dist.size(); ++i){
			const std::string filename=files[indices[dist.index(i)]];
			const Structure& strucl=struc[i];
			if(strucl.nAtoms()==0) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has zero atoms."));
			if(std::isinf(strucl.energy())) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has inf energy."));
			if(strucl.energy()!=strucl.energy()) throw std::runtime_error(std::string("ERROR: Structure \"")+filename+std::string(" has nan energy."));
			if(std::fabs(strucl.energy())<math::constant::ZERO) std::cout<<"*********** WARNING: Structure \""<<filename<<"\" has ZERO energy. ***********";
		}
		MPI_Barrier(WORLD.label());
		
		//======== save energy ========
		std::vector<double> energy(struc.size());
		for(int n=0; n<struc.size(); ++n){
			energy[n]=struc[n].energy();
		}
		
		//************************************************************************************
		// READ NN-POT
		//************************************************************************************
		
		//==== read the potential ====
		if(WORLD.rank()==0){
			std::cout<<"reading potential\n";
			NNP::read(file_nnp.c_str(),nnp);
			std::cout<<nnp<<"\n";
		}
		parallel::bcast(WORLD.label(),0,nnp);
		int nparams=0;
		for(int i=0; i<nnp.nspecies(); ++i) nparams+=nnp.nnh(i).nn().size();
		if(WORLD.rank()==0) std::cout<<"nparams = "<<nparams<<"\n";
		
		//======== set the types ========
		if(WORLD.rank()==0) std::cout<<"setting the types\n";
		for(int i=0; i<dist.size(); ++i){
			for(int n=0; n<struc[i].nAtoms(); ++n){
				struc[i].type(n)=nnp.index(struc[i].name(n));
			}
		}
		
		std::vector<Eigen::VectorXd> pvectors(nnp.nspecies());
		for(int n=0; n<nnp.nspecies(); ++n){
			pvectors[n].resize(nnp.nnh(n).nn().size());
		}
		
		//************************************************************************************
		// SET INPUTS
		//************************************************************************************
		
		//======== compute the symmetry functions ========
		clock.begin();
		if(WORLD.rank()==0) std::cout<<"setting the inputs (symmetry functions)\n";
		//compute symmetry functions
		for(int n=0; n<dist.size(); ++n){
			NeighborList nlist(struc[n],nnp.rc());
			NNP::init(nnp,struc[n]);
			NNP::symm(nnp,struc[n],nlist);
		}
		clock.end();
		time_symm=clock.duration();
		MPI_Barrier(WORLD.label());
		
		//************************************************************************************
		// READ PARAMETER HISTORY
		//************************************************************************************
		
		//==== read the parameters ====
		if(WORLD.rank()==0) std::cout<<"reading parameters\n";
		const int psize=nparams*20;
		char* pinput=new char[psize];
		reader=fopen(paramstr,"r");
		std::vector<Eigen::VectorXd> xdata;
		while(fgets(pinput,psize,reader)!=NULL){
			string::split(pinput,string::WS,strlist);
			if(strlist.size()!=nparams) throw std::runtime_error("Invalid number of parameters.");
			xdata.push_back(Eigen::VectorXd::Zero(nparams));
			for(int i=0; i<nparams; ++i) xdata.back()[i]=std::atof(strlist[i].c_str());
		}
		fclose(reader);
		reader=NULL;
		const Eigen::VectorXd pcenter=xdata.back();
		
		//************************************************************************************
		// COMPUTE PCA
		//************************************************************************************
		
		//==== PCA ===
		if(WORLD.rank()==0) std::cout<<"computing PCA\n";
		const int nobs=xdata.size();
		PCA pca(nobs,nparams);
		for(int i=0; i<nobs; ++i){
			pca.X().row(i)=xdata[i];
		}
		pca.compute();
		//store first two pc vectors
		const double w1=pca.w()[pca.ci(0)];
		const double w2=pca.w()[pca.ci(1)];
		const Eigen::VectorXd p1=pca.W().row(pca.ci(0));
		const Eigen::VectorXd p2=pca.W().row(pca.ci(1));
		if(WORLD.rank()==0) std::cout<<"PCA = \n"<<w1<<"\n"<<w2<<"\n";
		if(WORLD.rank()==0) std::cout<<"skew = "<<w1/w2<<"\n";
		//compute moore-penrose pseudo-inverse
		Eigen::MatrixXd pmat(nparams,2);
		pmat.col(0)=p1;
		pmat.col(1)=p2;
		Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> decomp(pmat);
		const Eigen::MatrixXd pinverse=decomp.pseudoInverse();
		
		//************************************************************************************
		// COMPUTE LOSS PATHWAY
		//************************************************************************************
		
		std::vector<Eigen::Vector3d> lp(xdata.size());
		for(int i=0; i<xdata.size(); ++i){
			const Eigen::MatrixXd vec=pinverse*(xdata[i]-pcenter);
			//split into nnp parameter vectors
			int pcount=0;
			for(int n=0; n<nnp.nspecies(); ++n){
				std::memcpy(pvectors[n].data(),xdata[i].data()+pcount,pvectors[n].size()*sizeof(double));
				pcount+=pvectors[n].size();
				nnp.nnh(n).nn()<<pvectors[n];
			}
			//compute loss
			double error=0;
			for(int n=0; n<struc.size(); ++n){
				NNP::energy(nnp,struc[n]);
				const double ediff=error_scale*(energy[n]-struc[n].energy())/struc[n].nAtoms();
				switch(loss){
					case Opt::Loss::MSE:{
						error+=0.5*ediff*ediff;
					} break;
					case Opt::Loss::MAE:{
						error+=std::fabs(ediff);
					} break;
					case Opt::Loss::HUBER:{
						error+=(sqrt(1.0+(ediff*ediff))-1.0);
					} break;
					default: break;
				}
			}
			lp[i]<<vec(0,0),vec(1,0),error;
		}
		//reduce
		{
			const int nsize=xdata.size();
			double* data_loc=new double[nsize];
			double* data_tot=new double[nsize];
			for(int i=0; i<nsize; ++i) data_loc[i]=lp[i][2];
			for(int i=0; i<nsize; ++i) data_tot[i]=0.0;
			MPI_Reduce(data_loc,data_tot,nsize,MPI_DOUBLE,MPI_SUM,0,WORLD.label());
			for(int i=0; i<nsize; ++i) lp[i][2]=data_tot[i]/nStruc;
			delete[] data_loc;
			delete[] data_tot;
		}
		
		//************************************************************************************
		// COMPUTE LOSS SURFACE
		//************************************************************************************
		
		//==== compute loss function ====
		if(WORLD.rank()==0) std::cout<<"computing loss function\n";
		//find max diff
		double pdmax=0;
		for(int i=0; i<xdata.size()-1; ++i){
			const double diff=(xdata[i]-pcenter).norm();
			if(diff>pdmax) pdmax=diff;
		}
		if(WORLD.rank()==0) std::cout<<"pdmax = "<<pdmax<<"\n";
		if(WORLD.rank()==0) std::cout<<"pdbe  = "<<(xdata.front()-xdata.back()).norm()<<"\n";
		if(plen==0) plen=pdmax;
		//compute rg
		Eigen::VectorXd rm=xdata.front();
		for(int i=1; i<xdata.size(); ++i) rm.noalias()+=xdata[i];
		double rg=0;
		for(int i=0; i<xdata.size(); ++i) rg+=(xdata[i]-rm).squaredNorm();
		rg=std::sqrt(rg/xdata.size());
		if(WORLD.rank()==0) std::cout<<"rg = "<<rg<<"\n";
		//generate grid
		const int npoints=(2*ngrid+1)*(2*ngrid+1);
		const double dp=plen/(1.0*ngrid);
		std::vector<Eigen::Vector3d> ll(npoints);
		int count=0;
		for(int i=-ngrid; i<=ngrid; ++i){
			for(int j=-ngrid; j<=ngrid; ++j){
				//generate parameter point
				const Eigen::VectorXd params=pcenter+i*dp*p1+j*dp*p2;
				//split into nnp parameter vectors
				int pcount=0;
				for(int n=0; n<nnp.nspecies(); ++n){
					std::memcpy(pvectors[n].data(),params.data()+pcount,pvectors[n].size()*sizeof(double));
					pcount+=pvectors[n].size();
					nnp.nnh(n).nn()<<pvectors[n];
				}
				//compute loss
				double error=0;
				for(int n=0; n<struc.size(); ++n){
					NNP::energy(nnp,struc[n]);
					const double ediff=error_scale*(energy[n]-struc[n].energy())/struc[n].nAtoms();
					switch(loss){
						case Opt::Loss::MSE:{
							error+=0.5*ediff*ediff;
						} break;
						case Opt::Loss::MAE:{
							error+=std::fabs(ediff);
						} break;
						case Opt::Loss::HUBER:{
							error+=(sqrt(1.0+(ediff*ediff))-1.0);
						} break;
						default: break;
					}
				}
				ll[count++]<<i*dp,j*dp,error;
			}
		}
		//reduce
		{
			double* data_loc=new double[npoints];
			double* data_tot=new double[npoints];
			for(int i=0; i<npoints; ++i) data_loc[i]=ll[i][2];
			for(int i=0; i<npoints; ++i) data_tot[i]=0.0;
			MPI_Reduce(data_loc,data_tot,npoints,MPI_DOUBLE,MPI_SUM,0,WORLD.label());
			for(int i=0; i<npoints; ++i) ll[i][2]=data_tot[i]/nStruc;
			delete[] data_loc;
			delete[] data_tot;
		}
		
		//************************************************************************************
		// WRITE
		//************************************************************************************
		
		//==== write loss function ====
		if(WORLD.rank()==0){
			std::cout<<"writing loss function\n";
			FILE* writer=fopen("loss.dat","w");
			if(writer==NULL) throw std::runtime_error("Could not open loss file.\n");
			for(int i=0; i<ll.size(); ++i){
				fprintf(writer,"%f %f %e\n",ll[i][0],ll[i][1],ll[i][2]);
			}
		}
		
		//==== write loss path ====
		if(WORLD.rank()==0){
			std::cout<<"writing loss path\n";
			FILE* writer=fopen("loss_path.dat","w");
			if(writer==NULL) throw std::runtime_error("Could not open loss file.\n");
			for(int i=0; i<lp.size(); ++i){
				fprintf(writer,"%f %f %e\n",lp[i][0],lp[i][1],lp[i][2]);
			}
		}
		
		//==== write pca scores ===
		if(WORLD.rank()==0){
			std::cout<<"writing pca scores\n";
			const double trace=pca.S().trace();
			FILE* writer=fopen("loss_pca.dat","w");
			if(writer==NULL) throw std::runtime_error("Could not open loss file.\n");
			const int nvars=pca.nvars();
			for(int i=0; i<nvars; ++i){
				fprintf(writer,"%f\n",pca.w()[i]/trace);
			}
		}
		
		//======== finalize mpi ========
		if(WORLD.rank()==0) std::cout<<"finalizing mpi\n";
		std::cout<<std::flush;
		MPI_Finalize();
	}catch(std::exception& e){
		std::cout<<"ERROR in nnp_train::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//======== free local variables ========
	delete[] paramfile;
	delete[] input;
	delete[] strbuf;
	
	return 0;
}
