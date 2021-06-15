// mpi
#include <mpi.h>
// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <iostream>
#include <exception>
#include <algorithm>
// ann - structure
#include "sim.hpp"
// ann - file i/o
#include "vasp.hpp"
#include "qe.hpp"
#include "ann.hpp"
#include "xyz.hpp"
#include "string.hpp"
// ann - units
#include "units.hpp"
// nn
#include "nn.hpp"
#include "nn_pot.hpp"
// ann - print
#include "print.hpp"
// ann - list
#include "list.hpp"
// ann - thread
#include "parallel.hpp"
// ann - compiler
#include "compiler.hpp"
// ann - time
#include "time.hpp"

parallel::Comm WORLD;//all processors

int main(int argc, char* argv[]){
	//==== file i/o ====
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input    =new char[string::M];
		char* simstr   =new char[string::M];
		char* potstr   =new char[string::M];
		char* strbuf   =new char[print::len_buf];
		std::vector<std::string> strlist;
	//==== simulation variables ====
		AtomType atomT;
		atomT.name=true; atomT.an=false; atomT.type=false; atomT.index=false;
		atomT.posn=true; atomT.force=true; atomT.symm=true; atomT.charge=false;
		FILE_FORMAT::type format;//format of training data
		Simulation sim;
		Interval interval;
		bool calc_force=false;
	//==== nn potential - opt ====
		NNPot nnpot;
	//==== timing ====
		Clock clock;
	//==== units ====
		units::System::type unitsys;
	//==== thread ====
		parallel::Dist dist;
	
	//==== initialize mpi ====
	MPI_Init(&argc,&argv);
	WORLD.label()=MPI_COMM_WORLD;
	MPI_Comm_size(WORLD.label(),&WORLD.size());
	MPI_Comm_rank(WORLD.label(),&WORLD.rank());
	
	try{
		if(argc!=2) throw std::invalid_argument("Invalid number of command-line arguments.");
		
		//==== start wall clock ====
		if(WORLD.rank()==0) clock.begin();
		
		//==== print title ====
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::title("ANN - COMPUTE",strbuf,' ')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
		}
		
		//==== print compiler information ====
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
		
		//==== print mathematical constants ====
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::printf("PI    = %.15f\n",math::constant::PI);
			std::printf("RadPI = %.15f\n",math::constant::RadPI);
			std::printf("Rad2  = %.15f\n",math::constant::Rad2);
			std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//==== print physical constants ====
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			std::printf("bohr-r  (A)  = %.12f\n",units::BOHR);
			std::printf("hartree (eV) = %.12f\n",units::HARTREE);
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//==== set mpi data ====
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
		
		//==== rank 0 reads parameters ====
		if(WORLD.rank()==0){
		
			//==== copy the parameter file ====
			std::cout<<"reading parameter file\n";
			std::strcpy(paramfile,argv[1]);
			
			//==== read in the general parameters ====
			reader=fopen(paramfile,"r");
			if(reader==NULL) throw std::runtime_error("I/O Error: could not open parameter file.");
			while(fgets(input,string::M,reader)!=NULL){
				string::trim_right(input,string::COMMENT);
				string::split(input,string::WS,strlist);
				if(strlist.size()==0) continue;
				string::to_upper(strlist.at(0));
				if(strlist.at(0)=="SIM"){
					std::strcpy(simstr,strlist.at(1).c_str());
				} else if(strlist.at(0)=="INTERVAL"){
					interval=Interval::read(strlist.at(1).c_str());
				} else if(strlist.at(0)=="NNPOT"){
					std::strcpy(potstr,strlist.at(1).c_str());
				} else if(strlist.at(0)=="FORMAT"){
					format=FILE_FORMAT::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="UNITS"){
					unitsys=units::System::read(string::to_upper(strlist.at(1)).c_str());
				} 
			}
			//close the file
			fclose(reader);
			reader=NULL;
			
			//==== print the parameters ====
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"SIM      = "<<simstr<<"\n";
			std::cout<<"INTERVAL = "<<interval<<"\n";
			std::cout<<"FORMAT   = "<<format<<"\n";
			std::cout<<"UNITS    = "<<unitsys<<"\n";
			std::cout<<"NNPOT    = "<<potstr<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			
			//==== check the parameters ====
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			
		}
		
		//==== broadcast the parameters ====
		if(WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		parallel::bcast(WORLD.label(),0,interval);
		MPI_Bcast(simstr,string::M,MPI_CHAR,0,WORLD.label());
		MPI_Bcast(potstr,string::M,MPI_CHAR,0,WORLD.label());
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.label());
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.label());
		
		//==== initialize the unit system ====
		if(WORLD.rank()==0) std::cout<<"initializing the unit system\n";
		units::consts::init(unitsys);
		
		//==== split the interval ====
		if(WORLD.rank()==0) std::cout<<"splitting the interval\n";
		Interval intloc=Interval::split(interval,WORLD.rank(),WORLD.size());
		for(int n=0; n<WORLD.size(); ++n){
			if(WORLD.rank()==n) std::cout<<"intloc["<<n<<"] = "<<intloc<<"\n"<<std::flush;
			MPI_Barrier(WORLD.label());
		}
		
		//==== read the data ====
		if(WORLD.rank()==0) std::cout<<"reading simulation\n";
		if(format==FILE_FORMAT::XDATCAR){
			VASP::XDATCAR::read(simstr,intloc,atomT,sim);
		} else if(format==FILE_FORMAT::XYZ){
			XYZ::read(simstr,intloc,atomT,sim);
		} else throw std::invalid_argument("Invalid file format");
		
		//==== read the potential ====
		if(WORLD.rank()==0) std::cout<<"reading potential\n";
		NNPot::read(potstr,nnpot);
		if(WORLD.rank()==0) std::cout<<nnpot<<"\n";
		
		//==== compute ====
		if(WORLD.rank()==0) std::cout<<"initializing symmetry functions\n";
		for(int t=0; t<sim.timesteps(); ++t){
			nnpot.init_symm(sim.frame(t));
		}
		if(WORLD.rank()==0) std::cout<<"computing symmetry functions\n";
		for(int t=0; t<sim.timesteps(); ++t){
			nnpot.calc_symm(sim.frame(t));
		}
		if(WORLD.rank()==0) std::cout<<"computing energy\n";
		double* energyloc=new double[sim.timesteps()];
		for(int t=0; t<sim.timesteps(); ++t){
			energyloc[t]=nnpot.energy(sim.frame(t),false)/sim.frame(t).nAtoms();
		}
		
		//==== print ====
		if(WORLD.rank()==0) std::cout<<"printing energy\n";
		int* ts=new int[WORLD.size()];
		int tsloc=sim.timesteps();
		MPI_Gather(&tsloc,1,MPI_INT,ts,1,MPI_INT,0,WORLD.label());
		int tstot=0;
		for(int n=0; n<WORLD.size(); ++n){
			tstot+=ts[n];
			if(WORLD.rank()==0) std::cout<<"ts["<<n<<"] = "<<ts[n]<<"\n";
		}
		MPI_Bcast(&tstot,1,MPI_INT,0,WORLD.label());
		MPI_Bcast(ts,WORLD.size(),MPI_INT,0,WORLD.label());
		if(WORLD.rank()==0) std::cout<<"tstot = "<<tstot<<"\n";
		int* offsets=new int[WORLD.size()];
		offsets[0]=0;
		if(WORLD.rank()==0) std::cout<<"offset["<<0<<"] = "<<offsets[0]<<"\n";
		for(int n=1; n<WORLD.size(); ++n){
			offsets[n]=offsets[n-1]+ts[n-1];
			if(WORLD.rank()==0) std::cout<<"offset["<<n<<"] = "<<offsets[n]<<"\n";
		}
		MPI_Bcast(offsets,WORLD.size(),MPI_INT,0,WORLD.label());
		double* energytot=new double[tstot];
		for(int t=0; t<tstot; ++t) energytot[t]=0;
		MPI_Gatherv(energyloc,tsloc,MPI_DOUBLE,energytot,ts,offsets,MPI_DOUBLE,0,WORLD.label());
		for(int t=0; t<tstot; ++t){
			if(WORLD.rank()==0) printf("energy/atom %i  = %.12f\n",t,energytot[t]);
		}
		
		delete[] energytot;
		delete[] energyloc;
		delete[] ts;
		delete[] offsets;
    	
		//==== stop the wall clock ====
		if(WORLD.rank()==0) clock.end();
		if(WORLD.rank()==0) std::cout<<"time = "<<clock.duration()<<"\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in nn_pot_compute::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//==== finalize mpi ====
	MPI_Finalize();
	
	
	//==== free memory ====
	delete[] paramfile;
	delete[] input;
	delete[] simstr;
	delete[] potstr;
	delete[] strbuf;
		
}