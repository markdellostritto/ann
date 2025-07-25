// mpi
#include <mpi.h>
// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <iostream>
#include <exception>
#include <algorithm>
// structure
#include "struc/structure.hpp"
#include "struc/neighbor.hpp"
// format
#include "format/file_struc.hpp"
#include "format/format.hpp"
#include "format/vasp_struc.hpp"
#include "format/qe_struc.hpp"
#include "format/xyz_struc.hpp"
#include "format/cp2k_struc.hpp"
// chem
#include "chem/units.hpp"
// ml
#include "ml/nn.hpp"
#include "nnp/nnp.hpp"
// string
#include "str/string.hpp"
#include "str/token.hpp"
#include "str/print.hpp"
// thread
#include "thread/comm.hpp"
#include "thread/dist.hpp"
#include "thread/mpif.hpp"
// util
#include "util/compiler.hpp"
#include "util/time.hpp"

thread::Comm WORLD;//all processors

int main(int argc, char* argv[]){
	//==== file i/o ====
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input    =new char[string::M];
		char* simstr   =new char[string::M];
		char* strbuf   =new char[print::len_buf];
		std::string potstr;
		Token token;
	//==== simulation variables ====
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.symm=true; atomT.charge=false;
		FILE_FORMAT::type format;//format of training data
		std::vector<Structure> strucs;
		std::vector<std::string> data;
		std::vector<std::string> files;
		bool calc_force=false;
		bool norm=false;
	//==== nn potential ====
		NNP nnp;
		bool force=false;
	//==== timing ====
		Clock clock;
	//==== units ====
		units::System unitsys=units::System::UNKNOWN;
	//==== thread ====
		thread::Dist dist;
	
	//==== initialize mpi ====
	MPI_Init(&argc,&argv);
	WORLD.mpic()=MPI_COMM_WORLD;
	MPI_Comm_size(WORLD.mpic(),&WORLD.size());
	MPI_Comm_rank(WORLD.mpic(),&WORLD.rank());
	
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
			std::printf("PI     = %.15f\n",math::constant::PI);
			std::printf("RadPI  = %.15f\n",math::constant::RadPI);
			std::printf("Rad2   = %.15f\n",math::constant::Rad2);
			std::printf("Log2   = %.15f\n",math::constant::LOG2);
			std::printf("Eps<D> = %.15e\n",std::numeric_limits<double>::epsilon());
			std::printf("Min<D> = %.15e\n",std::numeric_limits<double>::min());
			std::printf("Max<D> = %.15e\n",std::numeric_limits<double>::max());
			std::printf("Min<I> = %i\n",std::numeric_limits<int>::min());
			std::printf("Max<I> = %i\n",std::numeric_limits<int>::max());
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//==== print physical constants ====
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
			std::printf("alpha        = %.12f\n",units::ALPHA);
			std::printf("p/e mass     = %.12f\n",units::MPoME);
			std::printf("bohr-r  (A)  = %.12f\n",units::Bohr2Ang);
			std::printf("hartree (eV) = %.12f\n",units::Eh2Ev);
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//==== set mpi data ====
		{
			int* ranks=new int[WORLD.size()];
			MPI_Gather(&WORLD.rank(),1,MPI_INT,ranks,1,MPI_INT,0,WORLD.mpic());
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
				token.read(string::trim_right(input,string::COMMENT),string::WS);
				if(token.end()) continue; //skip empty lines
				const std::string tag=string::to_upper(token.next());
				if(tag=="DATA"){
					data.push_back(token.next());
				} else if(tag=="NNPOT"){
					potstr=token.next();
				} else if(tag=="FORMAT"){
					format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
				} else if(tag=="UNITS"){
					unitsys=units::System::read(string::to_upper(token.next()).c_str());
				} else if(tag=="FORCE"){
					force=string::boolean(token.next().c_str());
				} else if(tag=="NORM"){
					norm=string::boolean(token.next().c_str());
				}
			}
			//close the file
			fclose(reader);
			reader=NULL;
			
			//==== print the parameters ====
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
			std::cout<<"DATA   = \n"; for(int i=0; i<data.size(); ++i) std::cout<<"\t"<<data[i]<<"\n";
			std::cout<<"FORMAT = "<<format<<"\n";
			std::cout<<"UNITS  = "<<unitsys<<"\n";
			std::cout<<"FORCE  = "<<force<<"\n";
			std::cout<<"NORM   = "<<norm<<"\n";
			std::cout<<"NNPOT  = "<<potstr<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
			
			//==== check the parameters ====
			if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
			if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
			if(data.size()==0) throw std::invalid_argument("No data");
		
		}
		
		//==== broadcast the parameters ====
		if(WORLD.rank()==0) std::cout<<"broadcasting parameters\n";
		thread::bcast(WORLD.mpic(),0,potstr);
		MPI_Bcast(&unitsys,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&format,1,MPI_INT,0,WORLD.mpic());
		MPI_Bcast(&force,1,MPI_C_BOOL,0,WORLD.mpic());
		MPI_Bcast(&norm,1,MPI_C_BOOL,0,WORLD.mpic());
		
		//==== initialize the unit system ====
		if(WORLD.rank()==0) std::cout<<"initializing the unit system\n";
		units::consts::init(unitsys);
		
		//==== read the data ====
		if(WORLD.rank()==0){
			std::cout<<"reading data\n";
			for(int i=0; i<data.size(); ++i){
				//open the data file
				reader=fopen(data[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open data file: ")+data[i]);
				//read in the data
				while(fgets(input,string::M,reader)!=NULL){
					if(!string::empty(input)) files.push_back(std::string(string::trim(input)));
					std::cout<<files.back()<<"\n";
				}
				//close the file
				fclose(reader);
				reader=NULL;
			}
		}
		
		//==== broadcast the files ====
		thread::bcast(WORLD.mpic(),0,files);
		//==== generate the distribution over the files
		dist.init(WORLD.size(),WORLD.rank(),files.size());
		
		//==== read the structure ====
		if(WORLD.rank()==0) std::cout<<"reading structures\n";
		if(dist.size()>0){
			strucs.resize(dist.size());
			if(format==FILE_FORMAT::QE){
				for(int i=0; i<dist.size(); ++i){
					QE::OUT::read(files[dist.index(i)].c_str(),atomT,strucs[i]);
				}
			} else if(format==FILE_FORMAT::POSCAR){
				for(int i=0; i<dist.size(); ++i){
					VASP::POSCAR::read(files[dist.index(i)].c_str(),atomT,strucs[i]);
				}
			} else if(format==FILE_FORMAT::XYZ){
				for(int i=0; i<dist.size(); ++i){
					XYZ::read(files[dist.index(i)].c_str(),atomT,strucs[i]);
				}
			} else throw std::invalid_argument("Invalid file format.");
		}
		
		//==== read the potential ====
		if(WORLD.rank()==0) std::cout<<"reading potential\n";
		NNP::read(potstr.c_str(),nnp);
		if(WORLD.rank()==0) std::cout<<nnp<<"\n";
		
		//==== set atom properties ====
		if(WORLD.rank()==0) std::cout<<"setting atom properties\n";
		//index
		if(WORLD.rank()==0) std::cout<<"index\n";
		for(int i=0; i<strucs.size(); ++i){
			for(int n=0; n<strucs[i].nAtoms(); ++n){
				strucs[i].index(n)=n;
			}
		}
		//type
		if(WORLD.rank()==0) std::cout<<"set the type\n";
		for(int i=0; i<strucs.size(); ++i){
			for(int n=0; n<strucs[i].nAtoms(); ++n){
				strucs[i].type(n)=nnp.index(strucs[i].name(n));
			}
		}
		
		//==== compute ====
		if(WORLD.rank()==0) std::cout<<"initializing symmetry functions\n";
		for(int i=0; i<strucs.size(); ++i){
			NNP::init(nnp,strucs[i]);
		}
		if(WORLD.rank()==0) std::cout<<"computing\n";
		std::vector<double> energy_ref(strucs.size(),0.0);
		std::vector<double> energy_nnp(strucs.size(),0.0);
		for(int i=0; i<strucs.size(); ++i){
			energy_ref[i]=strucs[i].energy();
			NeighborList nlist;
			nlist.build(strucs[i],nnp.rc());
			NNP::symm(nnp,strucs[i],nlist);
			NNP::energy(nnp,strucs[i]);
			if(force) NNP::force(nnp,strucs[i],nlist);
			energy_nnp[i]=strucs[i].energy();
		}
		
		//==== print ====
		if(!norm){
			for(int n=0; n<WORLD.size(); ++n){
				if(n==WORLD.rank()){
					for(int i=0; i<strucs.size(); ++i){
						printf("energy %i %.6f %.6f\n",dist.index(i),energy_ref[i],energy_nnp[i]);
					}
				}
				MPI_Barrier(WORLD.mpic());
			}
		} else {
			for(int n=0; n<WORLD.size(); ++n){
				if(n==WORLD.rank()){
					for(int i=0; i<strucs.size(); ++i){
						printf("energy/atom %i %.6f %.6f\n",dist.index(i),energy_ref[i]/strucs[i].nAtoms(),energy_nnp[i]/strucs[i].nAtoms());
					}
				}
				MPI_Barrier(WORLD.mpic());
			}
		}
		if(force){
			for(int n=0; n<WORLD.size(); ++n){
				if(n==WORLD.rank()){
					for(int i=0; i<strucs.size(); ++i){
						std::cout<<"struc["<<dist.index(i)<<"]\n";
						for(int j=0; j<strucs[i].nAtoms(); ++j){
							std::cout<<"force["<<i<<"] = "<<strucs[i].force(j).transpose()<<"\n";
						}
					}
				}
				MPI_Barrier(WORLD.mpic());
			}
		}
		
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
	delete[] strbuf;
		
}