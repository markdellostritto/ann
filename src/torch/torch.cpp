// omp
#include "thread/openmp.hpp"
// c++
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <random>
// nnp
#include "nnp/type.hpp"
// torch
#include "torch/engine.hpp"
// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// struc
#include "struc/structure.hpp"
#include "analysis/group.hpp"
// format
#include "format/format.hpp"
#include "format/file_struc.hpp"
// str
#include "str/string.hpp"
#include "str/token.hpp"
#include "str/print.hpp"
// torch
#include "torch/job.hpp"
#include "torch/integrator.hpp"
#include "torch/pot_factory.hpp"
#include "torch/monte_carlo.hpp"
#include "torch/dump.hpp"
#include "torch/stochastic.hpp"
#include "torch/set_property_factory.hpp"
// util
#include "util/time.hpp"

int main(int argc, char* argv[]){
	//units
		units::System unitsys=units::System::UNKNOWN;
	//files
		std::string fparam;
		std::string fstruc;
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		Token token;
	//struc
		AtomType atomT;
		Structure struc;
		FILE_FORMAT::type format;
		Eigen::Vector3i nlat;
		bool super=false;
		int nstep=0;
	//property
		std::vector<std::shared_ptr<property::Base> > sprop;
	//job
		Job job;
	//groups
		std::vector<Group> groups;
	//engine
		Engine engine;
		Dump dump;
	//md
		std::shared_ptr<Integrator> intg;
	//mc
		Stochastic stochastic;
	//rand
		std::srand(std::time(NULL));
		int seed=std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
	//time
		Clock clock;
	//misc
		bool error=false;
	
	try{
		//==== check the arguments ====
		if(argc!=2) throw std::invalid_argument("Torch::main(int,char**): Invalid number of arguments.");
		
		//==== omp ====
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("OMP",strbuf)<<"\n";
		#pragma omp parallel
		{if(omp_get_thread_num()==0) std::cout<<"num threads = "<<omp_get_num_threads()<<"\n";}
		std::cout<<print::buf(strbuf)<<"\n";
		
		//==== open the parameter file ==== 
		fparam=argv[1];
		FILE* reader=fopen(fparam.c_str(),"r");
		if(reader==NULL) throw std::runtime_error("Torch::main(int,char**): Could not open parameter file.");
		
		//==== read the parameter file ==== 
		std::cout<<"reading general parameters\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);//trim comments
			Token token(input,string::WS); //split line into tokens
			if(token.end()) continue; //skip empty lines
			std::string tag=string::to_upper(token.next());
			if(tag=="JOB"){
				job=Job::read(string::to_upper(token.next()).c_str());
			} else if(tag=="UNITS"){
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
				units::Consts::init(unitsys);
			} else if(tag=="ATOM_TYPE"){
				atomT=AtomType::read(token);
				if(atomT.posn==false) throw std::invalid_argument("torch::main(int,char**): Atom type missing position.");
				if(atomT.mass==false) throw std::invalid_argument("torch::main(int,char**): Atom type missing mass.");
				if(atomT.index==false) throw std::invalid_argument("torch::main(int,char**): Atom type missing index.");
				if(atomT.type==false) throw std::invalid_argument("torch::main(int,char**): Atom type missing type.");
			} else if(tag=="FORMAT"){//simulation format
				format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
			} else if(tag=="FSTRUC"){
				fstruc=token.next();
				read_struc(fstruc.c_str(),format,atomT,struc);
			} else if(tag=="SUPER"){
				nlat[0]=std::atoi(token.next().c_str());
				nlat[1]=std::atoi(token.next().c_str());
				nlat[2]=std::atoi(token.next().c_str());
				super=true;
			} else if(tag=="GROUP"){
				Group group;
				group.read(token);
				groups.push_back(group);
			} else if(tag=="ENGINE"){
				engine.read(token);
			} else if(tag=="INTEGRATOR"){
				Integrator::read(intg,token);
			} else if(tag=="STOCHASTIC"){
				stochastic.read(token);
			} else if(tag=="POT"){
				std::shared_ptr<ptnl::Pot> pot;
				ptnl::read(pot,token);
				engine.pots().push_back(pot);
			} else if(tag=="PROPERTY"){
				std::shared_ptr<property::Base> prop;
				property::read(prop,token);
				sprop.push_back(prop);
			} else if(tag=="DUMP"){
				dump.read(token);
			} else if(tag=="NSTEP"){
				nstep=std::atoi(token.next().c_str());
			}
		}
		
		//==== print ====
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
		std::cout<<"ke = "<<units::Consts::ke()<<"\n";
		std::cout<<"kb = "<<units::Consts::kb()<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TORCH",strbuf)<<"\n";
		std::cout<<"JOB    = "<<job<<"\n";
		std::cout<<"UNITS  = "<<unitsys<<"\n";
		std::cout<<"ATOMT  = "<<atomT<<"\n";
		std::cout<<"NSTEP  = "<<nstep<<"\n";
		std::cout<<"FSTRUC = "<<fstruc<<"\n";
		std::cout<<"FORMAT = "<<format<<"\n";
		std::cout<<"DUMP   = "<<dump<<"\n";
		std::cout<<"STOCH  = "<<stochastic<<"\n";
		if(super) std::cout<<"NLAT   = "<<nlat.transpose()<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("GROUPS",strbuf)<<"\n";
		for(int i=0; i<groups.size(); ++i){
			std::cout<<groups[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("PROPERTIES",strbuf)<<"\n";
		for(int i=0; i<sprop.size(); ++i){
			std::cout<<sprop[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		if(intg!=NULL){
			switch(intg->name()){
				case Integrator::Name::QUICKMIN: std::cout<<static_cast<const Quickmin&>(*intg)<<"\n"; break;
				case Integrator::Name::VERLET: std::cout<<static_cast<const Verlet&>(*intg)<<"\n"; break;
				case Integrator::Name::VSCALE: std::cout<<static_cast<const VScale&>(*intg)<<"\n"; break;
				case Integrator::Name::BERENDSEN: std::cout<<static_cast<const Berendsen&>(*intg)<<"\n"; break;
				case Integrator::Name::LANGEVIN: std::cout<<static_cast<const Langevin&>(*intg)<<"\n"; break;
				default: std::cout<<"INTEGRATOR = UNKNOWN\n";
			}
		}
		
		//==== read the structure ====
		std::cout<<"reading the structure\n";
		read_struc(fstruc.c_str(),format,atomT,struc);
		std::cout<<"build the groups\n";
		for(int j=0; j<groups.size(); ++j){
			groups[j].build(struc);
		}
		
		//==== set the indices ====
		std::cout<<"setting the indices\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.index(i)=i;
		}
		
		//==== set initial properties ====
		std::cout<<"setting initial properties\n";
		for(int i=0; i<sprop.size(); ++i){
			//set group
			for(int j=0; j<groups.size(); ++j){
				if(sprop[i]->group().label()==groups[j].label()){
					sprop[i]->group()=groups[j];
					break;
				}
			}
			//set property
			sprop[i]->set(struc);
		}
		
		//==== make supercell ====
		if(super){
			std::cout<<"making supercell\n";
			Structure struc_super;
			Structure::super(struc,struc_super,nlat);
			struc=struc_super;
		}
		
		//==== print structure ====
		std::cout<<struc<<"\n";
		
		//==== resize the engine ====
		std::cout<<"resizing the engine\n";
		int ntypes=-1;
		for(int i=0; i<struc.nAtoms(); ++i){
			if(struc.type(i)>ntypes) ntypes=struc.type(i);
		}
		ntypes++;
		std::cout<<"ntypes = "<<ntypes<<"\n";
		engine.resize(ntypes);
		
		//==== read the coefficients ====
		std::cout<<"reading coefficients\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);//trim comments
			Token token(input,string::WS); //split line into tokens
			if(token.end()) continue; //skip empty lines
			std::string tag=string::to_upper(token.next());
			if(tag=="COEFF"){
				ptnl::coeff(engine.pots(),token);
			} 
		}
		
		//==== initialize the engine ====
		std::cout<<"initializing the engine\n";
		if(intg!=NULL) struc.dt()=intg->dt();
		engine.init();
		std::cout<<engine<<"\n";
		
		//==== close parameter file ==== 
		std::fclose(reader);
		reader=NULL;
		
		//==== compute ==== 
		clock.begin();
		switch(job){
			case Job::SP:{
				std::cout<<"JOB - SP\n";
				std::cout<<"building vlist\n";
				engine.vlist().build(struc);
				std::cout<<"computing energy\n";
				const double energy=engine.compute(struc);
				printf("energy = %.10f\n",energy);
				FILE* writer=fopen("out.dump","w");
				Dump::write(struc,writer);
				fclose(writer); writer=NULL;
			}break;
			case Job::MD:{
				std::cout<<"JOB - MD\n";
				FILE* writer=fopen("out.dump","w");
				if(writer==NULL) throw std::runtime_error("Could not open dump file.");
				printf("N T KE PE TE\n");
				for(int t=0; t<nstep; ++t){
					struc.t()=t;
					if(t%engine.vlist().stride()==0) engine.vlist().build(struc);
					//step
					intg->compute(struc,engine);
					//print
					if(t%dump.nprint()==0) printf("%i %4.5f %4.5f %4.5f %4.5f\n",t,struc.temp(),struc.ke(),struc.pe(),struc.ke()+struc.pe());
					//write
					if(t%dump.nwrite()==0) Dump::write(struc,writer);
				}
				fclose(writer); writer=NULL;
			}break;
			case Job::MC:{
				std::cout<<"JOB - MC\n";
				FILE* writer=fopen("out.dump","w");
				if(writer==NULL) throw std::runtime_error("Could not open dump file.");
				Metropolis& met=stochastic.met();
				met.k()=units::Consts::kb();
				//init
				engine.vlist().build(struc);
				double energy=engine.compute(struc);
				//run
				printf("N NSTEP NACCEPT RATIO ENERGY\n");
				for(int t=0; t<nstep; ++t){
					//mc step
					stochastic.step(struc,engine);
					const double ratio=met.nAccept()/(1.0*met.nStep());
					energy=0;
					//compute energy/force
					if(t%dump.nwrite()==0) energy=engine.compute(struc);
					else energy=engine.energy(struc);
					//write
					if(t%dump.nwrite()==0) Dump::write(struc,writer);
					//print
					if(t%dump.nprint()==0) printf("%i %i %i %4.5f %4.5f\n",t,met.nStep(),met.nAccept(),ratio,energy);
				}
				//close the file
				fclose(writer);
			}break;
			default:{
				std::cout<<"WARNING: Invalid job.";
			}break;
		}
		clock.end();
		std::cout<<"time = "<<clock.duration()<<"\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in Torch::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free memory
	std::cout<<"freeing memory\n";
	delete[] input;
	delete[] strbuf;
	
	if(error) return 1;
	else return 0;
}