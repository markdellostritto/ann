// c libraries
#include <cstdlib>
#include <cstdio>
// c++ libraries
#include <iostream>
// io
#include "str/token.hpp"
#include "str/parse.hpp"
#include "str/string.hpp"
// structure
#include "struc/sim.hpp"
// format
#include "format/file_sim.hpp"
#include "format/format.hpp"
// math
#include "math/const.hpp"

int main(int argc, char* argv[]){
	//simulation
		Simulation sim;
		Interval interval_in(1,-1,1);
		Interval interval_out(1,-1,1);
		FILE_FORMAT::type format_in;
		FILE_FORMAT::type format_out;
		std::vector<std::string> names;
	//file i/o
		std::string file_in;
		std::string file_out;
	//arguments
		std::vector<input::Arg> args;
	//atom type
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.frac=true; atomT.image=true;
	//offset
		std::string offset_str;
		Eigen::Vector3d offset=Eigen::Vector3d::Zero();
	//misc
		bool unwrap=false;
		bool error=false;
		
	try{
		//check the number of arguments
		if(argc==1) throw std::invalid_argument("No arguments provided.");
		
		//parse the arguments
		input::parse(argc,argv,args);
		for(int i=0; i<args.size(); ++i){
			if(args[i].key()=="fin") format_in=FILE_FORMAT::read(args[i].val(0).c_str());
			else if(args[i].key()=="fout") format_out=FILE_FORMAT::read(args[i].val(0).c_str());
			else if(args[i].key()=="in") file_in=args[i].val(0).c_str();
			else if(args[i].key()=="out") file_out=args[i].val(0).c_str();
			else if(args[i].key()=="cart") atomT.frac=false;
			else if(args[i].key()=="frac") atomT.frac=true;
			else if(args[i].key()=="interval") Interval::read(args[i].val(0).c_str(),interval_in);
			else if(args[i].key()=="offset") offset_str=args[i].val(0);
			else if(args[i].key()=="unwrap") unwrap=true;
			else if(args[i].key()=="names") names=args[i].vals();
		}
		
		//read offset
		std::cout<<"reading offset\n";
		if(!offset_str.empty()){
			Token token(offset_str.c_str(),":");
			offset[0]=std::atof(token.next().c_str());
			offset[1]=std::atof(token.next().c_str());
			offset[2]=std::atof(token.next().c_str());
		}
		
		//print parameters
		std::cout<<"format-in  = "<<format_in<<"\n";
		std::cout<<"format-out = "<<format_out<<"\n";
		std::cout<<"file-in    = "<<file_in<<"\n";
		std::cout<<"file-out   = "<<file_out<<"\n";
		std::cout<<"interval   = "<<interval_in<<"\n";
		std::cout<<"offset     = "<<offset.transpose()<<"\n";
		std::cout<<"unwrap     = "<<unwrap<<"\n";
		if(names.size()>0){
			std::cout<<"names      = ";
			for(int i=0; i<names.size(); ++i) std::cout<<names[i]<<" ";
			std::cout<<"\n";
		}
		
		//check parameters
		if(format_in==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid input format.");
		if(format_out==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid output format.");
		
		//read
		std::cout<<"reading structure\n";
		read_sim(file_in.c_str(),format_in,interval_in,atomT,sim);
		std::cout<<"reading completed\n";
		
		//apply offset
		if(offset.norm()>1e-6){
			std::cout<<"setting offset\n";
			Eigen::Vector3d r;
			for(int t=0; t<sim.timesteps(); ++t){
				for(int n=0; n<sim.frame(t).nAtoms(); ++n){
					sim.frame(t).posn(n).noalias()+=offset;
					sim.frame(t).modv(sim.frame(t).posn(n),r);
					sim.frame(t).posn(n)=r;
				}
			}
		}
		
		//apply names
		if(names.size()>0){
			for(int t=0; t<sim.timesteps(); ++t){
				for(int n=0; n<sim.frame(t).nAtoms(); ++n){
					if(sim.frame(t).type(n)<names.size()){
						sim.frame(t).name(n)=names[sim.frame(t).type(n)];
					}
				}
			}
		}
		
		//unwrap
		if(unwrap){
			std::cout<<"unwrapping\n";
			Simulation::set_image(sim);
			Simulation::unwrap(sim);
		}
		
		//write
		std::cout<<"writing structure\n";
		write_sim(file_out.c_str(),format_out,interval_out,atomT,sim);
		std::cout<<"writing completed\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in convert_sim::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) return 1;
	else return 0;
}