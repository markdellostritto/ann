//c libraries
#include <cstdlib>
#include <cstdio>
//c++ libraries
#include <iostream>
//ann - structure
#include "structure.hpp"
// ann - file i/o
#include "vasp.hpp"
#include "qe.hpp"
#include "ame.hpp"
// ann - math
#include "math_const.hpp"
// ann - string
#include "string.hpp"

#ifndef DEBUG_CONVERT
#define DEBUG_CONVERT 1
#endif

int main(int argc, char* argv[]){
	//simulation
		Simulation sim;
		FILE_FORMAT::type formatIn;
		FILE_FORMAT::type formatOut;
	//formats
		QE::Format formatQE;
		VASP::Format formatVASP;
		AME::Format formatAME;
	//interval
		Interval interval;
	//offset
		Eigen::Vector3d offset=Eigen::Vector3d::Zero();
	//file i/o
		std::string out;
	//arguments
		std::vector<std::string> strlist;
	//atom type
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.frac=true;
		sim.atomT()=atomT;
	//misc
		char* temp=(char*)malloc(sizeof(char)*500);
		bool error=false;
		bool sep=false;
		bool direct=true;
	
	try{
		//check the number of arguments
		if(argc==1) throw std::invalid_argument("No arguments provided.");
		
		//load the arguments
		strlist.resize(argc-1);
		for(unsigned int i=1; i<argc; ++i) strlist[i-1]=std::string(argv[i]);
		
		//read in the formats
		for(unsigned int i=0; i<strlist.size(); ++i){
			if(strlist[i]=="-format-in"){
				if(i==strlist.size()-1) throw std::invalid_argument("No input format provided.");
				else formatIn=FILE_FORMAT::read(strlist[i+1]);
			} else if(strlist[i]=="-format-out"){
				if(i==strlist.size()-1) throw std::invalid_argument("No output format provided.");
				else formatOut=FILE_FORMAT::read(strlist[i+1]);
			} else if(strlist[i]=="-out"){
				if(i==strlist.size()-1) throw std::invalid_argument("No output file provided.");
				else out=strlist[i+1];
			} else if(strlist[i]=="-cart"){
				atomT.frac=false;
			} else if(strlist[i]=="-frac"){
				atomT.frac=true;
			} else if(strlist[i]=="-ts"){
				if(i==strlist.size()-1) throw std::invalid_argument("No timestep provided.");
				else interval=Interval(std::atoi(strlist[i+1].c_str()),std::atoi(strlist[i+1].c_str()),1);
			} else if(strlist[i]=="-sep"){
				sep=true;
			} else if(strlist[i]=="-offset"){
				if(i==strlist.size()-1) throw std::invalid_argument("No offset provided.");
				std::vector<std::string> templist;
				string::split(strlist[i+1].c_str(),":",templist);
				if(templist.size()!=3) throw std::invalid_argument("Invalid offset format.");
				offset[0]=std::atof(templist[0].c_str());
				offset[1]=std::atof(templist[1].c_str());
				offset[2]=std::atof(templist[2].c_str());
			} else if(strlist[i]=="-interval"){
				if(i==strlist.size()-1) throw std::invalid_argument("No interval provided.");
				else interval=Interval::read(strlist[i+1].c_str());
			} 
		}
		
		//print parameters
		std::cout<<"interval   = "<<interval<<"\n";
		std::cout<<"format-in  = "<<formatIn<<"\n";
		std::cout<<"format-out = "<<formatOut<<"\n";
		std::cout<<"out        = "<<out<<"\n";
		std::cout<<"direct     = "<<direct<<"\n";
		std::cout<<"offset     = "<<offset.transpose()<<"\n";
		
		if(formatIn==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid input format.");
		if(formatOut==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid output format.");
		
		//read
		std::cout<<"Reading simulation...\n";
		if(formatIn==FILE_FORMAT::QE){
			if(DEBUG_CONVERT>0) std::cout<<"Reading QE...\n";
			QE::Format::read(strlist,formatQE);
			QE::read(formatQE,interval,atomT,sim);
		} else if(formatIn==FILE_FORMAT::XDATCAR || formatIn==FILE_FORMAT::POSCAR || formatIn==FILE_FORMAT::VASP_XML){
			if(DEBUG_CONVERT>0) std::cout<<"Reading VASP...\n";
			VASP::Format::read(strlist,formatVASP);
			VASP::read(formatVASP,interval,atomT,sim);
		} else if(formatIn==FILE_FORMAT::AME){
			if(DEBUG_CONVERT>0) std::cout<<"Reading AME...\n";
			AME::Format::read(strlist,formatAME);
			AME::read(formatAME.ame.c_str(),interval,atomT,sim);
		}
		
		//print the simulation
		std::cout<<"SIM = \n"<<sim<<"\n";
		
		//apply the offset
		if(offset.norm()>num_const::ZERO){
			std::cout<<"Applying the offset...\n";
			for(unsigned int t=0; t<sim.timesteps(); ++t){
				for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n){
					sim.frame(t).posn(n).noalias()+=offset;
					Cell::returnToCell(sim.frame(t).posn(n),sim.frame(t).posn(n),sim.frame(t).R(),sim.frame(t).RInv());
				}
			}
		}
		
		//write
		std::cout<<"Writing simulation...\n";
		if(formatOut==FILE_FORMAT::XDATCAR){
			if(DEBUG_CONVERT>0) std::cout<<"Writing XDATCAR...\n";
			if(!sep) VASP::XDATCAR::write(out.c_str(),Interval(0,-1,1),atomT,sim);
			else {
				for(unsigned int t=0; t<sim.timesteps(); ++t){
					std::string file=out.substr(0,out.find("."))+std::string("_")+std::to_string(t)+std::string(".vasp");
					VASP::POSCAR::write(file.c_str(),atomT,sim.frame(t));
				}
			}
		} else if(formatOut==FILE_FORMAT::POSCAR){
			if(DEBUG_CONVERT>0) std::cout<<"Writing POSCAR...\n";
			VASP::POSCAR::write(out.c_str(),atomT,sim.frame(interval.beg));
		} else if(formatOut==FILE_FORMAT::AME){
			if(DEBUG_CONVERT>0) std::cout<<"Writing AME...\n";
			if(!sep) AME::write(out.c_str(),Interval(0,-1,1),atomT,sim);
			else{
				for(unsigned int t=0; t<sim.timesteps(); ++t){
					std::string file=out.substr(0,out.find("."))+std::string("_")+std::to_string(t)+std::string(".ame");
					AME::write(file.c_str(),atomT,sim.frame(t));
				}
			}
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in convert::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	free(temp);
	
	if(error) return 1;
	else return 0;
}