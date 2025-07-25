// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <string>
#include <stdexcept>
#include <iostream>
// str
#include "str/string.hpp"
// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"
// format
#include "format/cp2k_sim.hpp"

namespace CP2K{

//*****************************************************
//FORMAT struct
//*****************************************************

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-xyz"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-xdatcar\" option.");
			else format.xyz=strlist[i+1];
		} else if(strlist[i]=="-inp"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-xdatcar\" option.");
			else format.input=strlist[i+1];
		}
	}
	return format;
}

//*****************************************************
//reading
//*****************************************************

Simulation& read(const Format& format, const Interval& interval, const AtomType& atomT, Simulation& sim){
	if(CP2K_PRINT_FUNC>0) std::cout<<"read(const Format&,const Interval&,const AtomType&,Simulation&):\n";
	char* input=new char[string::M];
	char* name=new char[string::M];
	FILE* reader=NULL;
	Eigen::Matrix3d lv;
	Eigen::Vector3d v;
	
	if(CP2K_PRINT_STATUS>0) std::cout<<"opening xyz file\n";
	reader=fopen(format.xyz.c_str(),"r");
	if(reader==NULL) throw std::runtime_error("Unable to open xyz file.");
	
	//read natoms
	if(CP2K_PRINT_STATUS>0) std::cout<<"reading natoms\n";
	fgets(input,string::M,reader);
	const int natoms=std::atoi(input);
	if(natoms<=0) throw std::runtime_error("Runtime Error: found zero atoms.");
	if(CP2K_PRINT_DATA>0) std::cout<<"natoms = "<<natoms<<"\n";
	
	//find the total number of timesteps
	if(CP2K_PRINT_STATUS>0) std::cout<<"reading timesteps\n";
	std::rewind(reader);
	int nlines=0;
	while(fgets(input,string::M,reader)) ++nlines;
	const int ts=nlines/(natoms+2);//natoms + natoms-line + comment-line
	if(CP2K_PRINT_DATA>0) std::cout<<"ts = "<<ts<<"\n";
	
	//set the interval
	if(CP2K_PRINT_STATUS>0) std::cout<<"setting interval\n";
	if(interval.beg()<0) throw std::invalid_argument("Invalid beginning timestep.");
	const int beg=interval.beg()-1;
	int end=interval.end()-1;
	if(interval.end()<0) end=ts+interval.end();
	const int tsint=end-beg+1;
	if(CP2K_PRINT_DATA>0) std::cout<<"interval = "<<beg<<":"<<end<<":"<<tsint<<"\n";
	
	//resize the simulation
	if(CP2K_PRINT_STATUS>0) std::cout<<"resizing simulation\n";
	sim.resize(tsint/interval.stride(),natoms,atomT);
	
	//read the positions
	if(CP2K_PRINT_STATUS>0) std::cout<<"reading position\n";
	std::rewind(reader);
	for(int t=0; t<beg; ++t){
		fgets(input,string::M,reader);//natoms
		fgets(input,string::M,reader);//comment line
		for(int n=0; n<natoms; ++n){
			fgets(input,string::M,reader);
		}
	}
	for(int t=0; t<sim.timesteps(); ++t){
		//read natoms
		fgets(input,string::M,reader);//natoms
		const int N=std::atoi(input);
		if(sim.frame(t).nAtoms()!=N) throw std::invalid_argument("Invalid number of atoms.");
		//read comment
		fgets(input,string::M,reader);
		//read positions
		for(int n=0; n<natoms; ++n){
			fgets(input,string::M,reader);
			std::sscanf(input,"%s %lf %lf %lf",name,&v[0],&v[1],&v[2]);
			if(atomT.name) sim.frame(t).name(n)=name;
			sim.frame(t).posn(n).noalias()=v;
		}
		//skip "stride-1" steps
		for(int tt=0; tt<interval.stride()-1; ++tt){
			fgets(input,string::M,reader);//natoms
			fgets(input,string::M,reader);//comment line
			for(int n=0; n<natoms; ++n) fgets(input,string::M,reader);
		}
	}
	
	//close the file
	fclose(reader);
	
	if(CP2K_PRINT_STATUS>0) std::cout<<"opening input file\n";
	reader=fopen(format.input.c_str(),"r");
	if(reader==NULL) throw std::runtime_error("Unable to open input file.");
	
	if(CP2K_PRINT_STATUS>0) std::cout<<"reading lattice vector\n";
	while(fgets(input,string::M,reader)){
		/*if(std::strstr(input,"ABC")!=NULL){
			std::vector<std::string> strlist;
			string::split(input,string::WS,strlist);
			lv=Eigen::Matrix3d::Zero();
			lv(2,2)=std::atof(strlist[strlist.size()-1].c_str());
			lv(1,1)=std::atof(strlist[strlist.size()-2].c_str());
			lv(0,0)=std::atof(strlist[strlist.size()-3].c_str());
			break;
		}*/
		if(std::strstr(input,"ABC")!=NULL){
			Token token(input,string::WS);
			std::vector<std::string> tokens;
			while(!token.end()) tokens.push_back(token.next());
			lv=Eigen::Matrix3d::Zero();
			lv(2,2)=std::atof(tokens[tokens.size()-1].c_str());
			lv(1,1)=std::atof(tokens[tokens.size()-2].c_str());
			lv(0,0)=std::atof(tokens[tokens.size()-3].c_str());
			break;
		}
	}
	if(CP2K_PRINT_DATA>0) std::cout<<"lv = \n"<<lv<<"\n";
	for(int t=0; t<sim.timesteps(); ++t){
		static_cast<Cell&>(sim.frame(t)).init(lv);
	}
	
	//close the file
	fclose(reader);
	
	if(CP2K_PRINT_STATUS>0) std::cout<<"return to cell\n";
	for(int t=0; t<sim.timesteps(); ++t){
		for(int n=0; n<sim.frame(t).nAtoms(); ++n){
			Cell::returnToCell(sim.frame(t).posn(n),sim.frame(t).posn(n),sim.frame(t).R(),sim.frame(t).RInv());
		}
	}
	
	//read forces
	if(atomT.force && format.fxyz.size()>0){
		std::cout<<"fxyz = "<<format.fxyz<<"\n";
		if(CP2K_PRINT_STATUS>0) std::cout<<"opening fxyz file\n";
		reader=fopen(format.fxyz.c_str(),"r");
		if(reader==NULL) throw std::runtime_error("Unable to open fxyz file.");
		
		//read the forces
		if(CP2K_PRINT_STATUS>0) std::cout<<"reading forces\n";
		for(int t=0; t<beg; ++t){
			fgets(input,string::M,reader);//natoms
			fgets(input,string::M,reader);//comment line
			for(int n=0; n<natoms; ++n){
				fgets(input,string::M,reader);
			}
		}
		for(int t=0; t<sim.timesteps(); ++t){
			//read natoms
			fgets(input,string::M,reader);//natoms
			const int N=std::atoi(input);
			if(sim.frame(t).nAtoms()!=N) throw std::invalid_argument("Invalid number of atoms.");
			//read comment
			fgets(input,string::M,reader);
			//read positions
			for(int n=0; n<natoms; ++n){
				fgets(input,string::M,reader);
				std::sscanf(input,"%s %lf %lf %lf",name,&v[0],&v[1],&v[2]);
				if(atomT.name) sim.frame(t).name(n)=name;
				sim.frame(t).force(n).noalias()=v;
			}
			//skip "stride-1" steps
			for(int tt=0; tt<interval.stride()-1; ++tt){
				fgets(input,string::M,reader);//natoms
				fgets(input,string::M,reader);//comment line
				for(int n=0; n<natoms; ++n) fgets(input,string::M,reader);
			}
		}
		
		//close the file
		fclose(reader);
	}
	
	if(CP2K_PRINT_STATUS>0) std::cout<<"simulation read\n";
	
	delete[] input;
	delete[] name;
	
	return sim;
}
	
//*****************************************************
//writing
//*****************************************************

/*const Simulation& write(const Format& format, const Interval& interval, const AtomType& atomT, const Simulation& sim){
	if(CP2K_PRINT_FUNC>0) std::cout<<"CP2K::write(const Format&,const Interval&,const AtomType&,const Simulation&):\n";
}*/

}