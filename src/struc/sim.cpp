//c++ libraries
#include <iostream>
// str
#include "str/print.hpp"
#include "str/token.hpp"
// sim
#include "struc/sim.hpp"

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Simulation& sim){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("SIMULATION",str)<<"\n";
	out<<"SIM   = "<<sim.name_<<"\n";
	out<<"TS    = "<<sim.timestep_<<"\n";
	out<<"T     = "<<sim.timesteps_<<"\n";
	out<<"AtomT = "<<sim.atomT_<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

void Simulation::defaults(){
	if(SIM_PRINT_FUNC>0) std::cout<<"Simulation::defaults():\n";
	name_=std::string("SYSTEM");
	timestep_=0;
	timesteps_=0;
}

void Simulation::clear(){
	if(SIM_PRINT_FUNC>0) std::cout<<"Simulation::clear():\n";
	frames_.clear();
	defaults();
}

void Simulation::resize(int ts, int nAtoms, const AtomType& atomT){
	if(SIM_PRINT_FUNC>0) std::cout<<"Simulation::resize(int,const std::vector<int>&,const std::vector<std::string>&,const AtomType&):\n";
	timesteps_=ts;
	atomT_=atomT;
	frames_.resize(timesteps_,Structure(nAtoms,atomT));
}

void Simulation::resize(int ts){
	if(SIM_PRINT_FUNC>0) std::cout<<"Simulation::resize(int):\n";
	timesteps_=ts;
	frames_.resize(timesteps_);
}

//==== static functions ====

void Simulation::set_image(Simulation & sim){
	for(int t=1; t<sim.timesteps(); ++t){
		for(int n=0; n<sim.frame(t).nAtoms(); ++n){
			const Eigen::Vector3d diff=sim.frame(t).RInv()*(sim.frame(t).posn(n)-sim.frame(t-1).posn(n));
			sim.frame(t).image(n)=sim.frame(t-1).image(n);
			if(diff[0]>0.5) sim.frame(t).image(n)[0]--;
			else if(diff[0]<-0.5) sim.frame(t).image(n)[0]++;
			if(diff[1]>0.5) sim.frame(t).image(n)[1]--;
			else if(diff[1]<-0.5) sim.frame(t).image(n)[1]++;
			if(diff[2]>0.5) sim.frame(t).image(n)[2]--;
			else if(diff[2]<-0.5) sim.frame(t).image(n)[2]++;
		}
	}
}

void Simulation::unwrap(Simulation & sim){
	for(int t=0; t<sim.timesteps(); ++t){
		for(int n=0; n<sim.frame(t).nAtoms(); ++n){
			sim.frame(t).posn(n).noalias()+=
				sim.frame(t).R().col(0)*sim.frame(t).image(n)[0]+
				sim.frame(t).R().col(1)*sim.frame(t).image(n)[1]+
				sim.frame(t).R().col(2)*sim.frame(t).image(n)[2];
			const Eigen::Vector3d offset=sim.frame(t).R().col(0)*sim.frame(t).image(n)[0]+
				sim.frame(t).R().col(1)*sim.frame(t).image(n)[1]+
				sim.frame(t).R().col(2)*sim.frame(t).image(n)[2];
			sim.frame(t).image(n).setZero();
		}
	}
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Simulation& sim){
		if(SIM_PRINT_FUNC>0) std::cout<<"nbytes(const Simulation&):\n";
		int size=0;
		size+=sizeof(int);//timesteps_
		size+=sizeof(int);//natoms
		size+=sizeof(double);//timestep
		size+=nbytes(sim.atomT());//atomT
		size+=nbytes(sim.name());//name
		for(int t=0; t<sim.timesteps(); ++t){
			size+=nbytes(sim.frame(t));
		}
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Simulation& sim, char* arr){
		if(SIM_PRINT_FUNC>0) std::cout<<"pack(const Simulation&,char*):\n";
		int pos=0,tmpInt=0;
		std::memcpy(arr+pos,&(tmpInt=sim.timesteps()),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&(tmpInt=sim.frame(0).nAtoms()),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&sim.timestep(),sizeof(double)); pos+=sizeof(double);
		pos+=pack(sim.atomT(),arr);
		pos+=pack(sim.name(),arr);
		for(int t=0; t<sim.timesteps(); ++t){
			pos+=pack(sim.frame(t),arr);
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Simulation& sim, const char* arr){
		if(SIM_PRINT_FUNC>0) std::cout<<"unpack(Simulation&,const char*):\n";
		int pos=0,nAtoms=0,ts=0;
		AtomType atomT;
		std::memcpy(&ts,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&nAtoms,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&sim.timestep(),arr+pos,sizeof(double)); pos+=sizeof(double);
		pos+=unpack(atomT,arr);
		pos+=unpack(sim.name(),arr);
		sim.resize(ts,nAtoms,atomT);
		for(int t=0; t<sim.timesteps(); ++t){
			pos+=unpack(sim.frame(t),arr);
		}
		return pos;
	}
	
}