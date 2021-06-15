//c++ libraries
#include <iostream>
// ann - print
#include "print.hpp"
// ann - sim
#include "sim.hpp"

//**********************************************************************************************
//Interval
//**********************************************************************************************

std::ostream& operator<<(std::ostream& out, const Interval& i){
	return out<<i.beg<<":"<<i.end<<":"<<i.stride;
}

Interval Interval::read(const char* str){
	std::vector<std::string> strlist;
	string::split(str,":",strlist);
	if(strlist.size()!=2 && strlist.size()!=3) throw std::invalid_argument("Interval::read(const char*): Invalid interval format.");
	Interval interval;
	if(string::to_upper(strlist.at(0))=="BEG") interval.beg=1;
	else interval.beg=std::atoi(strlist.at(0).c_str());
	if(string::to_upper(strlist.at(1))=="END") interval.end=-1;
	else interval.end=std::atoi(strlist.at(1).c_str());
	if(strlist.size()==3){
		interval.stride=std::atoi(strlist.at(2).c_str());
	}
	return interval;
}

Interval Interval::split(const Interval& interval, int rank, int nprocs){
	//const int ts=((interval.end-interval.beg)+1)/interval.stride;
	const int ts=((interval.end-interval.beg)+1);
	int ts_loc=ts/nprocs;
	int beg_loc=ts_loc*(rank)+1;
	int end_loc=ts_loc*(rank+1);
	if(rank<ts%nprocs){
		ts_loc++;
		beg_loc+=rank;
		end_loc+=rank+1;
	} else {
		beg_loc+=ts%nprocs;
		end_loc+=ts%nprocs;
	}
	Interval newint(beg_loc,end_loc,interval.stride);
	return newint;
}

//**********************************************
// Simulation
//**********************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Simulation& sim){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("SIMULATION",str)<<"\n";
	out<<"SIM   = "<<sim.name_<<"\n";
	out<<"TS    = "<<sim.timestep_<<"\n";
	out<<"T     = "<<sim.timesteps_<<"\n";
	out<<"CF    = "<<sim.cell_fixed_<<"\n";
	out<<"AtomT = "<<sim.atomT_<<"\n";
	out<<print::title("SIMULATION",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

void Simulation::defaults(){
	if(DEBUG_SIM>0) std::cout<<"Simulation::defaults():\n";
	name_=std::string("SYSTEM");
	timestep_=0;
	timesteps_=0;
}

void Simulation::clear(){
	if(DEBUG_SIM>0) std::cout<<"Simulation::clear():\n";
	frames_.clear();
	defaults();
}

void Simulation::resize(int ts, int nAtoms, const AtomType& atomT){
	if(DEBUG_SIM>0) std::cout<<"Simulation::resize(int,const std::vector<int>&,const std::vector<std::string>&,const AtomType&):\n";
	timesteps_=ts;
	atomT_=atomT;
	frames_.resize(timesteps_,Structure(nAtoms,atomT));
}

void Simulation::resize(int ts){
	if(DEBUG_SIM>0) std::cout<<"Simulation::resize(int):\n";
	timesteps_=ts;
	frames_.resize(timesteps_);
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Interval& obj){
		return 3*sizeof(int);
	}
	template <> int nbytes(const Simulation& sim){
		int size=0;
		size+=sizeof(int);//timesteps_
		size+=sizeof(int);//beg
		size+=sizeof(int);//end
		size+=sizeof(int);//stride
		size+=sizeof(int);//natoms
		size+=sizeof(double);//timestep
		size+=sizeof(bool);//cell_fixed_
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
	
	template <> int pack(const Interval& obj, char* arr){
		int pos=0;
		std::memcpy(arr+pos,&obj.beg,sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.end,sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.stride,sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	template <> int pack(const Simulation& sim, char* arr){
		int pos=0,tempInt=0;
		std::memcpy(arr+pos,&sim.timesteps(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&sim.beg(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&sim.end(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&sim.stride(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&(tempInt=sim.frame(0).nAtoms()),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&sim.timestep(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&sim.cell_fixed(),sizeof(bool)); pos+=sizeof(bool);
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
	
	template <> int unpack(Interval& obj, const char* arr){
		int pos=0;
		std::memcpy(&obj.beg,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.end,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.stride,arr+pos,sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	template <> int unpack(Simulation& sim, const char* arr){
		int pos=0,nAtoms=0,ts=0;
		AtomType atomT;
		std::memcpy(&ts,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&sim.beg(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&sim.end(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&sim.stride(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&nAtoms,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&sim.timestep(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&sim.cell_fixed(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		pos+=unpack(atomT,arr);
		pos+=unpack(sim.name(),arr);
		sim.resize(ts,nAtoms,atomT);
		for(int t=0; t<sim.timesteps(); ++t){
			pos+=unpack(sim.frame(t),arr);
		}
		return pos;
	}
	
}