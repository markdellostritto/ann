//c++ libraries
#include <iostream>
#include <stdexcept>
// sim
#include "struc/interval.hpp"
// string
#include "str/token.hpp"

//**********************************************************************************************
//Interval
//**********************************************************************************************

//==== static functions ====

std::ostream& operator<<(std::ostream& out, const Interval& i){
	return out<<i.beg_<<":"<<i.end_<<":"<<i.stride_;
}

Interval& Interval::read(const char* str, Interval& interval){
	Token token(str,":");
	int beg=0,end=0,stride=1;
	beg=std::atoi(token.next().c_str());
	if(!token.end()){
		end=std::atoi(token.next().c_str());
		if(!token.end()){
			stride=std::atoi(token.next().c_str());
		}
	} else end=beg;
	interval=Interval(beg,end,stride);
	return interval;
}


int Interval::index(int t, int ts){
	const int ts1=++ts;
	int i=(t%ts1+ts1)%ts1;
	return --i;
}

Interval Interval::split(const Interval& interval, int rank, int nprocs){
	const int ts=(interval.end()-interval.beg()+1);
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
	Interval newint(beg_loc,end_loc,interval.stride());
	return newint;
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Interval& obj){
		if(INTERVAL_PRINT_FUNC>0) std::cout<<"nbytes(const Interval&):\n";
		return 5*sizeof(int);
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Interval& obj, char* arr){
		if(INTERVAL_PRINT_FUNC>0) std::cout<<"pack(const Interval&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.beg(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.end(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.stride(),sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Interval& obj, const char* arr){
		if(INTERVAL_PRINT_FUNC>0) std::cout<<"unpack(Interval&,const char*):\n";
		int pos=0;
		int beg=0,end=0,stride=1;
		std::memcpy(&beg,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&end,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&stride,arr+pos,sizeof(int)); pos+=sizeof(int);
		obj=Interval(beg,end,stride);
		return pos;
	}
		
}