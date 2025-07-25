// c
#include <cmath>
// c++
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
// thread
#include "thread/comm.hpp"

namespace thread{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Comm& comm){
	return out<<"ncomm "<<comm.ncomm_<<" color "<<comm.color_<<" size "<<comm.size_<<" rank "<<comm.rank_;
}

//==== member functions ====

/**
* @param color - the color determining the new communicator
* @return the new split communicator
* Splits a given communicator into several new communicators which may contain fewer threads
* according to the variable "color", such that each thread with the same color is part of 
* the same communicator.
*/
Comm Comm::split(int color){
	//make communicator
	Comm comm;
	//split the current communicator
	MPI_Comm_split(mpic_,color,rank_,&comm.mpic());
	MPI_Comm_rank(comm.mpic(),&comm.rank());
	MPI_Comm_size(comm.mpic(),&comm.size());
	//set the color and number of communicators
	comm.color()=color;
	std::vector<int> colors(size_);
	MPI_Allgather(&color,1,MPI_INT,colors.data(),1,MPI_INT,mpic_);
	std::sort(colors.begin(),colors.end());
	int ncomm=1;
	for(int i=0; i<colors.size()-1; ++i){
		if(colors[i+1]!=colors[i]) ncomm++;
	}
	comm.ncomm()=ncomm;
	//return new communicator
	return comm;
}

/**
* @param ncomm - the number of sub-communicators
* @return the color of the sub-communicator associated with a given rank
* This function determines the color of the sub-communicator associated with a given
* rank if one were to split the current communicator into "ncomm" sub-communicators.
* The variable "ncomm" may be any positive integer less than or equal to size.
*/
int Comm::color(int ncomm)const{
	if(ncomm<=0 || ncomm>size_) throw std::invalid_argument("Comm::color(int): invalid number of sub-communicators.");
	const double nthreads=(1.0*size_)/(1.0*ncomm);//determine the fractional number of threads per sub-communicator
	double colorf=(1.0*rank_)/(1.0*nthreads);//determine the fractional color
	int colori=std::floor(colorf);//determine the integer color
	return colori;
}

/**
* @param nobj - the number of objects to be processed in a parallel task
* @return the number of communicators required to divide "nobj" objects 
* This function determines the number of communicators required to divide "nobj" 
* objects as equally as possible among all threads in the communicator.
* The variable "nobj" may be any positive integer.
* Typically the number of communicators will simply be "nobj" because each group
* of threads (e.g. communicator) will handle one object.
* However, this function guarantees there to be at least one thread per object,
* such that the maximum number of communicators is equal to the number of threads.
*/
int Comm::ncomm(int nobj)const{
	if(nobj<=0) throw std::invalid_argument("Comm::ncomm(int): invalid number of objects.");
	return (nobj>size_)?size_:nobj;
}

}

namespace serialize{

	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const thread::Comm& obj){
		int size=0;
		size+=sizeof(obj.rank());
		size+=sizeof(obj.size());
		size+=sizeof(obj.color());
		size+=sizeof(obj.ncomm());
		size+=sizeof(obj.mpic());
		return size;
	}

	//**********************************************
	// packing
	//**********************************************

	template <> int pack(const thread::Comm& obj, char* arr){
		int pos=0;
		std::memcpy(arr+pos,&obj.rank(),sizeof(obj.rank())); pos+=sizeof(obj.rank());
		std::memcpy(arr+pos,&obj.size(),sizeof(obj.size())); pos+=sizeof(obj.size());
		std::memcpy(arr+pos,&obj.color(),sizeof(obj.color())); pos+=sizeof(obj.color());
		std::memcpy(arr+pos,&obj.ncomm(),sizeof(obj.ncomm())); pos+=sizeof(obj.ncomm());
		std::memcpy(arr+pos,&obj.mpic(),sizeof(obj.mpic())); pos+=sizeof(obj.mpic());
		return pos;
	}

	//**********************************************
	// unpacking
	//**********************************************

	template <> int unpack(thread::Comm& obj, const char* arr){
		int pos=0;
		std::memcpy(&obj.rank(),arr+pos,sizeof(obj.rank())); pos+=sizeof(obj.rank());
		std::memcpy(&obj.size(),arr+pos,sizeof(obj.size())); pos+=sizeof(obj.size());
		std::memcpy(&obj.color(),arr+pos,sizeof(obj.color())); pos+=sizeof(obj.color());
		std::memcpy(&obj.ncomm(),arr+pos,sizeof(obj.ncomm())); pos+=sizeof(obj.ncomm());
		std::memcpy(&obj.mpic(),arr+pos,sizeof(obj.mpic())); pos+=sizeof(obj.mpic());
		return pos;
	}

}
