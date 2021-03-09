#pragma once
#ifndef PARALLEL_HPP
#define PARALLEL_HPP

// c++ libraries
#include <iosfwd>
// mpi
#include <mpi.h>
// ann - serialize
#include "serialize.hpp"

namespace parallel{

/**
* broadcast complex object to all ranks in the communicator
* the object is not specified, but must have serialization routines defined
* @param comm - mpi communicator 
* @param root - root processor which is broadcasting the object
* @param obj - object which will be broadcasted
*/
template <class T>
void bcast(MPI_Comm comm, int root, T& obj){
	int rank=0;
	MPI_Comm_rank(comm,&rank);
	//compute size
	int size=0;
	if(rank==root) size=serialize::nbytes(obj);
	//bcast size
	MPI_Bcast(&size,1,MPI_INT,root,comm);
	//pack object
	char* arr=new char[size];
	if(rank==root) serialize::pack(obj,arr);
	//bcast object
	MPI_Bcast(arr,size,MPI_CHAR,root,comm);
	//unpack object
	if(rank!=root) serialize::unpack(obj,arr);
	//clean up
	delete[] arr;
	MPI_Barrier(comm);
}

//*****************************************************************
// Distribution
//*****************************************************************

/**
* Stores the distribution of data over a given number of processors.
* The data is local, only the number of objects owned by the local processor
* as well as the offset of the data, as if indexed in a global array.
*/
class Dist{
private:
	int size_;
	int offset_;
public:
	//==== constructors/destructors ====
	Dist():size_(-1),offset_(-1){}
	Dist(int nprocs, int nrank, int nobj){init(nprocs,nrank,nobj);}
	~Dist(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Dist& dist);
	
	//==== access ====
	const int& size(){return size_;}
	const int& offset(){return offset_;}
	
	//==== member functions ====
	void init(int nprocs, int rank, int nobj);
	int index(int i){return offset_+i;}
	
	//==== static functions ====
	static int* size(int nrank, int nobj, int* size);
	static int* offset(int nrank, int nobj, int* offset);

};

//*****************************************************************
// Communicator
//*****************************************************************

/**
* Aggregates info on an MPI communicator.
* This class stores an MPI communicator as well as all information necessary
* to relate it to other communicators and its parent communicator, if it exits.
* This includes the rank of the processor with the local group, the color of
* the local group (i.e. group rank), the total size (number of processors) within 
* the local group, the total number of groups with the same label, and the label
* of the group.
* This class is intended to be used when one wishes to split processors into a series
* of groups which each will operate on a set of objects.  This class is most useful
* when there are more objects than processors, but can be used otherwise with some
* small extra overhead.
*/
class Comm{
private:
	int rank_;//the rank of the processor within the group
	int color_;//the color of the processor group (group rank)
	int size_;//the total number of processors in the group
	int ngroup_;//the total number of groups
	MPI_Comm label_;//the label of the group
public:
	//==== constructors/destructors ====
	Comm():size_(-1),rank_(-1),color_(-1),ngroup_(0),label_(MPI_COMM_NULL){}
	~Comm(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Comm& comm);
	
	//==== access ====
	int& rank(){return rank_;}
	const int& rank()const{return rank_;}
	int& size(){return size_;}
	const int& size()const{return size_;}
	int& color(){return color_;}
	const int& color()const{return color_;}
	int& ngroup(){return ngroup_;}
	const int& ngroup()const{return ngroup_;}
	MPI_Comm& label(){return label_;}
	const MPI_Comm& label()const{return label_;}
	
	//==== static functions ====
	static Comm& split(const Comm& world, Comm& group, int nobj);
};

}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const parallel::Comm& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const parallel::Comm& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(parallel::Comm& obj, const char* arr);
	
}

#endif