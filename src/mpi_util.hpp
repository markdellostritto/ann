#pragma once
#ifndef MPI_UTIL_HPP
#define MPI_UTIL_HPP

//c libraries
#include <cstdlib>
//mpi
#include <mpi.h>
//ann - serialize
#include "serialize.hpp"

namespace mpi_util{

/**
* broadcast complex object to all ranks in the communicator
* the object is not specified, but must have serialization routines defined
* @param comm - mpi communicator 
* @param obj - object which will be broadcasted
*/	
template <class T>
void bcast(MPI_Comm comm, T& obj){
	int rank=0;
	MPI_Comm_rank(comm,&rank);
	int size=0;
	if(rank==0) size=serialize::nbytes(obj);
	MPI_Bcast(&size,1,MPI_INT,0,comm);
	char* arr=new char[size];
	if(rank==0) serialize::pack(obj,arr);
	MPI_Bcast(arr,size,MPI_CHAR,0,comm);
	if(rank!=0) serialize::unpack(obj,arr);
	delete[] arr;
	MPI_Barrier(comm);
}

/**
* scatter vector of objects to all ranks in the communicator
* the objects will be distributed according to "thread_dist" and "thread_offset"
* @param send - the data to be scattered
* @param recv - the vector receiving the data being scattered
* @param thread_dist - the number of objects per process (size of communicator)
* @param thread_offset - the offset of objects for each process (size of communicator)
*/
template <class T>
void scatterv(std::vector<T>& send, std::vector<T>& recv, int* thread_dist, int* thread_offset){
	int nprocs=1,rank=0;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if(rank==0){
		for(int j=0; j<thread_dist[0]; ++j){
			recv[j]=send[j];
		}
	}
	for(int i=1; i<nprocs; ++i){
		for(int j=0; j<thread_dist[i]; ++j){
			int size=0;
			char* arr=NULL;
			if(rank==0) size=serialize::nbytes(send[thread_offset[i]+j]);
			if(rank==0) MPI_Send(&size,1,MPI_INT,i,0,MPI_COMM_WORLD);
			else if(rank==i) MPI_Recv(&size,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			if(rank==0 || rank==i) arr=new char[size];
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank==0) serialize::pack(send[thread_offset[i]+j],arr);
			if(rank==0) MPI_Send(arr,size,MPI_CHAR,i,0,MPI_COMM_WORLD);
			else if(rank==i) MPI_Recv(arr,size,MPI_CHAR,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			if(rank==i) serialize::unpack(recv[j],arr);
			if(rank==0 || rank==i) delete[] arr;
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
}

}

#endif