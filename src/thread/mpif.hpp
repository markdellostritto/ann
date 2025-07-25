#pragma once
#ifndef MPIF_HPP
#define MPIF_HPP

// mpi
#include <mpi.h>
// c++
#include <memory>

namespace thread{

/**
* broadcast complex object to all ranks in the communicator
* the object is not specified, but must have serialization routines defined
* @param comm - mpi communicator 
* @param root - root processor which is broadcasting the object
* @param obj - object which will be broadcasted
*/
template <class T>
void bcast(MPI_Comm comm, int root, T& obj){
	int rank=-1;
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

template <class T>
void bcast(MPI_Comm comm, int root, std::shared_ptr<T>& obj){
	int rank=-1;
	MPI_Comm_rank(comm,&rank);
	//make a copy of the object on root
	T copy; if(rank==root) copy=*obj;
	//compute size
	int size=0;
	if(rank==root) size=serialize::nbytes(copy);
	//bcast size
	MPI_Bcast(&size,1,MPI_INT,root,comm);
	//pack object
	char* arr=new char[size];
	if(rank==root) serialize::pack(copy,arr);
	//bcast object
	MPI_Bcast(arr,size,MPI_CHAR,root,comm);
	//unpack object
	if(rank!=root) serialize::unpack(copy,arr);
	//reset pointer not on root
	if(rank!=root) obj.reset(new T(copy));
	//clean up
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

template <class T>
void gather(T& obj, std::vector<T>& vec, MPI_Comm comm){
	//initialize MPI vars
	int nprocs=1;
	int rank=0;
	MPI_Comm_size(comm,&nprocs);
	MPI_Comm_rank(comm,&rank);
	//pack the object
	const int size=serialize::nbytes(obj);
	char* memarr=new char[size];
	serialize::pack(obj,memarr);
	//allocate total array
	int sizet=size*nprocs;
	char* memarrt=NULL;
	if(rank==0) memarrt=new char[sizet];
	//gather the data
	MPI_Gather(memarr,size,MPI_CHAR,memarrt,size,MPI_CHAR,0,comm);
	//unpack the data
	if(rank==0){
		int pos=0;
		vec.resize(nprocs);
		for(int i=0; i<nprocs; ++i){
			pos+=serialize::unpack(vec[i],memarrt+pos);
		}
	}
	//free memory
	delete[] memarr;
	if(rank==0) delete[] memarrt;
}

}

#endif