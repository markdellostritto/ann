#pragma once
#ifndef MPI_UTIL_HPP
#define MPI_UTIL_HPP

#include <cstdlib>
#include <mpi.h>
#include "serialize.hpp"

namespace mpi_util{
	
template <class T>
void bcast(T& obj){
	int rank=0;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	unsigned int size=0;
	if(rank==0) size=serialize::nbytes(obj);
	MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);
	char* arr=new char[size];
	if(rank==0) serialize::pack(obj,arr);
	MPI_Bcast(arr,size,MPI_CHAR,0,MPI_COMM_WORLD);
	if(rank!=0) serialize::unpack(obj,arr);
	delete[] arr;
	MPI_Barrier(MPI_COMM_WORLD);
}

}

#endif