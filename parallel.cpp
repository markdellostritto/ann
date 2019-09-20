#include "parallel.hpp"

namespace parallel{
	
std::vector<unsigned int>& gen_thread_dist(unsigned int nThreads, unsigned int nObj, std::vector<unsigned int>& thread_dist){
	thread_dist.clear();
	thread_dist.resize(nThreads);
	for(unsigned int i=0; i<nThreads; ++i){
		thread_dist[i]=nObj/nThreads;
	}
	unsigned int remainder=nObj%nThreads;
	for(unsigned int i=0; i<remainder; ++i){
		++thread_dist[i];
	}
	return thread_dist;
}

std::vector<unsigned int>& gen_thread_offset(unsigned int nThreads, unsigned int nObj, std::vector<unsigned int>& thread_offset){
	std::vector<unsigned int> thread_dist;
	gen_thread_dist(nThreads,nObj,thread_dist);
	thread_offset.resize(nThreads,0);
	for(unsigned int i=1; i<nThreads; ++i){
		thread_offset[i]=thread_offset[i-1]+thread_dist[i-1];
	}
	return thread_offset;
}

int* gen_thread_dist(unsigned int nThreads, unsigned int nObj, int* thread_dist){
	for(unsigned int i=0; i<nThreads; ++i){
		thread_dist[i]=nObj/nThreads;
	}
	unsigned int remainder=nObj%nThreads;
	for(unsigned int i=0; i<remainder; ++i){
		++thread_dist[i];
	}
	return thread_dist;
}

int* gen_thread_offset(unsigned int nThreads, unsigned int nObj, int* thread_offset){
	std::vector<unsigned int> thread_dist;
	gen_thread_dist(nThreads,nObj,thread_dist);
	thread_offset[0]=0;
	for(unsigned int i=1; i<nThreads; ++i){
		thread_offset[i]=thread_offset[i-1]+thread_dist[i-1];
	}
	return thread_offset;
}

//returns the number of data in a set of size N that are operated on by "rank" out of "nproc"
unsigned int thread_subset(unsigned int N, unsigned int rank, unsigned int nproc){
	unsigned int size=N/nproc;
	if(rank<N%nproc) ++size;
	return size;
}

//returns the offset of data in a set of size N that are operated on by "rank" out of "nproc"
unsigned int thread_offset(unsigned int N, unsigned int rank, unsigned int nproc){
	unsigned int offset=N/nproc*(rank);
	if(rank<=N%nproc) offset+=rank;
	else offset+=N%nproc;
	return offset;
}

}