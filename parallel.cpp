#include "parallel.hpp"

namespace parallel{
	
std::vector<unsigned int>& gen_thread_dist(unsigned int nThreads, unsigned int nObj, std::vector<unsigned int>& thread_dist){
	thread_dist.clear();
	thread_dist.resize(nThreads);
	for(unsigned int i=0; i<nThreads-1; ++i){
		thread_dist[i]=nObj/nThreads;
	}
	thread_dist.back()=nObj/nThreads+nObj%nThreads;
}

std::vector<unsigned int>& gen_thread_offset(unsigned int nThreads, unsigned int nObj, std::vector<unsigned int>& thread_offset){
	std::vector<unsigned int> thread_dist;
	gen_thread_dist(nThreads,nObj,thread_dist);
	thread_offset.resize(nThreads,0);
	for(unsigned int i=1; i<nThreads; ++i){
		thread_offset[i]=thread_offset[i-1]+thread_dist[i-1];
	}
}

}