#include <ostream>
#include "thread/dist.hpp"

namespace thread{

//*****************************************************************
// Distribution
//*****************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Dist& dist){
	return out<<"nrank "<<dist.nrank_<<" nobj "<<dist.nobj_<<" size "<<dist.size_<<" offset "<<dist.offset_;
}

//==== member functions ====

/**
* generate distribution of objects over processors
* @param nprocs - the total number of processors available for calculation
* @param rank - the rank of the current processor
* @param nobj - the number of objects which are split between the processors
* This function splits a number of objects between a number of processors.
* This function, and indeed this class, is intended when there are a larger number
* of objects than there are processors, though it will work otherwise.
* The objects are split between the processor with a remainder at first.
* One remainder is then added to each distribution until there are none left.
*/
void Dist::init(int nrank, int rank, int nobj){
	if(nrank<=0) throw std::runtime_error("thread::Dist::init(int,int,int): Number of ranks must be greater than zero.");
	if(rank<0) throw std::runtime_error("thread::Dist::init(int,int,int): Rank can't be negative.");
	if(nobj<0) throw std::runtime_error("thread::Dist::init(int,int,int): Number of objects can't be negative.");
	nrank_=nrank;
	nobj_=nobj;
	size_=nobj/nrank;
	offset_=nobj/nrank*(rank);
	if(rank<nobj%nrank){
		size_++;
		offset_+=rank;
	} else offset_+=nobj%nrank;
}

//==== static functions ====

/**
* generate distribution of objects over processors
* @param nrank - the total number of processors available for calculation
* @param nobj - the number of objects which are split between the processors
* @param size - array storing the number of objects owned by each rank
* This function splits a number of objects between a number of processors.
* Rather than generate a local instance of a Dist object, this generates
* the distribution of objects over nrank in an array for e.g. printing.
*/
int* Dist::size(int nrank, int nobj, int* size){
	for(int i=0; i<nrank; ++i){
		size[i]=nobj/nrank;
		if(i<nobj%nrank) ++size[i];
	}
	return size;
}

/**
* generate distribution of objects over processors
* @param nrank - the total number of processors available for calculation
* @param nobj - the number of objects which are split between the processors
* @param offset - offset in array storing the number of objects owned by each rank
* This function splits a number of objects between a number of processors.
* Rather than generate a local instance of a Dist object, this generates
* the offsets for the distribution of objects over nrank in an array for e.g. printing.
*/
int* Dist::offset(int nrank, int nobj, int* offset){
	for(int i=0; i<nrank; ++i){
		offset[i]=nobj/nrank*(i);
		if(i<nobj%nrank) offset[i]+=i;
		else offset[i]+=nobj%nrank;
	}
	return offset;
}

std::string& Dist::size(std::string& str, int nranks, int nobj){
	str.clear();
	for(int i=0; i<nranks; ++i){
		int size=nobj/nranks;
		if(i<nobj%nranks) ++size;
		str+=std::to_string(size);
		str+=" ";
	}
	return str;
}

std::string& Dist::offset(std::string& str, int nranks, int nobj){
	str.clear();
	for(int i=0; i<nranks; ++i){
		int offset=nobj/nranks*(i);
		if(i<nobj%nranks) offset+=i;
		else offset+=nobj%nranks;
		str+=std::to_string(offset);
		str+=" ";
	}
	return str;
}

}
