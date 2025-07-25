#pragma once
#ifndef PARALLEL_HPP
#define PARALLEL_HPP

// c++ libraries
#include <iosfwd>
// ann - serialize
#include "mem/serialize.hpp"

namespace thread{

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
	int nrank_;//total number of ranks
	int nobj_;//total number of objects
	int size_;//number of objects owned by the current rank
	int offset_;//the offset for the objects owned by the current rank in a global array of all objects
public:
	//==== constructors/destructors ====
	Dist():nrank_(-1),nobj_(-1),size_(-1),offset_(-1){}
	Dist(int nrank, int rank, int nobj){init(nrank,rank,nobj);}
	~Dist(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Dist& dist);
	
	//==== access ====
	const int& nrank()const{return nrank_;}
	const int& nobj()const{return nobj_;}
	const int& size()const{return size_;}
	const int& offset()const{return offset_;}
	int beg()const{return offset_;}
	int end()const{return offset_+size_;}
	
	//==== member functions ====
	void init(int nranks, int rank, int nobj);
	int index(int i)const{return offset_+i;}
	
	//==== static functions ====
	static int* size(int nrank, int nobj, int* size);
	static int* offset(int nrank, int nobj, int* offset);
	static std::string& size(std::string& str, int nranks, int nobj);
	static std::string& offset(std::string& str, int nranks, int nobj);
};

}

#endif