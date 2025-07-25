#ifndef COMM_HPP
#define COMM_HPP

// mpi
#include <mpi.h>
// c++ libraries
#include <iosfwd>
// mem
#include "mem/serialize.hpp"

namespace thread{

class Comm{
private:
	int rank_;//the rank of the processor within the group
	int size_;//the total number of processors in the group
	int color_;//the color of the communicator (communicator rank)
	int ncomm_;//the total number of communicators with the same name
	MPI_Comm mpic_;//the label of the group
public:
	//==== constructors/destructors ====
	Comm():size_(1),rank_(0),color_(0),ncomm_(1),mpic_(MPI_COMM_SELF){}
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
	int& ncomm(){return ncomm_;}
	const int& ncomm()const{return ncomm_;}
	int& ncolor(){return ncomm_;}
	const int& ncolor()const{return ncomm_;}
	MPI_Comm& mpic(){return mpic_;}
	const MPI_Comm& mpic()const{return mpic_;}
	
	//==== member functions ====
	Comm split(int color);
	int color(int ncomm)const;
	int ncomm(int nobj)const;
};

}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const thread::Comm& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const thread::Comm& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(thread::Comm& obj, const char* arr);
	
}

#endif