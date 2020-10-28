#include <ostream>
#include "parallel.hpp"

namespace parallel{

//*****************************************************************
// Distribution
//*****************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Dist& dist){
	return out<<"dist "<<dist.size_<<" "<<dist.offset_;
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
void Dist::init(int nprocs, int rank, int nobj){
	size_=nobj/nprocs;
	offset_=nobj/nprocs*(rank);
	if(rank<nobj%nprocs){
		size_++;
		offset_+=rank;
	} else offset_+=nobj%nprocs;
}

//==== static functions ====

int* Dist::size(int nrank, int nobj, int* size_){
	for(int i=0; i<nrank; ++i){
		size_[i]=nobj/nrank;
		if(i<nobj%nrank) ++size_[i];
	}
	return size_;
}

int* Dist::offset(int nrank, int nobj, int* offset_){
		for(int i=0; i<nrank; ++i){
			offset_[i]=nobj/nrank*(i);
			if(i<nobj%nrank) offset_[i]+=i;
			else offset_[i]+=nobj%nrank;
		}
		return offset_;
}


//*****************************************************************
// Communicator
//*****************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Comm& comm){
	return out<<"ngroup "<<comm.ngroup_<<" color "<<comm.color_<<" rank "<<comm.rank_<<" size "<<comm.size_;
}

//==== static functions ====

/**
* split a communicator into a series of groups based on a set of objects
* @param world - the communicator which will be split
* @param group - the communicator being initialized (subdivision of world)
* @param nobj - the number of objects which are split between the processors
* This function splits a given communicator into a set of smaller communicators.
* The size of the groups are determined by the number of objects.
* If there are more objects than there are processors in "world", then the size of
* each group will be 1, as each processor will be operating on multiple objects.
* If there are fewer objects than there are processors in "world", then the size of
* each group will be number of processors per object, as multiple processors
* will be operating on one object.
* This function, and indeed the class, is intended when multipling processors will
* be performing operations on a single object simultaneously.
* Thus, this function and class are intended for situations where there are a larger
* number of processors than there are objects, though it will work otherwise.
* NOTE: this sets only the size and color, the ngroup and rank must be set separately
*/
Comm& Comm::split(const Comm& world, Comm& group, int nobj){
	const int proc_per_obj=world.size()/nobj;//compute the processors per object
	const int size=(proc_per_obj==0)?1:proc_per_obj;//compute the size of the local group
	const int color=world.rank()/size;//compute the color of the local group
	group.size()=size;//set the size
	group.color()=color;//set the color
	return group;
}

}

namespace serialize{

	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const parallel::Comm& obj){
		int size=0;
		size+=sizeof(obj.rank());
		size+=sizeof(obj.color());
		size+=sizeof(obj.size());
		size+=sizeof(obj.ngroup());
		size+=sizeof(obj.label());
		return size;
	}

	//**********************************************
	// packing
	//**********************************************

	template <> int pack(const parallel::Comm& obj, char* arr){
		int pos=0;
		std::memcpy(arr+pos,&obj.rank(),sizeof(obj.rank())); pos+=sizeof(obj.rank());
		std::memcpy(arr+pos,&obj.color(),sizeof(obj.color())); pos+=sizeof(obj.color());
		std::memcpy(arr+pos,&obj.size(),sizeof(obj.size())); pos+=sizeof(obj.size());
		std::memcpy(arr+pos,&obj.ngroup(),sizeof(obj.ngroup())); pos+=sizeof(obj.ngroup());
		std::memcpy(arr+pos,&obj.label(),sizeof(obj.label())); pos+=sizeof(obj.label());
		return pos;
	}

	//**********************************************
	// unpacking
	//**********************************************

	template <> int unpack(parallel::Comm& obj, const char* arr){
		int pos=0;
		std::memcpy(&obj.rank(),arr+pos,sizeof(obj.rank())); pos+=sizeof(obj.rank());
		std::memcpy(&obj.color(),arr+pos,sizeof(obj.color())); pos+=sizeof(obj.color());
		std::memcpy(&obj.size(),arr+pos,sizeof(obj.size())); pos+=sizeof(obj.size());
		std::memcpy(&obj.ngroup(),arr+pos,sizeof(obj.ngroup())); pos+=sizeof(obj.ngroup());
		std::memcpy(&obj.label(),arr+pos,sizeof(obj.label())); pos+=sizeof(obj.label());
		return pos;
	}

}
