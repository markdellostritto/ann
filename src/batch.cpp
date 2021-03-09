// c++ libraries
#include <iostream>
#include <stdexcept>
// ann - string
#include "batch.hpp"
// ann - print
#include "print.hpp"

//***********************************************************************
// Batch
//***********************************************************************

//==== constructors/destructors ====

/**
* copy constructor
* @param batch - the object we are copying
*/
Batch::Batch(const Batch& batch):count_(0),size_(0),capacity_(0),elements_(NULL),data_(NULL){
	if(DEBUG_PRINT_FUNC>0) std::cout<<"Batch(const Batch&):\n";
	*this=batch;
}

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Batch& b){
	return out<<"batch count "<<b.count_<<" size "<<b.size_<<" capacity "<<b.capacity_;
}

/**
* assignment operator
* @param batch - the object we are copying
*/
Batch& Batch::operator=(const Batch& batch){
	if(DEBUG_PRINT_FUNC>0) std::cout<<"Batch::operator=(const Batch&)\n";
	Batch tmp(batch);
	clear();
	resize(tmp.size(),tmp.capacity());
	count_=tmp.count();
	for(int i=0; i<size_; ++i){
		elements_[i]=tmp.element(i);
	}
	for(int i=0; i<capacity_; ++i){
		data_[i]=tmp.data(i);
	}
	return *this;
}

//==== member functions ====

/**
* resize batch
* @param size - the size of the batch (number of elements in the batch)
* @param capacity - the total number of items from which we will draw the batch
*/
void Batch::resize(int size, int capacity){
	if(DEBUG_PRINT_FUNC>0) std::cout<<"Batch::resize(int,int)\n";
	clear();
	if(size<0) throw std::invalid_argument("Batch::resize(int,int): size cannot be negative.");
	if(capacity<0) throw std::invalid_argument("Batch::resize(int,int): capacity cannot be negative.");
	if(size>capacity) throw std::invalid_argument("Batch::resize(int,int): size must be smaller than capacity");
	size_=size;
	capacity_=capacity;
	if(size_>0){
		elements_=new int[size_];
		for(int i=0; i<size_; ++i) elements_[i]=i;
	}
	if(capacity_>0){
		data_=new int[capacity_];
		for(int i=0; i<capacity_; ++i) data_[i]=i;
	}
}

/**
* clear the batch
* delete the batch and set all sizes and counts to zero
*/
void Batch::clear(){
	if(DEBUG_PRINT_FUNC>0) std::cout<<"Batch::clear()\n";
	if(elements_!=NULL) delete[] elements_;
	if(data_!=NULL) delete[] data_;
	size_=0;
	capacity_=0;
	count_=0;
}

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const Batch& obj){
		if(DEBUG_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const Batch):\n";
		int size=0;
		size+=sizeof(int);//count
		size+=sizeof(int);//size
		size+=sizeof(int);//capacity
		size+=obj.size()*sizeof(int);//elements_
		size+=obj.capacity()*sizeof(int);//data_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Batch& obj, char* arr){
		if(DEBUG_PRINT_FUNC>0) std::cout<<"serialize::pack(const Batch,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.count(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.size(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.capacity(),sizeof(int)); pos+=sizeof(int);
		if(obj.size()>0) std::memcpy(arr+pos,obj.elements(),obj.size()*sizeof(int)); pos+=obj.size()*sizeof(int);
		if(obj.capacity()>0) std::memcpy(arr+pos,obj.data(),obj.capacity()*sizeof(int)); pos+=obj.capacity()*sizeof(int);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Batch& obj, const char* arr){
		if(DEBUG_PRINT_FUNC>0) std::cout<<"serialize::unpack(const Batch,const char*):\n";
		int pos=0,count=0,size=0,capacity=0;
		std::memcpy(&count,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&size,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&capacity,arr+pos,sizeof(int)); pos+=sizeof(int);
		obj.resize(size,capacity);
		obj.count()=count;
		if(size>0) std::memcpy(obj.elements(),arr+pos,obj.size()*sizeof(int)); pos+=obj.size()*sizeof(int);
		if(capacity>0) std::memcpy(obj.data(),arr+pos,obj.capacity()*sizeof(int)); pos+=obj.capacity()*sizeof(int);
		return pos;
	}
	
}