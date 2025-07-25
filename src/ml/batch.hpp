#pragma once
#ifndef BATCH_HPP
#define BATCH_HPP

// c++ libraries
#include <iosfwd>
// ann - string
#include "mem/serialize.hpp"

//***********************************************************************
// Batch
//***********************************************************************

#ifndef DEBUG_PRINT_FUNC
#define DEBUG_PRINT_FUNC 0
#endif

class Batch{
private:
	int count_;//count for the batch
	int size_;//size of the batch
	int capacity_;//number of elements in data set
	int* elements_;//batch elements
	int* data_;//all elements of the data set
public:
	//==== constructors/destructors ====
	Batch():count_(0),size_(0),capacity_(0),elements_(NULL),data_(NULL){if(DEBUG_PRINT_FUNC>0) std::cout<<"Batch():\n";}
	Batch(int size, int capacity):count_(0),size_(0),capacity_(0),elements_(NULL),data_(NULL){if(DEBUG_PRINT_FUNC>0) std::cout<<"Batch(int,int):\n";resize(size,capacity);}
	Batch(const Batch& batch);
	~Batch(){clear();}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Batch& batch);
	Batch& operator=(const Batch& batch);
	int& operator[](int i){return elements_[i];}
	const int& operator[](int i)const{return elements_[i];}
	
	//==== access ====
	const int& size()const{return size_;}
	const int& capacity()const{return capacity_;}
	int& count(){return count_;}
	const int& count()const{return count_;}
	int& element(int i){return elements_[i];}
	const int& element(int i)const{return elements_[i];}
	int& data(int i){return data_[i];}
	const int& data(int i)const{return data_[i];}
	int* elements(){return elements_;}
	const int* elements()const{return elements_;}
	int* data(){return data_;}
	const int* data()const{return data_;}
	
	//==== member functions ====
	void resize(int size, int capacity);
	void clear();
};

namespace serialize{
		
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const Batch& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Batch& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Batch& obj, const char* arr);
	
}

#endif
