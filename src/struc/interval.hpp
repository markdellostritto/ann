#pragma once
#ifndef INTERVAL_HPP
#define INTERVAL_HPP

// mem 
#include "mem/serialize.hpp"

#ifndef INTERVAL_PRINT_FUNC
#define INTERVAL_PRINT_FUNC 0
#endif

//**********************************************************************************************
// Interval
//**********************************************************************************************

class Interval{
private:
	int beg_;//timestep - beg
	int end_;//timestep - end
	int stride_;//stride
public:
	//==== constructors/destructors ====
	Interval():beg_(0),end_(0),stride_(1){}
	Interval(int b,int e):beg_(b),end_(e),stride_(1){}
	Interval(int b,int e,int s):beg_(b),end_(e),stride_(s){}
	~Interval(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Interval& i);
	
	//==== access ====
	const int& beg()const{return beg_;}
	const int& end()const{return end_;}
	const int& stride()const{return stride_;}
	
	//==== static functions ====
	static Interval& read(const char* str, Interval& interval);
	static int index(int t, int ts);
	static Interval split(const Interval& interval, int rank, int nproc);
};

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Interval& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Interval& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Interval& obj, const char* arr);
	
}
#endif