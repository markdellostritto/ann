#pragma once
#ifndef ITER_HPP
#define ITER_HPP

// c++
#include <iosfwd>
// eigen
#include <Eigen/Dense>
//serialization
#include "mem/serialize.hpp"
// opt
#include "opt/stop.hpp"
#include "opt/loss.hpp"

#ifndef OPT_ITER_PRINT_FUNC
#define OPT_ITER_PRINT_FUNC 0
#endif

namespace opt{

//***************************************************
// Iterator
//***************************************************

class Iterator{
private:
	//count
		int nPrint_;//print data every n steps
		int nWrite_;//write data every n steps
		int step_;//current step
		int count_;//current count
	//stopping
		int max_;//max steps
		Stop stop_;//the type of value determining the end condition
		Loss loss_;//loss function
		double tol_;//stop tolerance
public:
	//==== constructors/destructors ====
	Iterator(){defaults();}
	~Iterator(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Iterator& data);
	
	//==== access ====
	//count
		int& nPrint(){return nPrint_;}
		const int& nPrint()const{return nPrint_;}
		int& nWrite(){return nWrite_;}
		const int& nWrite()const{return nWrite_;}
		int& step(){return step_;}
		const int& step()const{return step_;}
		int& count(){return count_;}
		const int& count()const{return count_;}
	//stopping
		int& max(){return max_;}
		const int& max()const{return max_;}
		Stop& stop(){return stop_;}
		const Stop& stop()const{return stop_;}
		Loss& loss(){return loss_;}
		const Loss& loss()const{return loss_;}
		double& tol(){return tol_;}
		const double& tol()const{return tol_;}
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
};

}

namespace serialize{

	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const opt::Iterator& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const opt::Iterator& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(opt::Iterator& obj, const char* arr);
	
}

#endif