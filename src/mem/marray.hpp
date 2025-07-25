#ifndef MARRAY_HPP
#define MARRAY_HPP

//c++
#include <vector>
//eigen
#include <Eigen/Dense>
// mem
#include "mem/serialize.hpp"

template <class T>
class MArray{
private:
	int rank_;//tensor rank
	int size_;//total number of data points
	Eigen::VectorXi dim_;//tensor dimension
	Eigen::VectorXi len_;//stride length
	std::vector<T> val_;//tensor values
public:
	//==== constructors/destructors ====
	MArray(){}
	MArray(int rank, const Eigen::VectorXi& dim){resize(rank,dim);}
	MArray(int rank, const Eigen::VectorXi& dim, const T& vinit){resize(rank,dim,vinit);}
	~MArray(){}
	
	//==== operators ====
	T& operator()(const Eigen::VectorXi& index);
	const T& operator()(const Eigen::VectorXi& index)const;
	T& operator[](int i){return val_[i];}
	const T& operator[](int i)const{return val_[i];}
	
	//==== access ====
	const int& rank()const{return rank_;}
	const int& size()const{return size_;}
	const int& dim(int i)const{return dim_[i];}
	const Eigen::VectorXi& dim()const{return dim_;}
	std::vector<T>& val(){return val_;}
	const std::vector<T>& val()const{return val_;}
	
	//==== member functions ====
	void clear();
	void resize(int rank, const Eigen::VectorXi& dim);
	void resize(int rank, const Eigen::VectorXi& dim, const T& vinit);
	int index(const Eigen::VectorXi& i)const;
};

//==== operators ====

template <class T>
T& MArray<T>::operator()(const Eigen::VectorXi& index){
	int loc=0;
	for(int i=0; i<rank_; ++i) loc+=len_[i]*index[i];
	return val_[loc];
}

template <class T>
const T& MArray<T>::operator()(const Eigen::VectorXi& index)const{
	int loc=0;
	for(int i=0; i<rank_; ++i) loc+=len_[i]*index[i];
	return val_[loc];
}

template <class T>
int MArray<T>::index(const Eigen::VectorXi& in)const{
	int loc=0;
	for(int i=0; i<rank_; ++i) loc+=len_[i]*in[i];
	return loc;
}

//==== member functions =====

template <class T>
void MArray<T>::clear(){
	rank_=0;
	size_=0;
	dim_*=0;
	len_*=0;
	val_.clear();
}

template <class T>
void MArray<T>::resize(int rank, const Eigen::VectorXi& dim){
	if(rank<=0 || rank!=dim.size()) throw std::invalid_argument("Invalid rank/dimension");
	for(int i=0; i<dim.size(); ++i) if(dim[i]<=0) throw std::invalid_argument("Invalid dimension");
	//resize rank/dim
	rank_=rank;
	dim_=dim;
	size_=dim_.prod();
	//resize stride
	len_.resize(rank_);
	for(int i=0; i<rank_; ++i){
		len_[i]=1;
		for(int j=i+1; j<rank_; ++j){
			len_[i]*=dim_[j];
		}
	}
	//resize val
	val_.resize(dim_.prod());
}

template <class T>
void MArray<T>::resize(int rank, const Eigen::VectorXi& dim, const T& vinit){
	if(rank<=0 || rank!=dim.size()) throw std::invalid_argument("Invalid rank/dimension");
	for(int i=0; i<dim.size(); ++i) if(dim[i]<=0) throw std::invalid_argument("Invalid dimension");
	//resize rank/dim
	rank_=rank;
	dim_=dim;
	size_=dim_.prod();
	//resize stride
	len_.resize(rank_);
	for(int i=0; i<rank_; ++i){
		len_[i]=1;
		for(int j=i+1; j<rank_; ++j){
			len_[i]*=dim_[j];
		}
	}
	//resize val
	val_.resize(dim_.prod(),vinit);
}

namespace serialize{
	
	template <class T>
	int nbytes(const MArray<T>& obj){
		int size=0;
		size+=sizeof(int);
		size+=nbytes(obj.dim());
		for(int i=0; i<obj.size(); ++i){
			size+=nbytes(obj[i]);
		}
		return size;
	}
	
	template <class T>
	int pack(MArray<T>& obj, char* arr){
		int pos=0,tempInt;
		std::memcpy(arr+pos,&(tempInt=obj.rank()),sizeof(int)); pos+=sizeof(int);
		pos+=pack(obj.dim(),arr+pos);
		for(int i=0; i<obj.size(); ++i){
			pos+=pack(obj[i],arr+pos);
		}
		return pos;
	}
	
	template <class T>
	int unpack(MArray<T>& obj, const char* arr){
		int pos=0;
		int rank=-1;
		std::memcpy(&rank,arr+pos,sizeof(int)); pos+=sizeof(int);
		Eigen::VectorXi dim(rank,-1);
		pos+=unpack(dim,arr+pos);
		obj.resize(rank,dim);
		for(int i=0; i<obj.size(); ++i){
			pos+=unpack(obj[i],arr+pos);
		}
		return pos;
	}
	
}

#endif