#ifndef TENSORTRI_HPP
#define TENSORTRI_HPP

//c++
#include <vector>
#include <algorithm>
//eigen
#include <Eigen/Dense>
// mem
#include "mem/serialize.hpp"

//***********************************************************
// template <int R, class T> class TensorTri
//***********************************************************

/**
*	template <int R, class T> class TensorTri
*	R - rank of tensor
*	T - type of data stored
*	Class for storing data in a triangular (upper or lower) tensor.
*	This is equivalent to a triangular matrix, but generalized for any rank.
*	The diagonal entries are included.
*	Only the memory required to store the non-zero elements is requested.
*	Only square matrices/tensors are allowed.
*	Thus, the number of objects is equal to d*(d+1)/2*(d+2)/3*...*(d+R)/(R+1)
*	Access using any tuple will only access the non-zero elements,
*		regardless of the order of the indices.
*	Sequential access to the underyling linear array is also provided.
*/
template <int R, class T>
class TensorTri{
private:
	int dim_;//dimension
	int size_;//size
	std::vector<T> data_;//tensor values
public:
	//==== constructors/destructors ====
	TensorTri(){}
	TensorTri(int d){resize(d);}
	TensorTri(int d, const T& v){resize(d,v);}
	~TensorTri(){}
	
	//==== operators ====
	T& operator()(int i);
	const T& operator()(int i)const;
	T& operator()(int i, int j);
	const T& operator()(int i, int j)const;
	T& operator()(int i, int j, int k);
	const T& operator()(int i, int j, int k)const;
	T& operator()(const Eigen::VectorXi& index);
	const T& operator()(const Eigen::VectorXi& index)const;
	T& operator[](int i){return data_[i];}
	const T& operator[](int i)const{return data_[i];}
	TensorTri<R,T>& operator*=(const T& v);
	
	//==== access ====
	const int& rank()const{return R;}
	const int& size()const{return size_;}
	const int& dim()const{return dim_;}
	std::vector<T>& data(){return data_;}
	const std::vector<T>& data()const{return data_;}
	
	//==== member functions ====
	void clear();
	void resize(const Eigen::VectorXi& dim);
	void resize(const Eigen::VectorXi& dim, const T& vinit);
	void resize(int d);
	void resize(int d, const T& vinit);
	int index(const Eigen::VectorXi& i)const;
	int index(int i)const;
	int index(int i, int j)const;
	int index(int i, int j, int k)const;
};

//==== operators ====

template <int R, class T>
TensorTri<R,T>& TensorTri<R,T>::operator*=(const T& v){
	for(int i=0; i<size_; ++i) data_[i]*=v;
	return *this;
}

//==== operators - rank 1 ====

template <int R, class T>
T& TensorTri<R,T>::operator()(int i){
	static_assert(R==1,"TensorTri::operator(int): Rank 1 access only allowed for Rank 1 tensors.");
	return data_[i];
}

template <int R, class T>
const T& TensorTri<R,T>::operator()(int i)const{
	static_assert(R==1,"TensorTri::operator(int): Rank 1 access only allowed for Rank 1 tensors.");
	return data_[i];
}

//==== operators - rank 2 ====

template <int R, class T>
T& TensorTri<R,T>::operator()(int i, int j){
	static_assert(R==2,"TensorTri::operator(int,int): Rank 2 access only allowed for Rank 2 tensors.");
	if(i>j) return data_[i*(i+1)/2+j];
	else return data_[j*(j+1)/2+i];
}

template <int R, class T>
const T& TensorTri<R,T>::operator()(int i, int j)const{
	static_assert(R==2,"TensorTri::operator(int,int): Rank 2 access only allowed for Rank 2 tensors.");
	if(i>j) return data_[i*(i+1)/2+j];
	else return data_[j*(j+1)/2+i];
}

//==== operators - rank 3 ====

template <int R, class T>
T& TensorTri<R,T>::operator()(int i, int j, int k){
	static_assert(R==3,"TensorTri::operator(int,int,int): Rank 3 access only allowed for Rank 3 tensors.");
	if(i>j) std::swap(i,j);
	if(j>k) std::swap(j,k);
	if(i>j) std::swap(i,j);
	return data_[k*(k+1)*(k+2)/6+j*(j+1)/2+i];
}

template <int R, class T>
const T& TensorTri<R,T>::operator()(int i, int j, int k)const{
	static_assert(R==3,"TensorTri::operator(int,int,int): Rank 3 access only allowed for Rank 3 tensors.");
	if(i>j) std::swap(i,j);
	if(j>k) std::swap(j,k);
	if(i>j) std::swap(i,j);
	return data_[k*(k+1)*(k+2)/6+j*(j+1)/2+i];
}

//==== operators - rank R ====

template <int R, class T>
T& TensorTri<R,T>::operator()(const Eigen::VectorXi& in){
	throw std::runtime_error("TensorTri<R,T>::operator()(const Eigen::VectorXi&): not implemented yet");
	Eigen::VectorXi intmp=in;
	std::sort(intmp.data(),intmp.data()+intmp.size());
	int loc=0;
	for(int i=0; i<R; ++i){
		int s=0;
		int denom=1;
		for(int j=0; j<=i; ++j){
			s+=(intmp[i]+j)/(1.0*denom);
			denom+=1;
		}
		loc+=s;
	}
	return data_[loc];
}

template <int R, class T>
const T& TensorTri<R,T>::operator()(const Eigen::VectorXi& in)const{
	throw std::runtime_error("TensorTri<R,T>::operator()(const Eigen::VectorXi&): not implemented yet");
	Eigen::VectorXi intmp=in;
	std::sort(intmp.data(),intmp.data()+intmp.size());
	int loc=0;
	for(int i=0; i<R; ++i){
		int s=0;
		int denom=1;
		for(int j=0; j<=i; ++j){
			s+=(intmp[i]+j)/(1.0*denom);
			denom+=1;
		}
		loc+=s;
	}
	return data_[loc];
}

template <int R, class T>
int TensorTri<R,T>::index(const Eigen::VectorXi& in)const{
	throw std::runtime_error("TensorTri<R,T>::index(const Eigen::VectorXi&): not implemented yet");
	Eigen::VectorXi intmp=in;
	std::sort(intmp.data(),intmp.data()+intmp.size());
	int loc=0;
	for(int i=0; i<R; ++i){
		int s=0;
		int denom=1;
		for(int j=0; j<=i; ++j){
			s+=(intmp[i]+j)/(1.0*denom);
			denom+=1;
		}
		loc+=s;
	}
	return loc;
}

template <int R, class T>
int TensorTri<R,T>::index(int i)const{
	static_assert(R==1,"TensorTri::index(int): Rank 1 access only allowed for Rank 1 tensors.");
	return i;
}

template <int R, class T>
int TensorTri<R,T>::index(int i, int j)const{
	static_assert(R==2,"TensorTri::index(int,int): Rank 2 access only allowed for Rank 2 tensors.");
	if(i>j) return i*(i+1)/2+j;
	else return j*(j+1)/2+i;
}

template <int R, class T>
int TensorTri<R,T>::index(int i, int j, int k)const{
	static_assert(R==3,"TensorTri::index(int,int,int): Rank 3 access only allowed for Rank 3 tensors.");
	if(i>j) std::swap(i,j);
	if(j>k) std::swap(j,k);
	if(i>j) std::swap(i,j);
	return k*(k+1)*(k+2)/6+j*(j+1)/2+i;
}

//==== member functions =====

template <int R, class T>
void TensorTri<R,T>::clear(){
	size_=0;
	dim_*=0;
	data_.clear();
}

template <int R, class T>
void TensorTri<R,T>::resize(int d){
	if(d<=0) throw std::invalid_argument("TensorTri<R,T>::resize(int): Invalid dimension");
	int denom=1;
	size_=1;
	for(int i=0; i<R; ++i){
		size_*=(d+i)/(1.0*denom);
		denom+=1;
	}
	data_.resize(size_);
}

template <int R, class T>
void TensorTri<R,T>::resize(int d, const T& vinit){
	resize(d);
	data_.resize(size_,vinit);
}

#endif