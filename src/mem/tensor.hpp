#ifndef TENSOR_HPP
#define TENSOR_HPP

//c++
#include <vector>
//eigen
#include <Eigen/Dense>
// mem
#include "mem/serialize.hpp"

//# define TENSOR_ROW_MAJOR

//***********************************************************
// template <int R, class T> class Tensor
//***********************************************************

/**
*	template <int R, class T> class TensorTri
*	R - rank of tensor
*	T - type of data stored
*	Class for storing data in a mulidimensional tensor.
*	This is equivalent to a matrix, but generalized for any rank.
*	Sequential access to the underyling linear array is also provided.
*/
template <int R, class T>
class Tensor{
private:
	int size_;//total number of data points
	Eigen::VectorXi dim_;//tensor dimension
	std::vector<T> data_;//tensor values
public:
	//==== constructors/destructors ====
	Tensor(){}
	Tensor(int d){resize(d);}
	Tensor(const Eigen::VectorXi& dim){resize(dim);}
	Tensor(const Eigen::VectorXi& dim, const T& vinit){resize(dim,vinit);}
	~Tensor(){}
	
	//==== operators ====
	T& operator()(int i);
	const T& operator()(int i)const;
	T& operator()(int i, int j);
	const T& operator()(int i, int j)const;
	T& operator()(int i, int j, int k);
	const T& operator()(int i, int j, int k)const;
	T& operator()(const Eigen::Vector3i& index);
	const T& operator()(const Eigen::Vector3i& index)const;
	T& operator()(const Eigen::VectorXi& index);
	const T& operator()(const Eigen::VectorXi& index)const;
	T& operator()(const std::array<int,R>& index);
	const T& operator()(const std::array<int,R>& index)const;
	T& operator[](int i){return data_[i];}
	const T& operator[](int i)const{return data_[i];}
	Tensor<R,T>& operator*=(const T& v);
	
	//==== access ====
	const int& rank()const{return R;}
	const int& size()const{return size_;}
	const int& dim(int i)const{return dim_[i];}
	const Eigen::VectorXi& dim()const{return dim_;}
	std::vector<T>& data(){return data_;}
	const std::vector<T>& data()const{return data_;}
	
	//==== member functions ====
	void clear();
	void resize(const Eigen::VectorXi& dim, const T& vinit);
	void resize(const std::array<int,R>& dim, const T& vinit);
	void resize(const Eigen::VectorXi& dim){resize(dim,0.0);}
	void resize(const std::array<int,R>& dim){resize(dim,0.0);}
	void resize(int d, const T& vinit);
	void resize(int d){resize(d,0.0);}
	int index(int i)const;
	int index(int i, int j)const;
	int index(int i, int j, int k)const;
	int index(const Eigen::VectorXi& i)const;
	int index(const std::array<int,R>& i)const;
};

//==== operators ====

template <int R, class T>
Tensor<R,T>& Tensor<R,T>::operator*=(const T& v){
	for(int i=0; i<size_; ++i) data_[i]*=v;
	return *this;
}

//==== operators - rank 1 ====

template <int R, class T>
T& Tensor<R,T>::operator()(int i){
	static_assert(R==1,"Tensor::operator(int): Rank 1 access only allowed for Rank 1 tensors.");
	return data_[i];
}

template <int R, class T>
const T& Tensor<R,T>::operator()(int i)const{
	static_assert(R==1,"Tensor::operator(int): Rank 1 access only allowed for Rank 1 tensors.");
	return data_[i];
}

//==== operators - rank 2 ====

template <int R, class T>
T& Tensor<R,T>::operator()(int i, int j){
	static_assert(R==2,"Tensor::operator(int,int): Rank 2 access only allowed for Rank 2 tensors.");
	#ifdef TENSOR_ROW_MAJOR
		return data_[i*dim_[0]+j];
	#else
		return data_[j*dim_[1]+i];
	#endif
}

template <int R, class T>
const T& Tensor<R,T>::operator()(int i, int j)const{
	static_assert(R==2,"Tensor::operator(int,int): Rank 2 access only allowed for Rank 2 tensors.");
	#ifdef TENSOR_ROW_MAJOR
		return data_[i*dim_[0]+j];
	#else
		return data_[j*dim_[1]+i];
	#endif
}

//==== operators - rank 3 ====

template <int R, class T>
T& Tensor<R,T>::operator()(int i, int j, int k){
	static_assert(R==3,"Tensor::operator(int,int,int): Rank 3 access only allowed for Rank 3 tensors.");
	#ifdef TENSOR_ROW_MAJOR
		return data_[(i*dim_[0]+j)*dim_[1]+k];
	#else
		return data_[(k*dim_[2]+j)*dim_[1]+i];
	#endif
}

template <int R, class T>
const T& Tensor<R,T>::operator()(int i, int j, int k)const{
	static_assert(R==3,"Tensor::operator(int,int,int): Rank 3 access only allowed for Rank 3 tensors.");
	#ifdef TENSOR_ROW_MAJOR
		return data_[(i*dim_[0]+j)*dim_[1]+k];
	#else
		return data_[(k*dim_[2]+j)*dim_[1]+i];
	#endif
}

template <int R, class T>
T& Tensor<R,T>::operator()(const Eigen::Vector3i& index){
	static_assert(R==3,"Tensor::operator(const Eigen::Vector3i&): Rank 3 access only allowed for Rank 3 tensors.");
	#ifdef TENSOR_ROW_MAJOR
		return data_[(index[0]*dim_[0]+index[1])*dim_[1]+index[2]];
	#else
		return data_[(index[2]*dim_[2]+index[1])*dim_[1]+index[0]];
	#endif
}

template <int R, class T>
const T& Tensor<R,T>::operator()(const Eigen::Vector3i& index)const{
	static_assert(R==3,"Tensor::operator(const Eigen::Vector3i&): Rank 3 access only allowed for Rank 3 tensors.");
	#ifdef TENSOR_ROW_MAJOR
		return data_[(index[0]*dim_[0]+index[1])*dim_[1]+index[2]];
	#else
		return data_[(index[2]*dim_[2]+index[1])*dim_[1]+index[0]];
	#endif
}

//==== operators - rank R ====

template <int R, class T>
T& Tensor<R,T>::operator()(const Eigen::VectorXi& in){
	#ifdef TENSOR_ROW_MAJOR
		int loc=in[0];
		for(int i=1; i<R; ++i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#else
		int loc=in[R-1];
		for(int i=R-2; i>=0; --i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#endif
	return data_[loc];
}

template <int R, class T>
const T& Tensor<R,T>::operator()(const Eigen::VectorXi& in)const{
	#ifdef TENSOR_ROW_MAJOR
		int loc=in[0];
		for(int i=1; i<R; ++i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#else
		int loc=in[R-1];
		for(int i=R-2; i>=0; --i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#endif
	return data_[loc];
}

template <int R, class T>
T& Tensor<R,T>::operator()(const std::array<int,R>& in){
	#ifdef TENSOR_ROW_MAJOR
		int loc=in[0];
		for(int i=1; i<R; ++i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#else
		int loc=in[R-1];
		for(int i=R-2; i>=0; --i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#endif
	return data_[loc];
}

template <int R, class T>
const T& Tensor<R,T>::operator()(const std::array<int,R>& in)const{
	#ifdef TENSOR_ROW_MAJOR
		int loc=in[0];
		for(int i=1; i<R; ++i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#else
		int loc=in[R-1];
		for(int i=R-2; i>=0; --i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#endif
	return data_[loc];
}

//==== index - rank 1 =====

template <int R, class T>
int Tensor<R,T>::index(int i)const{
	static_assert(R==1,"Tensor::index(int): Rank 1 access only allowed for Rank 1 tensors.");
	return i;
}

//==== index - rank 2 =====

template <int R, class T>
int Tensor<R,T>::index(int i, int j)const{
	static_assert(R==2,"Tensor::index(int,int): Rank 2 access only allowed for Rank 2 tensors.");
	#ifdef TENSOR_ROW_MAJOR
		return i*dim_[0]+j;
	#else
		return j*dim_[1]+i;
	#endif
}

//==== index - rank 3 =====

template <int R, class T>
int Tensor<R,T>::index(int i, int j, int k)const{
	static_assert(R==3,"Tensor::index(int,int,int): Rank 3 access only allowed for Rank 3 tensors.");
	#ifdef TENSOR_ROW_MAJOR
		return (i*dim_[0]+j)*dim_[1]+k;
	#else
		return (k*dim_[2]+j)*dim_[1]+i;
	#endif
}

//==== index - rank R =====

template <int R, class T>
int Tensor<R,T>::index(const Eigen::VectorXi& in)const{
	#ifdef TENSOR_ROW_MAJOR
		int loc=in[0];
		for(int i=1; i<R; ++i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#else
		int loc=in[R-1];
		for(int i=R-2; i>=0; --i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#endif
	return loc;
}

template <int R, class T>
int Tensor<R,T>::index(const std::array<int,R>& in)const{
	#ifdef TENSOR_ROW_MAJOR
		int loc=in[0];
		for(int i=1; i<R; ++i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#else
		int loc=in[R-1];
		for(int i=R-2; i>=0; --i){
			loc*=dim_[i]; loc+=in[i]; 
		}
	#endif
	return loc;
}

//==== member functions =====

template <int R, class T>
void Tensor<R,T>::clear(){
	size_=0;
	dim_*=0;
	data_.clear();
}

template <int R, class T>
void Tensor<R,T>::resize(const Eigen::VectorXi& dim, const T& vinit){
	if(R!=dim.size()) throw std::invalid_argument("Tensor<R,T>::resize(const Eigen::VectorXi&,const T&): Invalid rank/dimension");
	for(int i=0; i<dim.size(); ++i) if(dim[i]<=0) throw std::invalid_argument("Tensor<R,T>::resize(const Eigen::VectorXi&, const T&): Invalid dimension");
	dim_=dim;
	size_=dim_.prod();
	data_.resize(size_,vinit);
}

template <int R, class T>
void Tensor<R,T>::resize(const std::array<int,R>& dim, const T& vinit){
	for(int i=0; i<dim.size(); ++i) if(dim[i]<=0) throw std::invalid_argument("Tensor<R,T>::resize(const std::array<int,R>&, const T&): Invalid dimension");
	dim_.resize(dim.size());
	for(int i=0; i<dim.size(); ++i) dim_[i]=dim[i];
	size_=dim_.prod();
	data_.resize(size_,vinit);
}

template <int R, class T>
void Tensor<R,T>::resize(int d, const T& vinit){
	if(d<=0) throw std::invalid_argument("Tensor<R,T>::resize(int,const T&): Invalid dimension");
	Eigen::VectorXi dim=Eigen::VectorXi::Constant(R,d);
	resize(dim,vinit);
}

#endif