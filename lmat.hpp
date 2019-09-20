#pragma once
#ifndef ANN_LMAT_HPP
#define ANN_LMAT_HPP

#include <vector>
#include "math_cmp.hpp"

template <class T>
class LMat{
private:
	unsigned int n_;
	std::vector<T> mat_;
public:
	//constructors/destructors
	LMat():n_(0){}
	LMat(unsigned int n){resize(n);}
	LMat(unsigned int n, const T& t){resize(n,t);}
	~LMat(){}
	
	//access
	unsigned int& n(){return n_;}
	const unsigned int& n()const{return n_;}
	T& operator()(unsigned int i, unsigned int j)noexcept;
	const T& operator()(unsigned int i, unsigned int j)const noexcept;
	T& operator[](unsigned int i)noexcept{return mat_[i];}
	const T& operator[](unsigned int i)const noexcept{return mat_[i];}
	
	//member functions
	void clear();
	unsigned int size()const noexcept{return (n_*(n_+1))/2;}
	void resize(unsigned int n){n_=n; mat_.resize(n*(n+1)/2);}
	void resize(unsigned int n, const T& t){n_=n; mat_.resize(n*(n+1)/2,t);}
	unsigned int index(unsigned int i, unsigned int j);
};

//access

template <class T>
T& LMat<T>::operator()(unsigned int i, unsigned int j) noexcept{
	unsigned int ii=cmp::min(i,j);
	unsigned int jj=cmp::max(i,j);
	return mat_[ii*(ii+1)/2+jj];
}

template <class T>
const T& LMat<T>::operator()(unsigned int i, unsigned int j)const noexcept{
	unsigned int ii=cmp::min(i,j);
	unsigned int jj=cmp::max(i,j);
	return mat_[ii*(ii+1)/2+jj];
}

//member functions

template <class T>
void LMat<T>::clear(){
	n_=0;
	mat_.clear();
}

template <class T>
unsigned LMat<T>::index(unsigned int i, unsigned int j){
	unsigned int ii=cmp::min(i,j);
	unsigned int jj=cmp::max(i,j);
	return ii*(ii+1)/2+jj;
}

template <class T> bool operator==(const LMat<T>& lmat1, const LMat<T>& lmat2){
	if(lmat1.n()!=lmat2.n()) return false;
	else{
		unsigned int size=lmat1.size();
		for(unsigned int i=0; i<size; ++i){
			if(lmat1[i]!=lmat2[i]) return false;
		}
		return true;
	}
}

template <class T> inline bool operator!=(const LMat<T>& lmat1, const LMat<T>& lmat2){
	return !(lmat1==lmat2);
}

#endif