#ifndef ANN_LMAT_HPP
#define ANN_LMAT_HPP

#include <vector>

template <class T>
class LMat{
private:
	unsigned int n_;
	std::vector<std::vector<T> > mat_;
public:
	//constructors/destructors
	LMat():n_(0){};
	LMat(unsigned int n){resize(n);};
	LMat(unsigned int n, const T& t){resize(n,t);};
	~LMat(){};
	
	//access
	unsigned int& n(){return n_;};
	const unsigned int& n()const{return n_;};
	T& operator()(unsigned int i, unsigned int j){return (i>=j)?mat_[i][j]:mat_[j][i];};
	const T& operator()(unsigned int i, unsigned int j)const{return (i>=j)?mat_[i][j]:mat_[j][i];}
	T& operator[](unsigned int i);
	const T& operator[](unsigned int i)const;
	
	//member functions
	void clear();
	unsigned int size()const{return (n_*(n_+1))/2;};
	void resize(unsigned int n);
	void resize(unsigned int n, const T& t);
	unsigned int index(unsigned int i, unsigned int j);
};

//access

template <class T>
T& LMat<T>::operator[](unsigned int n){
	for(unsigned int i=0; i<n_; ++i){
		if(n<mat_[i].size()) return mat_[i][n];
		else n-=mat_[i].size();
	}
	throw std::runtime_error("Invalid lmat index.");
}

template <class T>
const T& LMat<T>::operator[](unsigned int n)const{
	for(unsigned int i=0; i<n_; ++i){
		if(n<mat_[i].size()) return mat_[n];
		else n-=mat_[i].size();
	}
	throw std::runtime_error("Invalid lmat index.");
}
	
//member functions

template <class T>
void LMat<T>::clear(){
	n_=0;
	mat_.clear();
}

template <class T>
void LMat<T>::resize(unsigned int n){
	n_=n; mat_.resize(n_);
	for(unsigned int i=0; i<n_; ++i) mat_[i].resize(i+1);
}

template <class T>
void LMat<T>::resize(unsigned int n, const T& t){
	n_=n; mat_.resize(n_);
	for(unsigned int i=0; i<n_; ++i) mat_[i].resize(i+1,t);
}

template <class T>
unsigned LMat<T>::index(unsigned int i, unsigned int j){
	unsigned int ii=i,jj=j;
	if(i<j){ii=j;jj=i;}
	unsigned int index=0;
	for(unsigned int n=0; n<ii; ++n) index+=mat_[n].size();
	return index+jj;
}

#endif
