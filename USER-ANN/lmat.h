#ifndef ANN_LMAT_HPP
#define ANN_LMAT_HPP

#include <vector>
#include "math_cmp_ann.h"

template <class T>
class LMat{
private:
	int n_;
	std::vector<T> mat_;
public:
	//constructors/destructors
	LMat():n_(0){}
	LMat(int n){resize(n);}
	LMat(int n, const T& t){resize(n,t);}
	~LMat(){}
	
	//access
	int& n(){return n_;}
	const int& n()const{return n_;}
	T& operator()(int i, int j);
	const T& operator()(int i, int j)const;
	T& operator[](int i){return mat_[i];}
	const T& operator[](int i)const{return mat_[i];}
	
	//member functions
	void clear();
	int size()const{return (n_*(n_+1))/2;}
	void resize(int n){n_=n; mat_.resize(n*(n+1)/2);}
	void resize(int n, const T& t){n_=n; mat_.resize(n*(n+1)/2,t);}
	int index(int i, int j)const;
};

//access

template <class T>
T& LMat<T>::operator()(int i, int j){
	if(i<j) return mat_[i*(i+1)/2+j];
	else return mat_[j*(j+1)/2+i];
}

template <class T>
const T& LMat<T>::operator()(int i, int j)const{
	if(i<j) return mat_[i*(i+1)/2+j];
	else return mat_[j*(j+1)/2+i];
}

//member functions

template <class T>
void LMat<T>::clear(){
	n_=0;
	mat_.clear();
}

template <class T>
int LMat<T>::index(int i, int j)const{
	if(i<j) return i*(i+1)/2+j;
	else return j*(j+1)/2+i;
}

template <class T> bool operator==(const LMat<T>& lmat1, const LMat<T>& lmat2){
	if(lmat1.n()!=lmat2.n()) return false;
	else{
		const int size=lmat1.size();
		for(int i=0; i<size; ++i){
			if(lmat1[i]!=lmat2[i]) return false;
		}
		return true;
	}
}

template <class T> inline bool operator!=(const LMat<T>& lmat1, const LMat<T>& lmat2){
	return !(lmat1==lmat2);
}

#endif