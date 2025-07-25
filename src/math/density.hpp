#pragma once
#ifndef DENSITY_HPP
#define DENSITY_HPP

// c++
#include <array>
#include <stdexcept>
// memory
#include "mem/tensor.hpp"

template <int D> 
class Density{
private:
	//bins
	std::array<double,D> len_,min_,max_;
	std::array<int,D> nbins_,index_;
	//density
	Tensor<D,double> rho_;
	//count
	int c_,m_;
	double lp_;
public:
	//==== contructors/destructors ====
	Density<D>(){defaults();}
	~Density<D>(){}
	
	//==== access ====
	//count
	const int& c()const{return c_;}
	const int& m()const{return m_;}
	//bins
	const double& lp()const{return lp_;}
	const int& nbins(int i)const{return nbins_[i];}
	const double& len(int i)const{return len_[i];}
	const double& min(int i)const{return min_[i];}
	const double& max(int i)const{return min_[i];}
	std::array<int,D>& nbins()const{return nbins_;}
	std::array<double,D>& len()const{return len_;}
	std::array<double,D>& min()const{return min_;}
	std::array<double,D>& max()const{return max_;}
	//density
	const Tensor<D,double>& rho()const{return rho_;}
	std::array<double,D>& abscissa(const std::array<int,D>& index, std::array<double,D>& a)const;
	double ordinate(const std::array<int,D>& index)const;
	Eigen::Matrix<double,D,1>& abscissa(const Eigen::Matrix<int,D,1>& index, Eigen::Matrix<double,D,1>& a)const;
	double ordinate(const Eigen::Matrix<int,D,1>& index)const;
	
	//==== member functions ====
	void clear();
	void defaults(){clear();}
	void init(const std::array<double,D>& min, const std::array<double,D>& max, const std::array<int,D>& nbins);
	void init(const std::array<double,D>& min, const std::array<double,D>& max, const std::array<double,D>& len);
	void init(double min, double max, int nbins);
	void init(double min, double max, double len);
	std::array<int,D> bin(const std::array<double,D>& x);
	std::array<int,D>& bin(const std::array<double,D>& x, std::array<int,D>& index);
	std::array<int,D> bin(const Eigen::Matrix<double,D,1>& x);
	std::array<int,D>& bin(const Eigen::Matrix<double,D,1>& x, std::array<int,D>& index);
	void push(const std::array<double,D>& x){push(x,1.0);}
	void push(const std::array<double,D>& x, double y);
	void push(const Eigen::Matrix<double,D,1>& x){push(x,1.0);}
	void push(const Eigen::Matrix<double,D,1>& x, double y);
	std::array<double,D>& posnl(std::array<int,D>& index, std::array<double,D>& p);
	std::array<double,D>& posnr(std::array<int,D>& index, std::array<double,D>& p);
	std::array<double,D>& posna(std::array<int,D>& index, std::array<double,D>& p);
};

//==== member functions ===

template <int D>
void Density<D>::clear(){
	//bins
	nbins_.fill(0);
	len_.fill(0);
	min_.fill(0);
	max_.fill(0);
	//density
	rho_.clear();
	//count
	c_=0;
	m_=0;
}

template <int D>
void Density<D>::init(const std::array<double,D>& min, const std::array<double,D>& max, const std::array<int,D>& nbins){
	//check
	for(int i=0; i<D; ++i){
		if(min[i]>=max[i]) throw std::invalid_argument(
			"ERROR in Density::init(const std::array<double,D>&,const std::array<double,D>&,const std::array<int,D>&):\
			Invalid min/max, min must be less than max"
		);
	}
	for(int i=0; i<D; ++i){
		if(nbins[i]==0) throw std::invalid_argument(
			"ERROR in Density::init(const std::array<double,D>&,const std::array<double,D>&,const std::array<int,D>&):\
			Invalid nbins, must be greater than zero."
		);
	}
	//set
	c_=0;
	m_=0;
	min_=min;
	max_=max;
	nbins_=nbins;
	lp_=1.0;
	for(int i=0; i<D; ++i){
		len_[i]=(max_[i]-min_[i])/nbins_[i];
		lp_*=len_[i];
	}
	//rho
	rho_.resize(nbins_,0.0);
}

template <int D>
void Density<D>::init(const std::array<double,D>& min, const std::array<double,D>& max, const std::array<double,D>& len){
	//check
	for(int i=0; i<D; ++i){
		if(min[i]>=max[i]) throw std::invalid_argument(
			"ERROR in Density::init(const std::array<double,D>&,const std::array<double,D>&,const std::array<double,D>&):\
			Invalid min/max, min must be less than max"
		);
	}
	for(int i=0; i<D; ++i){
		if(len[i]==0) throw std::invalid_argument(
			"ERROR in Density::init(const std::array<double,D>&,const std::array<double,D>&,const std::array<double,D>&):\
			Invalid len, must be greater than zero."
		);
	}
	//set
	c_=0;
	m_=0;
	min_=min;
	max_=max;
	for(int i=0; i<D; ++i){
		nbins_[i]=(max_[i]-min_[i])/len[i];
	}
	lp_=1.0;
	for(int i=0; i<D; ++i){
		len_[i]=(max_[i]-min_[i])/nbins_[i];
		lp_*=len_[i];
	}
	//rho
	rho_.resize(nbins_,0.0);
}

template <int D>
void Density<D>::init(double min, double max, int nbins){
	std::array<double,D> mina; mina.fill(min);
	std::array<double,D> maxa; maxa.fill(max);
	std::array<int,D> nbinsa; nbinsa.fill(nbins);
	init(mina,maxa,nbinsa);
}

template <int D>
void Density<D>::init(double min, double max, double len){
	std::array<double,D> mina; mina.fill(min);
	std::array<double,D> maxa; maxa.fill(max);
	std::array<double,D> lena; lena.fill(len);
	init(mina,maxa,lena);
}

template <int D>
std::array<int,D> Density<D>::bin(const std::array<double,D>& x){
	std::array<int,D> index;
	for(int i=0; i<D; ++i){
		if(min_[i]<=x[i] && x[i]<=max_[i]){
			int uLim=nbins_[i];
			int lLim=0;
			int mid;
			while(uLim-lLim>1){
				mid=lLim+(uLim-lLim)/2;
				if(min_[i]+len_[i]*lLim<=x[i] && x[i]<=min_[i]+len_[i]*mid) uLim=mid;
				else lLim=mid;
			}
			index[i]=lLim;
		} else index[i]=-1;
	}
	return index;
}

template <int D>
std::array<int,D>& Density<D>::bin(const std::array<double,D>& x, std::array<int,D>& index){
	for(int i=0; i<D; ++i){
		if(min_[i]<=x[i] && x[i]<=max_[i]){
			int uLim=nbins_[i];
			int lLim=0;
			int mid;
			while(uLim-lLim>1){
				mid=lLim+(uLim-lLim)/2;
				if(min_[i]+len_[i]*lLim<=x[i] && x[i]<=min_[i]+len_[i]*mid) uLim=mid;
				else lLim=mid;
			}
			index[i]=lLim;
		} else index[i]=-1;
	}
	return index;
}

template <int D>
std::array<int,D> Density<D>::bin(const Eigen::Matrix<double,D,1>& x){
	std::array<int,D> index;
	for(int i=0; i<D; ++i){
		if(min_[i]<=x[i] && x[i]<=max_[i]){
			int uLim=nbins_[i];
			int lLim=0;
			int mid;
			while(uLim-lLim>1){
				mid=lLim+(uLim-lLim)/2;
				if(min_[i]+len_[i]*lLim<=x[i] && x[i]<=min_[i]+len_[i]*mid) uLim=mid;
				else lLim=mid;
			}
			index[i]=lLim;
		} else index[i]=-1;
	}
	return index;
}

template <int D>
std::array<int,D>& Density<D>::bin(const Eigen::Matrix<double,D,1>& x, std::array<int,D>& index){
	for(int i=0; i<D; ++i){
		if(min_[i]<=x[i] && x[i]<=max_[i]){
			int uLim=nbins_[i];
			int lLim=0;
			int mid;
			while(uLim-lLim>1){
				mid=lLim+(uLim-lLim)/2;
				if(min_[i]+len_[i]*lLim<=x[i] && x[i]<=min_[i]+len_[i]*mid) uLim=mid;
				else lLim=mid;
			}
			index[i]=lLim;
		} else index[i]=-1;
	}
	return index;
}

template <int D>
void Density<D>::push(const std::array<double,D>& x, double y){
	bin(x,index_);
	bool miss=false;
	for(int i=0; i<D; ++i){
		if(index_[i]<0){miss=true; break;}
	}
	if(!miss) rho_(index_)++;
	else m_++;
	c_++;
}

template <int D>
void Density<D>::push(const Eigen::Matrix<double,D,1>& x, double y){
	bin(x,index_);
	bool miss=false;
	for(int i=0; i<D; ++i){
		if(index_[i]<0){miss=true; break;}
	}
	if(!miss) rho_(index_)++;
	else m_++;
	c_++;
}

template <int D>
std::array<double,D>& Density<D>::abscissa(const std::array<int,D>& index, std::array<double,D>& a)const{
	for(int i=0; i<D; ++i){
		a[i]=min_[i]+len_[i]*(1.0*index[i]+0.5);
	}
	return a;
}

template <int D>
double Density<D>::ordinate(const std::array<int,D>& index)const{
	return rho_(index)/(c_*lp_);
}

template <int D>
Eigen::Matrix<double,D,1>& Density<D>::abscissa(const Eigen::Matrix<int,D,1>& index, Eigen::Matrix<double,D,1>& a)const{
	for(int i=0; i<D; ++i){
		a[i]=min_[i]+len_[i]*(1.0*index[i]+0.5);
	}
	return a;
}

template <int D>
double Density<D>::ordinate(const Eigen::Matrix<int,D,1>& index)const{
	return rho_(index)/(c_*lp_);
}

template <int D>
std::array<double,D>& Density<D>::posnl(std::array<int,D>& index, std::array<double,D>& p){
	for(int i=0; i<D; ++i){
		p[i]=min_[i]+len_[i]*i;
	}
}

template <int D>
std::array<double,D>& Density<D>::posnr(std::array<int,D>& index, std::array<double,D>& p){
	for(int i=0; i<D; ++i){
		p[i]=min_[i]+len_[i]*(i+1);
	}
}

template <int D>
std::array<double,D>& Density<D>::posna(std::array<int,D>& index, std::array<double,D>& p){
	for(int i=0; i<D; ++i){
		p[i]=min_[i]+len_[i]*(i+0.5);
	}
}

#endif