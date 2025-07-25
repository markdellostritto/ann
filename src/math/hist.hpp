#pragma once
#ifndef HIST_HPP
#define HIST_HPP

// c++
#include <iostream>
#include <vector>

class Histogram{
private:
	int nbins_,c_,m_;//nbins,count
	double len_,min_,max_;//bin len, min, max
	bool norm_;//whether to normalize histogram
	std::vector<double> hist_;
public:
	//==== constructors/destructors ===
	Histogram():nbins_(0),m_(0),c_(0),len_(0),min_(0),max_(0),norm_(true){}
	Histogram(double min, double max, int nbins):m_(0){init(min,max,nbins);}
	~Histogram(){};
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Histogram& hist);
	Histogram& operator+=(const Histogram& hist);
	
	//==== access ====
	bool& norm(){return norm_;};
	const bool& norm()const{return norm_;}
	const int& nbins()const{return nbins_;}
	const int& c()const{return c_;}
	const int& m()const{return m_;}
	const double& len()const{return len_;}
	const double& min()const{return min_;}
	const double& max()const{return min_;}
	double hist(int i)const{return hist_[i];}
	double abscissa(int i)const{return min_+len_*(1.0*i+0.5);}
	double ordinate(int i)const{return hist_[i]/c_;}
	
	//==== member functions ====
	void clear();
	void init(double min, double max, int nbins);
	void init(double min, double max, double len);
	int bin(double x);
	void push(double x){push(x,1.0);}
	void push(double x, double y);
	double avg()const;
	
};
bool operator==(const Histogram& hist1, const Histogram& hist2);
bool operator!=(const Histogram& hist1, const Histogram& hist2);

#endif