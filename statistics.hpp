#ifndef STATISTICS_H
#define STATISTICS_H

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <exception>
#include <vector>

#ifndef DEBUG_STATS
#define DEBUG_STATS 1
#endif

namespace stats{
	
	//***********************************************
	//Histogram Class
	//***********************************************
	
	class Histogram{
	private:
		unsigned int nData_;//total number of datums read in
		std::vector<double> hist_;//the histogram
		double len_;//the length of a bin
		double min_,max_;//the minimum and maximum values of the histogram
	public:
		//constructors/destructors
		Histogram(){defaults();};
		Histogram(unsigned int nBins, double min, double max);
		~Histogram(){};
		
		//operators
		friend std::ostream& operator<<(std::ostream& out, const Histogram& hist);
		double& operator[](unsigned int i){return hist_[i];};
		const double& operator[](unsigned int i)const{return hist_[i];};
		
		//static functions
		static unsigned int bin(const Histogram& hist, double datum);
		static double avg(const Histogram& hist);
		static double var(const Histogram& hist);
		static double sigma(const Histogram& hist);
		static double max_val(const Histogram& hist);
		static double min_val(const Histogram& hist);
		static unsigned int max_bin(const Histogram& hist);
		static unsigned int min_bin(const Histogram& hist);
		static double max_loc(const Histogram& hist);
		static double min_loc(const Histogram& hist);
		
		//access
		unsigned int nBins()const{return hist_.size();};
		unsigned int& nData(){return nData_;};
		const unsigned int& nData()const{return nData_;};
		double len()const{return len_;};
		double min()const{return min_;};
		double max()const{return max_;};
		const std::vector<double>& hist()const{return hist_;};
		
		//member functions
		void defaults();
		void clear(){defaults();};
		void push(double datum){++hist_[bin(*this,datum)];++nData_;};
		void push(double datum, double val){hist_[bin(*this,datum)]+=val;++nData_;};
		double abscissa(unsigned int i)const{return min_+len_*(1.0*i+0.5);};
		double ordinate(unsigned int i)const{return hist_[i];};
		
	};
	
	//operators
	
	bool operator==(const Histogram& hist1, const Histogram& hist2);
	bool operator!=(const Histogram& hist1, const Histogram& hist2);
	Histogram& operator+=(Histogram& hist1, const Histogram& hist2);
	Histogram& operator*=(Histogram& hist, double c);
	
	template <class T>
	T average(const std::vector<T>& v){
		T avg=0;
		for(unsigned int i=0; i<v.size(); ++i) avg+=v[i];
		return avg/v.size();
	}
	
	template <class T>
	T stddev(const std::vector<T>& v, T avg=0){
		if(avg==0) avg=average(v);
		T s=0;
		for(unsigned int i=0; i<v.size(); ++i) s+=(v[i]-avg)*(v[i]-avg);
		if(v.size()>1) s/=v.size()-1;
		return std::sqrt(s);
	}
	
	template <class T>
	T max(const std::vector<T>& v){
		T max=v[0];
		for(unsigned int i=1; i<v.size(); ++i){
			if(v[i]>max) max=v[i];
		}
		return max;
	}
	
	template <class T>
	T min(const std::vector<T>& v){
		T min=v[0];
		for(unsigned int i=1; i<v.size(); ++i){
			if(v[i]<min) min=v[i];
		}
		return min;
	}
	
	template <class T>
	T pcorr(const std::vector<T>& x, const std::vector<T>& y, double avgx=0, double avgy=0){
		if(x.size()!=y.size()) throw std::invalid_argument("Invalid vector sizes for correlation.");
		if(avgx==0) avgx=average(x);
		if(avgy==0) avgy=average(x);
		double n=0,dx=0,dy=0;
		for(unsigned int i=0; i<x.size(); ++i){
			n+=(x[i]-avgx)*(y[i]-avgy);
			dx+=(x[i]-avgx)*(x[i]-avgx);
			dy+=(y[i]-avgy)*(y[i]-avgy);
		}
		return n/std::sqrt(dx*dy);
	}
	
	template <class T>
	T lin_reg(const std::vector<T>& x, const std::vector<T>& y, double& m, double& b){
		if(x.size()!=y.size()) throw std::invalid_argument("Invalid vector sizes for linear regression.");
		double mx=average(x);
		double my=average(y);
		double sx=stddev(x,mx);
		double sy=stddev(y,my);
		double p=pcorr(x,y,mx,my);
		m=p*sy/sx;
		b=my-m*mx;
		double sst=0,ssr=0;
		for(unsigned int i=0; i<x.size(); ++i){
			double yy=m*x[i]+b;
			sst+=(y[i]-my)*(y[i]-my);
			ssr+=(yy-y[i])*(yy-y[i]);
		}
		return 1.0-ssr/sst;//r^2 coefficient
	}
}

#endif