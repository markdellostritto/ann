// c++
#include <stdexcept>
// math
#include "math/hist.hpp"

//==== operators ====

bool operator==(const Histogram& hist1, const Histogram& hist2){
	return (
		hist1.nbins()==hist2.nbins() &&
		hist1.len()==hist2.len() &&
		hist1.min()==hist2.min() && 
		hist1.max()==hist2.max()
	);
}

bool operator!=(const Histogram& hist1, const Histogram& hist2){
	return !(hist1==hist2);
}

Histogram& Histogram::operator+=(const Histogram& dist){
	//check that the dist is compatible
	if(*this!=dist) throw std::invalid_argument("Incompatible Histogram.");
	//average the two histograms
	for(int i=0; i<nbins_; ++i) hist_[i]=0.5*(hist_[i]+dist.hist(i));
	//update the global count
	c_+=dist.c();
	//return dist
	return *this;
}

std::ostream& operator<<(std::ostream& out, const Histogram& hist){
	return out<<"nbins "<<hist.nbins_<<" len "<<hist.len_<<" min "<<hist.min_<<" max "<<hist.max_<<" "<<hist.norm_;
}

//==== member functions ====

void Histogram::clear(){
	nbins_=0; c_=0;
	len_=0; min_=0; max_=0;
	norm_=true;
	hist_.clear();
}

void Histogram::init(double min, double max, int nbins){
	if(min>=max) throw std::runtime_error("Invalid Histogram limits.");
	if(nbins==0) throw std::runtime_error("Invalid Histogram nbins.");
	c_=0;
	min_=min;
	max_=max;
	nbins_=nbins;
	len_=(max-min)/nbins_;
	hist_.resize(nbins_,0);
}

void Histogram::init(double min, double max, double len){
	if(min>=max) throw std::runtime_error("Invalid Histogram limits.");
	if(len==0) throw std::runtime_error("Invalid Histogram len.");
	c_=0;
	min_=min;
	max_=max;
	nbins_=(max_-min_)/len;
	len_=(max-min)/nbins_;
	hist_.resize(nbins_,0);
}

int Histogram::bin(double x){
	if(min_<=x && x<=max_){
		int uLim=nbins_;
		int lLim=0;
		int mid;
		while(uLim-lLim>1){
			mid=lLim+(uLim-lLim)/2;
			if(min_+len_*lLim<=x && x<=min_+len_*mid) uLim=mid;
			else lLim=mid;
		}
		return lLim;
	} else return -1;
}

void Histogram::push(double x, double y){
	const int bin_=bin(x);
	if(bin_>0) hist_[bin_]+=y;
	else m_++;
	c_++;
}

double Histogram::avg()const{
	double avg_=0;
	for(int i=0; i<nbins_; ++i){
		avg_+=ordinate(i)*abscissa(i);
	}
	return avg_;
}