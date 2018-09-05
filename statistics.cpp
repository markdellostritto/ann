#include "statistics.hpp"

namespace stats{

//***********************************************
//Histogram Class
//***********************************************

//constructors/destructors

Histogram::Histogram(unsigned int nBins, double min, double max){
	hist_.resize(nBins,0);
	min_=min; max_=max;
	len_=(max_-min_)/nBins;
	nData_=0;
}

//operators

std::ostream& operator<<(std::ostream& out, const Histogram& hist){
	out<<"*************************** Histogram ***************************\n";
	out<<"N_DATA = "<<hist.nData_<<"\n";
	out<<"N_BINS = "<<hist.hist_.size()<<"\n";
	out<<"BIN_LEN = "<<hist.len_<<"\n";
	out<<"LIM = ("<<hist.min_<<","<<hist.max_<<")\n";
	out<<"*************************** Histogram ***************************\n";
}

//member functions

void Histogram::defaults(){
	nData_=0;
	len_=0;
	min_=0;
	max_=0;
	hist_.clear();
}

unsigned int Histogram::bin(const Histogram& hist, double datum){
	unsigned int uLim=hist.nBins();//upper limit for Newton's method
	unsigned int lLim=0;//lower limit for Newton's method
	unsigned int mid;//middle point for Newton's method
	
	while(uLim-lLim>1){
		mid=lLim+(uLim-lLim)/2;
		if(hist.min()+hist.len()*lLim<=datum && datum<=hist.min()+hist.len()*mid) uLim=mid;
		else lLim=mid;
	}
	
	return lLim;
}

//static functions

double Histogram::avg(const Histogram& hist){
	double norm=0,avg=0;
	for(unsigned int i=0; i<hist.nBins(); ++i){
		avg+=hist[i]*(hist.min()+hist.len()*(i+0.5));
		norm+=hist[i];
	}
	return (norm>0)?avg/norm:0.0;
}

double Histogram::var(const Histogram& hist){
	double avg=Histogram::avg(hist);
	double norm=0,s=0;
	for(unsigned int i=0; i<hist.nBins(); ++i){
		s+=((hist.min()+hist.len()*(i+0.5))-avg)*((hist.min()+hist.len()*(i+0.5))-avg)*hist[i];
		norm+=hist[i];
	}
	return s/norm;
}

double Histogram::sigma(const Histogram& hist){
	double avg=Histogram::avg(hist);
	double norm=0,s=0;
	for(unsigned int i=0; i<hist.nBins(); ++i){
		s+=((hist.min()+hist.len()*(i+0.5))-avg)*((hist.min()+hist.len()*(i+0.5))-avg)*hist[i];
		norm+=hist[i];
	}
	return std::sqrt(s/norm);
}

double Histogram::max_val(const Histogram& hist){
	double max=hist[0];
	for(unsigned int i=1; i<hist.nBins(); ++i) if(hist[i]>max) max=hist[i];
	return max;
}

double Histogram::min_val(const Histogram& hist){
	double min=hist[0];
	for(unsigned int i=1; i<hist.nBins(); ++i) if(hist[i]<min) min=hist[i];
	return min;
}

unsigned int Histogram::max_bin(const Histogram& hist){
	double max=hist[0];
	unsigned int indexMax=0;
	for(unsigned int i=1; i<hist.nBins(); ++i){
		if(hist[i]>max){
			max=hist[i];
			indexMax=i;
		}
	}
	return indexMax;
}

unsigned int Histogram::min_bin(const Histogram& hist){
	double min=hist[0];
	unsigned int indexMin=0;
	for(unsigned int i=1; i<hist.nBins(); ++i){
		if(hist[i]<min){
			min=hist[i];
			indexMin=i;
		}
	}
	return indexMin;
}

double Histogram::max_loc(const Histogram& hist){
	double max=hist[0];
	unsigned int indexMax=0;
	for(unsigned int i=1; i<hist.nBins(); ++i){
		if(hist[i]>max){
			max=hist[i];
			indexMax=i;
		}
	}
	return hist.min()+hist.len()*(indexMax+0.5);
}

double Histogram::min_loc(const Histogram& hist){
	double min=hist[0];
	unsigned int indexMin=0;
	for(unsigned int i=1; i<hist.nBins(); ++i){
		if(hist[i]<min){
			min=hist[i];
			indexMin=i;
		}
	}
	return hist.min()+hist.len()*(indexMin+0.5);
}

//operators

bool operator==(const Histogram& hist1, const Histogram& hist2){
	if(hist1.len()!=hist2.len()) return false;
	else if(hist1.min()!=hist2.min()) return false;
	else if(hist1.max()!=hist2.max()) return false;
	else return true;
}

bool operator!=(const Histogram& hist1, const Histogram& hist2){
	return !(hist1==hist2);
}

Histogram& operator+=(Histogram& hist1, const Histogram& hist2){
	if(hist1!=hist2) throw std::invalid_argument("Incompatible histograms: cannot sum.");
	for(unsigned int i=0; i<hist1.nBins(); ++i) hist1[i]+=hist2[i];
	hist1.nData()+=hist2.nData();
	return hist1;
}

Histogram& operator*=(Histogram& hist, double c){
	for(unsigned int i=0; i<hist.nBins(); ++i) hist[i]*=c;
	return hist;
}

}