// c++
#include <limits>
#include <iostream>
// math
#include "math/reduce.hpp"

//==== operators ====

Reduce<1>& Reduce<1>::operator+=(const Reduce<1>& r){
	//min
	if(r.min()<min_) min_=r.min();
	//max
	if(r.max()>max_) max_=r.max();
	//avg
	const int N1=N_;
	const int N2=r.N();
	const double avg1=avg_;
	const double avg2=r.avg();
	const double delta=avg2-avg1;
	const double m2_1=m2_;
	const double m2_2=r.m2();
	N_=N1+N2;
	if(N_>0){
		avg_=(N1*avg1+N2*avg2)/(N1+N2);
		m2_=m2_1+m2_2+delta*delta*N1*N2/(N1+N2);
	} else {
		avg_=0;
		m2_=0;
	}
	//return
	return *this;
}

Reduce<1> operator+(const Reduce<1>& r1, const Reduce<1>& r2){
	Reduce<1> rnew=r1;
	rnew+=r2;
	return rnew;
}

//==== member functions ====

void Reduce<1>::defaults(){
	N_=0;
	min_=std::numeric_limits<double>::max();
	max_=-std::numeric_limits<double>::max();
	avg_=0;
	m2_=0;
}

void Reduce<1>::push(double x){
	//min
	if(x<min_) min_=x;
	//max
	if(x>max_) max_=x;
	//avg/var
	const double delta_=(x-avg_);
	avg_+=delta_/(N_+1);
	m2_+=delta_*(x-avg_);
	//count
	N_++;
}

//***************************************************************************
// Reduction - 2D
//***************************************************************************

//==== operators ====

Reduce<2>& Reduce<2>::operator+=(const Reduce<2>& r){
	//avg
	const int N1=N_;
	const int N2=r.N();
	const double avgX1=avgX_;
	const double avgX2=r.avgX();
	const double avgY1=avgY_;
	const double avgY2=r.avgY();
	const double deltaX=avgX2-avgX1;
	const double deltaY=avgY2-avgY1;
	const double m2X_1=m2X_;
	const double m2X_2=r.m2X();
	const double m2Y_1=m2Y_;
	const double m2Y_2=r.m2Y();
	N_=N1+N2;
	if(N_>0){
		avgX_=(N1*avgX1+N2*avgX2)/(N1+N2);
		avgY_=(N1*avgY1+N2*avgY2)/(N1+N2);
		m2X_=m2X_1+m2X_2+deltaX*deltaX*N1*N2/(N1+N2);
		m2Y_=m2Y_1+m2Y_2+deltaY*deltaY*N1*N2/(N1+N2);
	} else {
		avgX_=0;
		avgY_=0;
		m2X_=0;
		m2Y_=0;
	}
	//return
	return *this;
}

//==== member functions ====

void Reduce<2>::defaults(){
	N_=0;
	avgX_=0;
	avgY_=0;
	m2X_=0;
	m2Y_=0;
	covar_=0;
}

void Reduce<2>::push(double x, double y){
	const double dX_=(x-avgX_);
	const double dY_=(y-avgY_);
	avgX_+=dX_/(N_+1);
	avgY_+=dY_/(N_+1);
	m2X_+=dX_*(x-avgX_);
	m2Y_+=dY_*(y-avgY_);
	covar_+=dX_*(y-avgY_);
	N_++;
}

//***************************************************************************
// serialization
//***************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Reduce<1>& obj){
		int size=0;
		size+=sizeof(int);//N_
		size+=sizeof(double);//max_
		size+=sizeof(double);//min_
		size+=sizeof(double);//avg_
		size+=sizeof(double);//m2_
		return size;
	}
	template <> int nbytes(const Reduce<2>& obj){
		int size=0;
		size+=sizeof(int);//N_
		size+=sizeof(double);//avgX_
		size+=sizeof(double);//avgY_
		size+=sizeof(double);//m2X_
		size+=sizeof(double);//m2Y_
		size+=sizeof(double);//covar_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Reduce<1>& obj, char* arr){
		int pos=0;
		int tmpI; double tmpD;
		std::memcpy(arr+pos,&(tmpI=obj.N()),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&(tmpD=obj.min()),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&(tmpD=obj.max()),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&(tmpD=obj.avg()),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&(tmpD=obj.var()),sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	template <> int pack(const Reduce<2>& obj, char* arr){
		int pos=0;
		int tmpI; double tmpD;
		std::memcpy(arr+pos,&(tmpI=obj.N()),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&(tmpD=obj.avgX()),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&(tmpD=obj.avgY()),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&(tmpD=obj.m2X()),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&(tmpD=obj.m2Y()),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&(tmpD=obj.covar()),sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Reduce<1>& obj, const char* arr){
		int pos=0;
		int N;
		double min,max,avg,m2;
		std::memcpy(&N,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&min,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&max,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&avg,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&m2,arr+pos,sizeof(double)); pos+=sizeof(double);
		obj.setN(N);
		obj.setmin(min);
		obj.setmax(max);
		obj.setavg(avg);
		obj.setm2(m2);
		return pos;
	}
	template <> int unpack(Reduce<2>& obj, const char* arr){
		int pos=0;
		int N;
		double avgX,avgY,m2X,m2Y,covar;
		std::memcpy(&N,arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&avgX,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&avgY,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&m2X,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&m2Y,arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&covar,arr+pos,sizeof(double)); pos+=sizeof(double);
		obj.setN(N);
		obj.setavgX(avgX);
		obj.setavgY(avgY);
		obj.setm2X(m2X);
		obj.setm2Y(m2Y);
		obj.setcovar(covar);
		return pos;
	}

}