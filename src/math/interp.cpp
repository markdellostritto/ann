//c
#include <cstring>
#include <cmath>
// c++
#include <iostream>
#include <stdexcept>
// math
#include "math/interp.hpp"

namespace math{
namespace interp{
	
//***************************************************
// Name
//***************************************************

std::ostream& operator<<(std::ostream& out, const Name& name){
	switch(name){
		case Name::CONST: out<<"CONST"; break;
		case Name::LINEAR: out<<"LINEAR"; break;
		case Name::AKIMA: out<<"AKIMA"; break;
		case Name::RBFI: out<<"RBFI"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Name::name(const Name& name){
	switch(name){
		case Name::CONST: return "CONST";
		case Name::LINEAR: return "LINEAR";
		case Name::AKIMA: return "AKIMA";
		case Name::RBFI: return "RBFI";
		default: return "UNKNOWN";
	}
}

Name Name::read(const char* str){
	if(std::strcmp(str,"CONST")==0) return Name::CONST;
	else if(std::strcmp(str,"LINEAR")==0) return Name::LINEAR;
	else if(std::strcmp(str,"AKIMA")==0) return Name::AKIMA;
	else if(std::strcmp(str,"RBFI")==0) return Name::RBFI;
	else return Name::UNKNOWN;
}

//***************************************************
// Base
//***************************************************

void Base::resize(int s){
	if(s<0) throw std::invalid_argument("math::interp::Base::resize(int): Invalid size.");
	size_=s;
	x_.resize(size_);
	y_.resize(size_);
	d_.resize(size_);
}

void Base::clear(){
	name_=Name::UNKNOWN;
}

int Base::search(double x){
	/*
	int l=0;
	int u=size_-1;
	int i=(u+l)/2;
	while(u-l>1){
		if(x<x_[i]) u=i;
		else l=i;
		i=(u+l)/2;
	}
	return i;
	*/
	return (int)((x-x_[0])/(x_[size_-1]-x_[0])*size_);
}

//***************************************************
// Const
//***************************************************

double Const::eval(double x){
	const int i=search(x);
	return y_[i];
}

//***************************************************
// Const
//***************************************************

double Linear::eval(double x){
	const int i=search(x);
	return y_[i]+d_[i]*(x-x_[i]);
}

//***************************************************
// AKIMA
//***************************************************

void Akima::resize(int s){
	Base::resize(s);
	s_.resize(size_);
	m_.resize(size_);
}

void Akima::init(){
	//set m_
	for(int i=0; i<size_-1; ++i){
		m_[i]=(y_[i+1]-y_[i])/(x_[i+1]-x_[i]);
	}
	m_[size_-1]=(y_[size_-1]-y_[size_-2])/(x_[size_-1]-x_[size_-2]);
	//set s_
	s_[0]=m_[0];
	s_[1]=0.5*(m_[0]+m_[1]);
	for(int i=2; i<size_-2; ++i){
		s_[i]=(std::fabs(m_[i+1]-m_[i])*m_[i-1]+std::fabs(m_[i-1]-m_[i-2])*m_[i])/(std::fabs(m_[i+1]-m_[i])+std::fabs(m_[i-1]-m_[i-2]));
	}
	s_[size_-2]=0.5*(m_[size_-3]+m_[size_-2]);
	s_[size_-1]=m_[size_-2];
}

double Akima::eval(double x){
	const int i=search(x);
	const double dx=(x-x_[i]);
	const double den=1.0/(x_[i+1]-x_[i]);
	return y_[i]+dx*(s_[i]+dx*den*((3.0*m_[i]-2.0*s_[i]-s_[i+1])+dx*den*(s_[i]+s_[i+1]-2.0*m_[i])));
}

//***************************************************
// RBFI
//***************************************************

void RBFI::resize(int s){
	Base::resize(s);
	m_.resize(size_,size_);
	z_.resize(size_);
}

void RBFI::init(){
	//build the matrix
	for(int j=0; j<size_; ++j){
		m_(j,j)=1.0;
		for(int i=j+1; i<size_; ++i){
			m_(i,j)=rbf_.val(x_[i]);
			m_(j,i)=m_(i,j);
		}
	}
	//solve
	//z_=m_.partialPivLu().solve(y_);
	z_=m_.llt().solve(y_);
}

double RBFI::eval(double x){
	double val=0;
	for(int i=0; i<size_; ++i){
		val+=z_[i]*rbf_.val(x-x_[i]);
	}
	return val;
}

}//end namspace interp
}//end namespace math