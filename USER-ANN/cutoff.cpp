// c libraries
#include <cstring>
// c++ libraries
#include <ostream> 
// ann - cutoff
#include "cutoff.h"

namespace cutoff{
	
//************************************************************
// CUTOFF NAMES
//************************************************************

Name::type Name::read(const char* str){
	if(std::strcmp(str,"COS")==0) return Name::COS;
	else if(std::strcmp(str,"TANH")==0) return Name::TANH;
	else return Name::UNKNOWN;
}

const char* Name::name(const Name::type& cutt){
	switch(cutt){
		case Name::COS: return "COS";
		case Name::TANH: return "TANH";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Name::type& cutt){
	switch(cutt){
		case Name::COS: out<<"COS"; break;
		case Name::TANH: out<<"TANH"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//************************************************************
// CUTOFF FUNCTIONS
//************************************************************
	
//==== Cos ====

//operators

std::ostream& operator<<(std::ostream& out, const Cos& obj){
	return out<<"COS rc "<<obj.rc_<<" rci "<<obj.rci_;
}

//member functions

/**
* compute the value of the cutoff function
* @param r - interatomic distance
* @return the value of the cutoff function
*/
double Cos::val(double r)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return (r>rc_)?0:0.5*(math::special::cos(r*prci_)+1.0);
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return (r>rc_)?0:0.5*(cos(r*prci_)+1.0);
	#endif
}

/**
* compute the gradient of the cutoff function
* @param r - interatomic distance
* @return the gradient of the cutoff function
*/
double Cos::grad(double r)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	return (r>rc_)?0:-0.5*prci_*math::special::sin(r*prci_);
	#elif (defined __ICC || defined __INTEL_COMPILER)
	return (r>rc_)?0:-0.5*prci_*sin(r*prci_);
	#endif
}

/**
* compute the value and gradient of the cutoff function
* @param r - interatomic distance
* @param v - stores value
* @param g - stores gradient
*/
void Cos::compute(double r, double& v, double& g)const{
	if(r<rc_){
		#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
		v=0.5*(math::special::cos(r*prci_)+1.0);
		g=-0.5*prci_*math::special::sin(r*prci_);
		#elif (defined __ICC || defined __INTEL_COMPILER)
		v=0.5*(cos(r*prci_)+1.0);
		g=-0.5*prci_*sin(r*prci_);
		#endif
	} else {
		v=0;
		g=0;
	}
}

//==== Tanh ====

//operators

std::ostream& operator<<(std::ostream& out, const Tanh& obj){
	return out<<"TANH rc "<<obj.rc_<<" rci "<<obj.rci_;
}

//member functions

/**
* compute the value of the cutoff function
* @param r - interatomic distance
* @return the value of the cutoff function
*/
double Tanh::val(double r)const{
	#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
	const double f=(r>rc_)?0:std::tanh(1.0-r*rci_); return f*f*f;
	#elif (defined __ICC || defined __INTEL_COMPILER)
	const double f=(r>rc_)?0:tanh(1.0-r*rci_); return f*f*f;
	#endif
}

/**
* compute the gradient of the cutoff function
* @param r - interatomic distance
* @return the gradient of the cutoff function
*/
double Tanh::grad(double r)const{
	if(r<rc_){
		#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
		const double f=std::tanh(1.0-r*rci_);
		#elif (defined __ICC || defined __INTEL_COMPILER)
		const double f=tanh(1.0-r*rci_);
		#endif
		return -3.0*f*f*(1.0-f*f)*rci_;
	} else return 0;
}

/**
* compute the value and gradient of the cutoff function
* @param r - interatomic distance
* @param v - stores value
* @param g - stores gradient
*/
void Tanh::compute(double r, double& v, double& g)const{
	if(r<rc_){
		#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
		const double f=std::tanh(1.0-r*rci_);
		#elif (defined __ICC || defined __INTEL_COMPILER)
		const double f=tanh(1.0-r*rci_);
		#endif
		v=f*f*f;
		g=-3.0*f*f*(1.0-f*f)*rci_;
	} else {
		v=0;
		g=0;
	}
}

}
