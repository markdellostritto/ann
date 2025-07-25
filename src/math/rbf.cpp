// c libaries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
// math
#include "math/rbf.hpp"

std::ostream& operator<<(std::ostream& out, const RBF::Name& name){
	switch(name){
		case RBF::Name::GAUSSIAN: out<<"GAUSSIAN"; break;
		case RBF::Name::MULTIQUADRIC: out<<"MULTIQUADRIC"; break;
		case RBF::Name::IMQUADRIC: out<<"IMQUADRIC"; break;
		case RBF::Name::IQUADRATIC: out<<"IQUADRATIC"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* RBF::Name::name(const RBF::Name& name){
	switch(name){
		case RBF::Name::GAUSSIAN: return "GAUSSIAN";
		case RBF::Name::MULTIQUADRIC: return "MULTIQUADRIC";
		case RBF::Name::IMQUADRIC: return "IMQUADRIC";
		case RBF::Name::IQUADRATIC: return "IQUADRATIC";
		default: return "UNKNOWN";
	}
}

RBF::Name RBF::Name::read(const char* str){
	if(std::strcmp(str,"GAUSSIAN")==0) return RBF::Name::GAUSSIAN;
	else if(std::strcmp(str,"MULTIQUADRIC")==0) return RBF::Name::MULTIQUADRIC;
	else if(std::strcmp(str,"IMQUADRIC")==0) return RBF::Name::IMQUADRIC;
	else if(std::strcmp(str,"IQUADRATIC")==0) return RBF::Name::IQUADRATIC;
	else return RBF::Name::UNKNOWN;
}

double RBF::operator()(double r)const{
	switch(name_){
		case RBF::Name::GAUSSIAN:
			return std::exp(-r*r*ieps2_);
		break;
		case RBF::Name::MULTIQUADRIC:
			return std::sqrt(r*r*ieps2_+1.0);
		break;
		case RBF::Name::IMQUADRIC:
			return 1.0/std::sqrt(r*r*ieps2_+1.0);
		break;
		case RBF::Name::IQUADRATIC:
			return 1.0/(r*r*ieps2_+1.0);
		break;
		default:
			throw std::invalid_argument("RBF::operator()(double): Invalid RBF Name.");
		break;
	}
}

double RBF::val(double r)const{
	switch(name_){
		case RBF::Name::GAUSSIAN:
			return std::exp(-r*r*ieps2_);
		break;
		case RBF::Name::MULTIQUADRIC:
			return std::sqrt(r*r*ieps2_+1.0);
		break;
		case RBF::Name::IMQUADRIC:
			return 1.0/std::sqrt(r*r*ieps2_+1.0);
		break;
		case RBF::Name::IQUADRATIC:
			return 1.0/(r*r*ieps2_+1.0);
		break;
		default:
			throw std::invalid_argument("RBF::operator()(double): Invalid RBF Name.");
		break;
	}
}

double RBF::val2(double r2)const{
	switch(name_){
		case RBF::Name::GAUSSIAN:
			return std::exp(-r2*ieps2_);
		break;
		case RBF::Name::MULTIQUADRIC:
			return std::sqrt(r2*ieps2_+1.0);
		break;
		case RBF::Name::IMQUADRIC:
			return 1.0/std::sqrt(r2*ieps2_+1.0);
		break;
		case RBF::Name::IQUADRATIC:
			return 1.0/(r2*ieps2_+1.0);
		break;
		default:
			throw std::invalid_argument("RBF::operator()(double): Invalid RBF Name.");
		break;
	}
}
