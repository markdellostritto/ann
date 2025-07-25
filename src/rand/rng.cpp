//c libraries
#include <cstring>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
// c++ libraries
#include <iostream>
// math
#include "rand/rng.hpp"

namespace rng{

namespace dist{
	
	//******************************************************
	// Distribution - Name
	//******************************************************

	std::ostream& operator<<(std::ostream& out, const Name& n){
		switch(n){
			case Name::UNIFORM: out<<"UNIFORM"; break;
			case Name::LAPLACE: out<<"LAPLACE"; break;
			case Name::NORMAL: out<<"NORMAL"; break;
			case Name::LOGNORMAL: out<<"LOGNORMAL"; break;
			case Name::SECH: out<<"SECH"; break;
			case Name::LOGISTIC: out<<"LOGISTIC"; break;
			case Name::COSINE: out<<"COSINE"; break;
			case Name::CAUCHY: out<<"CAUCHY"; break;
			default: out<<"UNKNOWN"; break;
		}
		return out;
	}

	const char* Name::name(const Name& n){
		switch(n){
			case Name::UNIFORM: return "UNIFORM";
			case Name::LAPLACE: return "LAPLACE";
			case Name::NORMAL: return "NORMAL";
			case Name::LOGNORMAL: return "LOGNORMAL";
			case Name::SECH: return "SECH";
			case Name::LOGISTIC: return "LOGISTIC";
			case Name::COSINE: return "COSINE";
			case Name::CAUCHY: return "CAUCHY";
			default: return "UNKNOWN";
		}
	}

	Name Name::read(const char* str){
		if(std::strcmp(str,"UNIFORM")==0) return Name::UNIFORM;
		else if(std::strcmp(str,"LAPLACE")==0) return Name::LAPLACE;
		else if(std::strcmp(str,"NORMAL")==0) return Name::NORMAL;
		else if(std::strcmp(str,"LOGNORMAL")==0) return Name::LOGNORMAL;
		else if(std::strcmp(str,"SECH")==0) return Name::SECH;
		else if(std::strcmp(str,"LOGISTIC")==0) return Name::LOGISTIC;
		else if(std::strcmp(str,"COSINE")==0) return Name::COSINE;
		else if(std::strcmp(str,"CAUCHY")==0) return Name::CAUCHY;
		else return Name::UNKNOWN;
	}
	
	//==== Uniform ====
	
	double Uniform::rand(std::mt19937& gen){
		return (double)gen()/gen.max()*(b_-a_)+a_;
	}
	
	double Uniform::pdf(double x){
		if(a_<x && x<b_) return 1.0;
		else return 0.0;
	}
	
	//==== Laplace ====
	
	double Laplace::rand(std::mt19937& gen){
		double p=0;
		while(p==0){p=(double)gen()/gen.max();}
		p-=0.5;
		return mu_-1.0*lambda_*math::special::sgn(p)*std::log(1.0-2.0*std::fabs(p));
	}
	
	double Laplace::pdf(double x){
		return 1.0/(2.0*lambda_)*exp(-fabs(x-mu_)/lambda_);
	}
	
	//==== Normal ====
	
	double Normal::rand(std::mt19937& gen){
		if(c_%2==0){
			double r1=0;
			while(r1==0){r1=(double)gen()/gen.max();}
			const double r2=(double)gen()/gen.max();
			const double mag=sigma_*sqrt(-2.0*log(r1));
			const double angle=2.0*PI*r2;
			x_=mag*cos(angle);
			y_=mag*sin(angle);
			c_++;
			return x_;
		} else {
			c_++;
			return y_;
		}
	}
	
	double Normal::pdf(double x){
		const double arg=(x-mu_)/sigma_;
		return 1.0/(Rad2*RadPI*sigma_)*exp(-0.5*arg*arg);
	}
	
	//==== Logistic ====

	double Logistic::rand(std::mt19937& gen){
		double p=0;
		while(p==0){p=(double)gen()/gen.max();}
		return mu_+sigma_*log(p/(1.0-p));
	}

	double Logistic::pdf(double x){
		const double fsech=math::special::sech((x-mu_)/(2.0*sigma_));
		return 1.0/(4.0*sigma_)*fsech*fsech;
	}

	//==== Sech ====

	double Sech::rand(std::mt19937& gen){
		double p=0;
		while(p==0){p=(double)gen()/gen.max();}
		return mu_+sigma_*2.0/PI*log(tan(0.5*PI*p));
	}

	double Sech::pdf(double x){
		return 1.0/(2.0*sigma_)*math::special::sech(0.5*PI*(x-mu_)/sigma_);
	}
	
	//==== Cosine ====

	double Cosine::rand(std::mt19937& gen){
		double p=0;
		while(p==0){p=(double)gen()/gen.max();}
		if(p<0.5){
			const double t=sqrt(-2.0*log(p));
			return mu_+sigma_*(0.820251-0.806770*t+0.093247*t*t);
		} else {
			const double t=sqrt(-2.0*log(1.0-p));
			return mu_-sigma_*(0.820251-0.806770*t+0.093247*t*t);
		}
	}

	double Cosine::pdf(double x){
		if(mu_-sigma_<x && x<mu_+sigma_){
			return 1.0/(2.0*sigma_)*(1.0+cos(PI*(x-mu_)/sigma_));
		} else return 0.0;
	}

} //end namespace dist
	
} //end namespace rng
