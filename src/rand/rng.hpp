#ifndef RNG_HPP
#define RNG_HPP

//c libraries
#if __cplusplus > 199711L
#include <cstdint>
#endif
// c libraries
#include <cmath>
// c++ libraries
#include <iosfwd>
#include <memory>
#include <random>
// math 
#include "math/const.hpp"
#include "math/special.hpp"

namespace rng{

namespace dist{
	
	using math::constant::PI;
	using math::constant::RadPI;
	using math::constant::Rad2;
	
	//******************************************************
	// Distribution - Name
	//******************************************************

	class Name{
	public:
		enum Type{
			UNKNOWN,
			UNIFORM,
			LAPLACE,
			NORMAL,
			LOGNORMAL,
			SECH,
			LOGISTIC,
			COSINE,
			CAUCHY
		};
		//constructor
		Name():t_(Type::UNKNOWN){}
		Name(Type t):t_(t){}
		//operators
		operator Type()const{return t_;}
		//member functions
		static Name read(const char* str);
		static const char* name(const Name& name);
	private:
		Type t_;
		//prevent automatic conversion for other built-in types
		//template<typename T> operator T() const;
	};
	std::ostream& operator<<(std::ostream& out, const Name& name);
	
	//******************************************************
	// Distribution - Functions
	//******************************************************
	
	//==== Base ====
	
	class Base{
	protected:
		Name name_;
	public:
		//==== constructors/destructors ====
		Base():name_(Name::UNKNOWN){}
		Base(Name name):name_(name){}
		virtual ~Base(){};
		
		//==== member functions ====
		virtual double rand(std::mt19937& gen)=0;
		virtual double pdf(double x)=0;
	};
	
	//==== Uniform ====
	
	class Uniform: public Base{
	private:
		double a_,b_;
	public:
		//==== constructors/destructors ====
		Uniform():Base(Name::UNIFORM),a_(0.0),b_(1.0){}
		Uniform(double a, double b):Base(Name::UNIFORM),a_(a),b_(b){}
		
		//==== access ====
		double& a(){return a_;}
		const double& a()const{return a_;}
		double& b(){return b_;}
		const double& b()const{return b_;}
		
		//==== member functions ====
		double rand(std::mt19937& gen);
		double pdf(double x);
	};
	
	//==== Laplace ====
	
	class Laplace: public Base{
	private:
		double mu_;
		double lambda_;
	public:
		//==== constructors/destructors ====
		Laplace():Base(Name::LAPLACE),mu_(0.0),lambda_(0.0){}
		Laplace(double lambda):Base(Name::LAPLACE),mu_(0.0),lambda_(lambda){}
		Laplace(double mu, double lambda):Base(Name::LAPLACE),mu_(mu),lambda_(lambda){}
		
		//==== access ====
		double& mu(){return mu_;}
		const double& mu()const{return mu_;}
		double& lambda(){return lambda_;}
		const double& lambda()const{return lambda_;}
		
		//==== member functions ====
		double rand(std::mt19937& gen);
		double pdf(double x);
	};
	
	//==== Normal ====
	
	class Normal: public Base{
	private:
		double mu_;
		double sigma_;
		double x_,y_;
		unsigned int c_;
	public:
		//==== constructors/destructors ====
		Normal():Base(Name::NORMAL),mu_(0.0),sigma_(0.0),c_(0){}
		Normal(double mu, double sigma):Base(Name::NORMAL),mu_(mu),sigma_(sigma),c_(0){}
		
		//==== access ====
		double& mu(){return mu_;}
		const double& mu()const{return mu_;}
		double& sigma(){return sigma_;}
		const double& sigma()const{return sigma_;}
		
		//==== member functions ====
		double rand(std::mt19937& gen);
		double pdf(double x);
	};
	
	//==== Logistic ====
	
	class Logistic: public Base{
	private:
		double mu_;
		double sigma_;
	public:
		//==== constructors/destructors ====
		Logistic():Base(Name::LOGISTIC),mu_(0.0),sigma_(0.0){}
		Logistic(double mu, double sigma):Base(Name::LOGISTIC),mu_(mu),sigma_(sigma){}
		
		//==== access ====
		double& mu(){return mu_;}
		const double& mu()const{return mu_;}
		double& sigma(){return sigma_;}
		const double& sigma()const{return sigma_;}
		
		//==== member functions ====
		double rand(std::mt19937& gen);
		double pdf(double x);
	};
	
	//==== Sech ====
	
	class Sech: public Base{
	private:
		double mu_;
		double sigma_;
	public:
		//==== constructors/destructors ====
		Sech():Base(Name::SECH),mu_(0.0),sigma_(0.0){}
		Sech(double mu, double sigma):Base(Name::SECH),mu_(mu),sigma_(sigma){}
		
		//==== access ====
		double& mu(){return mu_;}
		const double& mu()const{return mu_;}
		double& sigma(){return sigma_;}
		const double& sigma()const{return sigma_;}
		
		//==== member functions ====
		double rand(std::mt19937& gen);
		double pdf(double x);
	};
	
	//==== Cosine ====
	
	class Cosine: public Base{
	private:
		double mu_;
		double sigma_;
	public:
		//==== constructors/destructors ====
		Cosine():Base(Name::COSINE),mu_(0.0),sigma_(0.0){}
		Cosine(double mu, double sigma):Base(Name::COSINE),mu_(mu),sigma_(sigma){}
		
		//==== access ====
		double& mu(){return mu_;}
		const double& mu()const{return mu_;}
		double& sigma(){return sigma_;}
		const double& sigma()const{return sigma_;}
		
		//==== member functions ====
		double rand(std::mt19937& gen);
		double pdf(double x);
	};
	//Location–Scale Distributions: Linear Estimation and Probability Plotting Using MATLAB, Horst Rinne
	
} //end namespace dist

} //end namespace rng

#endif
