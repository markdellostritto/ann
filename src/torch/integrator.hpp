#pragma once
#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

// c++
#include <iosfwd>
#include <memory>
// chem
#include "chem/units.hpp"
// struc
#include "struc/structure.hpp"
// str
#include "str/token.hpp"
// torch
#include "torch/engine.hpp"

#ifndef INTEGRATOR_PRINT_FUNC
#define INTEGRATOR_PRINT_FUNC 0
#endif

//********************************************************************
// Integrator
//********************************************************************

class Integrator{
public:
	class Name{
	public:
		enum Type{
			QUICKMIN,
			VERLET,
			VSCALE,
			BERENDSEN,
			LANGEVIN,
			UNKNOWN
		};
		//constructor
		Name():t_(Type::UNKNOWN){}
		Name(Type t):t_(t){}
		//operators
		friend std::ostream& operator<<(std::ostream& out, const Name& name);
		operator Type()const{return t_;}
		//member functions
		static Name read(const char* str);
		static const char* name(const Name& name);
	private:
		Type t_;
	};
protected:
	static const double eps_;//divergence protection
	Name name_;//thermostat name
	double dt_;
public:
	//==== constructors/destructors ===
	Integrator():name_(Name::UNKNOWN){}
	Integrator(const Name& name):name_(name){}
	virtual ~Integrator(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Integrator& thermo);
	
	//==== access ====
	const Name& name()const{return name_;}
	double& dt(){return dt_;}
	const double& dt()const{return dt_;}
	
	//==== member functions ====
	void clear(){};
	
	//==== virtual functions ====
	virtual void read(Token& token);
	virtual void compute(Structure& struc, Engine& engine)=0;
	virtual std::ostream& print(std::ostream& out)const{}
	
	//==== static functions ====
	static double ke(const Structure& struc);
	static double temp(const Structure& struc);
	static void compute(Structure& struc);
	static std::shared_ptr<Integrator>& make(const Name& name, std::shared_ptr<Integrator>& therm);
	static std::shared_ptr<Integrator>& read(std::shared_ptr<Integrator>& therm, Token& token);
};

//***********************************************************
// Quickmin
//***********************************************************

class Quickmin: public Integrator{
protected:
	double tau_;//decay constant
public:
	//==== constructors/destructors ====
	Quickmin():Integrator(Name::QUICKMIN),tau_(0.0){}
	~Quickmin(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Quickmin& thermo);
	
	//==== member functions ====
	void read(Token& token);
	void compute(Structure& struc, Engine& engine);
	std::ostream& print(std::ostream& out)const;
	
};

//***********************************************************
// Verlet
//***********************************************************

class Verlet: public Integrator{
public:
	//==== constructors/destructors ====
	Verlet():Integrator(Name::VERLET){}
	~Verlet(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Verlet& thermo);
	
	//==== member functions ====
	void read(Token& token);
	void compute(Structure& struc, Engine& engine);
	std::ostream& print(std::ostream& out)const;	
};

//***********************************************************
// VScale
//***********************************************************

class VScale: public Integrator{
private:
	double T_;
	int tau_;
public:
	//==== constructors/destructors ====
	VScale():Integrator(Name::VSCALE){}
	~VScale(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const VScale& thermo);
	
	//==== access ====
	double& T(){return T_;}
	const double& T()const{return T_;}
	int& tau(){return tau_;}
	const int& tau()const{return tau_;}
	
	//==== member functions ====
	void read(Token& token);
	void compute(Structure& struc, Engine& engine);
	std::ostream& print(std::ostream& out)const;	
};

//***********************************************************
// Berendsen
//***********************************************************

class Berendsen: public Integrator{
private:
	double T_;
	double tau_;
public:
	//==== constructors/destructors ====
	Berendsen():Integrator(Name::BERENDSEN){}
	~Berendsen(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Berendsen& thermo);
	
	//==== access ====
	double& T(){return T_;}
	const double& T()const{return T_;}
	double& tau(){return tau_;}
	const double& tau()const{return tau_;}
	
	//==== member functions ====
	void read(Token& token);
	void compute(Structure& struc, Engine& engine);
	std::ostream& print(std::ostream& out)const;	
};

//***********************************************************
// Langevin
//***********************************************************

class Langevin: public Integrator{
private:
	double T_;
	double gamma_;//damping constant
public:
	//==== constructors/destructors ====
	Langevin():Integrator(Name::LANGEVIN){}
	~Langevin(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Langevin& thermo);
	
	//==== access ====
	double& T(){return T_;}
	const double& T()const{return T_;}
	double& gamma(){return gamma_;}
	const double& gamma()const{return gamma_;}
	
	//==== member functions ====
	void read(Token& token);
	void compute(Structure& struc, Engine& engine);
	std::ostream& print(std::ostream& out)const;	
};

#endif