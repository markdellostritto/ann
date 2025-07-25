#pragma once
#ifndef THERMOSTAT_HPP
#define THERMOSTAT_HPP

// c++
#include <iosfwd>
#include <memory>
// chem
#include "chem/units.hpp"
// struc
#include "struc/structure.hpp"
// str
#include "str/token.hpp"

//********************************************************************
// Thermostat
//********************************************************************

class Thermostat{
public:
	class Name{
	public:
		enum Type{
			NONE,
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
	int tp_;//period
	double T_;//target temperature
public:
	//==== constructors/destructors ===
	Thermostat():name_(Name::UNKNOWN){}
	Thermostat(const Name& name):name_(name){}
	virtual ~Thermostat(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Thermostat& thermo);
	
	//==== access ====
	const Name& name()const{return name_;}
	int& tp(){return tp_;}
	const int& tp()const{return tp_;}
	double& T(){return T_;}
	const double& T()const{return T_;}
	
	//==== member functions ====
	void clear();
	
	//==== virtual functions ====
	virtual void read(Token& token);
	virtual void rescale(Structure& struc)=0;
	virtual std::ostream& print(std::ostream& out)const{}
	
	//==== static functions ====
	static double ke(const Structure& struc);
	static double temp(const Structure& struc);
	static void compute(Structure& struc);
	static std::shared_ptr<Thermostat>& make(const Name& name, std::shared_ptr<Thermostat>& therm);
	static std::shared_ptr<Thermostat>& read(std::shared_ptr<Thermostat>& therm, Token& token);
};

//***********************************************************
// Velocity - Scaling
//***********************************************************

class VScale: public Thermostat{
public:
	//==== constructors/destructors ====
	VScale():Thermostat(Name::VSCALE){}
	~VScale(){}
	
	//==== member functions ====
	void read(Token& token);
	void rescale(Structure& struc);
	std::ostream& print(std::ostream& out)const;
	
};

//***********************************************************
// Berendsen
//***********************************************************

class Berendsen: public Thermostat{
protected:
	double tau_;//decay constant
public:
	//==== constructors/destructors ====
	Berendsen():Thermostat(Name::BERENDSEN),tau_(0.0){}
	~Berendsen(){}
	
	//==== member functions ====
	void read(Token& token);
	void rescale(Structure& struc);
	std::ostream& print(std::ostream& out)const;
	
};

//***********************************************************
// Langevin
//***********************************************************

class Langevin: public Thermostat{
protected:
	double gamma_;//damping factor
public:
	//==== constructors/destructors ====
	Langevin():Thermostat(Name::LANGEVIN):gamma_(0.0){}
	~Langevin(){}
	
	//==== member functions ====
	void read(Token& token);
	void rescale(Structure& struc);
	std::ostream& print(std::ostream& out)const;
	
};

#endif