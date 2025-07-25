// c
#include <cstring>
// c++
#include <iostream>
#include <string>
// str
#include "str/string.hpp"
// torch
#include "torch/thermostat.hpp"

//***********************************************************
// Name (Thermostat)
//***********************************************************

Thermostat::Name Thermostat::Name::read(const char* str){
	if(std::strcmp(str,"NONE")==0) return Thermostat::Name::NONE;
	else if(std::strcmp(str,"VSCALE")==0) return Thermostat::Name::VSCALE;
	else if(std::strcmp(str,"BERENDSEN")==0) return Thermostat::Name::BERENDSEN;
	else if(std::strcmp(str,"LANGEVIN")==0) return Thermostat::Name::LANGEVIN;
	else return Thermostat::Name::UNKNOWN;
}

const char* Thermostat::Name::name(const Thermostat::Name& t){
	switch(t){
		case Thermostat::Name::NONE: return "NONE";
		case Thermostat::Name::VSCALE: return "VSCALE";
		case Thermostat::Name::BERENDSEN: return "BERENDSEN";
		case Thermostat::Name::LANGEVIN: return "LANGEVIN";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Thermostat::Name& t){
	switch(t){
		case Thermostat::Name::NONE: out<<"NONE"; break;
		case Thermostat::Name::VSCALE: out<<"VSCALE"; break;
		case Thermostat::Name::BERENDSEN: out<<"BERENDSEN"; break;
		case Thermostat::Name::LANGEVIN: out<<"LANGEVIN"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//***********************************************************
// Thermostat
//***********************************************************

//==== constants ====

const double Thermostat::eps_=1e-14;

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Thermostat& thermo){
	out<<thermo.name_<<" T "<<thermo.T_<<" tp "<<thermo.tp_;
	thermo.print(out);
	return out;
}

//==== member functions ====

void Thermostat::clear(){
	T_=0;
	tp_=0;
}

void Thermostat::read(Token& token){
	//**** thermostat thermo-name temp ... ****
	//read temperature
	T_=std::atof(token.next().c_str());
	if(T_<=0) throw std::invalid_argument("Thermostat::read(Token&): invalid temperature.");
	//read period
	tp_=std::atoi(token.next().c_str());
	if(tp_<=0) std::invalid_argument("Thermostat::read(Token&): invalid period.");
}

//==== static functions ====

double Thermostat::ke(const Structure& struc){
	double energy=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		energy+=struc.mass(i)*struc.vel(i).squaredNorm();
	}
	return 0.5*energy;
}

double Thermostat::temp(const Structure& struc){
	return struc.ke()*(2.0/3.0)/(struc.nAtoms()*units::consts::kb());
}

void Thermostat::compute(Structure& struc){
	struc.ke()=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.ke()+=struc.mass(i)*struc.vel(i).squaredNorm();
	}
	struc.ke()*=0.5;
	struc.T()=struc.ke()*(2.0/3.0)/(struc.nAtoms()*units::consts::kb());
}

static std::shared_ptr<Thermostat>& make(const Thermostat::Name& name, std::shared_ptr<Thermostat>& therm){
	switch(name){
		case Thermostat::Name::VSCALE:{
			therm.reset(new VScale());
		} break;
		case Thermostat::Name::BERENDSEN:{
			therm.reset(new Berendsen());
		} break;
		default:{
			therm.reset();
		} break;
	}
	return therm;
}

std::shared_ptr<Thermostat>& Thermostat::read(std::shared_ptr<Thermostat>& therm, Token& token){
	//read name
	Thermostat::Name name=Thermostat::Name::read(string::to_upper(token.next()).c_str());
	switch(name){
		case Thermostat::Name::VSCALE: {
			therm.reset(new VScale());
			static_cast<VScale&>(*therm).read(token);
		} break;
		case Thermostat::Name::BERENDSEN: {
			therm.reset(new Berendsen());
			static_cast<Berendsen&>(*therm).read(token);
		} break;
		default:{
			throw std::invalid_argument("Thermostat::read(Token&): invalid thermostat name.");
		} break;
	}
	return therm;
}

//***********************************************************
// Velocity - Scaling
//***********************************************************

void VScale::rescale(Structure& struc){
	const double fac=sqrt(T_/(struc.T()+1e-16));
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.vel(i)*=fac;
	}
	compute(struc);
}

void VScale::read(Token& token){
	Thermostat::read(token);
}

std::ostream& VScale::print(std::ostream& out)const{
	return out;
}

//***********************************************************
// Berendsen
//***********************************************************

void Berendsen::rescale(Structure& struc){
	const double T=struc.T();
	const double t=struc.t();
	const double dt=struc.dt();
	const double fac=sqrt(1.0+dt/tau_*(T_/(T*(t-0.5*dt))-1.0));
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.vel(i)*=fac;
	}
}

void Berendsen::read(Token& token){
	Thermostat::read(token);
	while(!token.end()){
		const std::string tag=string::to_upper(token.next());
		if(tag=="TAU"){
			tau_=std::atof(token.next().c_str());
		}
	}
	if(tau_<=0) throw std::invalid_argument("Invalid decay constant.");
}

std::ostream& Berendsen::print(std::ostream& out)const{
	return out<<" tau "<<tau_;
}

//***********************************************************
// Langevin
//***********************************************************

void Langevin::rescale(Structure& struc){
	const double T=struc.T();
	const double t=struc.t();
	const double dt=struc.dt();
	const double c=sqrt((T_*units::consts::kb())/(dt*gamma_));
	for(int i=0; i<struc.nAtoms(); ++i){
		//damping force
		struc.force(i).noalias()-=struc.mass(i)/gamma_*struc.velocity(i);
		//random force
		Eigen::Vector3d rv=Eigen::Vector3d::Random();
		struc.force(i).noalias()+=c/sqrt(struc.mass(i))*rv/rv.norm();
	}
}

void Langevin::read(Token& token){
	Thermostat::read(token);
	while(!token.end()){
		const std::string tag=string::to_upper(token.next());
		if(tag=="GAMMA"){
			gamma_=std::atof(token.next().c_str());
		}
	}
	if(gamma_<=0) throw std::invalid_argument("Invalid damping constant.");
}

std::ostream& Langevin::print(std::ostream& out)const{
	return out<<" gamma "<<gamma_;
}