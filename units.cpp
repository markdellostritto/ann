// c libraries
#include <cstring>
// c++ libraries
#include <ostream>
#include <stdexcept>
// ann - units
#include "units.hpp"

namespace units{

//System

System::type System::read(const char* str){
	if(std::strcmp(str,"AU")==0) return System::AU;
	else if(std::strcmp(str,"METAL")==0) return System::METAL;
	else if(std::strcmp(str,"IDENTITY")==0) return System::IDENTITY;
	else return System::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, const System::type& t){
	switch(t){
		case System::AU: out<<"AU"; break;
		case System::METAL: out<<"METAL"; break;
		case System::IDENTITY: out<<"IDENTITY"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Dist::type& t){
	switch(t){
		case Dist::BOHR: out<<"BOHR"; break;
		case Dist::ANGSTROM: out<<"ANGSTROM"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Charge::type& t){
	switch(t){
		case Charge::ELECTRON: out<<"ELECTRON"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Mass::type& t){
	switch(t){
		case Mass::ELECTRON: out<<"ELECTRON"; break;
		case Mass::AMU: out<<"AMU"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Time::type& t){
	switch(t){
		case Time::AU: out<<"AU"; break;
		case Time::FEMTOSECONDS: out<<"FEMTOSECONDS"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//atomic units
const Time::type au::time=Time::AU;
const Dist::type au::dist=Dist::BOHR;
const Charge::type au::charge=Charge::ELECTRON;
const Mass::type au::mass=Mass::ELECTRON;
const double au::eps0=1.0/(4*3.14159);
const double au::mu0=0.0;
const double au::me=1.0;
const double au::mp=1.0;
const double au::qe=1.0;
const double au::hbar=1.0;
const double au::ke=1.0;
const double au::kb=3.1668114e-6;

//"metal units"
const Time::type metal::time=Time::FEMTOSECONDS;
const Dist::type metal::dist=Dist::ANGSTROM;
const Charge::type metal::charge=Charge::ELECTRON;
const Mass::type metal::mass=Mass::AMU;
const double metal::eps0=1.0/(4.0*PI*metal::ke);
const double metal::mu0=0.0;
const double metal::me=5.485799090e-4;
const double metal::mp=1.007276466879;
const double metal::qe=1.0;
const double metal::hbar=0.6582119514;
const double metal::ke=ANGpBOHR*EVpHARTREE;

const double metal::kb=8.6173303e-5;

//"identity units"
const Time::type identity::time=Time::FEMTOSECONDS;
const Dist::type identity::dist=Dist::ANGSTROM;
const Charge::type identity::charge=Charge::ELECTRON;
const Mass::type identity::mass=Mass::AMU;
const double identity::eps0=1.0;
const double identity::mu0=0.0;
const double identity::me=1.0;
const double identity::mp=1.0;
const double identity::qe=1.0;
const double identity::hbar=0.0;
const double identity::ke=1.0;
const double identity::kb=1.0;

//consts
System::type consts::system_=System::METAL;
Time::type consts::time_=Time::UNKNOWN;
Dist::type consts::dist_=Dist::UNKNOWN;
Charge::type consts::charge_=Charge::UNKNOWN;
Mass::type consts::mass_=Mass::UNKNOWN;
double consts::eps0_=0.0;
double consts::mu0_=0.0;
double consts::me_=0.0;
double consts::mp_=0.0;
double consts::qe_=0.0;
double consts::hbar_=0.0;
double consts::ke_=0.0;
double consts::kb_=0.0;

void consts::init(const System::type& t){
	if(t==System::AU){
		time_=au::time;
		dist_=au::dist;
		charge_=au::charge;
		mass_=au::mass;
		eps0_=au::eps0;
		mu0_=au::mu0;
		me_=au::me;
		mp_=au::mp;
		qe_=au::qe;
		hbar_=au::hbar;
		ke_=au::ke;
		kb_=au::kb;
	} else if(t==System::METAL){
		time_=metal::time;
		dist_=metal::dist;
		charge_=metal::charge;
		mass_=metal::mass;
		eps0_=metal::eps0;
		mu0_=metal::mu0;
		me_=metal::me;
		mp_=metal::mp;
		qe_=metal::qe;
		hbar_=metal::hbar;
		ke_=metal::ke;
		kb_=metal::kb;
	} else if(t==System::IDENTITY){
		time_=identity::time;
		dist_=identity::dist;
		charge_=identity::charge;
		mass_=identity::mass;
		eps0_=identity::eps0;
		mu0_=identity::mu0;
		me_=identity::me;
		mp_=identity::mp;
		qe_=identity::qe;
		hbar_=identity::hbar;
		ke_=identity::ke;
		kb_=identity::kb;
	} else throw std::invalid_argument("Invalid unit System.");
	system_=t;
}

std::ostream& operator<<(std::ostream& out, const consts& c){
	out<<"****************************************************\n";
	out<<"******************** UNITS ********************\n";
	out<<"SYSTEM = "<<c.system_<<"\n";
	out<<"TIME   = "<<c.time_<<"\n";
	out<<"DIST   = "<<c.dist_<<"\n";
	out<<"CHARGE = "<<c.charge_<<"\n";
	out<<"MASS   = "<<c.mass_<<"\n";
	out<<"EPS0   = "<<c.eps0_<<"\n";
	out<<"KE     = "<<c.ke_<<"\n";
	out<<"KB     = "<<c.kb_<<"\n";
	out<<"ME     = "<<c.me_<<"\n";
	out<<"MP     = "<<c.mp_<<"\n";
	out<<"QE     = "<<c.qe_<<"\n";
	out<<"HBAR   = "<<c.hbar_<<"\n";
	out<<"******************** UNITS ********************\n";
	out<<"****************************************************";
	return out;
}

}