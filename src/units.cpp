// c libraries
#include <cstring>
// c++ libraries
#include <ostream>
#include <stdexcept>
// ann - units
#include "units.hpp"
// ann - print
#include "print.hpp"

namespace units{

//System

System::type System::read(const char* str){
	if(std::strcmp(str,"AU")==0) return System::AU;
	else if(std::strcmp(str,"METAL")==0) return System::METAL;
	else if(std::strcmp(str,"LJ")==0) return System::LJ;
	else return System::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, const System::type& t){
	switch(t){
		case System::AU: out<<"AU"; break;
		case System::METAL: out<<"METAL"; break;
		case System::LJ: out<<"LJ"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Dist::type& t){
	switch(t){
		case Dist::BOHR: out<<"BOHR"; break;
		case Dist::ANGSTROM: out<<"ANGSTROM"; break;
		case Dist::LJ: out<<"LJ"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Charge::type& t){
	switch(t){
		case Charge::ELECTRON: out<<"ELECTRON"; break;
		case Charge::LJ: out<<"LJ"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Mass::type& t){
	switch(t){
		case Mass::ELECTRON: out<<"ELECTRON"; break;
		case Mass::AMU: out<<"AMU"; break;
		case Mass::LJ: out<<"LJ"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Time::type& t){
	switch(t){
		case Time::AU: out<<"AU"; break;
		case Time::FEMTOSECONDS: out<<"FEMTOSECONDS"; break;
		case Time::LJ: out<<"LJ"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//atomic units
const Time::type au::time=Time::AU;
const Dist::type au::dist=Dist::BOHR;
const Charge::type au::charge=Charge::ELECTRON;
const Mass::type au::mass=Mass::ELECTRON;
const double au::eps0=1.0/(4.0*PI);
const double au::mu0=0.0;
const double au::me=1.0;
const double au::mp=1.0;
const double au::qe=1.0;
const double au::hbar=1.0;
const double au::ke=1.0;
const double au::kb=metal::kb/EVpHARTREE;

//"metal units"
const Time::type metal::time=Time::FEMTOSECONDS;
const Dist::type metal::dist=Dist::ANGSTROM;
const Charge::type metal::charge=Charge::ELECTRON;
const Mass::type metal::mass=Mass::AMU;
const double metal::eps0=1.0/(4.0*PI*metal::ke);
const double metal::mu0=0.0;
const double metal::me=5.48579909065e-4;//NIST - Jan. 16 2020
const double metal::mp=1.007276466621;//NIST - Jan. 16 2020
const double metal::qe=1.0;
const double metal::hbar=6.582119569e-16;//NIST - Jan. 16 2020
const double metal::ke=ANGpBOHR*EVpHARTREE;
const double metal::kb=8.617333262e-5;//NIST - Jan. 16 2020

//"lj units"
const Time::type lj::time=Time::LJ;
const Dist::type lj::dist=Dist::LJ;
const Charge::type lj::charge=Charge::LJ;
const Mass::type lj::mass=Mass::LJ;
const double lj::eps0=1.0;
const double lj::mu0=0.0;
const double lj::me=1.0;
const double lj::mp=1.0;
const double lj::qe=1.0;
const double lj::hbar=0.0;
const double lj::ke=1.0;
const double lj::kb=1.0;

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
	} else if(t==System::LJ){
		time_=lj::time;
		dist_=lj::dist;
		charge_=lj::charge;
		mass_=lj::mass;
		eps0_=lj::eps0;
		mu0_=lj::mu0;
		me_=lj::me;
		mp_=lj::mp;
		qe_=lj::qe;
		hbar_=lj::hbar;
		ke_=lj::ke;
		kb_=lj::kb;
	} else throw std::invalid_argument("Invalid unit System.");
	system_=t;
}

std::ostream& operator<<(std::ostream& out, const consts& c){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("UNITS",str)<<"\n";
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
	out<<print::title("UNITS",str)<<"\n";
	out<<print::buf(str)<<"\n";
	return out;
}

}