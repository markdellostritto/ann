// c libraries
#include <cstring>
// c++ libraries
#include <ostream>
#include <stdexcept>
// ann -  math
#include "math/const.hpp"
// ann - units
#include "chem/units.hpp"
// ann - print
#include "str/print.hpp"

namespace units{

//*************************************************************************
//UNIT SYSTEM
//*************************************************************************

System System::read(const char* str){
	if(std::strcmp(str,"LJ")==0) return System::LJ;
	else if(std::strcmp(str,"AU")==0) return System::AU;
	else if(std::strcmp(str,"METAL")==0) return System::METAL;
	else return System::UNKNOWN;
}

const char* System::name(const System& t){
	switch(t){
		case System::LJ: return "LJ";
		case System::AU: return "AU";
		case System::METAL: return "METAL";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const System& t){
	switch(t){
		case System::LJ: out<<"LJ"; break;
		case System::AU: out<<"AU"; break;
		case System::METAL: out<<"METAL"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//*************************************************************************
//UNIT NAMES
//*************************************************************************

std::ostream& operator<<(std::ostream& out, const Dist::unit& t){
	switch(t){
		case Dist::LJ: out<<"LJ"; break;
		case Dist::BOHR: out<<"BOHR"; break;
		case Dist::ANGSTROM: out<<"ANGSTROM"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Charge::unit& t){
	switch(t){
		case Charge::LJ: out<<"LJ"; break;
		case Charge::ELECTRON: out<<"ELECTRON"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Mass::unit& t){
	switch(t){
		case Mass::LJ: out<<"LJ"; break;
		case Mass::ELECTRON: out<<"ELECTRON"; break;
		case Mass::DALTON: out<<"DALTON"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const Time::unit& t){
	switch(t){
		case Time::AU: out<<"AU"; break;
		case Time::FEMTOSECONDS: out<<"FEMTOSECONDS"; break;
		case Time::LJ: out<<"LJ"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//*************************************************************************
//STANDARD UNIT SYSTEMS
//*************************************************************************

//==== lj units ====
//units
const Time::unit lj::time=Time::LJ;
const Dist::unit lj::dist=Dist::LJ;
const Charge::unit lj::charge=Charge::LJ;
const Mass::unit lj::mass=Mass::LJ;
//fundamental constants
const double lj::eps0=1.0;
const double lj::mu0=0.0;
const double lj::me=1.0;
const double lj::mp=1.0;
const double lj::qe=1.0;
const double lj::hbar=0.0;
const double lj::ke=1.0;
const double lj::kb=1.0;
const double lj::mub=1.0;
const double lj::a0=1.0;
//conversion constants
const double lj::mvv_to_e=1.0;

//==== atomic units ====
//units
const Time::unit au::time=Time::AU;
const Dist::unit au::dist=Dist::BOHR;
const Charge::unit au::charge=Charge::ELECTRON;
const Mass::unit au::mass=Mass::ELECTRON;
//fundamental constants
const double au::eps0=1.0/(4.0*math::constant::PI);
const double au::mu0=0.0;
const double au::me=1.0;
const double au::mp=MPoME;
const double au::qe=1.0;
const double au::hbar=1.0;
const double au::ke=1.0;
const double au::kb=metal::kb*Ev2Eh;
const double au::mub=0.5;
const double au::a0=1.0;
//conversion constants
const double au::mvv_to_e=1.0;

//==== metal units ====
//units
const Time::unit metal::time=Time::FEMTOSECONDS;
const Dist::unit metal::dist=Dist::ANGSTROM;
const Charge::unit metal::charge=Charge::ELECTRON;
const Mass::unit metal::mass=Mass::DALTON;
//fundamental constants
const double metal::eps0=1.0/(4.0*math::constant::PI*metal::ke);
const double metal::mu0=0.0;
const double metal::me=5.48579909065e-4;
const double metal::mp=1.007276466621;
const double metal::qe=1.0;
const double metal::hbar=6.582119569e-1;
const double metal::ke=Bohr2Ang*Eh2Ev;
const double metal::kb=8.617333262e-5;
const double metal::mub=5.788381801226e-5;
const double metal::a0=0.529177210903;
//conversion constants
const double metal::mvv_to_e=1.03642696526805e2;

//*************************************************************************
//CONSTANTS
//*************************************************************************

//==== unit system ====

System Consts::system_=System::METAL;

//==== units ====

Time::unit Consts::time_=Time::UNKNOWN;
Dist::unit Consts::dist_=Dist::UNKNOWN;
Charge::unit Consts::charge_=Charge::UNKNOWN;
Mass::unit Consts::mass_=Mass::UNKNOWN;

//==== fundamental constants ====

double Consts::eps0_=0.0;
double Consts::mu0_=0.0;
double Consts::me_=0.0;
double Consts::mp_=0.0;
double Consts::qe_=0.0;
double Consts::hbar_=0.0;
double Consts::ke_=0.0;
double Consts::kb_=0.0;
double Consts::mub_=0.0;
double Consts::a0_=0.0;

//==== conversion constants ====

double Consts::mvv_to_e_=0.0;

//===== operators ====

std::ostream& operator<<(std::ostream& out, const Consts& c){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("UNITS",str)<<"\n";
	//unit system
	out<<"system = "<<c.system_<<"\n";
	//units
	out<<"time   = "<<c.time_<<"\n";
	out<<"dist   = "<<c.dist_<<"\n";
	out<<"charge = "<<c.charge_<<"\n";
	out<<"mass   = "<<c.mass_<<"\n";
	//fundamental constants
	out<<"esp0   = "<<c.eps0_<<"\n";
	out<<"ke     = "<<c.ke_<<"\n";
	out<<"kb     = "<<c.kb_<<"\n";
	out<<"me     = "<<c.me_<<"\n";
	out<<"mp     = "<<c.mp_<<"\n";
	out<<"qe     = "<<c.qe_<<"\n";
	out<<"hbar   = "<<c.hbar_<<"\n";
	out<<"a0     = "<<c.a0_<<"\n";
	//conversion constants
	out<<"mm2ve  = "<<c.mvv_to_e_<<"\n";
	out<<print::buf(str)<<"\n";
	return out;
}

//===== member functions ====

void Consts::init(const System& t){
	if(t==System::LJ){//lj units
		//units
		time_=lj::time;
		dist_=lj::dist;
		charge_=lj::charge;
		mass_=lj::mass;
		//fundamental constants
		eps0_=lj::eps0;
		mu0_=lj::mu0;
		me_=lj::me;
		mp_=lj::mp;
		qe_=lj::qe;
		hbar_=lj::hbar;
		ke_=lj::ke;
		kb_=lj::kb;
		mub_=lj::mub;
		a0_=lj::a0;
		//conversion constants
		mvv_to_e_=lj::mvv_to_e;
	} else if(t==System::AU){//atomic units
		//units
		time_=au::time;
		dist_=au::dist;
		charge_=au::charge;
		mass_=au::mass;
		//fundamental constants
		eps0_=au::eps0;
		mu0_=au::mu0;
		me_=au::me;
		mp_=au::mp;
		qe_=au::qe;
		hbar_=au::hbar;
		ke_=au::ke;
		kb_=au::kb;
		mub_=au::mub;
		a0_=au::a0;
		//conversion constants
		mvv_to_e_=au::mvv_to_e;
	} else if(t==System::METAL){//metal units
		//units
		time_=metal::time;
		dist_=metal::dist;
		charge_=metal::charge;
		mass_=metal::mass;
		//fundamental constants
		eps0_=metal::eps0;
		mu0_=metal::mu0;
		me_=metal::me;
		mp_=metal::mp;
		qe_=metal::qe;
		hbar_=metal::hbar;
		ke_=metal::ke;
		kb_=metal::kb;
		mub_=metal::mub;
		a0_=metal::a0;
		//conversion constants
		mvv_to_e_=metal::mvv_to_e;
	} else throw std::invalid_argument("units::Consts::init(const System&): Invalid unit System.");
	system_=t;
}

}