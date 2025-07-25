#pragma once
#ifndef UNITS_HPP
#define UNITS_HPP

// c++ libraries
#include <iosfwd>

namespace units{
	
	//*************************************************************************
	//FUNDAMENTAL CONSTANTS
	//*************************************************************************
	
	static const double ALPHA=0.0072973525693;//fine-structure constant
	static const double MPoME=1836.15267343102;//mass_proton/mass_electron NIST - Feb 10. 2024
	
	//*************************************************************************
	//CONVERSION CONSTANTS
	//*************************************************************************
	
	//==== energy ====
	
	//hartree <-> electronvolt
	static const double Eh2Ev=27.211386245988;
	static const double Ev2Eh=1.0/Eh2Ev;
	
	//==== distance ====
	
	//Angstrom <-> Bohr
	static const double Bohr2Ang=0.529177210903;
	static const double Ang2Bohr=1.0/Bohr2Ang;
	
	//==== time ====
	
	//AUT (atomic units - time) <-> Femtosecond
	static const double Aut2Fs=0.024188843265858;
	static const double Fs2Aut=1.0/Aut2Fs;
	
	//*************************************************************************
	//UNIT SYSTEM
	//*************************************************************************
	
	class System{
	public:
		//enum
		enum Type{
			LJ,//lj units (unitless)
			AU,//atomic units
			METAL,//metal units
			UNKNOWN
		};
		//constructor
		System():t_(Type::UNKNOWN){}
		System(Type t):t_(t){}
		//operators
		operator Type()const{return t_;}
		//member functions
		static System read(const char* str);
		static const char* name(const System& system);
	private:
		Type t_;
	};
	std::ostream& operator<<(std::ostream& out, const System& sys);
	
	//*************************************************************************
	//UNIT NAMES
	//*************************************************************************
	
	struct Dist{
		enum unit{
			LJ,
			BOHR,
			ANGSTROM,
			UNKNOWN
		};
	};
	std::ostream& operator<<(std::ostream& out, const Dist::unit& t);
	
	struct Charge{
		enum unit{
			LJ,
			ELECTRON,
			UNKNOWN
		};
	};
	std::ostream& operator<<(std::ostream& out, const Charge::unit& t);
	
	struct Mass{
		enum unit{
			LJ,
			ELECTRON,
			DALTON,
			UNKNOWN
		};
	};
	std::ostream& operator<<(std::ostream& out, const Mass::unit& t);
	
	struct Time{
		enum unit{
			LJ,
			AU,
			FEMTOSECONDS,
			UNKNOWN
		};
	};
	std::ostream& operator<<(std::ostream& out, const Time::unit& t);
	
	//*************************************************************************
	//STANDARD UNIT SYSTEMS
	//*************************************************************************
	
	struct au{
		//==== units ====
		static const Time::unit time;
		static const Dist::unit dist;
		static const Charge::unit charge;
		static const Mass::unit mass;
		//==== fundamental constants ====
		static const double eps0;//permittivity of vacuum
		static const double mu0;//permeability of vacuum
		static const double me;//electron rest mass
		static const double mp;//proton rest mass
		static const double qe;//electron fundamental charge
		static const double hbar;//reduced Planck's constant
		static const double ke;//Coulomb's constant
		static const double kb;//Boltzmann's constant
		static const double mub;//bohr magneton
		static const double a0;//bohr radius
		//==== conversion constants ====
		static const double mvv_to_e;
	};
	
	struct metal{
		//==== units ====
		static const Time::unit time;
		static const Dist::unit dist;
		static const Charge::unit charge;
		static const Mass::unit mass;
		//==== fundamental constants ====
		static const double eps0;//permittivity of vacuum
		static const double mu0;//permeability of vacuum
		static const double me;//electron rest mass
		static const double mp;//proton rest mass
		static const double qe;//electron fundamental charge
		static const double hbar;//reduced Planck's constant
		static const double ke;//Coulomb's constant
		static const double kb;//Boltzmann's constant
		static const double mub;//bohr magneton
		static const double a0;//bohr radius
		//==== conversion constants ====
		static const double mvv_to_e;
	};
	
	struct lj{
		//==== units ====
		static const Time::unit time;
		static const Dist::unit dist;
		static const Charge::unit charge;
		static const Mass::unit mass;
		//==== fundamental constants ====
		static const double eps0;//permittivity of vacuum
		static const double mu0;//permeability of vacuum
		static const double me;//electron rest mass
		static const double mp;//proton rest mass
		static const double qe;//electron fundamental charge
		static const double hbar;//reduced Planck's constant
		static const double ke;//Coulomb's constant
		static const double kb;//Boltzmann's constant
		static const double mub;//bohr magneton
		static const double a0;//bohr radius
		//==== conversion constants ====
		static const double mvv_to_e;
	};
	
	//*************************************************************************
	//CONSTANTS
	//*************************************************************************
	
	class Consts{
	private:
		//==== unit system ====
		static System system_;
		
		//==== units ====
		static Time::unit time_;//time
		static Dist::unit dist_;//distance
		static Charge::unit charge_;//charge
		static Mass::unit mass_;//mass
		
		//==== fundamental constants ====
		static double eps0_;//permittivity of vacuum
		static double mu0_;//permeability of vacuum
		static double me_;//electron rest mass
		static double mp_;//proton rest mass
		static double qe_;//electron fundamental charge
		static double hbar_;//reduced Planck's constant
		static double ke_;//Coulomb's constant
		static double kb_;//Boltzmann's constant
		static double mub_;//bohr magneton
		static double a0_;//bohr radius
		
		//==== conversion constants ====
		static double mvv_to_e_;//convert m*v^2 to energy
	public:
		//==== constructors/destructors ====
		Consts(){init(System::METAL);}
		Consts(const System& t){init(t);}
		~Consts(){};
		
		//==== operators ====
		friend std::ostream& operator<<(std::ostream& out, const Consts& c);
		
		//==== access ====
		//unit system
		static const System& system(){return system_;}
		//units
		static const Time::unit& time(){return time_;}
		static const Dist::unit& dist(){return dist_;}
		static const Charge::unit& charge(){return charge_;}
		static const Mass::unit& mass(){return mass_;}
		//fundamental constants
		static const double& eps0(){return eps0_;}
		static const double& mu0(){return mu0_;}
		static const double& me(){return me_;}
		static const double& mp(){return mp_;}
		static const double& qe(){return qe_;}
		static const double& hbar(){return hbar_;}
		static const double& ke(){return ke_;}
		static const double& kb(){return kb_;}
		static const double& mub(){return mub_;}
		static const double& a0(){return a0_;}
		//conversion constants
		static const double& mvv_to_e(){return mvv_to_e_;}
		
		//==== member functions ====
		static void init(const System& t);
		
	};
	
	
}

#endif
