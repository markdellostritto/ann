#pragma once
#ifndef UNITS_HPP
#define UNITS_HPP

// c++ libraries
#include <iosfwd>

namespace units{
	
	static const double BOHR=0.529177210903;//(A) NIST - Jan. 16 2020
	static const double HARTREE=27.211386245988;//(eV) NIST - Jan. 16 2020
	static const double PI=3.141592653589793;
	
	static const double ANGpBOHR=0.529177210903;//NIST - Jan. 16 2020
	static const double BOHRpANG=1.0/ANGpBOHR;
	static const double HARTREEpEV=1.0/27.211386245988;//NIST - Jan. 16 2020
	static const double EVpHARTREE=1.0/HARTREEpEV;
	
	struct System{
		enum type{
			AU,//atomic units
			METAL,//metal units
			LJ,//lennard jones units
			UNKNOWN
		};
		static System::type read(const char* str);
	};
	std::ostream& operator<<(std::ostream& out, const System::type& t);
	
	struct Dist{
		enum type{
			BOHR,
			ANGSTROM,
			LJ,
			UNKNOWN
		};
	};
	std::ostream& operator<<(std::ostream& out, const Dist::type& t);
	
	struct Charge{
		enum type{
			ELECTRON,
			LJ,
			UNKNOWN
		};
	};
	std::ostream& operator<<(std::ostream& out, const Charge::type& t);
	
	struct Mass{
		enum type{
			ELECTRON,
			AMU,
			LJ,
			UNKNOWN
		};
	};
	std::ostream& operator<<(std::ostream& out, const Mass::type& t);
	
	struct Time{
		enum type{
			AU,
			FEMTOSECONDS,
			LJ,
			UNKNOWN
		};
	};
	std::ostream& operator<<(std::ostream& out, const Time::type& t);
	
	struct au{
		static const Time::type time;
		static const Dist::type dist;
		static const Charge::type charge;
		static const Mass::type mass;
		static const double eps0;//permittivity of vacuum
		static const double mu0;//permeability of vacuum
		static const double me;//electron rest mass
		static const double mp;//proton rest mass
		static const double qe;//electron fundamental charge
		static const double hbar;//reduced Planck's constant
		static const double ke;//Coulomb's constant
		static const double kb;//Boltzmann's constant
	};
	
	struct metal{
		static const Time::type time;
		static const Dist::type dist;
		static const Charge::type charge;
		static const Mass::type mass;
		static const double eps0;//permittivity of vacuum
		static const double mu0;//permeability of vacuum
		static const double me;//electron rest mass
		static const double mp;//proton rest mass
		static const double qe;//electron fundamental charge
		static const double hbar;//reduced Planck's constant
		static const double ke;//Coulomb's constant
		static const double kb;//Boltzmann's constant
	};
	
	struct lj{
		static const Time::type time;
		static const Dist::type dist;
		static const Charge::type charge;
		static const Mass::type mass;
		static const double eps0;//permittivity of vacuum
		static const double mu0;//permeability of vacuum
		static const double me;//electron rest mass
		static const double mp;//proton rest mass
		static const double qe;//electron fundamental charge
		static const double hbar;//reduced Planck's constant
		static const double ke;//Coulomb's constant
		static const double kb;//Boltzmann's constant
	};
	
	class consts{
	private:
		static System::type system_;
		static Time::type time_;
		static Dist::type dist_;
		static Charge::type charge_;
		static Mass::type mass_;
		static double eps0_;//permittivity of vacuum
		static double mu0_;//permeability of vacuum
		static double me_;//electron rest mass
		static double mp_;//proton rest mass
		static double qe_;//electron fundamental charge
		static double hbar_;//reduced Planck's constant
		static double ke_;//Coulomb's constant
		static double kb_;//Boltzmann's constant
	public:
		consts(){init(System::METAL);};
		consts(const System::type& t){init(t);};
		~consts(){};
		
		friend std::ostream& operator<<(std::ostream& out, const consts& c);
		
		static void init(const System::type& t);
		
		static const System::type& system(){return system_;};
		static const Dist::type& dist(){return dist_;};
		static const Charge::type& charge(){return charge_;};
		static const Mass::type& mass(){return mass_;};
		static const double& esp0(){return eps0_;};
		static const double& mu0(){return mu0_;};
		static const double& me(){return me_;};
		static const double& mp(){return mp_;};
		static const double& qe(){return qe_;};
		static const double& hbar(){return hbar_;};
		static const double& ke(){return ke_;};
		static const double& kb(){return kb_;};
	};
	
	
}

#endif
