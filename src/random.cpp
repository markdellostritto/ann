//c libraries
#include <cstring>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#endif
//c++ libraries
#include <iostream>
//ann - random
#include "random.hpp"

namespace rng{

	namespace gen{
		
		//******************************************************
		// Random Engine Type
		//******************************************************

		std::ostream& operator<<(std::ostream& out, const Name::type& n){
			switch(n){
				case Name::LCG: out<<"LCG"; break;
				case Name::XOR: out<<"XOR"; break;
				case Name::MWC: out<<"MWC"; break;
				case Name::CG1: out<<"CG1"; break;
				case Name::CG2: out<<"CG2"; break;
				default: out<<"UNKNOWN"; break;
			}
			return out;
		}

		const char* Name::name(const Name::type& n){
			switch(n){
				case Name::LCG: return "LCG";
				case Name::XOR: return "XOR";
				case Name::MWC: return "MWC";
				case Name::CG1: return "CG1";
				case Name::CG2: return "CG2";
				default: return "UNKNOWN";
			}
		}

		Name::type Name::read(const char* str){
			if(std::strcmp(str,"LCG")==0) return Name::LCG;
			else if(std::strcmp(str,"XOR")==0) return Name::XOR;
			else if(std::strcmp(str,"MWC")==0) return Name::MWC;
			else if(std::strcmp(str,"CG1")==0) return Name::CG1;
			else if(std::strcmp(str,"CG2")==0) return Name::CG2;
			else return Name::UNKNOWN;
		}

		//******************************************************
		// Random Engine
		//******************************************************

		const double Engine::uint32i_=1.0/((double)UINT32_MAX);
		const double Engine::uint64i_=1.0/((double)UINT64_MAX);
		const uint64t Engine::mp61_=2305843009213693951;//mersenne prime 61

		//******************************************************
		// Linear Congruential Generator (LCG)
		//******************************************************

		const uint64t LCG::a_=3935559000370003845;
		const uint64t LCG::c_=2691343689449507681;
		void LCG::init(uint64t seed){
			s_=seed;
			v_=mp61_^s_;
		}
		void LCG::step(){
			v_=a_*v_+c_;
		}
		uint32t LCG::rand32(){
			step();
			return (uint32t)(v_>>32);
		}
		uint64t LCG::rand64(){
			step();
			return v_;
		}
		double LCG::randf(){
			step();
			return ((uint32t)(v_>>32))*uint32i_;
		}

		//******************************************************
		// XORShift method (XOR)
		//******************************************************

		void XOR::init(uint64t seed){
			s_=seed;
			v_=mp61_^s_;
		}
		void XOR::step(){
			v_^=v_>>21;
			v_^=v_<<35;
			v_^=v_>>4;
		}
		uint32t XOR::rand32(){
			step();
			return (uint32t)(v_>>32);
		}
		uint64t XOR::rand64(){
			step();
			return v_;
		}
		double XOR::randf(){
			step();
			return v_*uint64i_;
		}

		//******************************************************
		// Multiply With Carry method (MWC)
		//******************************************************

		const uint64t MWC::a=4294957665;
		void MWC::init(uint64t seed){
			s_=seed;
			v_=mp61_^s_;
		}
		void MWC::step(){
			v_=a*(v_&0xffffffff)+(v_>>32);
		}
		uint32t MWC::rand32(){
			step();
			return (uint32t)v_;
		}
		uint64t MWC::rand64(){
			step();
			return v_;
		}
		double MWC::randf(){
			step();
			return ((uint32t)v_)*uint32i_;
		}

		//******************************************************
		// Combined Generator 1 : LCG(XOR)
		//******************************************************

		const uint64t CG1::a_=3935559000370003845;
		const uint64t CG1::c_=2691343689449507681;
		void CG1::init(uint64t seed){
			s_=seed;
			v_=mp61_^s_;
		}
		void CG1::step(){
			v_^=v_>>21;
			v_^=v_<<35;
			v_^=v_>>4;
			w_=a_*v_+c_;
		}
		uint32t CG1::rand32(){
			step();
			return (uint32t)w_;
		}
		uint64t CG1::rand64(){
			step();
			return w_;
		}
		double CG1::randf(){
			step();
			return w_*uint64i_;
		}

		//******************************************************
		// Combined Generator 2 : XOR ^ MWC
		//******************************************************

		const uint64t CG2::a_=4294957665;
		void CG2::init(uint64t seed){
			s_=seed;
			v_=mp61_^s_;
		}
		void CG2::step(){
			v_^=v_>>17;
			v_^=v_<<31;
			v_^=v_>>8;
			w_=a_*(w_&0xffffffff)+(w_>>32);
			r_=v_^w_;
		}
		uint32t CG2::rand32(){
			step();
			return (uint32t)r_;
		}
		uint64t CG2::rand64(){
			step();
			return r_;
		}
		double CG2::randf(){
			step();
			return r_*uint64i_;
		}
		
	}
	
	namespace dist{
		
		//******************************************************
		// Distribution - Name
		//******************************************************

		std::ostream& operator<<(std::ostream& out, const Name::type& n){
			switch(n){
				case Name::UNIFORM: out<<"UNIFORM"; break;
				case Name::EXP: out<<"EXP"; break;
				case Name::NORMAL: out<<"NORMAL"; break;
				case Name::RAYLEIGH: out<<"RAYLEIGH"; break;
				case Name::LOGISTIC: out<<"LOGISTIC"; break;
				case Name::CAUCHY: out<<"CAUCHY"; break;
				default: out<<"UNKNOWN"; break;
			}
			return out;
		}

		const char* Name::name(const Name::type& n){
			switch(n){
				case Name::UNIFORM: return "UNIFORM";
				case Name::EXP: return "EXP";
				case Name::NORMAL: return "NORMAL";
				case Name::RAYLEIGH: return "RAYLEIGH";
				case Name::LOGISTIC: return "LOGISTIC";
				case Name::CAUCHY: return "CAUCHY";
				default: return "UNKNOWN";
			}
		}

		Name::type Name::read(const char* str){
			if(std::strcmp(str,"UNIFORM")==0) return Name::UNIFORM;
			else if(std::strcmp(str,"EXP")==0) return Name::EXP;
			else if(std::strcmp(str,"NORMAL")==0) return Name::NORMAL;
			else if(std::strcmp(str,"RAYLEIGH")==0) return Name::RAYLEIGH;
			else if(std::strcmp(str,"LOGISTIC")==0) return Name::LOGISTIC;
			else if(std::strcmp(str,"CAUCHY")==0) return Name::CAUCHY;
			else return Name::UNKNOWN;
		}

		//******************************************************
		// Distribution - Exponential
		//******************************************************

		double Exp::operator()(gen::Engine& engine){
			double u=0.0;
			do u=engine.randf(); while(u==0.0);
			return -log(u)/beta_;
		}

		//******************************************************
		// Distribution - Normal
		//******************************************************

		double Normal::operator()(gen::Engine& engine){
			if((c_++)%2==0){
				const double u=engine.randf();
				const double v=engine.randf();
				const double r=sqrt(-2.0*log(u));
				x_=r*cos(2.0*math::constant::PI*v);
				y_=r*sin(2.0*math::constant::PI*v);
				return mu_+sigma_*x_;
			} else return mu_+sigma_*y_;
		}

		//******************************************************
		// Distribution - Rayleigh
		//******************************************************

		double Rayleigh::operator()(gen::Engine& engine){
			return sqrt(-2.0*log(engine.randf()));
		}

		//******************************************************
		// Distribution - Logistic
		//******************************************************

		double Logistic::operator()(gen::Engine& engine){
			const double p=engine.randf();
			return mu_+math::constant::Rad3/math::constant::PI*sigma_*log(p/(1.0-p));
		}

		//******************************************************
		// Distribution - Cauchy
		//******************************************************

		double Cauchy::operator()(gen::Engine& engine){
			return mu_+sigma_*tan(math::constant::PI*(engine.randf()-0.5));
		}
		
	}
	
}