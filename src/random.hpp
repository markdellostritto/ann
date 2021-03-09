#ifndef RANDOM_HPP
#define RANDOM_HPP

//c libraries
#if __cplusplus > 199711L
#include <cstdint>
#endif
//c++ libraries
#include <iosfwd>
//math - const
#include "math_const.hpp"

namespace rng{

	namespace gen{

		#if __cplusplus <= 199711L
			typedef unsigned long int uint64t;
			typedef unsigned int uint32t;
			static const uint32t UINT32_MAX=~((uint32t)0);
			static const uint64t UINT64_MAX=~((uint64t)0);
		#else
			typedef uint64_t uint64t;
			typedef uint32_t uint32t;
		#endif

		//******************************************************
		// Random Engine Type
		//******************************************************

		struct Name{
			enum type{
				UNKNOWN=0,
				LCG=1,
				XOR=2,
				MWC=3,
				CG1=4,
				CG2=5
			};
			static type read(const char* str);
			static const char* name(const Name::type& t);
		};
		std::ostream& operator<<(std::ostream& out, const Name::type& v);

		//******************************************************
		// Random Engine
		//******************************************************

		class Engine{
		protected:
			static const double uint32i_;//inverse of largest possible 32-bit integer
			static const double uint64i_;//inverse of largest possible 64-bit integer
			static const uint64t mp61_;//mersenne prime 61
			uint64t s_;//seed
			uint64t v_;//current value
		public:
			//==== constructors/destructors ====
			Engine():s_(0),v_(mp61_){}
			Engine(uint64t s):s_(s),v_(mp61_){}
			virtual ~Engine(){}
			
			//==== member functions ====
			virtual void init(uint64t seed)=0;
			virtual void step()=0;
			virtual uint32t rand32()=0;
			virtual uint64t rand64()=0;
			virtual double randf()=0;
		};

		//******************************************************
		// Linear Congruential Generator (LCG)
		//******************************************************

		class LCG: public Engine{
		private:
			static const uint64t c_;//increment
			static const uint64t a_;//multiplier
		public:
			//==== constructors/destructors ====
			LCG(){}
			LCG(uint64t s){init(s);}
			~LCG(){}
			
			//==== access ====
			uint64t& s(){return s_;}
			const uint64t& s()const{return s_;}
			
			//==== member functions ====
			void init(uint64t seed);
			void step();
			uint32t rand32();
			uint64t rand64();
			double randf();
		};

		//******************************************************
		// XORShift method (XOR)
		//******************************************************

		class XOR: public Engine{
		public:
			//==== constructors/destructors ====
			XOR(){}
			XOR(uint64t s){init(s);}
			~XOR(){}
			
			//==== member functions ====
			void init(uint64t seed);
			void step();
			uint32t rand32();
			uint64t rand64();
			double randf();
		};

		//******************************************************
		// Multiply With Carry method (MWC)
		//******************************************************

		class MWC: public Engine{
		private:
			static const uint64t a;
		public:
			//==== constructors/destructors ====
			MWC(){}
			MWC(uint64t s){init(s);}
			~MWC(){}
			
			//==== member functions ====
			void init(uint64t seed);
			void step();
			uint32t rand32();
			uint64t rand64();
			double randf();
		};

		//******************************************************
		// Combined Generator 1 : LCG(XOR)
		//******************************************************

		class CG1: public Engine{
		private:
			static const uint64t c_;//increment
			static const uint64t a_;//multiplier
			uint64t w_;
		public:
			//==== constructors/destructors ====
			CG1(){}
			CG1(uint64t s){init(s);}
			~CG1(){}
			
			//==== member functions ====
			void init(uint64t seed);
			void step();
			uint32t rand32();
			uint64t rand64();
			double randf();
		};

		//******************************************************
		// Combined Generator 2 : XOR ^ MWC
		//******************************************************

		class CG2: public Engine{
		private:
			static const uint64t a_;
			uint64t w_,r_;
		public:
			//==== constructors/destructors ====
			CG2(){}
			CG2(uint64t s){init(s);}
			~CG2(){}
			
			//==== member functions ====
			void init(uint64t seed);
			void step();
			uint32t rand32();
			uint64t rand64();
			double randf();
		};
	
	}
	
	namespace dist{
		
		//******************************************************
		// Distribution - Name
		//******************************************************

		struct Name{
			enum type{
				UNKNOWN,
				UNIFORM,
				EXP,
				NORMAL,
				RAYLEIGH,
				LOGISTIC,
				CAUCHY
			};
			static type read(const char* str);
			static const char* name(const Name::type& t);
		};
		std::ostream& operator<<(std::ostream& out, const Name::type& v);

		//******************************************************
		// Distribution - Exponential
		//******************************************************

		class Exp{
		private:
			double beta_;
		public:
			Exp(double beta):beta_(beta){}
			~Exp(){}
			
			double operator()(gen::Engine& engine);
		};

		//******************************************************
		// Distribution - Normal
		//******************************************************

		class Normal{
		private:
			unsigned int c_;
			double mu_;
			double sigma_;
			double x_,y_;
		public:
			Normal(double mu, double sigma):c_(0),mu_(mu),sigma_(sigma),x_(0),y_(0){}
			~Normal(){}
			
			double operator()(gen::Engine& engine);
		};

		//******************************************************
		// Distribution - Rayleigh
		//******************************************************

		class Rayleigh{
		public:
			Rayleigh(){}
			~Rayleigh(){}
			
			double operator()(gen::Engine& engine);
		};

		//******************************************************
		// Distribution - Logistic
		//******************************************************

		class Logistic{
		private:
			double mu_;
			double sigma_;
		public:
			Logistic(double mu, double sigma):mu_(mu),sigma_(sigma){}
			~Logistic(){}
			
			double operator()(gen::Engine& engine);
		};

		//******************************************************
		// Distribution - Cauchy
		//******************************************************

		class Cauchy{
		private:
			double mu_;
			double sigma_;
		public:
			Cauchy(double mu, double sigma):mu_(mu),sigma_(sigma){}
			~Cauchy(){}
			
			double operator()(gen::Engine& engine);
		};
		
	}
}

#endif