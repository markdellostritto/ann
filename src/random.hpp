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

namespace RNG{

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

struct REngineN{
	enum type{
		UNKNOWN=0,
		LCG=1,
		XOR=2,
		MWC=3,
		CG1=4,
		CG2=5
	};
	static type read(const char* str);
	static const char* name(const REngineN::type& t);
};
std::ostream& operator<<(std::ostream& out, const REngineN::type& v);

//******************************************************
// Random Engine
//******************************************************

class REngine{
protected:
	static const double uint32i_;
	static const double uint64i_;
	static const uint64t mp61_;//mersenne prime 61
	uint64t s_;//seed
	uint64t v_;//current value
public:
	//==== constructors/destructors ====
	REngine():s_(0),v_(mp61_){}
	virtual ~REngine(){}
	
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

class LCG: public REngine{
private:
	static const uint64t c_;//increment
	static const uint64t a_;//multiplier
public:
	//==== constructors/destructors ====
	LCG(){}
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

class XOR: public REngine{
public:
	//==== constructors/destructors ====
	XOR(){}
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

class MWC: public REngine{
private:
	static const uint64t a;
public:
	//==== constructors/destructors ====
	MWC(){}
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

class CG1: public REngine{
private:
	static const uint64t c_;//increment
	static const uint64t a_;//multiplier
	uint64t w_;
public:
	//==== constructors/destructors ====
	CG1(){}
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

class CG2: public REngine{
private:
	static const uint64t a_;
	uint64t w_,r_;
public:
	//==== constructors/destructors ====
	CG2(){}
	~CG2(){}
	
	//==== member functions ====
	void init(uint64t seed);
	void step();
	uint32t rand32();
	uint64t rand64();
	double randf();
};

//******************************************************
// Distribution - Exponential
//******************************************************

class DistExp{
private:
	double beta_;
public:
	DistExp(double beta):beta_(beta){}
	~DistExp(){}
	
	double operator()(REngine& gen);
};

//******************************************************
// Distribution - Normal
//******************************************************

class DistNormal{
private:
	unsigned int c_;
	double mu_;
	double sigma_;
	double x_,y_;
public:
	DistNormal(double mu, double sigma):mu_(mu),sigma_(sigma){}
	~DistNormal(){}
	
	double operator()(REngine& gen);
};

//******************************************************
// Distribution - Rayleigh
//******************************************************

class DistRayleigh{
public:
	DistRayleigh(){}
	~DistRayleigh(){}
	
	double operator()(REngine& gen);
};

//******************************************************
// Distribution - Logistic
//******************************************************

class DistLogistic{
private:
	unsigned int c_;
	double mu_;
	double sigma_;
public:
	DistLogistic(double mu, double sigma):mu_(mu),sigma_(sigma){}
	~DistLogistic(){}
	
	double operator()(REngine& gen);
};

//******************************************************
// Distribution - Cauchy
//******************************************************

class DistCauchy{
private:
	unsigned int c_;
	double mu_;
	double sigma_;
public:
	DistCauchy(double mu, double sigma):mu_(mu),sigma_(sigma){}
	~DistCauchy(){}
	
	double operator()(REngine& gen);
};

}

#endif