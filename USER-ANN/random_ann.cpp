//c libraries
#include <cstring>
#include <cmath>
//c++ libraries
#include <iostream>
//ann - random
#include "random_ann.h"

namespace RNG{

//******************************************************
// Random Engine Type
//******************************************************

std::ostream& operator<<(std::ostream& out, const REngineN::type& n){
	switch(n){
		case REngineN::LCG: out<<"LCG"; break;
		case REngineN::XOR: out<<"XOR"; break;
		case REngineN::MWC: out<<"MWC"; break;
		case REngineN::CG1: out<<"CG1"; break;
		case REngineN::CG2: out<<"CG2"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* REngineN::name(const REngineN::type& n){
	switch(n){
		case REngineN::LCG: return "LCG";
		case REngineN::XOR: return "XOR";
		case REngineN::MWC: return "MWC";
		case REngineN::CG1: return "CG1";
		case REngineN::CG2: return "CG2";
		default: return "UNKNOWN";
	}
}

REngineN::type REngineN::read(const char* str){
	if(std::strcmp(str,"LCG")==0) return REngineN::LCG;
	else if(std::strcmp(str,"XOR")==0) return REngineN::XOR;
	else if(std::strcmp(str,"MWC")==0) return REngineN::MWC;
	else if(std::strcmp(str,"CG1")==0) return REngineN::CG1;
	else if(std::strcmp(str,"CG2")==0) return REngineN::CG2;
	else return REngineN::UNKNOWN;
}

//******************************************************
// Random Engine
//******************************************************

const double REngine::uint32i_=1.0/((double)UINT32_MAX);
const double REngine::uint64i_=1.0/((double)UINT64_MAX);
const uint64t REngine::mp61_=2305843009213693951;//mersenne prime 61

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

//******************************************************
// Distribution - Exponential
//******************************************************

double DistExp::operator()(REngine& gen){
	return -std::log(gen.randf())/beta_;
}

//******************************************************
// Distribution - Normal
//******************************************************

double DistNormal::operator()(REngine& gen){
	if((c_++)%2==0){
		const double u=gen.randf();
		const double v=gen.randf();
		const double r=std::sqrt(-2.0*std::log(u));
		x_=r*std::cos(2.0*math::constant::PI*v);
		y_=r*std::sin(2.0*math::constant::PI*v);
		return mu_+sigma_*x_;
	} else return mu_+sigma_*y_;
}

}