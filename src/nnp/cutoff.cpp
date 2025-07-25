// c libraries
#include <cstring>
// c++ libraries
#include <ostream> 
#include <stdexcept> 
// math
#include "math/special.hpp"
// nnp
#include "nnp/cutoff.hpp"

//************************************************************
// CUTOFF NAMES
//************************************************************

Cutoff::Name Cutoff::Name::read(const char* str){
	if(std::strcmp(str,"STEP")==0) return Cutoff::Name::STEP;
	else if(std::strcmp(str,"COS")==0) return Cutoff::Name::COS;
	else if(std::strcmp(str,"TANH")==0) return Cutoff::Name::TANH;
	else if(std::strcmp(str,"POLY3")==0) return Cutoff::Name::POLY3;
	else return Cutoff::Name::NONE;
}

const char* Cutoff::Name::name(const Cutoff::Name& name){
	switch(name){
		case Cutoff::Name::STEP: return "STEP";
		case Cutoff::Name::COS: return "COS";
		case Cutoff::Name::TANH: return "TANH";
		case Cutoff::Name::POLY3: return "POLY3";
		default: return "NONE";
	}
}

std::ostream& operator<<(std::ostream& out, const Cutoff::Name& name){
	switch(name){
		case Cutoff::Name::STEP: out<<"STEP"; break;
		case Cutoff::Name::COS: out<<"COS"; break;
		case Cutoff::Name::TANH: out<<"TANH"; break;
		case Cutoff::Name::POLY3: out<<"POLY3"; break;
		default: out<<"NONE"; break;
	}
	return out;
}

//************************************************************
// CUTOFF
//************************************************************

//==== constructors/destructors ====

Cutoff::Cutoff(Cutoff::Name name, double rc){
	if(name==Cutoff::Name::NONE) throw std::invalid_argument("Cutoff::Cutoff(Cutoff::Name&,double): Invalid Cutoff name.");
	if(rc<0) throw std::invalid_argument("Cutoff::Cutoff(Cutoff::Name&,double): Invalid Cutoff distance.");
	name_=name;
	rc_=rc;
	rci_=1.0/rc_;
	pirci_=math::constant::PI/rc_;
	
}

//==== member functions ====

double Cutoff::cutf(double dr)const{
	double val=0;
	switch(name_){
		case Cutoff::Name::STEP:{
			val=(dr<rc_)?1.0:0.0;
		} break;
		case Cutoff::Name::COS:{
			val=(dr<rc_)?0.5*(math::special::coscut(dr*pirci_)+1.0):0.0;
		} break;
		case Cutoff::Name::TANH:{
			const double f=(dr>rc_)?0.0:tanh(1.0-dr*rci_);
			val=f*f*f;
		} break;
		case Cutoff::Name::POLY3:{
			if(dr<rc_){
				dr*=rci_;
				val=(2.0*dr-3.0)*dr*dr+1.0;
			} else val=0;
		} break;
		default:
			throw std::invalid_argument("Cutoff:cutf(double): invalid Cutoff type.");
		break;
	}
	return val;
}

double Cutoff::cutg(double dr)const{
	double val=0;
	switch(name_){
		case Cutoff::Name::STEP:{
			val=0.0;
		} break;
		case Cutoff::Name::COS:{
			val=(dr<rc_)?-0.5*pirci_*math::special::sincut(dr*pirci_):0.0;
		} break;
		case Cutoff::Name::TANH:{
			if(dr<rc_){
				const double f=tanh(1.0-dr*rci_);
				val=-3.0*f*f*(1.0-f*f)*rci_;
			} else val=0;
		} break;
		case Cutoff::Name::POLY3:{
			if(dr<rc_){
				dr*=rci_;
				val=6.0*dr*(dr-1.0)*rci_;
			} else val=0;
		} break;
		default:
			throw std::invalid_argument("Cutoff:cutf(double): invalid Cutoff type.");
		break;
	}
	return val;
}

void Cutoff::compute(double dr, double& v, double& g)const{
	switch(name_){
		case Cutoff::Name::STEP:{
			v=(dr>rc_)?0.0:1.0;
			g=0.0;
		} break;
		case Cutoff::Name::COS:{
			if(dr<rc_){
				const double arg=dr*pirci_;
				v=0.5*(math::special::coscut(arg)+1.0);
				g=-0.5*pirci_*math::special::sincut(arg);
			} else {
				v=0.0;
				g=0.0;
			}
		} break;
		case Cutoff::Name::TANH:{
			if(dr<rc_){
				const double t=tanh(1.0-dr*rci_);
				v=t*t*t;
				g=-3.0*t*t*(1.0-t*t)*rci_;
			} else {
				v=0.0;
				g=0.0;
			}
		} break;
		case Cutoff::Name::POLY3:{
			if(dr<rc_){
				dr*=rci_;
				v=(2.0*dr-3.0)*dr*dr+1.0;
				g=6.0*dr*(dr-1.0)*rci_;
			} else {
				v=0.0;
				g=0.0;
			}
		} break;
		default:
			throw std::invalid_argument("Cutoff:cutf(double): invalid Cutoff type.");
		break;
	}
}

//==== serialize ====

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Cutoff& obj){
		int size=0;
		size+=sizeof(obj.name());//Cutoff name
		size+=sizeof(obj.rc());//Cutoff radius
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Cutoff& obj, char* arr){
		int pos=0;
		std::memcpy(arr+pos,&obj.name(),sizeof(obj.name())); pos+=sizeof(obj.name());//Cutoff name
		std::memcpy(arr+pos,&obj.rc(),sizeof(obj.rc())); pos+=sizeof(obj.rc());//Cutoff radius
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Cutoff& obj, const char* arr){
		int pos=0;
		Cutoff::Name name=Cutoff::Name::NONE;
		double rc=0;
		std::memcpy(&name,arr+pos,sizeof(name)); pos+=sizeof(name);
		std::memcpy(&rc,arr+pos,sizeof(rc)); pos+=sizeof(rc);
		obj=Cutoff(name,rc);
		return pos;
	}
	
}
