// c
#include <cstring>
// c++
#include <iostream>
// str
#include "str/string.hpp"
#include "str/token.hpp"
// math
#include "math/const.hpp"
// opt
#include "opt/decay.hpp"

//***************************************************
// decay method
//***************************************************

namespace opt{

//***************************************************
// decay name
//***************************************************

std::ostream& operator<<(std::ostream& out, const Decay::Name& name){
	switch(name){
		case Decay::Name::CONST: out<<"CONST"; break;
		case Decay::Name::EXP: out<<"EXP"; break;
		case Decay::Name::GAUSS: out<<"GAUSS"; break;
		case Decay::Name::SQRT: out<<"SQRT"; break;
		case Decay::Name::INV: out<<"INV"; break;
		case Decay::Name::STEP: out<<"STEP"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

Decay::Name Decay::Name::read(const char* str){
	if(std::strcmp(str,"CONST")==0) return Decay::Name::CONST;
	else if(std::strcmp(str,"EXP")==0) return Decay::Name::EXP;
	else if(std::strcmp(str,"GAUSS")==0) return Decay::Name::GAUSS;
	else if(std::strcmp(str,"SQRT")==0) return Decay::Name::SQRT;
	else if(std::strcmp(str,"INV")==0) return Decay::Name::INV;
	else if(std::strcmp(str,"STEP")==0) return Decay::Name::STEP;
	else return Decay::Name::UNKNOWN;
}

//***************************************************
// decay
//***************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Decay& decay){
	return out<<decay.name()<<" alpha "<<decay.alpha()<<" period "<<decay.period();
}

//==== member functions ====
	
double Decay::step(double gamma, const Iterator& iter)const{
	switch(name_){
		case Decay::Name::CONST:
			return gamma;
		break;
		case Decay::Name::EXP:
			return gamma*(1.0-alpha_);
		break;
		case Decay::Name::GAUSS:
			return gamma*(1.0-2.0*alpha_*iter.step());
		break;
		case Decay::Name::SQRT:
			return gamma*(1.0-0.5*alpha_/(1.0+alpha_*iter.step()));
		break;
		case Decay::Name::INV:
			return gamma*(1.0-alpha_/(1.0+alpha_*iter.step()));
		break;
		case Decay::Name::STEP:
			return (iter.step()>0 && (unsigned int)iter.step()%period_==0)?gamma*alpha_:gamma;
		break;
		default:
			throw std::invalid_argument("opt::Decay::step(const Objective&,const Iterator&): invalid decay method.");
		break;
	}
}

void Decay::read(Token& token){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"opt::Decay::read(Token&):\n";
	name_=Decay::Name::read(string::to_upper(token.next()).c_str());
	alpha_=std::atof(token.next().c_str());
	if(!token.end()) period_=std::atoi(token.next().c_str());
	if(name_==opt::Decay::Name::UNKNOWN) throw std::invalid_argument("opt::Decay::read(Token&): invalid name.");
	if(alpha_<0) throw std::invalid_argument("opt::Decay::read(Token&): invalid alpha.");
}

}

//**********************************************
// serialization
//**********************************************

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const opt::Decay& decay){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const opt::Decay&):\n";
	int size=0;
	size+=sizeof(opt::Decay::Name);//name
	size+=sizeof(int);//period
	size+=sizeof(double);//alpha
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const opt::Decay& decay, char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::pack(const opt::Decay&,char*):\n";
	int pos=0;
	std::memcpy(arr+pos,&decay.name(),sizeof(opt::Decay::Name)); pos+=sizeof(opt::Decay::Name);
	std::memcpy(arr+pos,&decay.period(),sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,&decay.alpha(),sizeof(double)); pos+=sizeof(double);
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(opt::Decay& decay, const char* arr){
	if(OPT_DECAY_PRINT_FUNC>0) std::cout<<"serialize::unpack(opt::Decay&,const char*):\n";
	int pos=0;
	std::memcpy(&decay.name(),arr+pos,sizeof(opt::Decay::Name)); pos+=sizeof(opt::Decay::Name);
	std::memcpy(&decay.period(),arr+pos,sizeof(int)); pos+=sizeof(int);
	std::memcpy(&decay.alpha(),arr+pos,sizeof(double)); pos+=sizeof(double);
	if(decay.name()==opt::Decay::Name::UNKNOWN) throw std::invalid_argument("opt::Decay::read(Token&): invalid name.");
	if(decay.alpha()<0) throw std::invalid_argument("opt::Decay::read(Token&): invalid alpha.");
	return pos;
}
	
}