//str
#include "str/string.hpp"
//torch
#include "torch/pot.hpp"

namespace ptnl{
	
Pot::Name Pot::Name::read(const char* str){
	if(std::strcmp(str,"PAULI")==0) return Pot::Name::PAULI;
	else if(std::strcmp(str,"LJ_CUT")==0) return Pot::Name::LJ_CUT;
	else if(std::strcmp(str,"LJ_LONG")==0) return Pot::Name::LJ_LONG;
	else if(std::strcmp(str,"LJ_SM")==0) return Pot::Name::LJ_SM;
	else if(std::strcmp(str,"LDAMP_CUT")==0) return Pot::Name::LDAMP_CUT;
	else if(std::strcmp(str,"LDAMP_DSF")==0) return Pot::Name::LDAMP_DSF;
	else if(std::strcmp(str,"LDAMP_LONG")==0) return Pot::Name::LDAMP_LONG;
	else if(std::strcmp(str,"COUL_CUT")==0) return Pot::Name::COUL_CUT;
	else if(std::strcmp(str,"COUL_WOLF")==0) return Pot::Name::COUL_WOLF;
	else if(std::strcmp(str,"COUL_DSF")==0) return Pot::Name::COUL_DSF;
	else if(std::strcmp(str,"COUL_LONG")==0) return Pot::Name::COUL_LONG;
	else if(std::strcmp(str,"GAUSS_CUT")==0) return Pot::Name::GAUSS_CUT;
	else if(std::strcmp(str,"GAUSS_DSF")==0) return Pot::Name::GAUSS_DSF;
	else if(std::strcmp(str,"GAUSS_LONG")==0) return Pot::Name::GAUSS_LONG;
	else if(std::strcmp(str,"QEQ_GL")==0) return Pot::Name::QEQ_GL;
	else if(std::strcmp(str,"SPIN_EX")==0) return Pot::Name::SPIN_EX;
	else if(std::strcmp(str,"NNPE")==0) return Pot::Name::NNPE;
	else return Pot::Name::UNKNOWN;
}

const char* Pot::Name::name(const Pot::Name& t){
	switch(t){
		case Pot::Name::PAULI: return "PAULI";
		case Pot::Name::LJ_CUT: return "LJ_CUT";
		case Pot::Name::LJ_LONG: return "LJ_LONG";
		case Pot::Name::LJ_SM: return "LJ_SM";
		case Pot::Name::LDAMP_CUT: return "LDAMP_CUT";
		case Pot::Name::LDAMP_DSF: return "LDAMP_DSF";
		case Pot::Name::LDAMP_LONG: return "LDAMP_LONG";
		case Pot::Name::COUL_CUT: return "COUL_CUT";
		case Pot::Name::COUL_WOLF: return "COUL_WOLF";
		case Pot::Name::COUL_DSF: return "COUL_DSF";
		case Pot::Name::COUL_LONG: return "COUL_LONG";
		case Pot::Name::GAUSS_CUT: return "GAUSS_CUT";
		case Pot::Name::GAUSS_DSF: return "GAUSS_DSF";
		case Pot::Name::GAUSS_LONG: return "GAUSS_LONG";
		case Pot::Name::QEQ_GL: return "QEQ_GL";
		case Pot::Name::SPIN_EX: return "SPIN_EX";
		case Pot::Name::NNPE: return "NNPE";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Pot::Name& t){
	switch(t){
		case Pot::Name::PAULI: out<<"PAULI"; break;
		case Pot::Name::LJ_CUT: out<<"LJ_CUT"; break;
		case Pot::Name::LJ_LONG: out<<"LJ_LONG"; break;
		case Pot::Name::LJ_SM: out<<"LJ_SM"; break;
		case Pot::Name::LDAMP_CUT: out<<"LDAMP_CUT"; break;
		case Pot::Name::LDAMP_DSF: out<<"LDAMP_DSF"; break;
		case Pot::Name::LDAMP_LONG: out<<"LDAMP_LONG"; break;
		case Pot::Name::COUL_CUT: out<<"COUL_CUT"; break;
		case Pot::Name::COUL_WOLF: out<<"COUL_WOLF"; break;
		case Pot::Name::COUL_DSF: out<<"COUL_DSF"; break;
		case Pot::Name::COUL_LONG: out<<"COUL_LONG"; break;
		case Pot::Name::GAUSS_CUT: out<<"GAUSS_CUT"; break;
		case Pot::Name::GAUSS_DSF: out<<"GAUSS_DSF"; break;
		case Pot::Name::GAUSS_LONG: out<<"GAUSS_LONG"; break;
		case Pot::Name::QEQ_GL: out<<"QEQ_GL"; break;
		case Pot::Name::SPIN_EX: out<<"SPIN_EX"; break;
		case Pot::Name::NNPE: out<<"NNPE"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

//==== operator ====

std::ostream& operator<<(std::ostream& out, const Pot& pot){
	return out<<pot.name_<<" "<<pot.rc_;
}

double operator-(const Pot& pot1, const Pot& pot2){
	return std::fabs(pot1.rc()-pot2.rc())+1.0*std::abs(pot1.ntypes()-pot2.ntypes());
}

//==== member functions ====

void Pot::read(Token& token){
	//pot name rc 6.0 ...
	rc_=std::atof(token.next().c_str());
	if(rc_<=0) throw std::invalid_argument("Pot::read(Token&): invalid cutoff.");
	rc2_=rc_*rc_;
}

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::Pot& obj){
		if(POT_PRINT_FUNC>0) std::cout<<"nbytes(const Pot&):\n";
		int size=0;
		size+=sizeof(double);//rcut_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::Pot& obj, char* arr){
		if(POT_PRINT_FUNC>0) std::cout<<"pack(const Pot&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.rc(),sizeof(double)); pos+=sizeof(double);//rcut_
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::Pot& obj, const char* arr){
		if(POT_PRINT_FUNC>0) std::cout<<"unpack(Pot&,const char*):\n";
		int pos=0;
		std::memcpy(&obj.rc(),arr+pos,sizeof(double)); pos+=sizeof(double);//rcut_
		obj.rc2()=obj.rc()*obj.rc();
		return pos;
	}
	
}