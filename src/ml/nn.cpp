// c libraries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
#include <ctime>
// c++ libraries
#include <iostream>
#include <random>
#include <chrono>
// math 
#include "math/special.hpp"
// str
#include "str/string.hpp"
#include "str/token.hpp"
#include "str/print.hpp"
// nn
#include "ml/nn.hpp"

namespace NN{

using math::constant::RadPI;
using math::constant::PI;
using math::constant::LOG2;

//***********************************************************************
// INITIALIZATION METHOD
//***********************************************************************

std::ostream& operator<<(std::ostream& out, const Init& init){
	switch(init){
		case Init::RAND: out<<"RAND"; break;
		case Init::LECUN: out<<"LECUN"; break;
		case Init::HE: out<<"HE"; break;
		case Init::XAVIER: out<<"XAVIER"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Init::name(const Init& init){
	switch(init){
		case Init::RAND: return "RAND";
		case Init::LECUN: return "LECUN";
		case Init::HE: return "HE";
		case Init::XAVIER: return "XAVIER";
		default: return "UNKNOWN";
	}
}

Init Init::read(const char* str){
	if(std::strcmp(str,"RAND")==0) return Init::RAND;
	else if(std::strcmp(str,"LECUN")==0) return Init::LECUN;
	else if(std::strcmp(str,"HE")==0) return Init::HE;
	else if(std::strcmp(str,"XAVIER")==0) return Init::XAVIER;
	else return Init::UNKNOWN;
}

//***********************************************************************
// NEURON
//***********************************************************************

//==== type ====

std::ostream& operator<<(std::ostream& out, const Neuron& neuron){
	switch(neuron){
		case Neuron::LINEAR: out<<"LINEAR"; break;
		case Neuron::SIGMOID: out<<"SIGMOID"; break;
		case Neuron::TANH: out<<"TANH"; break;
		case Neuron::ISRU: out<<"ISRU"; break;
		case Neuron::ARCTAN: out<<"ARCTAN"; break;
		case Neuron::RELU: out<<"RELU"; break;
		case Neuron::ELU: out<<"ELU"; break;
		case Neuron::TANHRE: out<<"TANHRE"; break;
		case Neuron::SQRE: out<<"SQRE"; break;
		case Neuron::SWISH: out<<"SWISH"; break;
		case Neuron::GELU: out<<"GELU"; break;
		case Neuron::MISH: out<<"MISH"; break;
		case Neuron::PFLU: out<<"PFLU"; break;
		case Neuron::SERF: out<<"SERF"; break;
		case Neuron::LOGISH: out<<"LOGISH"; break;
		case Neuron::SOFTPLUS: out<<"SOFTPLUS"; break;
		case Neuron::SQPLUS: out<<"SQPLUS"; break;
		case Neuron::ATISH: out<<"ATISH"; break;
		case Neuron::TEST: out<<"TEST"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Neuron::name(const Neuron& neuron){
	switch(neuron){
		case Neuron::LINEAR: return "LINEAR";
		case Neuron::SIGMOID: return "SIGMOID";
		case Neuron::TANH: return "TANH";
		case Neuron::ISRU: return "ISRU";
		case Neuron::ARCTAN: return "ARCTAN";
		case Neuron::RELU: return "RELU";
		case Neuron::ELU: return "ELU";
		case Neuron::TANHRE: return "TANHRE";
		case Neuron::SQRE: return "SQRE";
		case Neuron::SWISH: return "SWISH";
		case Neuron::GELU: return "GELU";
		case Neuron::MISH: return "MISH";
		case Neuron::PFLU: return "PFLU";
		case Neuron::SERF: return "SERF";
		case Neuron::LOGISH: return "LOGISH";
		case Neuron::SOFTPLUS: return "SOFTPLUS";
		case Neuron::SQPLUS: return "SQPLUS";
		case Neuron::ATISH: return "ATISH";
		case Neuron::TEST: return "TEST";
		default: return "UNKNOWN";
	}
}

Neuron Neuron::read(const char* str){
	if(std::strcmp(str,"LINEAR")==0) return Neuron::LINEAR;
	else if(std::strcmp(str,"SIGMOID")==0) return Neuron::SIGMOID;
	else if(std::strcmp(str,"TANH")==0) return Neuron::TANH;
	else if(std::strcmp(str,"ISRU")==0) return Neuron::ISRU;
	else if(std::strcmp(str,"ARCTAN")==0) return Neuron::ARCTAN;
	else if(std::strcmp(str,"RELU")==0) return Neuron::RELU;
	else if(std::strcmp(str,"ELU")==0) return Neuron::ELU;
	else if(std::strcmp(str,"TANHRE")==0) return Neuron::TANHRE;
	else if(std::strcmp(str,"SQRE")==0) return Neuron::SQRE;
	else if(std::strcmp(str,"SWISH")==0) return Neuron::SWISH;
	else if(std::strcmp(str,"GELU")==0) return Neuron::GELU;
	else if(std::strcmp(str,"MISH")==0) return Neuron::MISH;
	else if(std::strcmp(str,"PFLU")==0) return Neuron::PFLU;
	else if(std::strcmp(str,"SERF")==0) return Neuron::SERF;
	else if(std::strcmp(str,"LOGISH")==0) return Neuron::LOGISH;
	else if(std::strcmp(str,"SOFTPLUS")==0) return Neuron::SOFTPLUS;
	else if(std::strcmp(str,"SQPLUS")==0) return Neuron::SQPLUS;
	else if(std::strcmp(str,"ATISH")==0) return Neuron::ATISH;
	else if(std::strcmp(str,"TEST")==0) return Neuron::TEST;
	else return Neuron::UNKNOWN;
}

//***********************************************************************
// Activation functions
//***********************************************************************

//linear

void AFFP::af_lin(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i) a[i]=z[i];
}
void AFFPBP::af_lin(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i) a[i]=z[i];
	for(int i=0; i<size; ++i) d[i]=1.0;
}
void AFFPBP2::af_lin(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i) a[i]=z[i];
	for(int i=0; i<size; ++i) d[i]=1.0;
	for(int i=0; i<size; ++i) d2[i]=0.0;
}

//sigmoidal

void AFFP::af_sigmoid(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0){
			const double expf=exp(-c*z[i]);
			a[i]=1.0/(1.0+expf);
		} else {
			const double expf=exp(c*z[i]);
			a[i]=expf*1.0/(1.0+expf);
		}
	}	
}
void AFFPBP::af_sigmoid(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0){
			const double expf=exp(-c*z[i]);
			const double frac=1.0/(1.0+expf);
			a[i]=frac;
			d[i]=c*expf*frac*frac;
		} else {
			const double expf=exp(c*z[i]);
			const double frac=1.0/(1.0+expf);
			a[i]=expf*frac;
			d[i]=c*expf*frac*frac;
		}
	}
}
void AFFPBP2::af_sigmoid(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0){
			const double expf=exp(-c*z[i]);
			const double frac=1.0/(1.0+expf);
			a[i]=frac;
			d[i]=c*expf*frac*frac;
			d2[i]=-c*c*expf*(1.0-expf)*frac*frac*frac;
		} else {
			const double expf=exp(c*z[i]);
			const double frac=1.0/(1.0+expf);
			a[i]=expf*frac;
			d[i]=c*expf*frac*frac;
			d2[i]=-c*c*expf*(expf-1.0)*frac*frac*frac;
		}
	}	
}

void AFFP::af_tanh(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		a[i]=tanh(c*z[i]);
	}
}
void AFFPBP::af_tanh(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double ftanh=tanh(c*z[i]);
		a[i]=ftanh;
		d[i]=c*(1.0-ftanh*ftanh);
	}
}
void AFFPBP2::af_tanh(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double ftanh=tanh(c*z[i]);
		a[i]=ftanh;
		d[i]=c*(1.0-ftanh*ftanh);
		d2[i]=c*c*(-2.0*ftanh*(1.0-ftanh*ftanh));
	}
}

void AFFP::af_isru(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		a[i]=zz/sqrt(1.0+zz*zz);
	}
}
void AFFPBP::af_isru(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double isr=1.0/sqrt(1.0+zz*zz);
		a[i]=zz*isr;
		d[i]=c*isr*isr*isr;
	}
}
void AFFPBP2::af_isru(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double isr=1.0/sqrt(1.0+zz*zz);
		const double isr3=isr*isr*isr;
		if(zz<-100.0) a[i]=-1.0/(1.0+1.0/(zz*zz));
		else if(zz>100.0) a[i]=1.0/(1.0+1.0/(zz*zz));
		else a[i]=zz*isr;
		d[i]=c*isr3;
		d2[i]=-3.0*c*c*zz*isr3*isr*isr;
	}
}

void AFFP::af_arctan(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		a[i]=(2.0/math::constant::PI)*atan(c*z[i]*math::constant::PI/2.0);
	}
}
void AFFPBP::af_arctan(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i]*math::constant::PI/2.0;
		a[i]=(2.0/math::constant::PI)*atan(zz);
		d[i]=c/(1.0+zz*zz);
	}
}
void AFFPBP2::af_arctan(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i]*math::constant::PI/2.0;
		const double frac=1.0/(1.0+zz*zz);
		a[i]=(2.0/math::constant::PI)*atan(zz);
		d[i]=c*frac;
		d2[i]=-2.0*c*zz*frac*frac;
	}
}

void AFFP::af_relu(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>0.0){
			a[i]=z[i];
		} else {
			a[i]=0.0;
		}
	}
}
void AFFPBP::af_relu(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>0.0){
			a[i]=z[i];
			d[i]=1.0;
		} else {
			a[i]=0.0;
			d[i]=0.0;
		}
	}
}
void AFFPBP2::af_relu(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>0.0){
			a[i]=z[i];
			d[i]=1.0;
			d2[i]=0.0;
		} else {
			a[i]=0.0;
			d[i]=0.0;
			d2[i]=0.0;
		}
	}
}

void AFFP::af_elu(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0.0){
			a[i]=z[i];
		} else {
			a[i]=exp(c*z[i])-1.0;
		}
	}
}
void AFFPBP::af_elu(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0.0){
			a[i]=z[i];
			d[i]=1.0;
		} else {
			const double expf=exp(c*z[i]);
			a[i]=expf-1.0;
			d[i]=c*expf;
		}
	}
}
void AFFPBP2::af_elu(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0.0){
			a[i]=z[i];
			d[i]=1.0;
			d2[i]=0.0;
		} else {
			const double expf=exp(c*z[i]);
			a[i]=expf-1.0;
			d[i]=c*expf;
			d2[i]=c*c*expf;
		}
	}
}

void AFFP::af_tanhre(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>0){
			a[i]=z[i];
		} else {
			const double fexp=exp(2.0*c*z[i]);
			a[i]=(fexp-1.0)/(fexp+1.0);
		}
	}
}
void AFFPBP::af_tanhre(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0){
			a[i]=z[i];
			d[i]=1.0;
		} else {
			const double fexp=exp(2.0*c*z[i]);
			a[i]=(fexp-1.0)/(fexp+1.0);
			d[i]=c*(1.0-a[i]*a[i]);
		}
	}
}
void AFFPBP2::af_tanhre(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0){
			a[i]=z[i];
			d[i]=1.0;
			d2[i]=0.0;
		} else {
			const double fexp=exp(2.0*c*z[i]);
			a[i]=(fexp-1.0)/(fexp+1.0);
			d[i]=c*(1.0-a[i]*a[i]);
			d2[i]=-c*c*2.0*a[i]*d[i];
		}
	}
}

void AFFP::af_sqre(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0){
			a[i]=z[i];
		} else {
			a[i]=z[i]/sqrt(1.0+z[i]*z[i]);
		}
	}
}
void AFFPBP::af_sqre(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0){
			a[i]=z[i];
			d[i]=1.0;
		} else {
			const double den=1.0/sqrt(1.0+z[i]*z[i]);
			a[i]=z[i]*den;
			d[i]=den*den*den;
		}
	}
}
void AFFPBP2::af_sqre(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		if(z[i]>=0){
			a[i]=z[i];
			d[i]=1.0;
			d2[i]=0.0;
		} else {
			const double den=1.0/sqrt(1.0+z[i]*z[i]);
			const double den3=den*den*den;
			a[i]=z[i]*den;
			d[i]=den3;
			d2[i]=3.0*z[i]*den3*den*den;
		}
	}
}

//gated-switch

void AFFP::af_swish(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		if(zz>0.0){
			a[i]=z[i]/(1.0+exp(-zz));
		} else {
			const double expf=exp(zz);
			a[i]=z[i]*expf/(1.0+expf);
		}
	}
}
void AFFPBP::af_swish(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		if(zz>0.0){
			const double expf=exp(-zz);
			const double den=1.0/(1.0+expf);
			a[i]=z[i]*den;
			d[i]=(1.0+expf*(1.0+zz))*den*den;
		} else {
			const double expf=exp(zz);
			const double den=1.0/(1.0+expf);
			a[i]=z[i]*expf*den;
			d[i]=expf*(1.0+zz+expf)*den*den;
		}
	}
}
void AFFPBP2::af_swish(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		if(zz>0.0){
			const double expf=exp(-zz);
			const double den=1.0/(1.0+expf);
			a[i]=z[i]*den;
			d[i]=(1.0+expf*(1.0+zz))*den*den;
			d2[i]=c*expf*(2.0-zz+expf*(2.0+zz))*den*den*den;
		} else {
			const double expf=exp(zz);
			const double den=1.0/(1.0+expf);
			a[i]=z[i]*expf*den;
			d[i]=expf*(1.0+zz+expf)*den*den;
			d2[i]=c*expf*(2.0+zz+expf*(2.0-zz))*den*den*den;
		}
	}
}

void AFFP::af_gelu(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		a[i]=0.5*z[i]*(1.0+erf(0.5*RadPI*c*z[i]));
	}
}
void AFFPBP::af_gelu(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=0.5*RadPI*c*z[i];
		const double erfp1=(1.0+erf(zz));
		a[i]=0.5*z[i]*erfp1;
		d[i]=0.5*(erfp1+c*z[i]*exp(-zz*zz));
	}
}
void AFFPBP2::af_gelu(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=0.5*RadPI*c*z[i];
		const double erfp1=(1.0+erf(zz));
		a[i]=0.5*z[i]*erfp1;
		d[i]=0.5*(erfp1+c*z[i]*exp(-zz*zz));
		d2[i]=0.0;
	}
}

void AFFP::af_mish(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double expf=exp(zz);
		if(zz>0.0){
			a[i]=z[i]*(1.0-2.0/((1.0+expf)*(1.0+expf)+1.0));
		} else {
			const double ff=expf*(expf+2.0);
			a[i]=z[i]*ff/(ff+2.0);
		}
	}
	/*for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		if(zz>0.0){
			const double expf=exp(-zz);
			a[i]=z[i]*(2.0*expf+1.0)/(2.0*expf*(expf+1.0)+1.0);
		} else {
			const double expf=exp(zz);
			a[i]=z[i]*expf*(expf+2.0)/(expf*(expf+2.0)+2.0);
		}
	}*/
}
void AFFPBP::af_mish(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		if(zz>0.0){
			const double expf=exp(-zz);
			const double expf2=expf*expf;
			const double den=1.0/((1.0+expf)*(1.0+expf)+expf2);
			a[i]=z[i]*(1.0-2.0*expf2*den);
			d[i]=1.0-2.0*expf2*den*(1.0-2.0*zz*(1.0+expf)*den);
		} else {
			const double expf=exp(zz);
			const double den=1.0/(expf*(expf+2.0)+2.0);
			a[i]=z[i]*expf*(expf+2.0)*den;
			d[i]=1.0+2.0*den*(-1.0+2.0*zz*(1.0-(expf+2.0)*den));
		}
	}
}
void AFFPBP2::af_mish(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		if(zz>0.0){
			const double expf=exp(-zz);
			const double expf2=expf*expf;
			const double den=1.0/(1.0+2.0*(expf+expf2));
			a[i]=z[i]*(1.0-2.0*expf2*den);
			d[i]=1.0-2.0*expf2*den*(1.0-2.0*zz*(1.0+expf)*den);
			d2[i]=0.0;
		} else {
			const double expf=exp(zz);
			const double den=1.0/(expf*(expf+2.0)+2.0);
			a[i]=z[i]*expf*(expf+2.0)*den;
			d[i]=1.0+2.0*den*(-1.0+2.0*zz*(1.0-(expf+2.0)*den));
			d2[i]=0.0;
		}
	}
}

void AFFP::af_pflu(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		a[i]=0.5*z[i]*(1.0+zz/sqrt(1.0+zz*zz));
	}
}
void AFFPBP::af_pflu(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double z2=zz*zz;
		const double fsqrti=1.0/sqrt(z2+1.0);
		a[i]=0.5*z[i]*(1.0+zz*fsqrti);
		d[i]=0.5*(zz*(z2+2.0)*fsqrti*fsqrti*fsqrti+1.0);
	}
}
void AFFPBP2::af_pflu(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double z2=zz*zz;
		const double fsqrti=1.0/sqrt(z2+1.0);
		const double fsqrti3=fsqrti*fsqrti*fsqrti;
		a[i]=0.5*z[i]*(1.0+zz*fsqrti);
		d[i]=0.5*(zz*(z2+2.0)*fsqrti*fsqrti*fsqrti+1.0);
		d2[i]=-0.5*c*(z2-2.0)*fsqrti3*fsqrti*fsqrti;
	}
}

void AFFP::af_serf(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		a[i]=z[i]*std::erf(std::log1p(std::exp(z[i])));
	}
}
void AFFPBP::af_serf(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double fexp=std::exp(z[i]);
		const double flog=std::log1p(std::exp(z[i]));
		a[i]=z[i]*std::erf(flog);
		d[i]=a[i]/z[i]+2.0/RadPI*std::exp(-1.0*flog*flog)*z[i]/(1.0+1.0/fexp);
	}
}
void AFFPBP2::af_serf(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double fexp=std::exp(z[i]);
		const double flog=std::log1p(std::exp(z[i]));
		a[i]=z[i]*std::erf(flog);
		d[i]=a[i]/z[i]+2.0/RadPI*std::exp(-1.0*flog*flog)*z[i]/(1.0+1.0/fexp);
		d2[i]=0.0;
	}
}

void AFFP::af_logish(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	const double log2i=1.0/LOG2;
	for(int i=0; i<size; ++i){
		a[i]=log2i*z[i]*log(1.0+1.0/(1.0+exp(-z[i])));
	}
}
void AFFPBP::af_logish(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	const double log2i=1.0/LOG2;
	for(int i=0; i<size; ++i){
		const double expf=exp(-z[i]);
		const double logf=log(1.0+1.0/(1.0+expf));
		a[i]=log2i*z[i]*logf;
		d[i]=log2i*(logf+z[i]*expf/((1.0+expf)*(2.0+expf)));
	}
}
void AFFPBP2::af_logish(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	const double log2i=1.0/LOG2;
	for(int i=0; i<size; ++i){
		const double expf=exp(-z[i]);
		const double logf=log(1.0+1.0/(1.0+expf));
		a[i]=log2i*z[i]*logf;
		d[i]=log2i*(logf+z[i]*expf/((1.0+expf)*(2.0+expf)));
		d2[i]=0.0;
	}
}

//switch

void AFFP::af_softplus(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	const double ci=1.0/c;
	for(int i=0; i<size; ++i){
		if(z[i]>=0.0){
			a[i]=z[i]+ci*(log1p(exp(-c*z[i]))-math::constant::LOG2);
		} else {
			a[i]=ci*(log1p(exp(c*z[i]))-math::constant::LOG2);
		}
	}
}
void AFFPBP::af_softplus(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	const double ci=1.0/c;
	for(int i=0; i<size; ++i){
		if(z[i]>0.0){
			const double expf=exp(-c*z[i]);
			a[i]=z[i]+ci*(log1p(expf)-math::constant::LOG2);
			d[i]=1.0/(expf+1.0);
		} else {
			const double expf=exp(c*z[i]);
			a[i]=ci*(log1p(expf)-math::constant::LOG2);
			d[i]=expf/(expf+1.0);
		}
	}
}
void AFFPBP2::af_softplus(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	const double ci=1.0/c;
	for(int i=0; i<size; ++i){
		if(z[i]>=0.0){
			const double expf=exp(-c*z[i]);
			const double den=1.0/(expf+1.0);
			a[i]=z[i]+ci*(log1p(expf)-math::constant::LOG2);
			d[i]=den;
			d2[i]=c*expf*den*den;
		} else {
			const double expf=exp(c*z[i]);
			const double den=1.0/(expf+1.0);
			a[i]=ci*(log1p(expf)-math::constant::LOG2);
			d[i]=expf*den;
			d2[i]=c*expf*den*den;
		}
	}
}

void AFFP::af_sqplus(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	const double ci=1.0/c;
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		a[i]=0.5*ci*(zz-1.0+sqrt(1.0+zz*zz));
	}
}
void AFFPBP::af_sqplus(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	const double ci=1.0/c;
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double fsqrt=sqrt(1.0+zz*zz);
		a[i]=0.5*ci*(zz-1.0+fsqrt);
		d[i]=0.5*(zz/fsqrt+1.0);
	}
}
void AFFPBP2::af_sqplus(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	const double ci=1.0/c;
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double fsqrt=sqrt(1.0+zz*zz);
		a[i]=0.5*ci*(zz-1.0+fsqrt);
		d[i]=0.5*(zz/fsqrt+1.0);
		d2[i]=0.5*c/(fsqrt*fsqrt*fsqrt);
	}
}

void AFFP::af_atish(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	const double ci=1.0/c;
	for(int i=0; i<size; ++i){
		const double zz=1*0.5*PI*z[i];
		a[i]=z[i]*0.5*(1.0+2.0/PI*atan(zz));
	}
}
void AFFPBP::af_atish(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	const double ci=1.0/c;
	for(int i=0; i<size; ++i){
		const double zz=1*0.5*PI*z[i];
		a[i]=z[i]*0.5*(1.0+2.0/PI*atan(zz));
		d[i]=0.5*(1.0+z[i]/(1.0+zz*zz))+1.0/PI*atan(zz);
	}
}
void AFFPBP2::af_atish(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	const double ci=1.0/c;
	for(int i=0; i<size; ++i){
		const double zz=1*0.5*PI*z[i];
		a[i]=z[i]*0.5*(1.0+2.0/PI*atan(zz));
		d[i]=0.5*(1.0+z[i]/(1.0+zz*zz))+1.0/PI*atan(zz);
		d2[i]=0.0;
	}
}

//test

void AFFP::af_test(double c, const VecXd& z, VecXd& a){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double z2=zz*zz;
		const double z4=z2*z2;
		a[i]=0.5*z[i]*(1.0+zz/sqrt(sqrt(z4+1.0)));
	}
}
void AFFPBP::af_test(double c, const VecXd& z, VecXd& a, VecXd& d){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double z2=zz*zz;
		const double z4=z2*z2;
		const double den=1.0/sqrt(sqrt(z4+1.0));
		a[i]=0.5*z[i]*(1.0+zz*den);
		d[i]=0.5*(zz*den*(1.0+1.0/(z4+1.0))+1.0);
	}
}
void AFFPBP2::af_test(double c, const VecXd& z, VecXd& a, VecXd& d, VecXd& d2){
	const int size=z.size();
	for(int i=0; i<size; ++i){
		const double zz=c*z[i];
		const double z2=zz*zz;
		const double z4=z2*z2;
		const double den=1.0/sqrt(sqrt(z4+1.0));
		a[i]=0.5*z[i]*(1.0+zz*den);
		d[i]=0.5*(zz*den*(1.0+1.0/(z4+1.0))+1.0);
		d2[i]=0.0;
	}
}

//***********************************************************************
// ANN
//***********************************************************************

//==== operators ====

/**
* print network to screen
* @param out - output stream
* @param nn - neural network
* @return output stream
*/
std::ostream& operator<<(std::ostream& out, const ANN& nn){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ANN",str)<<"\n";
	out<<"nn     = "<<nn.nInp()<<" "; for(int n=0; n<nn.a_.size(); ++n) out<<nn.a_[n].size()<<" "; out<<"\n";
	out<<"size   = "<<nn.size()<<"\n";
	out<<"neuron = "<<nn.neuron_<<"\n";
	out<<"sharp  = "<<nn.c_<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

/**
* pack network parameters into serial array
* @param nn - neural network
* @param v - vector storing nn parameters
* @return v - vector storing nn parameters
*/
VecXd& operator>>(const ANN& nn, VecXd& v){
	if(NN_PRINT_FUNC>0) std::cout<<"operator>>(const ANN&, VecXd&):\n";
	int count=0;
	v=VecXd::Zero(nn.size());
	for(int l=0; l<nn.nlayer(); ++l){
		std::memcpy(v.data()+count,nn.b(l).data(),nn.b(l).size()*sizeof(double));
		count+=nn.b(l).size();
	}
	for(int l=0; l<nn.nlayer(); ++l){
		std::memcpy(v.data()+count,nn.w(l).data(),nn.w(l).size()*sizeof(double));
		count+=nn.w(l).size();
	}
	return v;
}

/**
* unpack network parameters from serial array
* @param nn - neural network
* @param v - vector storing nn parameters
* @return nn - neural network
*/
ANN& operator<<(ANN& nn, const VecXd& v){
	if(NN_PRINT_FUNC>0) std::cout<<"operator<<(ANN&,const VecXd&):\n";
	if(nn.size()!=v.size()) throw std::invalid_argument("Invalid size: vector and network mismatch.");
	int count=0;
	for(int l=0; l<nn.nlayer(); ++l){
		std::memcpy(nn.b(l).data(),v.data()+count,nn.b(l).size()*sizeof(double));
		count+=nn.b(l).size();
	}
	for(int l=0; l<nn.nlayer(); ++l){
		std::memcpy(nn.w(l).data(),v.data()+count,nn.w(l).size()*sizeof(double));
		count+=nn.w(l).size();
	}
	return nn;
}

//==== member functions ====

/**
* set the default values
*/
void ANN::defaults(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::defaults():\n";
	//layers
		nlayer_=0;
		c_=1.0;
	//input/output
		inp_.resize(0);
		ins_.resize(0);
		out_.resize(0);
		inpw_.resize(0);
		inpb_.resize(0);
		outw_.resize(0);
		outb_.resize(0);
	//node weights and biases
		z_.clear();
		a_.clear();
		b_.clear();
		w_.clear();
	//gradients - nodes
		dadz_.clear();
		d2adz2_.clear();
	//neuron
		neuron_=Neuron::UNKNOWN;
		affp_.clear();
		affpbp_.clear();
		affpbp2_.clear();
}

/**
* clear all values
* note that parameters like neuron are unchanged
*/
void ANN::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::clear():\n";
	//layers
		nlayer_=-1;
	//input/output
		inp_.resize(0);
		ins_.resize(0);
		out_.resize(0);
		inpw_.resize(0);
		inpb_.resize(0);
		outw_.resize(0);
		outb_.resize(0);
	//node weights and biases
		z_.clear();
		a_.clear();
		b_.clear();
		w_.clear();
	//gradients - nodes
		dadz_.clear();
		d2adz2_.clear();
	//neuron
		affp_.clear();
		affpbp_.clear();
		affpbp2_.clear();
}

/**
* compute and return the size of the network - the number of adjustable parameters
* @return the size of the network - the number of adjustable parameters
*/
int ANN::size()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::size():\n";
	int s=0;
	for(int n=0; n<nlayer_; ++n) s+=b_[n].size();
	for(int n=0; n<nlayer_; ++n) s+=w_[n].size();
	return s;
}

/**
* compute and return the number of bias parameters 
* @return the number of bias parameters 
*/
int ANN::nBias()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::nBias():\n";
	int s=0;
	for(int n=0; n<nlayer_; ++n) s+=b_[n].size();
	return s;
}

/**
* compute and return the number of weight parameters 
* @return the number of weight parameters 
*/
int ANN::nWeight()const{
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::nWeight():\n";
	int s=0;
	for(int n=0; n<nlayer_; ++n) s+=w_[n].size();
	return s;
}

/**
* resize the network - no hidden layers
* @param init - object containing requisite initialization parameters
* @param nInp - number of inputs of the newtork
* @param nOut - the number of outputs of the network
*/
void ANN::resize(const ANNP& annp, int nInp, int nOut){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNP&,int,int):\n";
	std::vector<int> nNodes_(1,nOut);
	resize(annp,nInp,nNodes_);
}

/**
* resize the network - given separate hidden layers and output layer
* @param annp - object containing requisite initialization parameters
* @param nInp - number of inputs of the newtork
* @param nOut - the number of outputs of the network
* @param nNodes - the number of nodes in each hidden layer
*/
void ANN::resize(const ANNP& annp, int nInp, const std::vector<int>& nNodes, int nOut){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNP&,int,const std::vector<int>&,int):\n";
	std::vector<int> nNodes_(nNodes.size()+1);
	for(int n=0; n<nNodes.size(); ++n) nNodes_[n]=nNodes[n];
	nNodes_.back()=nOut;
	resize(annp,nInp,nNodes_);
}

/**
* resize the network - given combined hidden layers and output layer
* @param annp - object containing requisite initialization parameters
* @param nNodes - the number of nodes in each layer of the network
*/
void ANN::resize(const ANNP& annp, const std::vector<int>& nNodes){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNP&,const std::vector<int>&):\n";
	int nInp=nNodes.front();
	std::vector<int> nNodes_(nNodes.size()-1);
	for(int i=0; i<nNodes_.size(); ++i) nNodes_[i]=nNodes[i+1];
	resize(annp,nInp,nNodes_);
}

void ANN::resize(const ANNP& annp, int nInp, const std::vector<int>& nNodes){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNP&,const std::vector<int>&):\n";
	//initialize the random number generator
		if(annp.sigma_b()<=0) throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization deviation - bias");
		if(annp.sigma_w()<=0) throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization deviation - weight");
		std::mt19937 rngen(annp.seed()<0?std::chrono::system_clock::now().time_since_epoch().count():annp.seed());
		std::uniform_real_distribution<double> uniform(-1.0,1.0);
	//clear the network
		clear();
	//number of layers
		nlayer_=nNodes.size();
		c_=annp.c();
		if(nlayer_<1) throw std::invalid_argument("ANN::resize(const ANNP&,int,const std::vector<int>&): Invalid number of layers.");
	//check parameters
		for(int n=0; n<nNodes.size(); ++n){
			if(nNodes[n]<=0) throw std::invalid_argument("ANN::resize(const ANNP&,int,const std::vector<int>&): Invalid layer size.");
		}
	//input/output
		inp_=VecXd::Zero(nInp);
		ins_=VecXd::Zero(nInp);
		out_=VecXd::Zero(nNodes.back());
	//pre/post conditioning
		inpw_=VecXd::Constant(inp_.size(),1);
		inpb_=VecXd::Constant(inp_.size(),0);
		outw_=VecXd::Constant(out_.size(),1);
		outb_=VecXd::Constant(out_.size(),0);
	//gradients - nodes
		dadz_.resize(nlayer_);
		d2adz2_.resize(nlayer_);
		for(int n=0; n<nlayer_; ++n){
			dadz_[n]=VecXd::Zero(nNodes[n]);
			d2adz2_[n]=VecXd::Zero(nNodes[n]);
		}
	//nodes
		a_.resize(nlayer_);
		z_.resize(nlayer_);
		for(int n=0; n<nlayer_; ++n){
			a_[n]=VecXd::Zero(nNodes[n]);
			z_[n]=VecXd::Zero(nNodes[n]);
		}
	//biases and weights - resizing
		b_.resize(nlayer_);
		for(int n=0; n<nlayer_; ++n){
			b_[n]=VecXd::Zero(nNodes[n]);
		}
		w_.resize(nlayer_);
		w_[0]=MatXd::Zero(nNodes[0],nInp);
		for(int n=1; n<nlayer_; ++n){
			w_[n]=MatXd::Zero(nNodes[n],nNodes[n-1]);
		}
		//weight(n) * layer(n) -> layer(n+1), thus size(weight) = (layer(n+1) rows * layer(n) cols)
	//bias - initialization
		if(annp.dist_b()==rng::dist::Name::UNIFORM){
			for(int n=0; n<nlayer_; ++n){
				const int ni=(n>0)?nNodes[n-1]:nInp;
				const int no=nNodes[n];
				double s=std::sqrt(3.0*annp.sigma_b());
				switch(annp.init()){
					case Init::RAND:   s*=1.0; break;
					case Init::LECUN:  s*=std::sqrt(1.0/ni); break;
					case Init::HE:     s*=std::sqrt(2.0/ni); break;
					case Init::XAVIER: s*=std::sqrt(2.0/(no+ni)); break;
					default: throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization scheme."); break;
				}
				std::uniform_real_distribution<double> dist(-s,s);
				for(int m=0; m<b_[n].size(); ++m){
					b_[n].data()[m]=dist(rngen);
				}
			}
		} else if(annp.dist_b()==rng::dist::Name::NORMAL){
			for(int n=0; n<nlayer_; ++n){
				const int ni=(n>0)?nNodes[n-1]:nInp;
				const int no=nNodes[n];
				double s=annp.sigma_b();
				switch(annp.init()){
					case Init::RAND:   s*=1.0; break;
					case Init::LECUN:  s*=std::sqrt(1.0/ni); break;
					case Init::HE:     s*=std::sqrt(2.0/ni); break;
					case Init::XAVIER: s*=std::sqrt(2.0/(no+ni)); break;
					default: throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization scheme."); break;
				}
				std::normal_distribution<double> dist(0.0,s);
				for(int m=0; m<b_[n].size(); ++m){
					b_[n].data()[m]=dist(rngen);
				}
			}
		} else if(annp.dist_b()==rng::dist::Name::LAPLACE){
			for(int n=0; n<nlayer_; ++n){
				const int ni=(n>0)?nNodes[n-1]:nInp;
				const int no=nNodes[n];
				double s=annp.sigma_b();
				switch(annp.init()){
					case Init::RAND:   s*=1.0; break;
					case Init::LECUN:  s*=std::sqrt(1.0/ni); break;
					case Init::HE:     s*=std::sqrt(2.0/ni); break;
					case Init::XAVIER: s*=std::sqrt(2.0/(no+ni)); break;
					default: throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization scheme."); break;
				}
				std::exponential_distribution<double> dist(s);
				for(int m=0; m<b_[n].size(); ++m){
					b_[n].data()[m]=dist(rngen);
				}
			}
		} else if(annp.dist_b()==rng::dist::Name::CAUCHY){
			for(int n=0; n<nlayer_; ++n){
				const int ni=(n>0)?nNodes[n-1]:nInp;
				const int no=nNodes[n];
				double s=annp.sigma_b();
				switch(annp.init()){
					case Init::RAND:   s*=1.0; break;
					case Init::LECUN:  s*=std::sqrt(1.0/ni); break;
					case Init::HE:     s*=std::sqrt(2.0/ni); break;
					case Init::XAVIER: s*=std::sqrt(2.0/(no+ni)); break;
					default: throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization scheme."); break;
				}
				std::cauchy_distribution<double> dist(0.0,s);
				for(int m=0; m<b_[n].size(); ++m){
					b_[n].data()[m]=dist(rngen);
				}
			}
		} 
	//weights - initialization
		if(annp.dist_w()==rng::dist::Name::UNIFORM){
			for(int n=0; n<nlayer_; ++n){
				const int ni=(n>0)?nNodes[n-1]:nInp;
				const int no=nNodes[n];
				double s=std::sqrt(3.0*annp.sigma_w());
				switch(annp.init()){
					case Init::RAND:   s*=1.0; break;
					case Init::LECUN:  s*=std::sqrt(1.0/ni); break;
					case Init::HE:     s*=std::sqrt(2.0/ni); break;
					case Init::XAVIER: s*=std::sqrt(2.0/(no+ni)); break;
					default: throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization scheme."); break;
				}
				std::uniform_real_distribution<double> dist(-s,s);
				for(int m=0; m<w_[n].size(); ++m){
					w_[n].data()[m]=dist(rngen);
				}
			}
		} else if(annp.dist_w()==rng::dist::Name::NORMAL){
			for(int n=0; n<nlayer_; ++n){
				const int ni=(n>0)?nNodes[n-1]:nInp;
				const int no=nNodes[n];
				double s=annp.sigma_w();
				switch(annp.init()){
					case Init::RAND:   s*=1.0; break;
					case Init::LECUN:  s*=std::sqrt(1.0/ni); break;
					case Init::HE:     s*=std::sqrt(2.0/ni); break;
					case Init::XAVIER: s*=std::sqrt(2.0/(no+ni)); break;
					default: throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization scheme."); break;
				}
				std::normal_distribution<double> dist(0.0,s);
				for(int m=0; m<w_[n].size(); ++m){
					w_[n].data()[m]=dist(rngen);
				}
			}
		} else if(annp.dist_w()==rng::dist::Name::LAPLACE){
			for(int n=0; n<nlayer_; ++n){
				const int ni=(n>0)?nNodes[n-1]:nInp;
				const int no=nNodes[n];
				double s=annp.sigma_w();
				switch(annp.init()){
					case Init::RAND:   s*=1.0; break;
					case Init::LECUN:  s*=std::sqrt(1.0/ni); break;
					case Init::HE:     s*=std::sqrt(2.0/ni); break;
					case Init::XAVIER: s*=std::sqrt(2.0/(no+ni)); break;
					default: throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization scheme."); break;
				}
				std::exponential_distribution<double> dist(s);
				for(int m=0; m<w_[n].size(); ++m){
					w_[n].data()[m]=dist(rngen);
				}
			}
		} else if(annp.dist_w()==rng::dist::Name::CAUCHY){
			for(int n=0; n<nlayer_; ++n){
				const int ni=(n>0)?nNodes[n-1]:nInp;
				const int no=nNodes[n];
				double s=annp.sigma_w();
				switch(annp.init()){
					case Init::RAND:   s*=1.0; break;
					case Init::LECUN:  s*=std::sqrt(1.0/ni); break;
					case Init::HE:     s*=std::sqrt(2.0/ni); break;
					case Init::XAVIER: s*=std::sqrt(2.0/(no+ni)); break;
					default: throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization scheme."); break;
				}
				std::cauchy_distribution<double> dist(0.0,s);
				for(int m=0; m<w_[n].size(); ++m){
					w_[n].data()[m]=dist(rngen);
				}
			}
		} else throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid probability distribution.");
	//neuron
		neuron_=annp.neuron();
		switch(neuron_){
			//linear
			case Neuron::LINEAR:{
				affp_.resize(nlayer_,AFFP::af_lin);
				affpbp_.resize(nlayer_,AFFPBP::af_lin);
				affpbp2_.resize(nlayer_,AFFPBP2::af_lin);
			}break;
			//sigmoidal
			case Neuron::SIGMOID:{
				affp_.resize(nlayer_,AFFP::af_sigmoid);
				affpbp_.resize(nlayer_,AFFPBP::af_sigmoid);
				affpbp2_.resize(nlayer_,AFFPBP2::af_sigmoid);
			}break;
			case Neuron::TANH:{
				affp_.resize(nlayer_,AFFP::af_tanh);
				affpbp_.resize(nlayer_,AFFPBP::af_tanh);
				affpbp2_.resize(nlayer_,AFFPBP2::af_tanh);
			}break;
			case Neuron::ISRU:{
				affp_.resize(nlayer_,AFFP::af_isru);
				affpbp_.resize(nlayer_,AFFPBP::af_isru);
				affpbp2_.resize(nlayer_,AFFPBP2::af_isru);
			}break;
			case Neuron::ARCTAN:{
				affp_.resize(nlayer_,AFFP::af_arctan);
				affpbp_.resize(nlayer_,AFFPBP::af_arctan);
				affpbp2_.resize(nlayer_,AFFPBP2::af_arctan);
			}break;
			case Neuron::RELU:{
				affp_.resize(nlayer_,AFFP::af_relu);
				affpbp_.resize(nlayer_,AFFPBP::af_relu);
				affpbp2_.resize(nlayer_,AFFPBP2::af_relu);
			}break;
			case Neuron::ELU:{
				affp_.resize(nlayer_,AFFP::af_elu);
				affpbp_.resize(nlayer_,AFFPBP::af_elu);
				affpbp2_.resize(nlayer_,AFFPBP2::af_elu);
			}break;
			case Neuron::TANHRE:{
				affp_.resize(nlayer_,AFFP::af_tanhre);
				affpbp_.resize(nlayer_,AFFPBP::af_tanhre);
				affpbp2_.resize(nlayer_,AFFPBP2::af_tanhre);
			}break;
			case Neuron::SQRE:{
				affp_.resize(nlayer_,AFFP::af_sqre);
				affpbp_.resize(nlayer_,AFFPBP::af_sqre);
				affpbp2_.resize(nlayer_,AFFPBP2::af_sqre);
			}break;
			//gated-switch
			case Neuron::SWISH:{
				affp_.resize(nlayer_,AFFP::af_swish);
				affpbp_.resize(nlayer_,AFFPBP::af_swish);
				affpbp2_.resize(nlayer_,AFFPBP2::af_swish);
			}break;
			case Neuron::MISH:{
				affp_.resize(nlayer_,AFFP::af_mish);
				affpbp_.resize(nlayer_,AFFPBP::af_mish);
				affpbp2_.resize(nlayer_,AFFPBP2::af_mish);
			}break;
			case Neuron::GELU:{
				affp_.resize(nlayer_,AFFP::af_gelu);
				affpbp_.resize(nlayer_,AFFPBP::af_gelu);
				affpbp2_.resize(nlayer_,AFFPBP2::af_gelu);
			}break;
			case Neuron::PFLU:{
				affp_.resize(nlayer_,AFFP::af_pflu);
				affpbp_.resize(nlayer_,AFFPBP::af_pflu);
				affpbp2_.resize(nlayer_,AFFPBP2::af_pflu);
			}break;
			case Neuron::SERF:{
				affp_.resize(nlayer_,AFFP::af_serf);
				affpbp_.resize(nlayer_,AFFPBP::af_serf);
				affpbp2_.resize(nlayer_,AFFPBP2::af_serf);
			}break;
			case Neuron::LOGISH:{
				affp_.resize(nlayer_,AFFP::af_logish);
				affpbp_.resize(nlayer_,AFFPBP::af_logish);
				affpbp2_.resize(nlayer_,AFFPBP2::af_logish);
			}break;
			//switch
			case Neuron::SOFTPLUS:{
				affp_.resize(nlayer_,AFFP::af_softplus);
				affpbp_.resize(nlayer_,AFFPBP::af_softplus);
				affpbp2_.resize(nlayer_,AFFPBP2::af_softplus);
			}break;
			case Neuron::SQPLUS:{
				affp_.resize(nlayer_,AFFP::af_sqplus);
				affpbp_.resize(nlayer_,AFFPBP::af_sqplus);
				affpbp2_.resize(nlayer_,AFFPBP2::af_sqplus);
			}break;
			case Neuron::ATISH:{
				affp_.resize(nlayer_,AFFP::af_atish);
				affpbp_.resize(nlayer_,AFFPBP::af_atish);
				affpbp2_.resize(nlayer_,AFFPBP2::af_atish);
			}break;
			case Neuron::TEST:{
				affp_.resize(nlayer_,AFFP::af_test);
				affpbp_.resize(nlayer_,AFFPBP::af_test);
				affpbp2_.resize(nlayer_,AFFPBP2::af_test);
			}break;
			default: throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid neuron."); break;
		}
		//final layer is typically linear
		affp_.back()=AFFP::af_lin;
		affpbp_.back()=AFFPBP::af_lin;
		affpbp2_.back()=AFFPBP2::af_lin;
}

void ANN::reset(const ANNP& annp){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::resize(const ANNP&,const std::vector<int>&):\n";
	//initialize the random number generator
		if(annp.sigma_b()<=0) throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization deviation - bias");
		if(annp.sigma_w()<=0) throw std::invalid_argument("ANN::resize(const ANNP&,const std::vector<int>&): Invalid initialization deviation - weight");
		std::mt19937 rngen(annp.seed()<0?std::chrono::system_clock::now().time_since_epoch().count():annp.seed());
		std::uniform_real_distribution<double> uniform(-1.0,1.0);
	//gradients - nodes
		for(int n=0; n<nlayer_; ++n){
			dadz_[n].setZero();
			d2adz2_[n].setZero();
		}
	//nodes
		for(int n=0; n<nlayer_; ++n){
			a_[n].setZero();
			z_[n].setZero();
		}
}

/**
* execute the network
* @return out_ - the output of the network
*/
const VecXd& ANN::fp(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::fp():\n";
	//scale the input
	ins_.noalias()=inpw_.cwiseProduct(inp_+inpb_);
	//propagate the inputs
	z_[0]=b_[0];
	z_[0].noalias()+=w_[0]*ins_;
	(*affp_[0])(c_,z_[0],a_[0]);
	for(int l=1; l<nlayer_; ++l){
		z_[l]=b_[l];
		z_[l].noalias()+=w_[l]*a_[l-1];
		(*affp_[l])(c_,z_[l],a_[l]);
	}
	//scale the output
	out_=outb_;
	out_.noalias()+=a_.back().cwiseProduct(outw_);
	//return the output
	return out_;
}

/**
* execute the network
* @return out_ - the output of the network
*/
const VecXd& ANN::fpbp(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::fpbp():\n";
	//scale the input
	ins_.noalias()=inpw_.cwiseProduct(inp_+inpb_);
	//propagate the inputs
	z_[0]=b_[0];
	z_[0].noalias()+=w_[0]*ins_;
	(*affpbp_[0])(c_,z_[0],a_[0],dadz_[0]);
	for(int l=1; l<nlayer_; ++l){
		z_[l]=b_[l];
		z_[l].noalias()+=w_[l]*a_[l-1];
		(*affpbp_[l])(c_,z_[l],a_[l],dadz_[l]);
	}
	//scale the output
	out_=outb_;
	out_.noalias()+=a_.back().cwiseProduct(outw_);
	//return the output
	return out_;
}

/**
* execute the network
* @return out_ - the output of the network
*/
const VecXd& ANN::fpbp2(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::execute():\n";
	//scale the input
	ins_.noalias()=inpw_.cwiseProduct(inp_+inpb_);
	//propagate the inputs
	z_[0]=b_[0];
	z_[0].noalias()+=w_[0]*ins_;
	(*affpbp2_[0])(c_,z_[0],a_[0],dadz_[0],d2adz2_[0]);
	for(int l=1; l<nlayer_; ++l){
		z_[l]=b_[l];
		z_[l].noalias()+=w_[l]*a_[l-1];
		(*affpbp2_[l])(c_,z_[l],a_[l],dadz_[l],d2adz2_[l]);
	}
	//scale the output
	out_=outb_;
	out_.noalias()+=a_.back().cwiseProduct(outw_);
	//return the output
	return out_;
}

//==== static functions ====

/**
* write the network to file
* @param file - the file name where the network is to be written
* @param nn - the neural network to be written
*/
void ANN::write(const char* file, const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::write(const char*,const ANN&):\n";
	//local variables
	FILE* writer=NULL;
	//open the file
	writer=std::fopen(file,"w");
	if(writer!=NULL){
		ANN::write(writer,nn);
		std::fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for writing.\n"));
}

/**
* write the network to file
* @param writer - file pointer
* @param nn - the neural network to be written
*/
void ANN::write(FILE* writer, const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::write(FILE*,const ANN&):\n";
	//print the configuration
	fprintf(writer,"nn %i ",nn.nInp());
	for(int i=0; i<nn.nlayer(); ++i) fprintf(writer,"%i ",nn.nNodes(i));
	fprintf(writer,"\n");
	//print the neuron
	fprintf(writer,"neuron %s %f\n",Neuron::name(nn.neuron()),nn.c());
	//print the scaling layers
	fprintf(writer,"inpw ");
	for(int i=0; i<nn.nInp(); ++i) fprintf(writer,"%.15f ",nn.inpw()[i]);
	fprintf(writer,"\n");
	fprintf(writer,"outw ");
	for(int i=0; i<nn.nOut(); ++i) fprintf(writer,"%.15f ",nn.outw()[i]);
	fprintf(writer,"\n");
	//print the biasing layers
	fprintf(writer,"inpb ");
	for(int i=0; i<nn.nInp(); ++i) fprintf(writer,"%.15f ",nn.inpb()[i]);
	fprintf(writer,"\n");
	fprintf(writer,"outb ");
	for(int i=0; i<nn.nOut(); ++i) fprintf(writer,"%.15f ",nn.outb()[i]);
	fprintf(writer,"\n");
	//print the biases
	for(int n=0; n<nn.nlayer(); ++n){
		fprintf(writer,"bias[%i] ",n+1);
		for(int i=0; i<nn.b(n).size(); ++i){
			fprintf(writer,"%.15f ",nn.b(n)[i]);
		}
		fprintf(writer,"\n");
	}
	//print the weights
	for(int n=0; n<nn.nlayer(); ++n){
		fprintf(writer,"weight[%i,%i] ",n,n+1);
		for(int i=0; i<nn.w(n).rows(); ++i){
			for(int j=0; j<nn.w(n).cols(); ++j){
				fprintf(writer,"%.15f ",nn.w(n)(i,j));
			}
		}
		fprintf(writer,"\n");
	}
}

/**
* read the network from file
* @param file - the file name where the network is to be read
* @param nn - the neural network to be read
*/
void ANN::read(const char* file, ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::read(const char*,ANN&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		ANN::read(reader,nn);
		std::fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for reading.\n"));
}

/**
* read the network from file
* @param reader - file pointer
* @param nn - the neural network to be read
*/
void ANN::read(FILE* reader, ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"ANN::read(FILE*,ANN&):\n";
	//==== local variables ====
	const int MAX=5000;
	const int N_DIGITS=32;//max number of digits in number
	int b_max=0;//max number of biases for a given layer
	int w_max=0;//max number of weights for a given layer
	char* input=new char[MAX];
	char* b_str=NULL;//bias string
	char* w_str=NULL;//weight string
	char* i_str=NULL;//input string
	char* o_str=NULL;//output string
	std::vector<int> nodes;
	Token token;
	ANNP annp;
	//==== clear the network ====
	if(NN_PRINT_STATUS>0) std::cout<<"clearing the network\n";
	nn.clear();
	//==== load the configuration ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading configuration\n";
	token.read(fgets(input,MAX,reader),string::WS); token.next();
	while(!token.end()) nodes.push_back(std::atoi(token.next().c_str()));
	if(NN_PRINT_DATA>0){for(int i=0; i<nodes.size(); ++i) std::cout<<nodes[i]<<" "; std::cout<<"\n";}
	//==== set the nueron ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading neuron\n";
	token.read(fgets(input,MAX,reader),string::WS); token.next();
	annp.neuron()=Neuron::read(token.next().c_str());
	annp.c()=std::atof(token.next().c_str());
	if(annp.neuron()==Neuron::UNKNOWN) throw std::invalid_argument("ANN::read(FILE*,ANN&): Invalid neuron.");
	//==== resize the nueral newtork ====
	if(NN_PRINT_STATUS>0) std::cout<<"resizing neural network\n";
	nn.resize(annp,nodes);
	if(NN_PRINT_STATUS>1) std::cout<<"nn = "<<nn<<"\n";
	w_max=nn.nNodes(0)*nn.nInp();
	for(int i=0; i<nn.nlayer(); ++i) b_max=(b_max>nn.nNodes(i))?b_max:nn.nNodes(i);
	for(int i=1; i<nn.nlayer(); ++i) w_max=(w_max>nn.nNodes(i)*nn.nNodes(i-1))?w_max:nn.nNodes(i)*nn.nNodes(i-1);
	if(NN_PRINT_DATA>0) std::cout<<"b_max "<<b_max<<" w_max "<<w_max<<"\n";
	b_str=new char[b_max*N_DIGITS];
	w_str=new char[w_max*N_DIGITS];
	i_str=new char[nn.nInp()*N_DIGITS];
	o_str=new char[nn.nOut()*N_DIGITS];
	//==== read the scaling layers ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading input/output scaling layers\n";
	token.read(fgets(i_str,nn.nInp()*N_DIGITS,reader),string::WS); token.next();
	for(int j=0; j<nn.nInp(); ++j) nn.inpw()[j]=std::atof(token.next().c_str());
	token.read(fgets(o_str,nn.nOut()*N_DIGITS,reader),string::WS); token.next();
	for(int j=0; j<nn.nOut(); ++j) nn.outw()[j]=std::atof(token.next().c_str());
	//==== read the biasing layers ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading input/output biasing layers\n";
	token.read(fgets(i_str,nn.nInp()*N_DIGITS,reader),string::WS); token.next();
	for(int j=0; j<nn.nInp(); ++j) nn.inpb()[j]=std::atof(token.next().c_str());
	token.read(fgets(o_str,nn.nOut()*N_DIGITS,reader),string::WS); token.next();
	for(int j=0; j<nn.nOut(); ++j) nn.outb()[j]=std::atof(token.next().c_str());
	//==== read in the biases ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading biases\n";
	for(int n=0; n<nn.nlayer(); ++n){
		token.read(fgets(b_str,b_max*N_DIGITS,reader),string::WS); token.next();
		for(int i=0; i<nn.b(n).size(); ++i){
			nn.b(n)[i]=std::atof(token.next().c_str());
		}
	}
	//==== read in the weights ====
	if(NN_PRINT_STATUS>0) std::cout<<"reading weights\n";
	for(int n=0; n<nn.nlayer(); ++n){
		token.read(fgets(w_str,w_max*N_DIGITS,reader),string::WS); token.next();
		for(int i=0; i<nn.w(n).rows(); ++i){
			for(int j=0; j<nn.w(n).cols(); ++j){
				nn.w(n)(i,j)=std::atof(token.next().c_str());
			}
		}
	}
	//==== free local variables ====
	if(input!=NULL) delete[] input;
	if(b_str!=NULL) delete[] b_str;
	if(w_str!=NULL) delete[] w_str;
	if(i_str!=NULL) delete[] i_str;
	if(o_str!=NULL) delete[] o_str;
}

//***********************************************************************
// ANNP
//***********************************************************************

/**
* set the default parameters for ANNP
*/
void ANNP::defaults(){
	if(NN_PRINT_FUNC>0) std::cout<<"ANNP::defaults():\n";
	sigma_b_=0.001;
	sigma_w_=1.0;
	c_=1.0;
	dist_b_=rng::dist::Name::NORMAL;
	dist_w_=rng::dist::Name::NORMAL;
	init_=Init::RAND;
	seed_=-1;
	neuron_=NN::Neuron::UNKNOWN;
}

/**
* print ANNP parameters to screen
* @param out - output stream
* @param annp - ANNP instance
* @return output stream
*/
std::ostream& operator<<(std::ostream& out, const ANNP& annp){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ANNP",str)<<"\n";
	out<<"seed    = "<<annp.seed_<<"\n";
	out<<"dist_b  = "<<annp.dist_b_<<"\n";
	out<<"dist_w  = "<<annp.dist_w_<<"\n";
	out<<"sigma_b = "<<annp.sigma_b_<<"\n";
	out<<"sigma_w = "<<annp.sigma_w_<<"\n";
	out<<"init    = "<<annp.init_<<"\n";
	out<<"neuron  = "<<annp.neuron_<<"\n";
	out<<"sharp   = "<<annp.c_<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== static functions ====

/**
* read ANNP parameters from a text file
* @param file - name of file
* @param annp - ANNP instance which will be read
*/
void ANNP::read(const char* file, ANNP& annp){
	if(NN_PRINT_FUNC>0) std::cout<<"ANNP::read(const char*,ANNP&):\n";
	//local variables
	FILE* reader=NULL;
	//open the file
	reader=std::fopen(file,"r");
	if(reader!=NULL){
		ANNP::read(reader,annp);
		std::fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("ERROR: Could not open \"")+std::string(file)+std::string("\" for reading.\n"));
}

/**
* read ANNP parameters from a FILE pointer
* @param file - name of file
* @param annp - ANNP instance which will be read
*/
void ANNP::read(FILE* reader, ANNP& annp){
	if(NN_PRINT_FUNC>0) std::cout<<"ANNP::read(FILE*,ANNP&):\n";
	//==== local variables ====
	char* input=new char[string::M];
	Token token;
	//==== rewind reader ====
	std::rewind(reader);
	//==== read parameters ====
	while(fgets(input,string::M,reader)!=NULL){
		token.read(string::trim_right(input,string::COMMENT),string::WS);
		if(token.end()) continue;//skip if empty
		const std::string tag=string::to_upper(token.next());
		if(tag=="SEED"){//random seed
			annp.seed()=std::atoi(token.next().c_str());
		} else if(tag=="SIGMA_B"){//initialization deviation - bias
			annp.sigma_b()=std::atof(token.next().c_str());
		} else if(tag=="SIGMA_W"){//initialization deviation - weight
			annp.sigma_w()=std::atof(token.next().c_str());
		} else if(tag=="DIST_B"){//initialization distribution - bias
			annp.dist_b()=rng::dist::Name::read(string::to_upper(token.next()).c_str());
		} else if(tag=="DIST_W"){//initialization distribution - weight
			annp.dist_w()=rng::dist::Name::read(string::to_upper(token.next()).c_str());
		} else if(tag=="INIT"){//initialization
			annp.init()=NN::Init::read(string::to_upper(token.next()).c_str());
		} else if(tag=="NEURON"){//neuron function
			annp.neuron()=NN::Neuron::read(string::to_upper(token.next()).c_str());
		} else if(tag=="SHARPNESS"){//neuron function
			annp.c()=std::atof(token.next().c_str());
		}
	}
	//==== free local variables ====
	delete[] input;
}

//***********************************************************************
// Cost
//***********************************************************************

/**
* clear all local data
*/
void Cost::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::clear():\n";
	dcdz_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the cost function
*/
void Cost::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::resize(const ANN&):\n";
	dcdz_.resize(nn.nlayer());
	for(int n=0; n<nn.nlayer(); ++n){
		dcdz_[n]=VecXd::Zero(nn.nNodes(n));
	}
	grad_.resize(nn.size());
}

/**
* compute gradient of error given the derivative of the cost function w.r.t. the output (dcdo)
* @param nn - the neural network for which we will compute the gradient
* @param dcdo - the derivative of the cost function w.r.t. the output (dc/do)
* @return grad - the gradient of the cost function w.r.t. each parameter of the network
*/
const VecXd& Cost::grad(const ANN& nn, const VecXd& dcdo){
	if(NN_PRINT_FUNC>0) std::cout<<"Cost::grad(const ANN&,const VecXd&):\n";
	const int nlayer=nn.nlayer();
	//compute delta for the output layer
	const int size=nn.outw().size();
	// concise implementation:
	//for(int i=0; i<size; ++i) dcdz_[nlayer-1][i]=nn.outw()[i]*dcdo[i]*nn.dadz(nlayer-1)[i];
	// efficient implementation:
	dcdz_[nlayer-1]=dcdo;
	for(int i=0; i<size; ++i) dcdz_[nlayer-1][i]*=nn.outw()[i]*nn.dadz(nlayer-1)[i];
	//back-propogate the error
	for(int l=nlayer-1; l>0; --l){
		// concise implementation:
		//dcdz_[l-1].noalias()=nn.dadz(l-1).cwiseProduct(nn.w(l).transpose()*dcdz_[l]);
		// efficient implementation:
		dcdz_[l-1].noalias()=nn.w(l).transpose()*dcdz_[l];
		const int s=dcdz_[l-1].size();
		for(int i=0; i<s; ++i) dcdz_[l-1][i]*=nn.dadz(l-1)[i];
	}
	int count=0;
	//gradient w.r.t bias
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. bias\n";
	for(int l=0; l<nlayer; ++l){
		const int s=dcdz_[l].size();
		for(int n=0; n<s; ++n){
			grad_[count++]=dcdz_[l][n];//bias(l,n)
		}
	}
	//gradient w.r.t. edges
	if(NN_PRINT_STATUS>1) std::cout<<"computing gradient w.r.t. edges\n";
	//l=0
	{
		//const int ncols=nn.w(0).cols();//mathematical defn.
		//const int nrows=nn.w(0).rows();//mathematical defn.
		const int ncols=nn.ins().size();//comp. efficient defn.
		for(int m=0; m<ncols; ++m){
			const double a=nn.ins()(m);
			const int nrows=dcdz_[0].size();//comp. efficient defn.
			for(int n=0; n<nrows; ++n){
				grad_[count++]=dcdz_[0][n]*a;//weight(0,n,m)
			}
		}
	}
	//l>0
	for(int l=1; l<nlayer; ++l){
		//const int ncols=nn.w(l).cols();//mathematical defn.
		//const int nrows=nn.w(l).rows();//mathematical defn.
		const int ncols=nn.a(l-1).size();//comp. efficient defn.
		for(int m=0; m<ncols; ++m){
			const double a=nn.a(l-1)(m);
			const int nrows=dcdz_[l].size();//comp. efficient defn.
			for(int n=0; n<nrows; ++n){
				grad_[count++]=dcdz_[l][n]*a;//weight(l,n,m)
			}
		}
	}
	//return the gradient
	return grad_;
}

//***********************************************************************
// DODZ
//***********************************************************************

/**
* clear all local data
*/
void DODZ::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"DODZ::clear():\n";
	dodz_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the gradient
*/
void DODZ::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DODZ::resize(const ANN&):\n";
	dodi_=MatXd::Zero(nn.out().size(),nn.inp().size());
	dodz_.resize(nn.nlayer());
	for(int n=0; n<nn.nlayer(); ++n){
		dodz_[n]=MatXd::Zero(nn.out().size(),nn.nNodes(n));
	}
}

/**
* compute the gradient of output w.r.t. each node input (dO/dZ)
* @param nn - the neural network for which we will compute the gradient
*/
void DODZ::grad(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DODZ::grad(const ANN&):\n";
	//back-propogate the gradient (n.b. do/dz_{o}=outw_ "gradient of out_ w.r.t. the input of out_ is outw_")
	//==== mathematically concise implementation ====
	dodz_.back()=nn.outw().cwiseProduct(nn.dadz(nn.nlayer()-1)).asDiagonal();
	for(int l=nn.nlayer()-1; l>0; --l){
		dodz_[l-1].noalias()=dodz_[l]*nn.w(l)*nn.dadz(l-1).asDiagonal();
	}
	dodi_=dodz_[0]*nn.w(0)*nn.inpw().asDiagonal();
	//==== computationally efficient implementation ====
	/*dodz_.back()=nn.outw().cwiseProduct(nn.dadz(nn.nlayer()-1)).asDiagonal();
	for(int l=nn.nlayer()-1; l>0; --l){
		dodz_[l-1].noalias()=dodz_[l]*nn.w(l);
		const int ncols=dodz_[l-1].cols();
		for(int i=0; i<ncols; ++i){
			dodz_[l-1].col(i)*=nn.dadz(l-1)[i];
		}
	}
	dodi_.noalias()=dodz_[0]*nn.w(0);
	const int ncols=dodi_.cols();
	for(int i=0; i<ncols; ++i){
		dodi_.col(i)*=nn.inpw()[i];
	}*/
}

//***********************************************************************
// DODP
//***********************************************************************

/**
* clear all local data
*/
void DODP::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"DODP::clear():\n";
	dodz_.clear();
	dodb_.clear();
	dodw_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the gradient
*/
void DODP::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DODP::resize(const ANN&):\n";
	dodz_.resize(nn.nlayer());
	for(int n=0; n<nn.nlayer(); ++n){
		dodz_[n]=MatXd::Zero(nn.out().size(),nn.nNodes(n));
	}
	dodb_.resize(nn.nOut());
	for(int n=0; n<nn.nOut(); ++n){
		dodb_[n].resize(nn.nlayer());
		for(int l=0; l<nn.nlayer(); ++l){
			dodb_[n][l]=VecXd::Zero(nn.b(l).size());
		}
	}
	dodw_.resize(nn.nOut());
	for(int n=0; n<nn.nOut(); ++n){
		dodw_[n].resize(nn.nlayer());
		for(int l=0; l<nn.nlayer(); ++l){
			dodw_[n][l]=MatXd::Zero(nn.w(l).rows(),nn.w(l).cols());
		}
	}
	dodp_=MatXd::Zero(nn.nOut(),nn.nBias()+nn.nWeight());
}

/**
* compute the gradient of output w.r.t. each network parameter (weight and bias)
* @param nn - the neural network for which we will compute the gradient
*/
void DODP::grad(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DODP::grad(const ANN&):\n";
	//back-propogate the gradient (n.b. do/dz_{o}=outw_ "gradient of out_ w.r.t. the input of out_ is outw_")
	dodz_.back()=nn.outw().cwiseProduct(nn.dadz(nn.nlayer()-1)).asDiagonal();
	for(int l=nn.nlayer()-1; l>0; --l){
		dodz_[l-1].noalias()=dodz_[l]*nn.w(l)*nn.dadz(l-1).asDiagonal();
	}
	//compute the gradient of the output w.r.t. the biases
	for(int n=0; n<nn.nOut(); ++n){
		for(int l=0; l<nn.nlayer(); ++l){
			for(int i=0; i<nn.b(l).size(); ++i){
				dodb_[n][l](i)=dodz_[l](n,i);
			}
		}
	}
	//compute the gradient of the output w.r.t. the weights
	for(int n=0; n<nn.nOut(); ++n){
		for(int l=0; l<nn.nlayer(); ++l){
			for(int j=0; j<nn.w(l).cols(); ++j){
				const double a=(l>0)?nn.a(l-1)[j]:nn.ins()[j];
				for(int i=0; i<nn.w(l).rows(); ++i){
					dodw_[n][l](i,j)=dodz_[l](n,i)*a;
				}
			}
		}
	}
	//pack the gradient
	for(int n=0; n<nn.nOut(); ++n){
		int c=0;
		for(int l=0; l<nn.nlayer(); ++l){
			for(int i=0; i<nn.b(l).size(); ++i){
				dodp_(n,c++)=dodb_[n][l](i);
			}
		}
		for(int l=0; l<nn.nlayer(); ++l){
			for(int j=0; j<nn.w(l).cols(); ++j){
				for(int i=0; i<nn.w(l).rows(); ++i){
					dodp_(n,c++)=dodw_[n][l](i,j);
				}
			}
		}
	}
}

//***********************************************************************
// DZDI
//***********************************************************************

/**
* clear all local data
*/
void DZDI::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"DZDI::clear():\n";
	dzdi_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the gradient
*/
void DZDI::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DZDI::resize(const ANN&):\n";
	dzdi_.resize(nn.nlayer());
	for(int n=0; n<nn.nlayer(); ++n){
		dzdi_[n]=MatXd::Zero(nn.nNodes(n),nn.nInp());
	}
}

/**
* compute the gradient of each node input (z) w.r.t. the network inputs (i) (dz/di)
* @param nn - the neural network for which we will compute the gradient
*/
void DZDI::grad(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"DZDI::grad(const ANN&):\n";
	dzdi_[0]=nn.w(0)*nn.dadz(0).asDiagonal()*nn.inpw().asDiagonal();
	dzdi_[0]=nn.w(0)*nn.inpw().asDiagonal();
	for(int i=1; i<nn.nlayer(); ++i){
		dzdi_[i]=nn.w(i)*nn.dadz(i-1).asDiagonal()*dzdi_[i-1];
	}
}

//***********************************************************************
// D2ODZDI
//***********************************************************************

/**
* clear all local data
*/
void D2ODZDI::clear(){
	if(NN_PRINT_FUNC>0) std::cout<<"D2ODZDI::clear():\n";
	dOdZ_.clear();
	dZdI_.clear();
	d2odzdi_.clear();
	d2odbdi_.clear();
	d2odwdi_.clear();
	d2odpdi_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the gradient
*/
void D2ODZDI::resize(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"D2ODZDI::resize(const ANN&):\n";
	dOdZ_.resize(nn);
	dZdI_.resize(nn);
	
	d2odzdi_.resize(nn.nlayer());
	for(int l=0; l<nn.nlayer(); ++l){
		d2odzdi_[l].resize(nn.nNodes(l));
		for(int n=0; n<nn.nNodes(l); ++n){
			d2odzdi_[l][n]=MatXd::Zero(nn.nOut(),nn.nInp());
		}
	}
	d2odbdi_.resize(nn.nlayer());
	for(int l=0; l<nn.nlayer(); ++l){
		d2odbdi_[l].resize(nn.b(l).size());
		for(int n=0; n<nn.b(l).size(); ++n){
			d2odbdi_[l][n]=MatXd::Zero(nn.nOut(),nn.nInp());
		}
	}
	d2odwdi_.resize(nn.nlayer());
	for(int l=0; l<nn.nlayer(); ++l){
		d2odwdi_[l].resize(nn.w(l).rows());
		for(int n=0; n<nn.w(l).rows(); ++n){
			d2odwdi_[l][n].resize(nn.w(l).cols());
			for(int m=0; m<nn.w(l).cols(); ++m){
				d2odwdi_[l][n][m]=MatXd::Zero(nn.nOut(),nn.nInp());
			}
		}
	}
	d2odpdi_.resize(nn.size());
	for(int i=0; i<nn.size(); ++i){
		d2odpdi_[i]=MatXd::Zero(nn.nOut(),nn.nInp());
	}
}

/**
* compute the gradient of the gradient of (o) w.r.t. (z) and (i) 
* @param nn - the neural network for which we will compute the gradient
*/
void D2ODZDI::grad(const ANN& nn){
	if(NN_PRINT_FUNC>0) std::cout<<"D2ODZDI::grad(const ANN&):\n";
	int count=0;
	//compute dOdZ
	dOdZ_.grad(nn);
	//compute dZdI
	dZdI_.grad(nn);
	//final layer
	const int nlm1=nn.nlayer()-1;
	for(int j=0; j<nn.nNodes(nlm1); ++j){
		d2odzdi_[nlm1][j].row(j).noalias()=nn.d2adz2(nlm1)(j)*dZdI_.dzdi(nlm1).row(j);
	}
	for(int l=nn.nlayer()-2; l>=0; l--){
		for(int j=0; j<nn.nNodes(l); ++j){
			for(int i=0; i<nn.nOut(); ++i){
				for(int k=0; k<nn.nInp(); ++k){
					for(int n=0; n<nn.nNodes(l+1); ++n){
						d2odzdi_[l][j](i,k)+=
							d2odzdi_[l+1][n](i,k)*nn.w(l+1)(n,j)*nn.dadz(l)(j)
							+dOdZ_.dodz(l+1)(i,n)*nn.w(l+1)(n,j)*nn.d2adz2(l)(j)*dZdI_.dzdi(l)(j,k);
					}
				}
			}
		}
	}
	//bias
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<nn.b(l).size(); ++n){
			d2odbdi_[l][n]=d2odzdi_[l][n];
			
		}
	}
	/*
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<nn.b(l).size(); ++n){
			std::cout<<"d2odbdi_["<<l<<"]["<<n<<"] = "<<d2odbdi_[l][n]<<"\n";
		}
	}
	*/
	//weight
	for(int l=0; l<nn.nlayer(); ++l){
		for(int m=0; m<nn.w(l).cols(); ++m){
			const double a=(l>0)?nn.a(l-1)(m):nn.ins()(m);
			for(int n=0; n<nn.w(l).rows(); ++n){
				d2odwdi_[l][n][m]=d2odzdi_[l][n]*a;
			}
		}
	}
	for(int m=0; m<nn.w(0).cols(); ++m){
		for(int n=0; n<nn.w(0).rows(); ++n){
			for(int i=0; i<nn.nOut(); ++i){
				d2odwdi_[0][n][m](i,m)+=dOdZ_.dodz(0)(i,n);
			}
		}
	}
	for(int l=1; l<nn.nlayer(); ++l){
		for(int m=0; m<nn.w(l).cols(); ++m){
			for(int n=0; n<nn.w(l).rows(); ++n){
				for(int i=0; i<nn.nOut(); ++i){
					for(int k=0; k<nn.nInp(); ++k){
						d2odwdi_[l][n][m](i,k)+=dOdZ_.dodz(l)(i,n)*nn.dadz(l-1)(m)*dZdI_.dzdi(l-1)(m,k);
					}
				}
			}
		}
	}
	/*
	for(int l=0; l<nn.nlayer(); ++l){
		for(int m=0; m<nn.w(l).cols(); ++m){
			for(int n=0; n<nn.w(l).rows(); ++n){
				std::cout<<"d2odwdi_["<<l<<"]["<<n<<"]["<<m<<"] = "<<d2odwdi_[l][n][m]<<"\n";
			}
		}
	}
	*/
	int c=0;
	for(int l=0; l<nn.nlayer(); ++l){
		for(int n=0; n<nn.b(l).size(); ++n){
			d2odpdi_[c++]=d2odbdi_[l][n];
		}
	}
	for(int l=0; l<nn.nlayer(); ++l){
		for(int m=0; m<nn.w(l).cols(); ++m){
			for(int n=0; n<nn.w(l).rows(); ++n){
				d2odpdi_[c++]=d2odwdi_[l][n][m];
			}
		}
	}
}

//***********************************************************************
// D2ODZDIN
//***********************************************************************

/**
* clear all local data
*/
void D2ODZDIN::clear(){
	nnc_.clear();
	dOdZ_.clear();
	d2odpdi_.clear();
}

/**
* resize data for a given neural network
* @param nn - the neural network for which we will compute the gradient
*/
void D2ODZDIN::resize(const ANN& nn){
	//gradient of the ouput with respect to the input
	dOdZ_.resize(nn);
	//second derivative
	d2odpdi_.resize(nn.size());
}

/**
* compute the gradient of the gradient of (o) w.r.t. (z) and (i) numerically
* @param nn - the neural network for which we will compute the gradient
*/
void D2ODZDIN::grad(const ANN& nn){
	//local variables
	int count=0;
	//make copy of the network 
	nnc_=nn;
	//loop over all biases
	for(int l=0; l<nnc_.nlayer(); ++l){
		for(int n=0; n<nnc_.b(l).size(); ++n){
			const double delta=nnc_.b(l)[n]/100.0;
			//point 1
			nnc_.b(l)[n]=nn.b(l)[n]-delta;
			nnc_.fpbp();
			dOdZ_.grad(nnc_);
			pt1_=dOdZ_.dodi();
			//point 2
			nnc_.b(l)[n]=nn.b(l)[n]+delta;
			nnc_.fpbp();
			dOdZ_.grad(nnc_);
			pt2_=dOdZ_.dodi();
			//reset
			nnc_.b(l)[n]=nn.b(l)[n];
			//gradient
			d2odpdi_[count++].noalias()=0.5*(pt2_-pt1_)/delta;
			//std::cout<<"d2odpdi_["<<count-1<<"] = "<<d2odpdi_[count-1]<<"\n";
		}
	}
	//loop over all weights
	for(int l=0; l<nnc_.nlayer(); ++l){
		for(int n=0; n<nn.w(l).size(); ++n){
			const double delta=nnc_.w(l)(n)/1000.0;
			//point 1
			nnc_.w(l)(n)=nn.w(l)(n)-delta;
			nnc_.fpbp();
			dOdZ_.grad(nnc_);
			pt1_=dOdZ_.dodi();
			//point 2
			nnc_.w(l)(n)=nn.w(l)(n)+delta;
			nnc_.fpbp();
			dOdZ_.grad(nnc_);
			pt2_=dOdZ_.dodi();
			//reset
			nnc_.w(l)(n)=nn.w(l)(n);
			//gradient
			d2odpdi_[count++].noalias()=0.5*(pt2_-pt1_)/delta;
			//std::cout<<"d2odpdi_["<<count-1<<"] = "<<d2odpdi_[count-1]<<"\n";
		}
	}
}

}

//***********************************************************************
// serialization
//***********************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const NN::ANN& obj){
		if(NN_PRINT_FUNC>0) std::cout<<"nbytes(const NN::ANN&):\n";
		int size=0;
		size+=sizeof(NN::Neuron);//neuron type
		size+=sizeof(int);//nlayer_
		size+=sizeof(double);//c_
		if(obj.nlayer()>0){
			size+=sizeof(int)*(obj.nlayer()+1);//number of nodes in each layer
			for(int l=0; l<obj.nlayer(); ++l) size+=obj.b(l).size()*sizeof(double);//bias
			for(int l=0; l<obj.nlayer(); ++l) size+=obj.w(l).size()*sizeof(double);//weight
			size+=obj.nInp()*sizeof(double);//pre-scale
			size+=obj.nInp()*sizeof(double);//pre-bias
			size+=obj.nOut()*sizeof(double);//post-scale
			size+=obj.nOut()*sizeof(double);//post-bias
		}
		return size;
	}
	
	template <> int nbytes(const NN::ANNP& obj){
		if(NN_PRINT_FUNC>0) std::cout<<"nbytes(const NN::ANNP&):\n";
		int size=0;
		size+=sizeof(int);//seed_
		size+=sizeof(rng::dist::Name);//dist - bias
		size+=sizeof(rng::dist::Name);//dist - weight
		size+=sizeof(NN::Init);
		size+=sizeof(NN::Neuron);//neuron type
		size+=sizeof(double);//sigma_b_
		size+=sizeof(double);//sigma_w_
		size+=sizeof(double);//c_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const NN::ANN& obj, char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"pack(const NN::ANN&,char*):\n";
		int pos=0;
		int tmp=0;
		//neuron type
		std::memcpy(arr+pos,&(obj.neuron()),sizeof(NN::Neuron)); pos+=sizeof(NN::Neuron);
		std::memcpy(arr+pos,&(obj.c()),sizeof(double)); pos+=sizeof(double);
		//nlayer_
		std::memcpy(arr+pos,&(tmp=obj.nlayer()),sizeof(int)); pos+=sizeof(int);
		if(obj.nlayer()>0){
			//number of inputs
			std::memcpy(arr+pos,&(tmp=obj.nInp()),sizeof(int)); pos+=sizeof(int);
			//number of nodes in each layer
			for(int l=0; l<obj.nlayer(); ++l){
				std::memcpy(arr+pos,&(tmp=obj.nNodes(l)),sizeof(int)); pos+=sizeof(int);
			}
			//bias
			for(int l=0; l<obj.nlayer(); ++l){
				const int s=obj.b(l).size()*sizeof(double);
				std::memcpy(arr+pos,obj.b(l).data(),s); pos+=s;
			}
			//weights
			for(int l=0; l<obj.nlayer(); ++l){
				const int s=obj.w(l).size()*sizeof(double);
				std::memcpy(arr+pos,obj.w(l).data(),s); pos+=s;
			}
			//inputs
			const int sinp=obj.nInp()*sizeof(double);
			//pre-scale
			std::memcpy(arr+pos,obj.inpw().data(),sinp); pos+=sinp;
			//pre-bias
			std::memcpy(arr+pos,obj.inpb().data(),sinp); pos+=sinp;
			//inputs
			const int sout=obj.nOut()*sizeof(double);
			//post-scale
			std::memcpy(arr+pos,obj.outw().data(),sout); pos+=sout;
			//post-bias
			std::memcpy(arr+pos,obj.outb().data(),sout); pos+=sout;
		}
		//return bytes written
		return pos;
	}
	
	template <> int pack(const NN::ANNP& obj, char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"pack(const NN::ANNP&,char*):\n";
		int pos=0;
		int tmp=0;
		std::memcpy(arr+pos,&obj.seed(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.dist_b(),sizeof(rng::dist::Name)); pos+=sizeof(rng::dist::Name);
		std::memcpy(arr+pos,&obj.dist_w(),sizeof(rng::dist::Name)); pos+=sizeof(rng::dist::Name);
		std::memcpy(arr+pos,&obj.init(),sizeof(NN::Init)); pos+=sizeof(NN::Init);
		std::memcpy(arr+pos,&obj.neuron(),sizeof(NN::Neuron)); pos+=sizeof(NN::Neuron);
		std::memcpy(arr+pos,&obj.sigma_b(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.sigma_w(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.c(),sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(NN::ANN& obj, const char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"unpack(NN::ANN&,const char*):\n";
		//local variables
		int pos=0;
		int nlayer=0,nInp=0;
		double c=1.0;
		std::vector<int> nNodes;
		NN::Neuron neuron=NN::Neuron::UNKNOWN;
		//neuron type
		std::memcpy(&neuron,arr+pos,sizeof(NN::Neuron)); pos+=sizeof(NN::Neuron);
		std::memcpy(&c,arr+pos,sizeof(double)); pos+=sizeof(double);
		//nlayer
		std::memcpy(&nlayer,arr+pos,sizeof(int)); pos+=sizeof(int);
		if(nlayer>0){
			nNodes.resize(nlayer,0);
			//number of inputs
			std::memcpy(&nInp,arr+pos,sizeof(int)); pos+=sizeof(int);
			//number of nodes in each layer
			for(int i=0; i<nlayer; ++i){
				std::memcpy(&nNodes[i],arr+pos,sizeof(int)); pos+=sizeof(int);
			}
			//resize the network
			NN::ANNP annp;
			annp.neuron()=neuron;
			annp.c()=c;
			obj.resize(annp,nInp,nNodes);
			//bias
			for(int l=0; l<obj.nlayer(); ++l){
				const int s=obj.b(l).size()*sizeof(double);
				std::memcpy(obj.b(l).data(),arr+pos,s); pos+=s;
			}
			//weights
			for(int l=0; l<obj.nlayer(); ++l){
				const int s=obj.w(l).size()*sizeof(double);
				std::memcpy(obj.w(l).data(),arr+pos,s); pos+=s;
			}
			//inputs
			const int sinp=obj.nInp()*sizeof(double);
			//pre-scale
			std::memcpy(obj.inpw().data(),arr+pos,sinp); pos+=sinp;
			//pre-bias
			std::memcpy(obj.inpb().data(),arr+pos,sinp); pos+=sinp;
			//outputs
			const int sout=obj.nOut()*sizeof(double);
			//post-scale
			std::memcpy(obj.outw().data(),arr+pos,sout); pos+=sout;
			//post-bias
			std::memcpy(obj.outb().data(),arr+pos,sout); pos+=sout;
		}
		//return bytes read
		return pos;
	}
	
	template <> int unpack(NN::ANNP& obj, const char* arr){
		if(NN_PRINT_FUNC>0) std::cout<<"unpack(NN::ANNP&,const char*):\n";
		//local variables
		int pos=0;
		int size=0;
		std::memcpy(&obj.seed(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.dist_b(),arr+pos,sizeof(rng::dist::Name)); pos+=sizeof(rng::dist::Name);
		std::memcpy(&obj.dist_w(),arr+pos,sizeof(rng::dist::Name)); pos+=sizeof(rng::dist::Name);
		std::memcpy(&obj.init(),arr+pos,sizeof(NN::Init)); pos+=sizeof(NN::Init);
		std::memcpy(&obj.neuron(),arr+pos,sizeof(NN::Neuron)); pos+=sizeof(NN::Neuron);
		std::memcpy(&obj.sigma_b(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.sigma_w(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.c(),arr+pos,sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
	
}