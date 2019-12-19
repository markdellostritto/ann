//c++ libraries
#include <ostream>
#include <stdexcept>
//ann - math
#include "math_func_ann.h"
// ann - math - special
#include "math_special_ann.h"

namespace special{
	
	//**************************************************************
	//trig (fdlibm)
	//**************************************************************
	
	//cosine function
	double cos(double x){
		x*=x;
		return 1.0+x*(-0.5+x*(cos_const[0]+x*(cos_const[1]+x*(cos_const[2]+x*(cos_const[3]+x*(cos_const[4]+x*cos_const[5]))))));
	}
	
	//sine function
	double sin(double x){
		const double r=x*x;
		return x*(1.0+r*(sin_const[0]+r*(sin_const[1]+r*(sin_const[2]+r*(sin_const[3]+r*(sin_const[4]+r*sin_const[5]))))));
	}
	
	//**************************************************************
	//Logarithm
	//**************************************************************
	
	double logp1(double x){
		const double y=x/(x+2.0);
		const double y2=y*y;
		return 2.0*y*(logp1c[0]+y2*(logp1c[1]+y2*(logp1c[2]+y2*(logp1c[3]+y2*logp1c[4]))));
	}
	
	//**************************************************************
	//Softplus
	//**************************************************************
	
	double softplus(double x){
		if(x>=1){
			return x+logp1(std::exp(-x));
		} else {
			const double exp=std::exp(x);
			const double f=exp/(exp+2.0);
			const double f2=f*f;
			return 2.0*f*(sfpc[0]+f2*(sfpc[1]+f2*(sfpc[2]+f2*(sfpc[3]+f2*sfpc[4]))));
		}
	}
	
	//**************************************************************
	//Complementary Error Function - Approximations
	//**************************************************************
	
	const double erfa_const::a1[5]={1.0,0.278393,0.230389,0.000972,0.078108};
	const double erfa_const::a2[5]={0.0,0.3480242,-0.0958798,0.7478556,0.47047};
	const double erfa_const::a3[7]={1.0,0.0705230784,0.0422820123,0.0092705272,0.0001520143,0.0002765672,0.0000430638};
	const double erfa_const::a4[7]={0.0,0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429,0.3275911};
	
	double erfa1(double x){double s=sign(x); x*=s; x=function::poly<4>(x,erfa_const::a1); x*=x; x*=x; return s*(1.0-1.0/x);}
	double erfa2(double x){double s=sign(x); x*=s; return s*(1.0-function::poly<3>(1.0/(1.0+erfa_const::a2[4]*x),erfa_const::a2)*std::exp(-x*x));}
	double erfa3(double x){double s=sign(x); x*=s; x=function::poly<6>(x,erfa_const::a3); x*=x; x*=x; x*=x; x*=x; return s*(1.0-1.0/x);}
	double erfa4(double x){double s=sign(x); x*=s; return s*(1.0-function::poly<5>(1.0/(1.0+erfa_const::a4[6]*x),erfa_const::a4)*std::exp(-x*x));}
	
}
