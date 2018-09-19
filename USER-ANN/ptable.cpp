#include "ptable.h"

namespace PTable{
	
//*********************************************
//Function
//*********************************************

//************** NAME ***************
const char* name(unsigned int an){return ELEMENT_NAME[an-1];}
//********** ATOMIC_NUMBER **********
unsigned int an(const char* name){
	for(unsigned int i=0; i<NUM_ELEMENTS; i++){
		if(std::strcmp(name,ELEMENT_NAME[i])==0) return i+1;
	}
	return 0;
}
unsigned int an(double mass){
	double min=100;
	unsigned int an=0;
	for(int i=0; i<NUM_ELEMENTS; ++i){
		if(std::fabs(mass-ELEMENT_MASS[i])<min){
			min=std::fabs(mass-ELEMENT_MASS[i]);
			an=i+1;
		}
	}
	return an;
}
//************** MASS ***************
double mass(unsigned int an){return ELEMENT_MASS[an-1];}
//************* RADIUS **************
double atomicRadius(unsigned int an){return ATOMIC_RADII[an-1];}
double covalentRadius(unsigned int an){return COVALENT_RADII[an-1];}

}

