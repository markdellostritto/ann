// c libraries
#include <cstring>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// ann - ptable
#include "chem/ptable.hpp"

namespace ptable{
	
//*********************************************
//Function
//*********************************************

//************** NAME ***************

const char* name(int an){return NAME[an-1];}

//********** ATOMIC_NUMBER **********

int an(const char* name){
	for(int i=0; i<N_ELEMENTS; i++){
		if(std::strcmp(name,NAME[i])==0) return i+1;
	}
	return 0;
}

int an(double mass){
	double min=100;
	int an=0;
	for(int i=0; i<N_ELEMENTS; ++i){
		if(fabs(mass-MASS[i])<min){
			min=fabs(mass-MASS[i]);
			an=i+1;
		}
	}
	return an;
}

//************** MASS ***************

double mass(int an){
	return MASS[an-1];
}

//************* RADIUS **************

double radius_covalent(int an){return RADIUS_COVALENT[an-1];}
double radius_vdw(int an){
	if(RADIUS_VDW[an-1]>0.0) return RADIUS_VDW[an-1];
	else return pow(ALPHA[an-1],1.0/3.0)*0.529177210903;
}

//==== Electric ====

double IE(int an){return IONIZATION_ENERGY[an-1];}

double AN(int an){return ELECTRON_AFFINITY[an-1];}

double CHI(int an){
	return 0.5*(IONIZATION_ENERGY[an-1]+ELECTRON_AFFINITY[an-1]);
}

double ETA(int an){
	return IONIZATION_ENERGY[an-1]-ELECTRON_AFFINITY[an-1];
}

double chi_uff(int an){return CHI_UFF[an-1];}

double eta_uff(int an){return ETA_UFF[an-1];}

double alpha(int an){return ALPHA[an-1];}

}

