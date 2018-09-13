#include "ptable.hpp"

namespace PTable{
	
//*********************************************
//Function
//*********************************************

//name
const char* name(unsigned int an){return ELEMENT_NAME[an-1];}
//period
unsigned int period(unsigned int an){return PERIOD[an-1];}
//atomic number 
int an(const char* name){
	for(unsigned int i=0; i<NUM_ELEMENTS; i++){
		if(std::strcmp(name,ELEMENT_NAME[i])==0) return i+1;
	}
	return -1;
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
//********** MASS **********
//mass 
double mass(unsigned int an){return ELEMENT_MASS[an-1];}
//********** RADIUS **********
//covalent radius
double atomicRadius(unsigned int an){return ATOMIC_RADII[an-1];};
//covalent radius
double covalentRadius(unsigned int an){return COVALENT_RADII[an-1];};
//********** IONIZATION_ENERGY **********
//ionization energy
double ionizationEnergy(unsigned int an){return IONIZATION_ENERGY[an-1];}
//ionization energy - Hinze
double ionizationEnergyHinze(unsigned int an, unsigned int vs=0){
	if(an==1) return 13.5999;
	else if(an==2) return 54.402;
	else if(an==3) return 5.392;
	else if(an==4) return 8.552;
	else if(an==5) return 11.25;
	//else if(an==6 && vs==1) return 11.036;
	//else if(an==6 && vs==2) return 17.277;
	//else if(an==6 && vs==3) return 15.477;
	//else if(an==6 && vs==4) return 14.423;
	else if(an==6) return 14.423;
	else if(an==7) return 16.974;
	else if(an==8) return 20.166;
	else if(an==9) return 22.853;
	else if(an==10) return 44.274;
	else if(an==11) return 5.144;
	else if(an==12) return 7.029;
	else if(an==13) return 8.860;
	else if(an==14) return 11.384;
	else if(an==15) return 13.431;
	else if(an==16) return 14.049;
	else if(an==17) return 15.921;
	else if(an==18) return 29.357;
	else if(an==19) return 4.341;
	else if(an==20) return 5.747;
	else if(an==29) return 7.726;
	else if(an==30) return 8.254;
	else if(an==31) return 9.726;
	else if(an==32) return 11.544;
	else if(an==33) return 12.35;
	else if(an==34) return 12.831;
	else if(an==35) return 14.653;
	else if(an==36) return 26.376;
	else if(an==37) return 4.176;
	else if(an==38) return 5.369;
	else if(an==47) return 7.576;
	else if(an==48) return 7.790;
	else if(an==49) return 9.073;
	else if(an==50) return 10.346;
	else if(an==51) return 11.983;
	else if(an==52) return 11.483;
	else if(an==53) return 13.197;
	else if(an==54) return 26.376;
	else return 0;
}
//********** ELECTRON_AFFINITY **********
//electron affinity
double electronAffinity(unsigned int an){return ELECTRON_AFFINITY[an-1];}
//electron affinity - Hinze
double electronAffinityHinze(unsigned int an, unsigned int vs=0){
	if(an==1) return 0.755;
	else if(an==2) return 24.578;
	else if(an==3) return 0.620;
	else if(an==4) return 1.108;
	else if(an==5) return 1.130;
	//else if(an==6 && vs==1) return 0.4;
	//else if(an==6 && vs==2) return 3.677;
	//else if(an==6 && vs==3) return 2.315;
	//else if(an==6 && vs==4) return 1.719;
	else if(an==6) return 1.719;
	else if(an==7) return 2.39;
	else if(an==8) return 3.644;
	else if(an==9) return 4.469;
	else if(an==10) return 21.588;
	else if(an==11) return 0.560;
	else if(an==12) return 0.929;
	else if(an==13) return 1.806;
	else if(an==14) return 2.580;
	else if(an==15) return 2.785;
	else if(an==16) return 3.377;
	else if(an==17) return 4.183;
	else if(an==18) return 15.755;
	else if(an==19) return 0.501;
	else if(an==20) return 0.997;
	else if(an==29) return 1.226;
	else if(an==30) return 1.407;
	else if(an==31) return 1.926;
	else if(an==32) return 2.564;
	else if(an==33) return 2.534;
	else if(an==34) return 3.401;
	else if(an==35) return 4.057;
	else if(an==36) return 13.996;
	else if(an==37) return 0.484;
	else if(an==38) return 0.979;
	else if(an==47) return 1.303;
	else if(an==48) return 1.091;
	else if(an==49) return 1.707;
	else if(an==50) return 3.680;
	else if(an==51) return 2.217;
	else if(an==52) return 3.607;
	else if(an==53) return 3.741;
	else if(an==54) return 12.128;
	else return 0;
}
//********** ELECTRONEGATIVITY **********
//electronegativity - Mulliken
double electronegativityMulliken(unsigned int an){return 0.5*(IONIZATION_ENERGY[an-1]+ELECTRON_AFFINITY[an-1]);}
//electronegativity - Pauling
double electronegativityPauling(unsigned int an){return ELECTRONEGATIVITY_PAULING[an-1];}
//electronegativity - Allen
double electronegativityAllen(unsigned int an){return ELECTRONEGATIVITY_ALLEN[an-1];}
//electronegativity - Hinze
double electronegativityHinze(unsigned int an, unsigned int vs=0){return 0.5*(ionizationEnergyHinze(an,vs)+electronAffinityHinze(an,vs));}
//electronegativity - type
ElectronegativityType::type ElectronegativityType::load(const char* str){
	if(std::strcmp(str,"MULLIKEN")==0) return ElectronegativityType::MULLIKEN;
	else if(std::strcmp(str,"PAULING")==0) return ElectronegativityType::PAULING;
	else if(std::strcmp(str,"ALLEN")==0) return ElectronegativityType::ALLEN;
	else if(std::strcmp(str,"HINZE")==0) return ElectronegativityType::HINZE;
	else return ElectronegativityType::UNKNOWN;
}
std::ostream& operator<<(std::ostream& out, const ElectronegativityType::type& t){
	if(t==ElectronegativityType::MULLIKEN) out<<"MULLIKEN";
	else if(t==ElectronegativityType::PAULING) out<<"PAULING";
	else if(t==ElectronegativityType::ALLEN) out<<"ALLEN";
	else if(t==ElectronegativityType::HINZE) out<<"HINZE";
	return out;
}
//********** HARDNESS **********
//hardness
double hardness(unsigned int an){return 0.5*(IONIZATION_ENERGY[an-1]-ELECTRON_AFFINITY[an-1]);}
//********** IDEMPOTENTIAL **********
//idempotential (coulomb self-energy)
double idempotential(unsigned int an){return IONIZATION_ENERGY[an-1]-ELECTRON_AFFINITY[an-1];}
//idempotential Hinze(coulomb self-energy)
double idempotentialHinze(unsigned int an, unsigned int vs=0){return ionizationEnergyHinze(an,vs)+electronAffinityHinze(an,vs);}
//shell configuration
std::vector<unsigned int>& shell(unsigned int an, std::vector<unsigned int>& c){
	switch (an){
		case 1:{unsigned int tmp[]={1}; c=std::vector<unsigned int>(tmp,tmp+1); break;}
		case 2:{unsigned int tmp[]={2}; c=std::vector<unsigned int>(tmp,tmp+1); break;}
		case 3:{unsigned int tmp[]={2,1}; c=std::vector<unsigned int>(tmp,tmp+2); break;}
		case 4:{unsigned int tmp[]={2,2}; c=std::vector<unsigned int>(tmp,tmp+2); break;}
		case 5:{unsigned int tmp[]={2,3}; c=std::vector<unsigned int>(tmp,tmp+2); break;}
		case 6:{unsigned int tmp[]={2,4}; c=std::vector<unsigned int>(tmp,tmp+2); break;}
		case 7:{unsigned int tmp[]={2,5}; c=std::vector<unsigned int>(tmp,tmp+2); break;}
		case 8:{unsigned int tmp[]={2,6}; c=std::vector<unsigned int>(tmp,tmp+2); break;}
		case 9:{unsigned int tmp[]={2,7}; c=std::vector<unsigned int>(tmp,tmp+2); break;}
		case 10:{unsigned int tmp[]={2,8}; c=std::vector<unsigned int>(tmp,tmp+2); break;}
		case 11:{unsigned int tmp[]={2,8,1}; c=std::vector<unsigned int>(tmp,tmp+3); break;}
		case 12:{unsigned int tmp[]={2,8,2}; c=std::vector<unsigned int>(tmp,tmp+3); break;}
		case 13:{unsigned int tmp[]={2,8,3}; c=std::vector<unsigned int>(tmp,tmp+3); break;}
		case 14:{unsigned int tmp[]={2,8,4}; c=std::vector<unsigned int>(tmp,tmp+3); break;}
		case 15:{unsigned int tmp[]={2,8,5}; c=std::vector<unsigned int>(tmp,tmp+3); break;}
		case 16:{unsigned int tmp[]={2,8,6}; c=std::vector<unsigned int>(tmp,tmp+3); break;}
		case 17:{unsigned int tmp[]={2,8,7}; c=std::vector<unsigned int>(tmp,tmp+3); break;}
		case 18:{unsigned int tmp[]={2,8,8}; c=std::vector<unsigned int>(tmp,tmp+3); break;}
		case 19:{unsigned int tmp[]={2,8,8,1}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 20:{unsigned int tmp[]={2,8,8,2}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 21:{unsigned int tmp[]={2,8,9,2}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 22:{unsigned int tmp[]={2,8,10,2}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 23:{unsigned int tmp[]={2,8,11,2}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 24:{unsigned int tmp[]={2,8,13,1}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 25:{unsigned int tmp[]={2,8,13,2}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 26:{unsigned int tmp[]={2,8,14,2}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 27:{unsigned int tmp[]={2,8,15,2}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 28:{unsigned int tmp[]={2,8,16,2}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 29:{unsigned int tmp[]={2,8,18,1}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 30:{unsigned int tmp[]={2,8,18,2}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 31:{unsigned int tmp[]={2,8,18,3}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 32:{unsigned int tmp[]={2,8,18,4}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 33:{unsigned int tmp[]={2,8,18,5}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 34:{unsigned int tmp[]={2,8,18,6}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 35:{unsigned int tmp[]={2,8,18,7}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 36:{unsigned int tmp[]={2,8,18,8}; c=std::vector<unsigned int>(tmp,tmp+4); break;}
		case 37:{unsigned int tmp[]={2,8,18,8,1}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 38:{unsigned int tmp[]={2,8,18,8,2}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 39:{unsigned int tmp[]={2,8,18,9,2}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 40:{unsigned int tmp[]={2,8,18,10,2}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 41:{unsigned int tmp[]={2,8,18,12,1}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 42:{unsigned int tmp[]={2,8,18,13,1}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 43:{unsigned int tmp[]={2,8,18,13,2}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 44:{unsigned int tmp[]={2,8,18,15,1}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 45:{unsigned int tmp[]={2,8,18,16,1}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 46:{unsigned int tmp[]={2,8,18,18,0}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 47:{unsigned int tmp[]={2,8,18,18,1}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 48:{unsigned int tmp[]={2,8,18,18,2}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 49:{unsigned int tmp[]={2,8,18,18,3}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 50:{unsigned int tmp[]={2,8,18,18,4}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 51:{unsigned int tmp[]={2,8,18,18,5}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 52:{unsigned int tmp[]={2,8,18,18,6}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 53:{unsigned int tmp[]={2,8,18,18,7}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 54:{unsigned int tmp[]={2,8,18,18,8}; c=std::vector<unsigned int>(tmp,tmp+5); break;}
		case 55:{unsigned int tmp[]={2,8,18,18,8,1}; c=std::vector<unsigned int>(tmp,tmp+6); break;}
		default:{break;}
	}
	return c;
}
//********** Z_EFF **********
//zEff - outer shell
double zEff(unsigned int an){
	std::vector<unsigned int> c;
	shell(an,c);
	double z=((double)an);
	if(c.size()==1) z-=0.30*(c.back()-1.0);
	else {z-=0.35*(c.back()-1.0);
		if(c.size()>1) z-=0.85*c[c.size()-1-1];
		for(int i=c.size()-1-2; i>=0; i--) z-=1.0*c[i];
	}
	return z;
}
//zEff

double zEff(unsigned int an, unsigned int shellv){
	double z=an;
	std::vector<unsigned int> c;
	shell(an,c);
	if(shellv==1) z-=0.30*(c.back()-1.0);
	else {
		z-=0.35*(c[shellv-1]-1.0);
		if(shellv>1) z-=0.85*c[shellv-1-1];
		for(int i=shellv-1-2; i>=0; --i) z-=1.0*c[i];
	}
	return z;
}
double zSlaterC(unsigned int an){
	return zEff(an)/(2.0*period(an)-1.0);
}
//cm5 parameters
double cm5(unsigned int an){return CM5[an-1];};

}