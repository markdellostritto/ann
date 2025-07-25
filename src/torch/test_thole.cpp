// c++ libraries
#include <iostream>
// structure
#include "struc/structure.hpp"
// chem
#include "chem/units.hpp"
// torch
#include "torch/thole.hpp"

int main(int argc, char* argv[]){
	//units
		units::System unitsys=units::System::UNKNOWN;
	//atom format
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=false; atomT.alpha=true; atomT.radius=true;
	//structure
		Structure struc;
	//Thole
		Thole thole;
	
	//set the unit system
	
	units::Consts::init(units::System::AU);
	//make the structure
	const int nAtoms=3;
	struc.resize(nAtoms,atomT);
	struc.name(0)="O";
	struc.name(1)="H";
	struc.name(2)="H";
	struc.posn(0)<<0.0,0,0,0.116452;
	struc.posn(1)<<0.0, 0.759929,-0.465806;
	struc.posn(2)<<0.0,-0.759929,-0.465806;
	struc.alpha(0)=5.7494;
	struc.alpha(1)=2.7927;
	struc.alpha(2)=2.7927;
	struc.radius(0)=0.84;
	struc.radius(1)=0.42;
	struc.radius(2)=0.42;
	for(int i=0;i<nAtoms;++i) struc.posn(i)/=0.529;
	std::cout<<struc<<"\n";
	
	//compute thole
	thole.resize(nAtoms);
	struc.atot()=thole.compute(struc);
	std::cout<<"alpha-tot = \n"<<struc.atot()<<"\n";
	
	return 0;
}