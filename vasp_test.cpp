#include <iostream>
#include <cstdlib>
#include "atom.hpp"
#include "structure.hpp"
#include "vasp.hpp"
#include "serialize.hpp"
int main(int argc, char* argv[]){
	
	bool test_xml=true;
	
	if(test_xml){
	std::cout<<"***********************************************\n";
	std::cout<<"***************** TEST - XML *****************\n";
	try{
		// local variables
		typedef Atom<Name,AN,Index,Species,Position,Velocity,Force> AtomT;
		Structure<AtomT> struc;
		
		//load xml file
		std::cout<<"Loading XML file...\n";
		VASP::XML::load("Si.xml",struc);
		
		//print the simulation
		std::cout<<struc<<"\n";
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			std::cout<<"\t"<<struc.atom(n).name()<<struc.atom(n).index()+1<<" "<<struc.atom(n).posn().transpose()<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - XML\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"***************** TEST - XML *****************\n";
	std::cout<<"***********************************************\n";
	}
}