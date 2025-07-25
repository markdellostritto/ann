// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <string>
#include <stdexcept>
#include <iostream>
// ann - structure
#include "struc/structure.hpp"
// ann - strings
#include "str/string.hpp"
// ann - chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// ann - format
#include "format/raw_struc.hpp"

namespace RAW{

//*****************************************************
//reading
//*****************************************************

void read(const char* xyzfile, const AtomType& atomT, Structure& struc){
	if(RAW_PRINT_FUNC>0) std::cout<<"XYZ::read(const char*,const AtomType&,Structure&):\n";
}

//*****************************************************
//writing
//*****************************************************

void write(const char* file, const AtomType& atomT, const Structure& struc){
	if(RAW_PRINT_FUNC>0) std::cout<<"write(const char*,const AtomType&,const Structure&):\n";
	FILE* writer=NULL;
	
	//coord
	if(RAW_PRINT_STATUS>0) std::cout<<"writing coord\n";
	writer=fopen("coord.raw","w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file: coord.raw");
	for(int i=0; i<struc.nAtoms(); ++i){
		fprintf(writer,"%17.8f %17.8f %17.8f ",struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2]);
	}
	fprintf(writer,"\n");
	fclose(writer); writer=NULL;
	
	//force
	if(RAW_PRINT_STATUS>0) std::cout<<"writing force\n";
	writer=fopen("force.raw","w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file: force.raw");
	for(int i=0; i<struc.nAtoms(); ++i){
		fprintf(writer,"%17.8f %17.8f %17.8f ",struc.force(i)[0],struc.force(i)[1],struc.force(i)[2]);
	}
	fprintf(writer,"\n");
	fclose(writer); writer=NULL;
	
	//energy
	if(RAW_PRINT_STATUS>0) std::cout<<"writing energy\n";
	writer=fopen("energy.raw","w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file: energy.raw");
	fprintf(writer,"%19.10f\n",struc.pe());
	fclose(writer); writer=NULL;
	
	//box
	if(RAW_PRINT_STATUS>0) std::cout<<"writing box\n";
	writer=fopen("box.raw","w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file: box.raw");
	for(int i=0; i<3; i++){
		for(int j=0; j<3; ++j){
			fprintf(writer,"%19.10f ",struc.R()(j,i));
		}
	}
	fprintf(writer,"\n");
	fclose(writer); writer=NULL;
	
	//type
	if(RAW_PRINT_STATUS>0) std::cout<<"writing type\n";
	writer=fopen("type.raw","w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file: type.raw");
	for(int i=0; i<struc.nAtoms(); ++i){
		fprintf(writer,"%i ",struc.type(i));
	}
	fprintf(writer,"\n");
	fclose(writer); writer=NULL;
}

	
}