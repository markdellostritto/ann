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
#include "format/gauss_struc.hpp"

namespace GAUSSIAN{

//*****************************************************
//FORMAT struct
//*****************************************************

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-gauss"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-xdatcar\" option.");
			else format.gauss=strlist[i+1];
		}
	}
	return format;
}

//*****************************************************
//reading
//*****************************************************

void read(const char* xyzfile, const AtomType& atomT, Structure& struc){
	if(GAUSS_PRINT_FUNC>0) std::cout<<"GAUSSIAN::read(const char*,const AtomType&,Structure&):\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* name=new char[string::M];
		Token token;
	//strings
		const char* str_natoms="NAtoms";
		const char* str_posn="Input orientation:";
		const char* str_energy="SCF Done:";
		const char* str_dipole="Electric dipole moment (input orientation):";
		const char* str_alpha="Dipole polarizability, Alpha (input orientation)";
	//atom info
		double pe=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//units
		double s_len=0.0,s_energy=0.0,s_mass=0.0,s_len_mu=0.0;
		if(units::Consts::system()==units::System::LJ){
			s_len=1.0;
			s_energy=1.0;
			s_mass=1.0;
		} else if(units::Consts::system()==units::System::AU){
			s_len=units::Ang2Bohr;
			s_len_mu=1.0;
			s_energy=1.0;
			s_mass=units::MPoME;
		} else if(units::Consts::system()==units::System::METAL){
			s_len=1.0;
			s_len_mu=units::Bohr2Ang;
			s_energy=units::Eh2Ev;
			s_mass=1.0;
		} 
		else throw std::runtime_error("Invalid units.");
		
	//open file
	if(GAUSS_PRINT_STATUS>0) std::cout<<"opening file\n";
	reader=fopen(xyzfile,"r");
	if(reader==NULL) throw std::runtime_error(std::string("ERROR in GAUSSIAN::read(const char*,const AtomType&,Structure&): Could not open file: ")+std::string(xyzfile));
	
	//read the number of atoms
	int nAtoms=0;
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,str_natoms)!=NULL){
			token.read(input,string::WS); token.next();
			nAtoms=std::atoi(token.next().c_str());
		}
	}
	std::rewind(reader);
	
	//read the rest of the data
	while(fgets(input,string::M,reader)!=NULL){
		//positions
		if(std::strstr(input,str_posn)!=NULL){
			//skip 4 lines
			for(int i=0; i<4; i++) fgets(input,string::M,reader);
			//read positions
			std::vector<int> an;
			std::vector<std::string> name;
			std::vector<Eigen::Vector3d> posn;
			for(int i=0; i<nAtoms; ++i){
				token.read(fgets(input,string::M,reader),string::WS); token.next();
				an.push_back(std::atoi(token.next().c_str())); token.next();
				name.push_back(std::string(ptable::name(an.back())));
				posn.push_back(Eigen::Vector3d::Zero());
				posn.back()[0]=std::atof(token.next().c_str())*s_len;
				posn.back()[1]=std::atof(token.next().c_str())*s_len;
				posn.back()[2]=std::atof(token.next().c_str())*s_len;
			}
			//resize structure
			struc.resize(nAtoms,atomT);
			//set data
			if(atomT.an) for(int i=0; i<nAtoms; ++i) struc.an(i)=an[i];
			if(atomT.name) for(int i=0; i<nAtoms; ++i) struc.name(i)=name[i];
			if(atomT.posn) for(int i=0; i<nAtoms; ++i) struc.posn(i)=posn[i];
		}
		
		//energy
		if(std::strstr(input,str_energy)!=NULL){
			token.read(input,string::WS);
			for(int i=0; i<4; i++) token.next();
			struc.pe()=std::atof(token.next().c_str())*s_energy;
		}
		
		//dipole moment
		if(std::strstr(input,str_dipole)!=NULL){
			Token tokenf;
			//skip 3 lines
			for(int i=0; i<3; i++) fgets(input,string::M,reader);
			//read dipole
			token.read(fgets(input,string::M,reader),string::WS); token.next(); tokenf.read(token.next(),"D");
			struc.mutot()[0]=std::atof(tokenf.next().c_str())*s_len_mu*pow(10.0,std::atof(tokenf.next().c_str()));
			token.read(fgets(input,string::M,reader),string::WS); token.next(); tokenf.read(token.next(),"D");
			struc.mutot()[1]=std::atof(tokenf.next().c_str())*s_len_mu*pow(10.0,std::atof(tokenf.next().c_str()));
			token.read(fgets(input,string::M,reader),string::WS); token.next(); tokenf.read(token.next(),"D");
			struc.mutot()[2]=std::atof(tokenf.next().c_str())*s_len_mu*pow(10.0,std::atof(tokenf.next().c_str()));
		}
		
		//polarizability
		if(std::strstr(input,str_alpha)!=NULL){
			Token tokenf;
			//skip 5 lines
			for(int i=0; i<5; i++) fgets(input,string::M,reader);
			//read alpha
			const double slm3=s_len_mu*s_len_mu*s_len_mu;
			token.read(fgets(input,string::M,reader),string::WS); token.next(); tokenf.read(token.next(),"D");
			struc.atot()(0,0)=std::atof(tokenf.next().c_str())*slm3*pow(10.0,std::atof(tokenf.next().c_str()));
			token.read(fgets(input,string::M,reader),string::WS); token.next(); tokenf.read(token.next(),"D");
			struc.atot()(1,0)=std::atof(tokenf.next().c_str())*slm3*pow(10.0,std::atof(tokenf.next().c_str()));
			struc.atot()(0,1)=struc.atot()(1,0);
			token.read(fgets(input,string::M,reader),string::WS); token.next(); tokenf.read(token.next(),"D");
			struc.atot()(1,1)=std::atof(tokenf.next().c_str())*slm3*pow(10.0,std::atof(tokenf.next().c_str()));
			token.read(fgets(input,string::M,reader),string::WS); token.next(); tokenf.read(token.next(),"D");
			struc.atot()(2,0)=std::atof(tokenf.next().c_str())*slm3*pow(10.0,std::atof(tokenf.next().c_str()));
			struc.atot()(0,2)=struc.atot()(2,0);
			token.read(fgets(input,string::M,reader),string::WS); token.next(); tokenf.read(token.next(),"D");
			struc.atot()(2,1)=std::atof(tokenf.next().c_str())*slm3*pow(10.0,std::atof(tokenf.next().c_str()));
			struc.atot()(1,2)=struc.atot()(2,1);
			token.read(fgets(input,string::M,reader),string::WS); token.next(); tokenf.read(token.next(),"D");
			struc.atot()(2,2)=std::atof(tokenf.next().c_str())*slm3*pow(10.0,std::atof(tokenf.next().c_str()));
		}
	}
	
	//close file
	fclose(reader);
	reader=NULL;
	
	//free memory
	delete[] input;
	delete[] name;
}

//*****************************************************
//writing
//*****************************************************

void write(const char* file, const AtomType& atomT, const Structure& struc){
	if(GAUSS_PRINT_FUNC>0) std::cout<<"GAUSSIAN::write(const char*,const AtomType&,const Structure&):\n";
	throw std::runtime_error("NOT YET IMPLEMENTED");
}

	
}