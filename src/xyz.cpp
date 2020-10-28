// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <string>
#include <stdexcept>
#include <iostream>
// ann - math
#include "math_func.hpp"
// ann - strings
#include "string.hpp"
// ann - units
#include "units.hpp"
// ann - ptable
#include "ptable.hpp"
// ann - structure
#include "structure.hpp"
// ann - xyz
#include "xyz.hpp"

namespace XYZ{

//*****************************************************
//FORMAT struct
//*****************************************************

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-xyz"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-xdatcar\" option.");
			else format.xyz=strlist[i+1];
		}
	}
	return format;
}

void unwrap(Structure& struc){
	if(XYZ_PRINT_FUNC>0) std::cout<<"unwrap(Structure&):\n";
	Eigen::Vector3d drp,dr;
	std::vector<double> rcov(struc.nAtoms());
	
	if(XYZ_PRINT_STATUS>0) std::cout<<"finding covalent radii\n";
	for(int i=0; i<struc.nAtoms(); ++i) rcov[i]=ptable::radius_cov(ptable::an(struc.name(i).c_str()));
	for(int i=0; i<struc.nAtoms(); ++i) if(rcov[i]<=0) throw std::runtime_error("No covalent radius.");
	
	if(XYZ_PRINT_STATUS>0) std::cout<<"unwrapping coordinates\n";
	for(int i=0; i<struc.nAtoms(); ++i){
		const double RRI=ptable::radius_cov(ptable::an(struc.name(i).c_str()));
		for(int j=i+1; j<struc.nAtoms(); ++j){
			if(struc.name(j)=="H"){
				const double RRJ=ptable::radius_cov(ptable::an(struc.name(j).c_str()));
				const double R0=RRI+RRJ;
				Cell::diff(struc.posn(i),struc.posn(j),drp,struc.R(),struc.RInv());
				dr=struc.posn(i)-struc.posn(j);
				if(drp.norm()<R0 && dr.norm()>R0){
					if(std::fabs(drp[0])<R0 && std::fabs(dr[0])>R0) struc.posn(j)[0]+=math::func::sign(struc.posn(i)[0]-struc.posn(j)[0])*struc.R()(0,0);
					if(std::fabs(drp[1])<R0 && std::fabs(dr[1])>R0) struc.posn(j)[1]+=math::func::sign(struc.posn(i)[1]-struc.posn(j)[1])*struc.R()(1,1);
					if(std::fabs(drp[2])<R0 && std::fabs(dr[2])>R0) struc.posn(j)[2]+=math::func::sign(struc.posn(i)[2]-struc.posn(j)[2])*struc.R()(2,2);
				}
			}
		}
	}
}

//*****************************************************
//reading
//*****************************************************

void read(const char* xyzfile, const AtomType& atomT, Structure& struc){
	if(XYZ_PRINT_FUNC>0) std::cout<<"XYZ::read(const char*,const AtomType&,Structure&):\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* name=new char[string::M];
		std::vector<std::string> strlist;
	//atom info
		int nAtoms=0;
		Eigen::Vector3d posn;
		double energy=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//units
		double s_len=0.0,s_energy=0.0;
		if(units::consts::system()==units::System::AU){
			s_len=units::BOHRpANG;
			s_energy=units::HARTREEpEV;
			
		} else if(units::consts::system()==units::System::METAL){
			s_len=1.0;
			s_energy=1.0;
		}
		else throw std::runtime_error("Invalid units.");
		
	//open file
	if(XYZ_PRINT_STATUS>0) std::cout<<"opening file\n";
	reader=fopen(xyzfile,"r");
	if(reader==NULL) throw std::runtime_error(std::string("ERROR in XYZ::read(const char*,const AtomType&,Structure&): Could not open file: ")+std::string(xyzfile));
	
	//read natoms
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading natoms\n";
	fgets(input,string::M,reader);
	nAtoms=std::atoi(input);
	if(nAtoms<=0) throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): found zero atoms.");
	
	//read in cell length and energy
	fgets(input,string::M,reader);
	string::split(input,string::WS,strlist);
	if(strlist.size()==1+1){
		energy=std::atof(strlist.at(1).c_str());
	} else if(strlist.size()==1+1+3){
		energy=std::atof(strlist.at(1).c_str());
		lv(0,0)=std::atof(strlist.at(2).c_str());
		lv(1,1)=std::atof(strlist.at(3).c_str());
		lv(2,2)=std::atof(strlist.at(4).c_str());
	}
	
	//resize the structure
	if(XYZ_PRINT_STATUS>0) std::cout<<"resizing structure\n";
	struc.resize(nAtoms,atomT);
	
	//read in names and positions
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading names and posns\n";
	for(int i=0; i<nAtoms; ++i){
		fgets(input,string::M,reader);
		std::sscanf(input,"%s %lf %lf %lf",name,&posn[0],&posn[1],&posn[2]);
		if(struc.atomType().name) struc.name(i)=name;
		if(struc.atomType().posn) struc.posn(i).noalias()=posn*s_len;
	}
	
	//set the cell
	if(XYZ_PRINT_STATUS>0) std::cout<<"setting cell\n";
	if(lv.norm()>0) static_cast<Cell&>(struc).init(lv);
	
	//set the energy
	if(XYZ_PRINT_STATUS>0) std::cout<<"setting energy\n";
	struc.energy()=s_energy*energy;
	
	//close the file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
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
	if(XYZ_PRINT_FUNC>0) std::cout<<"write(const char*,const AtomType&,const Structure&):\n";
	FILE* writer=NULL;
	
	//unwrap structure
	if(XYZ_PRINT_STATUS>0) std::cout<<"unwrapping structure\n";
	Structure strucTemp=struc;
	//unwrap(strucTemp);
	
	//open file
	if(XYZ_PRINT_STATUS>0) std::cout<<"opening file\n";
	writer=fopen(file,"w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file: \""+std::string(file)+"\"");
	
	//write xyz
	if(XYZ_PRINT_STATUS>0) std::cout<<"writing structure\n";
	fprintf(writer,"%i\n",strucTemp.nAtoms());
	fprintf(writer,"SIMULATION %f\n",strucTemp.energy());
	for(int i=0; i<strucTemp.nAtoms(); ++i){
		fprintf(writer,"%s %f %f %f\n",strucTemp.name(i).c_str(),
			strucTemp.posn(i)[0],strucTemp.posn(i)[1],strucTemp.posn(i)[2]
		);
	}
	
	//close file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(writer);
	writer=NULL;
}

	
}