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
#include "format/xyz_struc.hpp"

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
		Token token;
	//atom info
		int nAtoms=0;
		double pe=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//units
		double s_len=0.0,s_energy=0.0,s_mass=0.0;
		if(units::Consts::system()==units::System::LJ){
			s_len=1.0;
			s_energy=1.0;
			s_mass=1.0;
		} else if(units::Consts::system()==units::System::AU){
			s_len=units::Ang2Bohr;
			s_energy=units::Ev2Eh;
			s_mass=units::MPoME;
		} else if(units::Consts::system()==units::System::METAL){
			s_len=1.0;
			s_energy=1.0;
			s_mass=1.0;
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
	
	//read header
	int ni=0;
	Eigen::Vector3i ri; ri<<1,2,3;
	Eigen::Vector3i fi; fi<<-1,-1,-1;
	fgets(input,string::M,reader);
	string::to_upper(input);
	if(std::strstr(input,"PROPERTIES")!=NULL){
		token.read(std::strstr(input,"PROPERTIES")," \r\t\n="); token.next();
		const std::string propstr=token.next();
		Token proptok=Token(propstr,":");
		int index=0;
		while(!proptok.end()){
			const std::string tag=proptok.next();
			if(tag=="SPECIES"){
				if(proptok.next()!="S") throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid name data type.");
				else if(std::atoi(proptok.next().c_str())!=1) throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid name length.");
				else {
					ni=index++;
				}
			} else if(tag=="POS"){
				if(proptok.next()!="R") throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid position data type.");
				else if(std::atoi(proptok.next().c_str())!=3) throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid position length.");
				else {
					ri[0]=index++;
					ri[1]=index++;
					ri[2]=index++;
				}
			} else if(tag=="FORCES"){
				if(proptok.next()!="R") throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid force data type.");
				else if(std::atoi(proptok.next().c_str())!=3) throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid force length.");
				else {
					fi[0]=index++;
					fi[1]=index++;
					fi[2]=index++;
				}
			}
		}
	}
	if(std::strstr(input,"POTENTIAL_ENERGY")!=NULL){
		token.read(std::strstr(input,"POTENTIAL_ENERGY")," \r\t\n="); token.next();
		pe=std::atof(token.next().c_str());
	}
	if(std::strstr(input,"LATTICE")!=NULL){
		token.read(std::strstr(input,"LATTICE")," \r\t\n=\"");
		token.next();
		lv(0,0)=std::atof(token.next().c_str());
		lv(1,0)=std::atof(token.next().c_str());
		lv(2,0)=std::atof(token.next().c_str());
		lv(0,1)=std::atof(token.next().c_str());
		lv(1,1)=std::atof(token.next().c_str());
		lv(2,1)=std::atof(token.next().c_str());
		lv(0,2)=std::atof(token.next().c_str());
		lv(1,2)=std::atof(token.next().c_str());
		lv(2,2)=std::atof(token.next().c_str());
		lv*=s_len;
	}
	
	//resize the structure
	if(XYZ_PRINT_STATUS>0) std::cout<<"resizing structure\n";
	struc.resize(nAtoms,atomT);
	
	//read in names and positions
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading names and posns\n";
	if(atomT.force && fi.minCoeff()>0){
		Eigen::Vector3d posn;
		Eigen::Vector3d force;
		for(int i=0; i<nAtoms; ++i){
			token.read(fgets(input,string::M,reader),string::WS);
			const std::string aname=token.next();
			posn[0]=std::atof(token.next().c_str());
			posn[1]=std::atof(token.next().c_str());
			posn[2]=std::atof(token.next().c_str());
			force[0]=std::atof(token.next().c_str());
			force[1]=std::atof(token.next().c_str());
			force[2]=std::atof(token.next().c_str());
			if(struc.atomType().name) struc.name(i)=aname;
			if(struc.atomType().posn) struc.posn(i).noalias()=posn*s_len;
			if(struc.atomType().force) struc.force(i).noalias()=force*s_energy/s_len;
		}
	} else {
		Eigen::Vector3d posn;
		for(int i=0; i<nAtoms; ++i){
			token.read(fgets(input,string::M,reader),string::WS);
			const std::string aname=token.next();
			posn[0]=std::atof(token.next().c_str());
			posn[1]=std::atof(token.next().c_str());
			posn[2]=std::atof(token.next().c_str());
			if(struc.atomType().name) struc.name(i)=aname;
			if(struc.atomType().posn) struc.posn(i).noalias()=posn*s_len;
		}
	}
	
	//close the file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(reader);
	reader=NULL;
	
	//set the cell
	if(lv.norm()>0){
		if(XYZ_PRINT_STATUS>0) std::cout<<"setting cell\n";
		static_cast<Cell&>(struc).init(lv);
		for(int i=0; i<nAtoms; ++i){
			Cell::returnToCell(struc.posn(i),struc.posn(i),struc.R(),struc.RInv());
		}
	}
	
	//set the energy
	if(XYZ_PRINT_STATUS>0) std::cout<<"setting energy\n";
	struc.pe()=s_energy*pe;
	
	//set an
	if(atomT.an && atomT.name){
		for(int i=0; i<nAtoms; ++i){
			struc.an(i)=ptable::an(struc.name(i).c_str());
		}
	}
	
	//set mass
	if(atomT.an && atomT.mass){
		for(int i=0; i<nAtoms; ++i){
			struc.mass(i)=ptable::mass(struc.an(i))*s_mass;
		}
	} else if(atomT.name && atomT.mass){
		for(int i=0; i<nAtoms; ++i){
			const int an=ptable::an(struc.name(i).c_str());
			struc.mass(i)=ptable::mass(an)*s_mass;
		}
	}
	
	//set radius
	if(atomT.an && atomT.radius){
		for(int i=0; i<nAtoms; ++i){
			struc.radius(i)=ptable::radius_covalent(struc.an(i));
		}
	} else if(atomT.name && atomT.radius){
		for(int i=0; i<nAtoms; ++i){
			const int an=ptable::an(struc.name(i).c_str());
			struc.radius(i)=ptable::radius_covalent(an);
		}
	}
	
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
	
	//open file
	if(XYZ_PRINT_STATUS>0) std::cout<<"opening file\n";
	writer=fopen(file,"w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file: \""+std::string(file)+"\"");
	
	//write xyz
	if(XYZ_PRINT_STATUS>0) std::cout<<"writing structure\n";
	fprintf(writer,"%i\n",struc.nAtoms());
	const Eigen::Matrix3d& R=struc.R();
	if(atomT.force){
		fprintf(writer,"Properties=species:S:1:pos:R:3:forces:R:3 potential_energy=%f pbc=\"T T T\" Lattice=\"%f %f %f %f %f %f %f %f %f\"\n",
			struc.pe(),R(0,0),R(1,0),R(2,0),R(0,1),R(1,1),R(1,2),R(0,2),R(1,2),R(2,2)
		);
		for(int i=0; i<struc.nAtoms(); ++i){
			fprintf(writer,"%-2s %19.10f %19.10f %19.10f %19.10f %19.10f %19.10f\n",struc.name(i).c_str(),
				struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2],
				struc.force(i)[0],struc.force(i)[1],struc.force(i)[2]
			);
		}
	} else {
		fprintf(writer,"Properties=species:S:1:pos:R:3 potential_energy=%f pbc=\"T T T\" Lattice=\"%f %f %f %f %f %f %f %f %f\"\n",
			struc.pe(),R(0,0),R(1,0),R(2,0),R(0,1),R(1,1),R(1,2),R(0,2),R(1,2),R(2,2)
		);
		for(int i=0; i<struc.nAtoms(); ++i){
			fprintf(writer,"%-2s %19.10f %19.10f %19.10f\n",struc.name(i).c_str(),
				struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2]
			);
		}
	}
	
	//close file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(writer);
	writer=NULL;
}

	
}