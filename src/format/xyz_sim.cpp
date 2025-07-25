// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <string>
#include <stdexcept>
#include <iostream>
// str
#include "str/string.hpp"
// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// format
#include "format/xyz_sim.hpp"

namespace XYZ{

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	if(XYZ_PRINT_FUNC>0) std::cout<<"XYZ::read(const char*,Interval&,const AtomType&,Simulation&):\n";
	const int M=1000;
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[M];
		char* name=new char[M];
		Token token;
	//atom info	
		int nAtoms=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//positions
		Eigen::Vector3d r;
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
	reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Runtime Error: Could not open file.");
	
	//read natoms
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading natoms\n";
	fgets(input,M,reader);
	nAtoms=std::atoi(input);
	if(nAtoms<=0) throw std::runtime_error("Runtime Error: found zero atoms.");
	if(XYZ_PRINT_DATA>0) std::cout<<"natoms = "<<nAtoms<<"\n";
	
	//find the total number of timesteps
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading timesteps\n";
	std::rewind(reader);
	int nlines=0;
	while(fgets(input,M,reader)) ++nlines;
	int ts=nlines/(nAtoms+2);//natoms + natoms-line + comment-line
	if(XYZ_PRINT_DATA>0) std::cout<<"ts = "<<ts<<"\n";
	
	//set the interval
	if(XYZ_PRINT_STATUS>0) std::cout<<"setting interval\n";
	const int ibeg=Interval::index(interval.beg(),ts);
	const int iend=Interval::index(interval.end(),ts);
	const int len=iend-ibeg+1;
	const int nsteps=len/interval.stride();
	
	//resize the simulation
	if(XYZ_PRINT_STATUS>0) std::cout<<"resizing simulation\n";
	sim.resize(nsteps,nAtoms,atomT);
	
	//read the simulation
	if(XYZ_PRINT_STATUS>0) std::cout<<"reading simulation\n";
	std::rewind(reader);
	for(int t=0; t<ibeg; ++t){
		fgets(input,M,reader);//natoms
		const int N=std::atoi(input);
		fgets(input,M,reader);//comment line
		for(int n=0; n<N; ++n){
			fgets(input,M,reader);
		}
	}
	for(int t=0; t<sim.timesteps(); ++t){
		//read natoms
		fgets(input,M,reader);//natoms
		const int N=std::atoi(input);
		if(sim.frame(t).nAtoms()!=N) throw std::invalid_argument("Error in XYZ::read(const char*,Interval&,const AtomType&,Simulation&): Invalid number of atoms.");
		//read header
		double pe=0;
		int ni=0;
		Eigen::Vector3i ri; ri<<1,2,3;
		Eigen::Vector3i fi; fi<<-1,-1,-1;
		fgets(input,M,reader);
		string::to_upper(input);
		if(std::strstr(input,"PROPERTIES")!=NULL){
			token.read(std::strstr(input,"PROPERTIES")," \r\t\n=");
			const std::string propstr=token.next();
			Token proptok=Token(propstr,":");
			int index=0;
			while(!proptok.end()){
				const std::string tag=proptok.next();
				if(tag=="SPECIES"){
					if(token.next()!="S") throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid name data type.");
					else if(std::atoi(token.next().c_str())!=1) throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid name length.");
					else {
						ni=index++;
					}
				} else if(tag=="POS"){
					if(token.next()!="R") throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid position data type.");
					else if(std::atoi(token.next().c_str())!=3) throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid position length.");
					else {
						ri[0]=index++;
						ri[1]=index++;
						ri[2]=index++;
					}
				} else if(tag=="FORCES"){
					if(token.next()!="R") throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid force data type.");
					else if(std::atoi(token.next().c_str())!=3) throw std::runtime_error("ERROR in XYZ::read(const char*,const AtomType&,Structure&): invalid force length.");
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
		sim.frame(t).pe()=pe*s_energy;
		//read positions
		for(int n=0; n<N; ++n){
			fgets(input,M,reader);
			std::sscanf(input,"%s %lf %lf %lf",name,&r[0],&r[1],&r[2]);
			sim.frame(t).posn(n).noalias()=s_len*r;
			if(atomT.name) sim.frame(t).name(n)=name;
		}
		//set the cell
		if(XYZ_PRINT_STATUS>0) std::cout<<"setting cell\n";
		if(lv.norm()>0) static_cast<Cell&>(sim.frame(t)).init(lv);
		//skip "stride-1" steps
		for(int tt=0; tt<interval.stride()-1; ++tt){
			fgets(input,M,reader);//natoms
			const int NN=std::atoi(input);
			fgets(input,M,reader);//comment line
			for(int n=0; n<NN; ++n) fgets(input,M,reader);
		}
	}
	
	//set an
	if(atomT.name && atomT.an){
		for(int t=0; t<sim.timesteps(); ++t){
			Structure& struc=sim.frame(t);
			for(int i=0; i<nAtoms; ++i){
				struc.an(i)=ptable::an(struc.name(i).c_str());
			}
		}
	}
	
	//set mass
	if(atomT.mass){
		if(atomT.an){
			for(int t=0; t<sim.timesteps(); ++t){
				Structure& struc=sim.frame(t);
				for(int i=0; i<nAtoms; ++i){
					struc.mass(i)=ptable::mass(struc.an(i))*s_mass;
				}
			}
		} else if(atomT.name){
			for(int t=0; t<sim.timesteps(); ++t){
				Structure& struc=sim.frame(t);
				for(int i=0; i<nAtoms; ++i){
					const int an=ptable::an(struc.name(i).c_str());
					struc.mass(i)=ptable::mass(an)*s_mass;
				}
			}
		}
	}
	
	//close file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(reader);
	reader=NULL;
	
	//free memory
	delete[] input;
	delete[] name;
}

void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim){
	if(XYZ_PRINT_FUNC>0) std::cout<<"write(const char*,Interval&,const AtomType&,Simulation&):\n";
	FILE* writer=NULL;
	
	//open file
	if(XYZ_PRINT_STATUS>0) std::cout<<"opening file\n";
	writer=fopen(file,"w");
	if(writer==NULL) throw std::runtime_error("Runtime Error: Could not open file.");
	
	//check timing info
	if(XYZ_PRINT_STATUS>0) std::cout<<"setting interval\n";
	const int ibeg=Interval::index(interval.beg(),sim.timesteps());
	const int iend=Interval::index(interval.end(),sim.timesteps());
	
	//write simulation
	if(XYZ_PRINT_STATUS>0) std::cout<<"writing simulation\n";
	if(atomT.force){
		for(int t=ibeg; t<=iend; ++t){
			Structure struc=sim.frame(t);
			fprintf(writer,"%i\n",struc.nAtoms());
			if(struc.R().norm()>0){
				const Eigen::Matrix3d& R=struc.R();
				fprintf(writer,"Properties=species:S:1:pos:R:3:forces:R:3 potential_energy=%f pbc=\"T T T\" Lattice=\"%f %f %f %f %f %f %f %f %f\"\n",
					struc.pe(),R(0,0),R(1,0),R(2,0),R(0,1),R(1,1),R(1,2),R(0,2),R(1,2),R(2,2)
				);
			} else {
				fprintf(writer,"%s\n",sim.name().c_str());
			}
			for(int i=0; i<struc.nAtoms(); ++i){
				fprintf(writer,"  %-2s %19.10f %19.10f %19.10f %19.10f %19.10f %19.10f\n",struc.name(i).c_str(),
					struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2],
					struc.force(i)[0],struc.force(i)[1],struc.force(i)[2]
				);
			}
		}
	} else {
		for(int t=ibeg; t<=iend; ++t){
			Structure struc=sim.frame(t);
			fprintf(writer,"%i\n",struc.nAtoms());
			if(struc.R().norm()>0){
				const Eigen::Matrix3d& R=struc.R();
				fprintf(writer,"Properties=species:S:1:pos:R:3 potential_energy=%f pbc=\"T T T\" Lattice=\"%f %f %f %f %f %f %f %f %f\"\n",
					struc.pe(),R(0,0),R(1,0),R(2,0),R(0,1),R(1,1),R(1,2),R(0,2),R(1,2),R(2,2)
				);
			} else {
				fprintf(writer,"%s\n",sim.name().c_str());
			}
			for(int i=0; i<struc.nAtoms(); ++i){
				fprintf(writer,"  %-2s %19.10f %19.10f %19.10f\n",struc.name(i).c_str(),
					struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2]
				);
			}
		}
	}
	
	//close file
	if(XYZ_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(writer);
	writer=NULL;
}
	
}