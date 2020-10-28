//c libraries
#include <cstdio>
#include <ctime>
//c++ libraries
#include <iostream>
// eigen libraries
#include <Eigen/Dense>
// ann - structure
#include "structure.hpp"
// ann - string
#include "string.hpp"
// ann - units
#include "units.hpp"
// ann - ame
#include "ame.hpp"

namespace AME{

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-ame"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-ame\" option.");
			else format.ame=strlist[i+1];
		}
	}
	return format;
}

void read(const char* file, const AtomType& atomT, Structure& struc){
	const char* funcName="read(const char*,Structure&,const AtomType&)";
	if(DEBUG_AME_PRINT_FUNC>0) std::cout<<NAMESPACE<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
	//simulation flags
		bool frac;//fractiona/Cartesian coordinates
	//cell info
		Eigen::Matrix3d lv;
	//atom info
		int nspecies=0;//the number of atomic species
		std::vector<int> numbers;//the number of atoms in each species
		std::vector<std::string> names;//the names of each species
		int natoms=0;
	//misc
		bool error=false;
		std::vector<std::string> strlist;
	//units
		double s_posn=0.0,s_energy=0.0;
		if(units::consts::system()==units::System::AU){
			s_posn=units::BOHRpANG;
			s_energy=units::HARTREEpEV;
		}
		else if(units::consts::system()==units::System::METAL){
			s_posn=1.0;
			s_energy=1.0;
		}
		else throw std::runtime_error("Invalid units.");
		
	try{
		//open the file
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: \"")+std::string(file)+std::string("\"\n"));
		
		//read in the timestep
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"reading timestep\n";
		fgets(input,string::M,reader);
		std::strtok(input,string::WS);
		double ts=std::atof(std::strtok(NULL,string::WS));
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"TIMESTEP = "<<ts<<"\n";
		
		//read in number of frames
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"reading number of frames\n";
		fgets(input,string::M,reader);
		std::strtok(input,string::WS);
		int N=std::atof(std::strtok(NULL,string::WS));
		if(N==0) throw std::invalid_argument("Invalid number of frames.");
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"N        = "<<N<<"\n";
		
		//read in the cell
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"reading lattice vectors\n";
		fgets(input,string::M,reader);
		std::strtok(input,string::WS);
		lv(0,0)=std::atof(std::strtok(NULL,string::WS));
		lv(1,0)=std::atof(std::strtok(NULL,string::WS));
		lv(2,0)=std::atof(std::strtok(NULL,string::WS));
		lv(0,1)=std::atof(std::strtok(NULL,string::WS));
		lv(1,1)=std::atof(std::strtok(NULL,string::WS));
		lv(2,1)=std::atof(std::strtok(NULL,string::WS));
		lv(0,2)=std::atof(std::strtok(NULL,string::WS));
		lv(1,2)=std::atof(std::strtok(NULL,string::WS));
		lv(2,2)=std::atof(std::strtok(NULL,string::WS));
		lv*=s_posn;
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"LV = "<<lv<<"\n";
		static_cast<Cell&>(struc).init(lv);
		
		//read in atom names
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"reading species names\n";
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		nspecies=strlist.size()-1;
		if(nspecies==0) throw std::invalid_argument("Invalid number of species.");
		names.resize(nspecies);
		for(int i=0; i<nspecies; ++i){
			names[i]=strlist[i+1];
		}
		if(DEBUG_AME_PRINT_DATA>0){std::cout<<"NAMES = "; for(int i=0; i<nspecies; ++i) std::cout<<names[i]<<" "; std::cout<<"\n";}
		
		//read in atom numbers
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"reading species numbers\n";
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		if(strlist.size()-1!=nspecies) throw std::invalid_argument("Invalid number of species.");
		numbers.resize(nspecies);
		for(int i=0; i<nspecies; ++i){
			numbers[i]=std::atoi(strlist[i+1].c_str());
			natoms+=numbers[i];
		}
		if(DEBUG_AME_PRINT_DATA>0){std::cout<<"NUMBERS = "; for(int i=0; i<nspecies; ++i) std::cout<<numbers[i]<<" "; std::cout<<"\n";}
		
		//read in coordinate
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"reading coordinate\n";
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		if(string::to_upper(strlist[1])=="FRAC") frac=true;
		else if(string::to_upper(strlist[1])=="CART") frac=false;
		else throw std::invalid_argument("Invalid coordinate.");
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"COORD = "<<frac<<"\n";
		
		//resize structure
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"resizing structure\n";
		struc.resize(natoms,atomT);
		
		//read in energy
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"reading energy\n";
		fgets(input,string::M,reader);
		std::strtok(input,string::WS);
		struc.energy()=std::atof(std::strtok(NULL,string::WS))*s_energy;
		
		//read in positions
		if(atomT.posn){
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"reading positions\n";
			fgets(input,string::M,reader);
			for(int i=0; i<struc.nAtoms(); ++i){
				fgets(input,string::M,reader);
				struc.posn(i)[0]=std::atof(std::strtok(input,string::WS));
				struc.posn(i)[1]=std::atof(std::strtok(NULL,string::WS));
				struc.posn(i)[2]=std::atof(std::strtok(NULL,string::WS));
				if(DEBUG_AME_PRINT_DATA>1) std::cout<<"posn["<<i<<"] = "<<struc.posn(i).transpose()<<"\n";
			}
			if(frac){
				for(int i=0; i<struc.nAtoms(); ++i){
					struc.posn(i)=struc.R()*struc.posn(i);
				}
			} else {
				for(int i=0; i<struc.nAtoms(); ++i){
					struc.posn(i)*=s_posn;
				}
			}
		}
		
		//read in forces
		if(atomT.force){
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"reading forces\n";
			fgets(input,string::M,reader);
			if(input!=NULL){
				for(int i=0; i<struc.nAtoms(); ++i){
					fgets(input,string::M,reader);
					struc.force(i)[0]=std::atof(std::strtok(input,string::WS));
					struc.force(i)[1]=std::atof(std::strtok(NULL,string::WS));
					struc.force(i)[2]=std::atof(std::strtok(NULL,string::WS));
					if(DEBUG_AME_PRINT_DATA>1) std::cout<<"force["<<i<<"] = "<<struc.force(i).transpose()<<"\n";
				}
				if(frac){
					for(int i=0; i<struc.nAtoms(); ++i){
						struc.force(i)=struc.RInv()*struc.force(i)*s_energy;
					}
				} else {
					for(int i=0; i<struc.nAtoms(); ++i){
						struc.force(i)*=s_energy/s_posn;
					}
				}
			}
		}
		
		//close the file
		fclose(reader);
		reader=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//==== free all local variables ====
	if(reader!=NULL) fclose(reader);
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Exception: Could not read data.");
}

void write(const char* file, const AtomType& atomT, const Structure& struc){
	/*const char* funcName="read(const char*,const Structure&,const AtomType&)";
	if(DEBUG_AME_PRINT_FUNC>0) std::cout<<NAMESPACE<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* writer=NULL;
	//misc
		bool error=false;
	//units
		double s_posn=0.0,s_energy=0.0;
		if(units::consts::system()==units::System::AU){
			s_posn=units::ANGpBOHR;
			s_energy=units::EVpHARTREE;
		}
		else if(units::consts::system()==units::System::METAL){
			s_posn=1.0;
			s_energy=1.0;
		}
		else throw std::runtime_error("Invalid units.");
		
	try{
		//open the file
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: \"")+std::string(file)+std::string("\"\n"));
		
		//write the timestep
		fprintf(writer,"TIMESTEP 0\n");
		
		//write the number of timesteps
		fprintf(writer,"N 1\n");
		
		//write the cell
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"writing lattice vectors\n";
		fprintf(writer,"CELL ");
		fprintf(writer,"%f ",struc.R()(0,0)*s_posn);
		fprintf(writer,"%f ",struc.R()(1,0)*s_posn);
		fprintf(writer,"%f ",struc.R()(2,0)*s_posn);
		fprintf(writer,"%f ",struc.R()(0,1)*s_posn);
		fprintf(writer,"%f ",struc.R()(1,1)*s_posn);
		fprintf(writer,"%f ",struc.R()(2,1)*s_posn);
		fprintf(writer,"%f ",struc.R()(0,2)*s_posn);
		fprintf(writer,"%f ",struc.R()(1,2)*s_posn);
		fprintf(writer,"%f ",struc.R()(2,2)*s_posn);
		fprintf(writer,"\n");
		
		//write atom names
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"writing species names\n";
		fprintf(writer,"ATOMS ");
		for(int i=0; i<struc.nSpecies(); ++i) fprintf(writer,"%s ",struc.species(i).c_str());
		fprintf(writer,"\n");
		
		//write atom numbers
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"writing species numbers\n";
		fprintf(writer,"NUMBERS ");
		for(int i=0; i<struc.nSpecies(); ++i) fprintf(writer,"%i ",struc.nAtoms(i));
		fprintf(writer,"\n");
		
		//write coordinates
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"writing coordinate\n";
		fprintf(writer,"COORD FRAC\n");
		
		//write energy
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"writing energy\n";
		fprintf(writer,"ENERGY %.8f\n",struc.energy()*s_energy);
		
		//write positions
		fprintf(writer,"POSNS\n");
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"writing positions\n";
		if(atomT.posn){
			for(int i=0; i<struc.nAtoms(); ++i){
				Eigen::Vector3d posn=struc.RInv()*struc.posn(i);
				fprintf(writer,"\t%14.12f %14.12f %14.12f\n",posn[0],posn[1],posn[2]);
			}
		} else {
			for(int i=0; i<struc.nAtoms(); ++i){
				fprintf(writer,"\t0.0 0.0 0.0\n");
			}
		}
		
		//write forces
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"writing forces\n";
		if(atomT.force){
			double ftot=0;
			for(int i=0; i<struc.nAtoms(); ++i) ftot+=struc.force(i).norm();
			if(ftot/struc.nAtoms()>1e-6){
				//write forces only if non-zero
				fprintf(writer,"FORCES\n");
				for(int i=0; i<struc.nAtoms(); ++i){
					Eigen::Vector3d force=struc.R()*struc.force(i)*s_energy;
					fprintf(writer,"\t%14.12f %14.12f %14.12f\n",force[0],force[1],force[2]);
				}
			}
		}
		
		//close the file
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"closing the file\n";
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//==== free all local variables ====
	if(writer!=NULL) fclose(writer);
	
	if(error) throw std::runtime_error("I/O Exception: Could not write data.");*/
}

}