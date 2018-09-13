#ifndef VASP_HPP
#define VASP_HPP

// c libraries
#include <cstdio>
// c++ libraries
#include <string>
#include <stdexcept>
// eigen libraries
#include <Eigen/Dense>
// local libraries - structure
#include "cell.hpp"
#include "structure.hpp"
// local libraries - strings
#include "string.hpp"
// local libraries - units
#include "units.hpp"

#ifndef DEBUG_VASP
#define DEBUG_VASP 0
#endif

#ifndef __cplusplus
	#error A C++ compiler is required
#endif

namespace VASP{

//static variables
static const int HEADER_SIZE=7;//number of lines in the header before the atomic positions
static const char* NAMESPACE_GLOBAL="VASP";

//utilities
template <class AtomT> void setAN(AtomT& atom, std::false_type){};
template <class AtomT> void setAN(AtomT& atom, std::true_type){
	atom.an()=PTable::an(atom.name().c_str());
}

//*****************************************************
//FORMAT struct
//*****************************************************

struct Format{
	std::string xdatcar;//xdatcar
	std::string poscar;//poscar
	std::string xml;//poscar
	std::string outcar;//outcar
	std::string procar;//procar
	std::string eigenval;//eigenval
	std::string energy;//energy
	int beg,end,stride;//interval
	Format():beg(1),end(-1),stride(1){};
	static Format& load(const std::vector<std::string>& strlist, Format& format);
};

namespace POSCAR{

//static variables
static const char* NAMESPACE_LOCAL="POSCAR";

template <class AtomT>
void load(const char* file, Structure<AtomT>& struc){
	const char* funcName="load<AtomT>(const char*,Structure<AtomT>&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<funcName<<":\n";
	/* local function variables */
	//file i/o
		FILE* reader=NULL;
		char* input=(char*)malloc(sizeof(char)*string::M);
		char* temp=(char*)malloc(sizeof(char)*string::M);
	//simulation flags
		bool direct;//whether the coordinates are in direct or Cartesian coordinates
	//cell
		Cell cell;
		Eigen::Matrix3d lv;
		double scale=1;
	//atom info
		unsigned int nSpecies=0;//the number of atomic species
		std::vector<unsigned int> nAtoms;//the number of atoms in each species
		std::vector<std::string> names;//the names of each species
		unsigned int N=0;
	//units
		double s=0.0;
		if(units::consts::system()==units::System::AU) s=units::BOHRpANG;
		else if(units::consts::system()==units::System::METAL) s=1.0;
		else throw std::runtime_error("Invalid units.");
	//misc
		bool error=false;
		
	try{
		//open the file
		if(DEBUG_VASP>1) std::cout<<"Opening file...\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("Unable to open file.");
		
		//clear the simulation
		struc.clear();
		//set the periodicity (always true for VASP)
		struc.periodic()=true;
		
		/* read in the system name */
		if(DEBUG_VASP>1) std::cout<<"Reading system name...\n";
		struc.system()=std::string(string::trim(fgets(input,string::M,reader)));
		
		/* load the simulation cell */
		if(DEBUG_VASP>1) std::cout<<"Loading simulation cell...\n";
		scale=std::atof(fgets(input, string::M, reader));
		if(scale<0) scale=std::pow(-1.0*scale,1.0/3.0);
		//read in the lattice vectors of the system
		/*
			Here we take the transpose of the lattice vector matrix.
			The reason is that VASP stores the lattice vector matrix such that the lattice vectors
			are the rows of the matrix.  However, it's more appropriate for operations on coordinates
			for the lattice vectors to be columns of the matrix.
		*/
		if(DEBUG_VASP>1) std::cout<<"Reading in the lattice vectors...\n";
		for(unsigned int i=0; i<3; ++i){
			fgets(input, string::M, reader);
			lv(0,i)=std::atof(std::strtok(input,string::WS));
			for(unsigned int j=1; j<3; ++j){
				lv(j,i)=std::atof(std::strtok(NULL,string::WS));
			}
		}
		//initialize the cell
		struc.cell().init(s*lv,scale);
		
		/* load the atom info */
		//read in the number of species
		if(DEBUG_VASP>1) std::cout<<"Reading in the number of species...\n";
		fgets(input, string::M, reader);
		nSpecies=string::substrN(input,string::WS);
		names.resize(nSpecies);
		nAtoms.resize(nSpecies);
		//read in the species names
		if(DEBUG_VASP>1) std::cout<<"Reading in the species names...\n";
		names[0]=std::string(std::strtok(input,string::WS));
		for(unsigned int i=1; i<nSpecies; ++i){
			names[i]=std::string(std::strtok(NULL,string::WS));
		}
		//read in the species numbers
		if(DEBUG_VASP>1) std::cout<<"Reading in the species numbers...\n";
		fgets(input, string::M, reader);
		nAtoms[0]=std::atoi(std::strtok(input,string::WS));
		for(unsigned int i=1; i<nSpecies; ++i){
			nAtoms[i]=std::atoi(std::strtok(NULL,string::WS));
		}
		
		/* load the coordinate type */
		fgets(input, string::M, reader);
		if(input[0]=='D') direct=true;
		else direct=false;
		
		/* print data to screen */
		if(DEBUG_VASP>1){
			std::cout<<"NAME = "<<struc.system()<<"\n";
			std::cout<<"DIRECT = "<<(direct?"T":"F")<<"\n";
			std::cout<<"CELL = \n"<<struc.cell()<<"\n";
			std::cout<<"SPECIES = ";
			for(unsigned int i=0; i<names.size(); ++i) std::cout<<names[i]<<" "; std::cout<<"\n";
			std::cout<<"NUMBERS = ";
			for(unsigned int i=0; i<nAtoms.size(); ++i) std::cout<<nAtoms[i]<<" "; std::cout<<"\n";
		}
		
		/* resize the simulation */
		if(DEBUG_VASP>0) std::cout<<"Allocating memory...\n";
		struc.resize(nAtoms,names);
		
		/* load positions */
		if(DEBUG_VASP>1) std::cout<<"Loading positions...\n";
		//load the positions
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			fgets(input,string::M,reader);
			struc.atom_posn()[n][0]=s*std::atof(std::strtok(input,string::WS));
			struc.atom_posn()[n][1]=s*std::atof(std::strtok(NULL,string::WS));
			struc.atom_posn()[n][2]=s*std::atof(std::strtok(NULL,string::WS));
		}
		
		/* convert to cartesian coordinates (if necessary) */
		if(DEBUG_VASP>1) std::cout<<"Converting to Cartesian coordinates...\n";
		if(direct){
			for(unsigned int n=0; n<struc.nAtoms(); ++n){
				struc.atom_posn()[n]=struc.cell().R()*struc.atom_posn()[n];
			}
		}
		
		fclose(reader);
		reader=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free all local variables
	if(reader!=NULL) fclose(reader);
	free(input);
	free(temp);
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

template <class AtomT>
void print(const char* file, const Structure<AtomT>& struc, bool direct=true){
	static const char* funcName="load<AtomT>(const char*,Structure<AtomT>&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	/* local function variables */
	//file i/o
		FILE* writer=NULL;
	//misc
		bool error=false;
	
	try{
		//open the file
		if(DEBUG_VASP>1) std::cout<<"Opening file...\n";
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error("Unable to open file.");
		
		//print the system name
		if(DEBUG_VASP>1) std::cout<<"Printing name...\n";
		fprintf(writer,"%s\n",struc.system().c_str());
		
		//print the simulation cell (transpose)
		if(DEBUG_VASP>1) std::cout<<"Printing cell...\n";
		fprintf(writer," %10.4f\n",struc.cell().scale());
		for(unsigned int i=0; i<3; ++i){
			for(unsigned int j=0; j<3; ++j){
				fprintf(writer," %11.6f",struc.cell().R()(j,i)/struc.cell().scale());
			}
			fprintf(writer,"\n");
		}
		
		//print the atom info
		if(DEBUG_VASP>1) std::cout<<"Printing atom info...\n";
		for(unsigned int i=0; i<struc.nSpecies(); ++i){
			fprintf(writer," %4s",struc.atomNames(i).c_str());
		}
		fprintf(writer,"\n");
		for(unsigned int i=0; i<struc.nSpecies(); ++i){
			fprintf(writer," %4i",struc.nAtoms(i));
		}
		fprintf(writer,"\n");
		
		//print the positions
		if(DEBUG_VASP>1) std::cout<<"Printing posns...\n";
		if(!direct){
			fprintf(writer,"Cart\n");
			for(unsigned int n=0; n<struc.nAtoms(); ++n){
				fprintf(writer, "%.8f %.8f %.8f\n",
					struc.atom_posn()[n][0],
					struc.atom_posn()[n][1],
					struc.atom_posn()[n][2]
				);
			}
		} else {
			fprintf(writer,"Direct\n");
			for(unsigned int n=0; n<struc.nAtoms(); ++n){
				Eigen::Vector3d posn=struc.cell().RInv()*struc.atom_posn()[n];
				fprintf(writer, "%.8f %.8f %.8f\n",posn[0],posn[1],posn[2]);
			}
		}
		
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	if(writer!=NULL) fclose(writer);
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

}

namespace XML{

//static variables
static const char* NAMESPACE_LOCAL="XML";

template <class AtomT> void load_forces(std::false_type, FILE* reader, Structure<AtomT>& struc, unsigned int t){};
template <class AtomT> void load_forces(std::true_type, FILE* reader, Structure<AtomT>& struc, unsigned int t){
	static const char* funcName="load_forces(std::true_type,FILE*,Structure<AtomT>&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//==== local variables ====
	//string utilities
		const char* force_str="<varray name=\"forces\" >";
		char* input=(char*)malloc(sizeof(char)*string::M);
		char* temp=(char*)malloc(sizeof(char)*string::M);
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
	
	rewind(reader);
	//==== read in the forces ====
	if(DEBUG_VASP>0) std::cout<<"Reading in forces...\n";
	std::rewind(reader);
	unsigned tt=0;
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"<calculation>")!=NULL){
			while(fgets(input,string::M,reader)!=NULL){
				if(std::strstr(input,"</calculation>")!=NULL){
					throw std::invalid_argument("No forces provided in calculation.");
				} else if(std::strstr(input,"forces")!=NULL){
					if(tt<t) continue;
					for(unsigned int n=0; n<struc.nAtoms(); ++n){
						fgets(input,string::M,reader); std::strtok(input,string::WS);
						struc.atom_force()[n][0]=std::atof(std::strtok(NULL,string::WS))*s_energy/s_posn;
						struc.atom_force()[n][1]=std::atof(std::strtok(NULL,string::WS))*s_energy/s_posn;
						struc.atom_force()[n][2]=std::atof(std::strtok(NULL,string::WS))*s_energy/s_posn;
					}
					++tt;
					break;
				}
			}
		}
	}
	
	//==== free local variables ====
	free(temp);
	free(input);
}

template <class AtomT>
void load(const char* file, Structure<AtomT>& struc, unsigned int t=0){
	static const char* funcName="load<AtomT>(const char*,Structure<AtomT>&,(int))";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=(char*)malloc(sizeof(char)*string::M);
		char* temp=(char*)malloc(sizeof(char)*string::M);
		std::string str;
	//simulation flags
		bool direct;//whether the coordinates are in direct or Cartesian coordinates
	//time info
		unsigned int ts=0;//number of timesteps
		unsigned int interval=0;//requested interval
	//cell info
		double scale=1;
		Eigen::Matrix3d lv;
		Cell cell;
	//atom info
		unsigned int nSpecies=0;//the number of atomic species
		std::vector<unsigned int> nAtoms;//the number of atoms in each species
		std::vector<std::string> atomNames;//the names of each species
		unsigned int N=0;
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
	//misc
		bool error=false;
		unsigned int tt;
	
	try{
		
		//==== open the file ====
		if(DEBUG_VASP>0) std::cout<<"Opening the file...\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Could not open file.");
		
		//==== load the number of timesteps ====
		if(DEBUG_VASP>0) std::cout<<"Loading the timesteps...\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"<calculation>")!=NULL) ++ts;
		}
		if(DEBUG_VASP>0) std::cout<<"ts = "<<ts<<"\n";
		if(t>ts) throw std::invalid_argument("Invalid timestep");
		
		//==== read in the atom info ====
		if(DEBUG_VASP>0) std::cout<<"Reading in the atom info...\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"atominfo")!=NULL){
				fgets(input,string::M,reader);
				std::strtok(input,">");
				N=std::atoi(std::strtok(NULL,"<"));
				for(unsigned int i=0; i<6; ++i) fgets(input,string::M,reader);
				for(unsigned int i=0; i<N; ++i){
					fgets(input,string::M,reader);
					std::strtok(input,">"); std::strtok(NULL,">");
					std::string name=std::strtok(NULL,"<");
					bool match=false;
					for(unsigned int j=0; j<atomNames.size(); ++j){
						if(name==atomNames[j]){
							++nAtoms[j];
							match=true; 
							break;
						}
					}
					if(!match){
						atomNames.push_back(name);
						nAtoms.push_back(1);
					}
				}
				break;
			}
		}
		if(DEBUG_VASP>0){
			std::cout<<"ATOM_NAMES = "; for(unsigned int i=0; i<atomNames.size(); ++i) std::cout<<atomNames[i]<<" "; std::cout<<"\n";
			std::cout<<"ATOM_NUMBERS = "; for(unsigned int i=0; i<nAtoms.size(); ++i) std::cout<<nAtoms[i]<<" "; std::cout<<"\n";
		}
		if(atomNames.size()!=nAtoms.size()) throw std::runtime_error("Mismatch in atom names/numbers.");
		nSpecies=atomNames.size();
		
		//==== resize the simulation ====
		if(DEBUG_VASP>0) std::cout<<"Resizing the simulation...\n";
		struc.resize(nAtoms,atomNames);
		
		//==== load the cells ====
		if(DEBUG_VASP>0) std::cout<<"Loading the cells...\n";
		tt=0;
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"<calculation>")!=NULL){
				while(fgets(input,string::M,reader)!=NULL){
					if(std::strstr(input,"</calculation>")!=NULL){
						throw std::invalid_argument("No cell provided in calculation.");
					} else if(std::strstr(input,"basis")!=NULL){
						if(tt<t) continue;
						fgets(input,string::M,reader);
						std::strtok(input,string::WS);
						lv(0,0)=std::atof(std::strtok(NULL,string::WS));
						lv(1,0)=std::atof(std::strtok(NULL,string::WS));
						lv(2,0)=std::atof(std::strtok(NULL,string::WS));
						fgets(input,string::M,reader);
						std::strtok(input,string::WS);
						lv(0,1)=std::atof(std::strtok(NULL,string::WS));
						lv(1,1)=std::atof(std::strtok(NULL,string::WS));
						lv(2,1)=std::atof(std::strtok(NULL,string::WS));
						fgets(input,string::M,reader);
						std::strtok(input,string::WS);
						lv(0,2)=std::atof(std::strtok(NULL,string::WS));
						lv(1,2)=std::atof(std::strtok(NULL,string::WS));
						lv(2,2)=std::atof(std::strtok(NULL,string::WS));
						lv*=s_posn;
						struc.cell().init(lv);
						break;
					}
				}
			}
		}
		
		//==== read in the positions ====
		if(DEBUG_VASP>0) std::cout<<"Reading in positions...\n";
		std::rewind(reader);
		tt=0;
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"<calculation>")!=NULL){
				while(fgets(input,string::M,reader)!=NULL){
					if(std::strstr(input,"</calculation>")!=NULL){
						throw std::invalid_argument("No positions provided in calculation.");
					} else if(std::strstr(input,"positions")!=NULL){
						if(tt<t) continue;
						for(unsigned int n=0; n<struc.nAtoms(); ++n){
							fgets(input,string::M,reader); std::strtok(input,string::WS);
							struc.atom_posn()[n][0]=std::atof(std::strtok(NULL,string::WS));
							struc.atom_posn()[n][1]=std::atof(std::strtok(NULL,string::WS));
							struc.atom_posn()[n][2]=std::atof(std::strtok(NULL,string::WS));
						}
						break;
					}
				}
			}
		}
		
		//==== read in the energies ====
		if(DEBUG_VASP>0) std::cout<<"Reading in the energies...\n";
		std::rewind(reader);
		tt=0;
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"<calculation>")!=NULL){
				if(tt<t) continue;
				double energy=0;
				while(fgets(input,string::M,reader)!=NULL){
					if(std::strstr(input,"</calculation>")!=NULL) break;
					if(std::strstr(input,"e_fr_energy")!=NULL){
						std::strtok(input,">");
						energy=std::atof(std::strtok(NULL,"<"));
					}
				}
				struc.energy()=energy*s_energy;
				break;
			}
		}
		
		//==== read in the forces ====
		if(DEBUG_VASP>0) std::cout<<"Reading in the forces...\n";
		load_forces(std::is_base_of<Force,AtomT>(),reader,struc,t);
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//==== transform to Cartesian coordinates ====
	if(DEBUG_VASP>0) std::cout<<"Transforming to Cartesian coordinates...\n";
	for(unsigned int n=0; n<struc.nAtoms(); ++n){
		struc.atom_posn()[n]=struc.cell().R()*struc.atom_posn()[n];
		Cell::returnToCell(struc.atom_posn()[n],struc.atom_posn()[n],struc.cell().R(),struc.cell().RInv());
	}
	
	//==== free all local variables ====
	if(reader!=NULL) fclose(reader);
	free(input);
	free(temp);
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

}

}

#endif
