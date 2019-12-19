// c libraries
#include <ctime>
// c++ libraries
#include <iostream>
#include <string>
#include <stdexcept>
// eigen libraries
#include <Eigen/Dense>
// ann - structure
#include "structure.hpp"
// ann - strings
#include "string.hpp"
// ann - units
#include "units.hpp"
// ann - math
#include "math_const.hpp"
// ann - qe
#include "qe.hpp"

namespace QE{

//*****************************************************
//FORMAT struct
//*****************************************************

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(unsigned int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-pos"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-pos\" option.");
			else format.filePos=strlist[i+1];
		} else if(strlist[i]=="-in"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-in\" option.");
			else format.fileIn=strlist[i+1];
		} else if(strlist[i]=="-cel"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-cel\" option.");
			else format.fileCel=strlist[i+1];
		} else if(strlist[i]=="-evp"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-evp\" option.");
			else format.fileEvp=strlist[i+1];
		} else if(strlist[i]=="-qeout"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-evp\" option.");
			else format.fileOut=strlist[i+1];
		}
	}
	return format;
}

//*****************************************************
//CEL format
//*****************************************************

namespace CEL{

void read_cell(FILE* reader, Simulation& sim){
	const char* func_name="read_cell(const char*,SimI&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	//==== local function variables ====
	//file i/o
		char* input=new char[string::M];
	//cell
		Eigen::Matrix3d lv;
	//misc
		bool error=false;
	//units
		double s_posn=0.0;
		if(units::consts::system()==units::System::AU) s_posn=1.0;
		else if(units::consts::system()==units::System::METAL) s_posn=units::ANGpBOHR;
		else throw std::runtime_error("Invalid units.");
		
	try{
		//rewind the reader
		std::cout<<"rewinding reader...\n";
		std::rewind(reader);
		
		//skip to the beginning
		std::cout<<"skipping to beginning...\n";
		for(unsigned int t=0; t<sim.beg(); ++t){
			if(QE_PRINT_STATUS>1) std::cout<<"t = "<<t<<"\n";
			fgets(input,string::M,reader);//header
			for(unsigned int n=0; n<3; ++n) fgets(input,string::M,reader);
		}
		//read in the data
		std::cout<<"reading data...\n";
		for(unsigned int t=0; t<sim.timesteps(); ++t){
			if(QE_PRINT_STATUS>1) std::cout<<"t = "<<t<<"\n";
			fgets(input,string::M,reader);//header
			//first line
			fgets(input,string::M,reader);
			lv(0,0)=std::atof(std::strtok(input,string::WS))*s_posn;
			lv(0,1)=std::atof(std::strtok(NULL,string::WS))*s_posn;
			lv(0,2)=std::atof(std::strtok(NULL,string::WS))*s_posn;
			//second line
			fgets(input,string::M,reader);
			lv(1,0)=std::atof(std::strtok(input,string::WS))*s_posn;
			lv(1,1)=std::atof(std::strtok(NULL,string::WS))*s_posn;
			lv(1,2)=std::atof(std::strtok(NULL,string::WS))*s_posn;
			//third line
			fgets(input,string::M,reader);
			lv(2,0)=std::atof(std::strtok(input,string::WS))*s_posn;
			lv(2,1)=std::atof(std::strtok(NULL,string::WS))*s_posn;
			lv(2,2)=std::atof(std::strtok(NULL,string::WS))*s_posn;
			//set the cell
			sim.frame(t).init(lv);
			for(unsigned int tt=1; tt<sim.stride(); ++tt){
				fgets(input,string::M,reader);//header
				for(unsigned int n=0; n<3; ++n) fgets(input,string::M,reader);
			}
		}
		
		lv=sim.frame(0).R();
		sim.cell_fixed()=true;
		for(unsigned int t=1; t<sim.timesteps(); ++t){
			if((lv-sim.frame(t).R()).norm()>num_const::ZERO){
				sim.cell_fixed()=false; break;
			}
		}
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] input;
}
	
}

//*****************************************************
//IN format
//*****************************************************

namespace IN{

Cell& read_cell(FILE* reader, Cell& cell){
	const char* func_name="read_cell(const char*,SimAtomic<AtomT>&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	/* local function variables */
	//file i/o
		char* input=new char[string::M];
		char* tag=new char[string::M];
		char* option=new char[string::M];
	//cell
		int ibrav=0;
		std::vector<double> lvp(6,0);
		Eigen::Matrix3d lv;
	//misc
		bool error=false;
	//units
		double s_posn=0.0;
		if(units::consts::system()==units::System::AU) s_posn=1.0;
		else if(units::consts::system()==units::System::METAL) s_posn=units::ANGpBOHR;
		else throw std::runtime_error("Invalid units.");
		
	try{
		//rewind the reader
		std::rewind(reader);
		//read in ibrav
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_all(input);//remove all spaces
			string::copy_left(tag,input,"=");//copy the tag
			if(std::strcmp(tag,"ibrav")==0){
				ibrav=std::atoi(std::strpbrk(input,"=")+1);
				break;
			}
		}
		if(QE_PRINT_STATUS>1) std::cout<<"IBRAV = "<<ibrav<<"\n";
		
		//rewind the reader
		std::rewind(reader);
		if(ibrav==0){
			//read in the cell parameters
			while(fgets(input,string::M,reader)!=NULL){
				string::trim(input);//remove beg/end spaces
				string::copy_left(tag,input,string::WS);//copy the tag
				if(std::strpbrk(input,string::WS)!=NULL){//if there's an option...
					std::strtok(input,string::WS);
					string::trim_all(string::to_upper(std::strcpy(option,std::strtok(NULL,string::WS))));//copy the option
				}
				if(std::strcmp(tag,"CELL_PARAMETERS")==0){
					//read in the unit, convert to angstrom in necessary
					if(!string::empty(option)){
						if(units::consts::system()==units::System::METAL){
							if(std::strcmp(option,"ANGSTROM")==0) s_posn=1;
							else if(std::strcmp(option,"BOHR")==0) s_posn=units::ANGpBOHR;
						} else if(units::consts::system()==units::System::AU){
							if(std::strcmp(option,"ANGSTROM")==0) s_posn=units::BOHRpANG;
							else if(std::strcmp(option,"BOHR")==0) s_posn=1.0;
						} else throw std::runtime_error("Invalid units.");
					}
					if(QE_PRINT_STATUS>1) std::cout<<"option = "<<option<<"\n";
					//first lattice vector
					fgets(input,string::M,reader);
					lv(0,0)=std::atof(std::strtok(input,string::WS))*s_posn;
					lv(1,0)=std::atof(std::strtok(NULL,string::WS))*s_posn;
					lv(2,0)=std::atof(std::strtok(NULL,string::WS))*s_posn;
					//second lattice vector
					fgets(input,string::M,reader);
					lv(0,1)=std::atof(std::strtok(input,string::WS))*s_posn;
					lv(1,1)=std::atof(std::strtok(NULL,string::WS))*s_posn;
					lv(2,1)=std::atof(std::strtok(NULL,string::WS))*s_posn;
					//third lattice vector
					fgets(input,string::M,reader);
					lv(0,2)=std::atof(std::strtok(input,string::WS))*s_posn;
					lv(1,2)=std::atof(std::strtok(NULL,string::WS))*s_posn;
					lv(2,2)=std::atof(std::strtok(NULL,string::WS))*s_posn;
					//break
					break;
				}
			}
		} else {
			//read in the cell parameters
			while(fgets(input,string::M,reader)!=NULL){
				string::trim_all(input);//remove all spaces
				string::copy_left(tag,input,"(");//copy the tag
				if(std::strcmp(tag,"celldm")==0){
					std::strtok(input," \t\n()=");
					int index=std::atoi(std::strtok(NULL," \t\n()="));
					double val=std::atof(std::strtok(NULL," \t\n()="));
					lvp[index]=val;
				}
			}
			if(ibrav==0){
				lv=Eigen::Matrix3d::Identity()*lvp[0];
			} else if(ibrav==1){
				lv(0,0)=-1; lv(1,0)=0; lv(2,0)=1;
				lv(0,1)=0; lv(1,1)=1; lv(2,1)=1;
				lv(0,2)=-1; lv(1,2)=1; lv(2,2)=0;
				lv*=lvp[0]*0.5;
			} else if(ibrav==2){
				lv(0,0)=1; lv(1,0)=1; lv(2,0)=1;
				lv(0,1)=-1; lv(1,1)=1; lv(2,1)=1;
				lv(0,2)=-1; lv(1,2)=-1; lv(2,2)=1;
				lv*=lvp[0]*0.5;
			} else if(ibrav==4){
				lv(0,0)=1; lv(1,0)=0; lv(2,0)=0;
				lv(0,1)=-0.5; lv(1,1)=0.5*std::sqrt(3.0); lv(2,1)=0;
				lv(0,2)=0; lv(1,2)=0; lv(2,2)=lvp[2];
				lv*=lvp[0];
			} else if(ibrav==6){
				lv=Eigen::Matrix3d::Identity();
				lv(2,2)=lvp[2];
				lv*=lvp[0];
			} else if(ibrav==8){
				lv=Eigen::Matrix3d::Identity();
				lv(0,0)=lvp[0];
				lv(1,1)=lvp[1];
				lv(2,2)=lvp[2];
			} else if(ibrav==12){
				lv=Eigen::Matrix3d::Identity();
				lv(0,0)=lvp[0];
				lv(0,1)=lvp[0]*lvp[1]*std::cos(lvp[3]);
				lv(1,1)=lvp[0]*lvp[1]*std::cos(lvp[3]);
				lv(2,2)=lvp[0]*lvp[2];
			} else throw std::invalid_argument("Invalid IBRAV.");
			//scale units
			lv*=s_posn;
		}
		
		if(QE_PRINT_STATUS>1) std::cout<<"lv = \n"<<lv<<"\n";
		
		//initialize the cell
		cell.init(lv);
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	delete[] input;
	delete[] tag;
	delete[] option;
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
	else return cell;
}

void load_atoms(FILE* reader, std::vector<std::string>& species, std::vector<unsigned int>& atomNumbers){
	const char* func_name="load_atoms(FILE*,std::vector<std::string>&,std::vector<std::string>&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	/* local function variables */
	//file i/o
		char* input=new char[string::M];
		char* temp=new char[string::M];
		char* tag=new char[string::M];
	//atoms
		unsigned int nAtoms=0;
		unsigned int nTypes=0;
	//misc
		bool error=false;
		
	try{
		//rewind the reader
		std::rewind(reader);
		
		//read in the atom info
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_all(string::trim_right(input,","));
			string::copy_left(temp,input,"=");
			if(std::strcmp(temp,"nat")==0){
				nAtoms=std::atoi(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"ntyp")==0){
				nTypes=std::atoi(std::strpbrk(input,"=")+1);
			}
		}
		
		//rewind the reader
		std::rewind(reader);
		
		//read in the atom types
		species.clear();
		atomNumbers.clear();
		while(fgets(input,string::M,reader)!=NULL){
			string::trim(input);//remove all spaces
			string::copy_left(tag,input,string::WS);//copy the tag
			if(std::strcmp(tag,"ATOMIC_POSITIONS")==0){
				for(unsigned int n=0; n<nAtoms; ++n){
					fgets(input,string::M,reader);
					if(string::substrN(input,string::WS)!=4) throw std::runtime_error("Invalid atom format.");
					//read in the name
					std::string name=std::string(std::strtok(input,string::WS));
					bool match=false;
					for(unsigned int i=0; i<species.size(); ++i){
						if(name==species[i]){
							match=true;
							++atomNumbers[i];
						}
					}
					if(!match){
						species.push_back(name);
						atomNumbers.push_back(1);
					}
				}
				break;
			}
		}
		
		unsigned int nAtomsList=0;
		for(unsigned int i=0; i<atomNumbers.size(); ++i) nAtomsList+=atomNumbers[i];
		
		if(QE_PRINT_STATUS>0){
			std::cout<<"nTypes     = "<<nTypes<<"\n";
			std::cout<<"nAtomNames = "<<species.size()<<"\n";
			std::cout<<"nAtoms     = "<<nAtoms<<"\n";
			std::cout<<"nAtomsList = "<<nAtomsList<<"\n";
			std::cout<<"species  = \n";
			for(unsigned int i=0; i<species.size(); ++i){
				std::cout<<"\t"<<species[i]<<"\n";
			}
		}
		
		if(species.size()!=nTypes) throw std::runtime_error("Incompatible number of types and atom specification.");
		if(nAtoms!=nAtomsList) throw std::runtime_error("Incompatible number of atoms and atom specification.");
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	delete[] input;
	delete[] temp;
	delete[] tag;
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
}

double load_timestep(FILE* reader){
	const char* func_name="load_timestep(FILE*)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	/* local function variables */
	//file i/o
		char* input=new char[string::M];
		char* temp=new char[string::M];
	//timestep
		double ts;
		unsigned int print=0;
	//misc
		bool error=false;
		
	try{
		//rewind the reader
		std::rewind(reader);
		
		//read in the cell
		if(QE_PRINT_STATUS>0) std::cout<<"reading the timestep\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_all(string::trim_right(input,","));
			string::copy_left(temp,input,"=");
			if(std::strcmp(temp,"dt")==0){
				ts=std::atof(std::strpbrk(input,"=")+1);
			} else if(std::strcmp(temp,"iprint")==0){
				print=std::atoi(std::strpbrk(input,"=")+1);
			}
		}
		
		if(QE_PRINT_STATUS>0){
			std::cout<<"timestep = "<<ts<<"\n";
			std::cout<<"print    = "<<print<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	delete[] input;
	delete[] temp;
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
	else return ts*print*0.02418884326505;//convert from a.u. to fs
}

}

//*****************************************************
//POS format
//*****************************************************

namespace POS{

unsigned int load_timesteps(FILE* reader){
	const char* func_name="load_timesteps(const char*)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	/* local function variables */
	//file i/o
		char* input=new char[string::M];
	//timesteps
		unsigned int ts=0;
	//misc
		bool error=false;
		
	try{
		//rewind the reader
		std::rewind(reader);
		
		//read in the timesteps
		while(fgets(input,string::M,reader)!=NULL){
			if(string::substrN(input,string::WS)==2) ++ts;
		}
		
		if(QE_PRINT_STATUS>0){
			std::cout<<"ts = "<<ts<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
	return ts;
}

void load_posns(FILE* reader, Simulation& sim){
	const char* func_name="load_posns(const char*,Simulation&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	/* local function variables */
	//file i/o
		char* input=new char[string::M];
	//misc
		bool error=false;
	//units
		double s_posn=0.0;
		if(units::consts::system()==units::System::AU) s_posn=1.0;
		else if(units::consts::system()==units::System::METAL) s_posn=units::ANGpBOHR;
		else throw std::runtime_error("Invalid units.");
		
	try{
		//rewind to beginning of file
		std::rewind(reader);
		
		//skip to the beginning
		for(unsigned int t=0; t<sim.beg(); ++t){
			if(QE_PRINT_STATUS>1) std::cout<<"T = "<<t<<"\n";
			else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
			fgets(input,string::M,reader);//header
			for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n) fgets(input,string::M,reader);
		}
		//read in the data
		for(unsigned int t=0; t<sim.timesteps(); ++t){
			if(QE_PRINT_STATUS>1) std::cout<<"T = "<<t<<"\n";
			else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
			fgets(input,string::M,reader);//header
			for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n){
				fgets(input,string::M,reader);
				sim.frame(t).posn(n)[0]=s_posn*std::atof(std::strtok(input,string::WS));
				sim.frame(t).posn(n)[1]=s_posn*std::atof(std::strtok(NULL,string::WS));
				sim.frame(t).posn(n)[2]=s_posn*std::atof(std::strtok(NULL,string::WS));
			}
			for(unsigned int tt=1; tt<sim.stride(); ++tt){
				fgets(input,string::M,reader);//header
				for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n){
					fgets(input,string::M,reader);
				}
			}
		}
		
		//return to cell
		for(unsigned int t=0; t<sim.timesteps(); ++t){
			for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n){
				Cell::returnToCell(
					sim.frame(t).posn(n),sim.frame(t).posn(n),
					sim.frame(t).R(),sim.frame(t).RInv()
				);
			}
		}
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
}

}

//*****************************************************
//FOR format
//*****************************************************

namespace FOR{

unsigned int load_timesteps(FILE* reader){
	const char* func_name="load_timesteps(const char*)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	/* local function variables */
	//file i/o
		char* input=new char[string::M];
	//timesteps
		unsigned int ts=0;
	//misc
		bool error=false;
		
	try{
		//rewind the reader
		std::rewind(reader);
		
		//read in the timesteps
		while(fgets(input,string::M,reader)!=NULL){
			if(string::substrN(input,string::WS)==2) ++ts;
		}
		
		if(QE_PRINT_STATUS>0){
			std::cout<<"ts = "<<ts<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
	return ts;
}

void read_forces(FILE* reader, Simulation& sim){
	const char* func_name="read_forces(const char*,Simulation&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	/* local function variables */
	//file i/o
		char* input=new char[string::M];
	//misc
		bool error=false;
	//units
		double s_energy=0.0,s_posn=0.0;
		if(units::consts::system()==units::System::AU){
			s_posn=1.0;
			s_energy=1.0;
		} else if(units::consts::system()==units::System::METAL){
			s_posn=units::ANGpBOHR;
			s_energy=units::HARTREEpEV;
		} else throw std::runtime_error("Invalid units.");
		
	try{
		//rewind to beginning of file
		std::rewind(reader);
		
		//skip to the beginning
		for(unsigned int t=0; t<sim.beg(); ++t){
			if(QE_PRINT_STATUS>1) std::cout<<"T = "<<t<<"\n";
			else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
			fgets(input,string::M,reader);//header
			for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n) fgets(input,string::M,reader);
		}
		//read in the data
		for(unsigned int t=0; t<sim.timesteps(); ++t){
			if(QE_PRINT_STATUS>1) std::cout<<"T = "<<t<<"\n";
			else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
			fgets(input,string::M,reader);//header
			const double fac=s_energy/s_posn;
			for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n){
				fgets(input,string::M,reader);
				sim.frame(t).force(n)[0]=std::atof(std::strtok(input,string::WS))*fac;
				sim.frame(t).force(n)[1]=std::atof(std::strtok(NULL,string::WS))*fac;
				sim.frame(t).force(n)[2]=std::atof(std::strtok(NULL,string::WS))*fac;
			}
			for(unsigned int tt=1; tt<sim.stride(); ++tt){
				fgets(input,string::M,reader);//header
				for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n){
					fgets(input,string::M,reader);
				}
			}
		}
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
}

}

//*****************************************************
//EVP format
//*****************************************************

namespace EVP{
	
void read_energy(FILE* reader, Simulation& sim){
	const char* func_name="read_energy(FILE*,SimAtomic<AtomT>&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	/* local function variables */
	//file i/o
		char* input=new char[string::M];
	//timesteps
		unsigned int ts=0;
	//misc
		bool error=false;
	//units
		double s_energy=0.0;
		if(units::consts::system()==units::System::AU) s_energy=1.0;
		else if(units::consts::system()==units::System::METAL) s_energy=units::HARTREEpEV;
		else throw std::runtime_error("Invalid units.");
		
	try{
		//rewind the reader
		if(QE_PRINT_STATUS>1) std::cout<<"rewinding reader\n";
		std::rewind(reader);
		
		fgets(input,string::M,reader);//skip the header
		//skip to beginning
		if(QE_PRINT_STATUS>1) std::cout<<"moving to beginning\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strpbrk(input,"#")!=NULL) continue;//skip any lines with "#"
			if(ts==sim.beg()){
				std::strtok(input,string::WS);
				std::strtok(NULL,string::WS);
				std::strtok(NULL,string::WS);
				std::strtok(NULL,string::WS);
				std::strtok(NULL,string::WS);
				sim.frame(ts++).energy()=s_energy*std::atof(std::strtok(NULL,string::WS));
				for(unsigned int tt=1; tt<sim.stride(); ++tt) fgets(input,string::M,reader);
				break;
			} else ++ts;
		}
		ts=1;
		if(QE_PRINT_STATUS>1) std::cout<<"reading energy\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strpbrk(input,"#")!=NULL) continue;//skip any lines with "#"
			if(string::empty(input)) continue;//skip any empty lines
			if(ts==sim.timesteps()) break;
			std::strtok(input,string::WS);
			std::strtok(NULL,string::WS);
			std::strtok(NULL,string::WS);
			std::strtok(NULL,string::WS);
			std::strtok(NULL,string::WS);
			sim.frame(ts++).energy()=s_energy*std::atof(std::strtok(NULL,string::WS));
			for(unsigned int tt=1; tt<sim.stride(); ++tt) fgets(input,string::M,reader);
		}
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
}

}

//*****************************************************
//OUT format
//*****************************************************

namespace OUT{
	
void read(const char* file, const AtomType& atomT, Structure& struc){
	const char* func_name="read(FILE*,const AtomType&,Structure&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	/*local variables*/
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
	//structure
		unsigned int natomst=0;
		std::vector<unsigned int> natoms;
		unsigned int nspecies=0;
		std::vector<std::string> species;
	//cell
		double alat=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//format strings
		const char* str_energy="!    total energy";
		const char* str_lv="crystal axes";
		const char* str_alat="lattice parameter";
		const char* str_posn="site n.";
		const char* str_force="Forces";
		const char* str_stress="total   stress";
		const char* str_natoms="number of atoms/cell";
		const char* str_nspecies="number of atomic types";
		const char* str_species="atomic species  ";
	//units
		double s_posn=0.0,s_energy=0.0;
		if(units::consts::system()==units::System::AU){
			s_posn=1.0;
			s_energy=1.0;
		}
		else if(units::consts::system()==units::System::METAL){
			s_posn=units::ANGpBOHR;
			s_energy=0.5*units::EVpHARTREE;//QE energy: Rydberg
		}
		else throw std::runtime_error("Invalid units.");
	//misc
		bool error=false;
	
	try{
		//open the file
		if(QE_PRINT_STATUS>0) std::cout<<"opening the file: "<<file<<"\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("ERROR: Could not open file: \"")+std::string(file)+std::string("\"\n"));
		
		if(QE_PRINT_STATUS>0) std::cout<<"reading simulation info\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,str_energy)!=NULL){
				struc.energy()=s_energy*std::atof(string::trim_right(string::trim_left(input,"="),"Ry"));
			} else if(std::strstr(input,str_lv)!=NULL){
				std::vector<std::string> strlist;
				fgets(input,string::M,reader); string::replace(input,')',' '); string::replace(input,'(',' ');
				string::split(input,string::WS,strlist);
				lv(0,0)=std::atof(strlist.at(3).c_str());
				lv(1,0)=std::atof(strlist.at(4).c_str());
				lv(2,0)=std::atof(strlist.at(5).c_str());
				fgets(input,string::M,reader); string::replace(input,')',' '); string::replace(input,'(',' ');
				string::split(input,string::WS,strlist);
				lv(0,1)=std::atof(strlist.at(3).c_str());
				lv(1,1)=std::atof(strlist.at(4).c_str());
				lv(2,1)=std::atof(strlist.at(5).c_str());
				fgets(input,string::M,reader); string::replace(input,')',' '); string::replace(input,'(',' ');
				string::split(input,string::WS,strlist);
				lv(0,2)=std::atof(strlist.at(3).c_str());
				lv(1,2)=std::atof(strlist.at(4).c_str());
				lv(2,2)=std::atof(strlist.at(5).c_str());
			} else if(std::strstr(input,str_alat)!=NULL){
				alat=s_posn*std::atof(string::trim_right(string::trim_left(input,"="),"a"));
			} else if(std::strstr(input,str_natoms)!=NULL){
				natomst=std::atoi(string::trim_left(input,"="));
			} else if(std::strstr(input,str_nspecies)!=NULL){
				nspecies=std::atoi(string::trim_left(input,"="));
				natoms.resize(nspecies,0);
			} else if(std::strstr(input,str_species)!=NULL){
				for(unsigned int i=0; i<nspecies; ++i){
					fgets(input,string::M,reader);
					species.push_back(std::string(std::strtok(input,string::WS)));
				}
			} else if(std::strstr(input,str_posn)!=NULL){
				std::vector<std::string> strlist;
				for(unsigned int i=0; i<natomst; ++i){
					fgets(input,string::M,reader);
					string::split(input,string::WS,strlist);
					std::string name=strlist.at(1);
					for(unsigned int j=0; j<species.size(); ++j){
						if(name==species[j]){++natoms[j];break;}
					}
				}
			}
		}
		
		//print parameters
		if(QE_PRINT_STATUS>0){
			std::cout<<"ATOM    = "<<atomT<<"\n";
			std::cout<<"NATOMST = "<<natomst<<"\n";
			std::cout<<"SPECIES = "; for(unsigned int i=0; i<species.size(); ++i) std::cout<<species[i]<<" "; std::cout<<"\n";
			std::cout<<"NATOMS  = "; for(unsigned int i=0; i<natoms.size(); ++i) std::cout<<natoms[i]<<" "; std::cout<<"\n";
			std::cout<<"ENERGY  = "<<struc.energy()<<"\n";
			std::cout<<"ALAT    = "<<alat<<"\n";
			std::cout<<"LV      = \n"<<lv<<"\n";
		}
		
		//check the parameters
		if(nspecies<=0) throw std::runtime_error("ERROR reading number of species.");
		if(natomst<=0) throw std::runtime_error("ERROR reading number of atoms.");
		if(species.size()!=nspecies) throw std::runtime_error("ERROR reading species names.");
		if(alat<=0) throw std::runtime_error("ERROR reading alat.");
		if(lv.norm()==0) throw std::runtime_error("ERROR reading alat.");
		if(std::fabs(lv.determinant())<num_const::ZERO) throw std::runtime_error("Invalid lattice vector matrix.");
		
		//resize the structure
		if(QE_PRINT_STATUS>0) std::cout<<"resizing structure\n";
		struc.resize(natoms,species,atomT);
		static_cast<Cell&>(struc).init(alat*lv);
		
		//read in stress
		#ifdef INCLUDE_VIRIAL
		if(QE_PRINT_STATUS>0) std::cout<<"reading virial\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,str_stress)!=NULL){
				std::vector<std::string> strlist;
				const double scale=s_energy/(s_posn*s_posn*s_posn);
				fgets(input,string::M,reader);
				string::split(input,string::WS,strlist);
				struc.virial()(0,0)=std::atof(strlist[0].c_str())*scale;
				struc.virial()(0,1)=std::atof(strlist[1].c_str())*scale;
				struc.virial()(0,2)=std::atof(strlist[2].c_str())*scale;
				fgets(input,string::M,reader);
				string::split(input,string::WS,strlist);
				struc.virial()(1,0)=std::atof(strlist[0].c_str())*scale;
				struc.virial()(1,1)=std::atof(strlist[1].c_str())*scale;
				struc.virial()(1,2)=std::atof(strlist[2].c_str())*scale;
				fgets(input,string::M,reader);
				string::split(input,string::WS,strlist);
				struc.virial()(2,0)=std::atof(strlist[0].c_str())*scale;
				struc.virial()(2,1)=std::atof(strlist[1].c_str())*scale;
				struc.virial()(2,2)=std::atof(strlist[2].c_str())*scale;
			}
		}
		#endif
		
		//read positions
		if(atomT.posn){
			if(QE_PRINT_STATUS>0) std::cout<<"reading positions\n";
			std::rewind(reader);
			while(fgets(input,string::M,reader)!=NULL){
				if(std::strstr(input,str_posn)!=NULL){
					std::vector<std::string> strlist;
					for(unsigned int i=0; i<natoms.size(); ++i) natoms[i]=0;
					for(unsigned int i=0; i<natomst; ++i){
						fgets(input,string::M,reader);
						string::replace(input,'(',' ');
						string::replace(input,')',' ');
						string::split(input,string::WS,strlist);
						std::string name=strlist.at(1);
						int si=-1;
						for(int j=species.size()-1; j>=0; --j){
							if(name==species[j]){si=j;break;}
						}
						if(si<0) throw std::runtime_error("Invalid species.");
						struc.posn(si,natoms[si])[0]=std::atof(strlist[5].c_str());
						struc.posn(si,natoms[si])[1]=std::atof(strlist[6].c_str());
						struc.posn(si,natoms[si])[2]=std::atof(strlist[7].c_str());
						++natoms[si];
					}
				}
			}
			//convert to cartesian coordinates
			for(unsigned int i=0; i<natomst; ++i) struc.posn(i)*=alat;
		}
		
		//read in forces
		if(atomT.force){
			if(QE_PRINT_STATUS>0) std::cout<<"reading forces\n";
			std::rewind(reader);
			while(fgets(input,string::M,reader)!=NULL){
				if(std::strstr(input,str_force)!=NULL){
					std::vector<std::string> strlist;
					fgets(input,string::M,reader);
					const double fac=s_energy/s_posn;
					for(unsigned int i=0; i<natomst; ++i){
						fgets(input,string::M,reader);
						string::split(input,string::WS,strlist);
						unsigned int atom=std::atoi(strlist.at(1).c_str())-1;
						const unsigned int type=std::atoi(strlist.at(3).c_str())-1;
						for(unsigned int j=0; j<type; ++j) atom-=struc.nAtoms(j);
						struc.force(type,atom)[0]=std::atof(strlist.at(6).c_str())*fac;
						struc.force(type,atom)[1]=std::atof(strlist.at(7).c_str())*fac;
						struc.force(type,atom)[2]=std::atof(strlist.at(8).c_str())*fac;
					}
				}
			}
		}
		
		//close the file
		fclose(reader);
		reader=NULL;
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Exception: Could not read data.");
}

}

Simulation& read(const Format& format, const Interval& interval, const AtomType& atomT, Simulation& sim){
	const char* func_name="read(const Format&,const Interval&,const AtomType&,Simulation&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<func_name<<":\n";
	/* local function variables */
	//file i/o
		FILE* reader=NULL;
	//simulation parameters
		Cell cell;
		unsigned int ts=0;
		int tsint=0;
		std::vector<std::string> species;
		std::vector<unsigned int> atomNumbers;
	//timing
		clock_t start,stop;
		double time;
	//misc
		bool error=false;
	
	try{		
		//start the timer
		start=std::clock();
		
		if(!format.fileOut.empty()){
			
			Structure struc;
			OUT::read(format.fileOut.c_str(),atomT,struc);
			sim.resize(1,struc.nAtomsV(),struc.species(),atomT);
			sim.frame(0)=struc;
			sim.timesteps()=1;
			sim.cell_fixed()=true;
			
		} else {
		
			if(format.fileIn.empty() && format.fileCel.empty()) throw std::runtime_error("No cell information included.");
			if(format.filePos.empty()) throw std::runtime_error("No position information included.");
			
			if(!format.fileIn.empty()){
				//open the "in" file
				reader=fopen(format.fileIn.c_str(),"r");
				if(reader==NULL) throw std::runtime_error("I/O Error: Could not open \"in\" file.");
				//read the cell
				IN::read_cell(reader,cell);
				//read the atom info
				IN::load_atoms(reader,species,atomNumbers);
				//read the timestep
				sim.timestep()=IN::load_timestep(reader);
				//close the "in" file
				fclose(reader); reader=NULL;
			}
			
			if(!format.filePos.empty()){
				//open the "pos" file
				reader=fopen(format.filePos.c_str(),"r");
				if(reader==NULL) throw std::runtime_error("I/O Error: Could not open \"pos\" file.");
				//read the number of timesteps
				ts=POS::load_timesteps(reader);
				//close the "pos" file
				fclose(reader); reader=NULL;
			} else throw std::runtime_error("Found no POS file.");
			
			//set the interval
			if(interval.beg<0) throw std::invalid_argument("Invalid beginning timestep.");
			sim.beg()=interval.beg-1;
			if(interval.end<0){
				sim.end()=ts+interval.end;
			} else {
				if(interval.end>ts) throw std::invalid_argument("Invalid ending timestep.");
				sim.end()=interval.end-1;
			}
			tsint=sim.end()-sim.beg()+1;
			
			//resize the simulation
			sim.resize(tsint/interval.stride,atomNumbers,species,atomT);
			sim.stride()=interval.stride;
			if(QE_PRINT_STATUS>0) std::cout<<"interval = "<<interval.beg<<":"<<interval.end<<":"<<interval.stride<<"\n";
			if(QE_PRINT_STATUS>0) std::cout<<"SIM = \n"<<sim<<"\n";
			
			if(!format.fileCel.empty()){
				//open the "cel" file
				reader=fopen(format.fileCel.c_str(),"r");
				if(reader==NULL) throw std::runtime_error("I/O Error: Could not open \"cel\" file.");
				//read the cell
				CEL::read_cell(reader,sim);
				//close the "cel" file
				fclose(reader); reader=NULL;
			}
			
			if(!format.fileEvp.empty()){
				//open the "evp" file
				reader=fopen(format.fileEvp.c_str(),"r");
				if(reader==NULL) throw std::runtime_error("I/O Error: Could not open \"evp\" file.");
				//read the energy
				EVP::read_energy(reader,sim);
				//close the "evp" file
				fclose(reader); reader=NULL;
			}
			
			if(!format.filePos.empty()){
				//open the "pos" file
				reader=fopen(format.filePos.c_str(),"r");
				if(reader==NULL) throw std::runtime_error("I/O Error: Could not open \"pos\" file.");
				//read the positions
				POS::load_posns(reader,sim);
				//close the "pos" file
				fclose(reader); reader=NULL;
			}
			
			if(!format.fileFor.empty()){
				//open the "for" file
				reader=fopen(format.fileFor.c_str(),"r");
				if(reader==NULL) throw std::runtime_error("I/O Error: Could not open \"for\" file.");
				//read the positions
				FOR::read_forces(reader,sim);
				//close the "for" file
				fclose(reader); reader=NULL;
			}
			
		}
		
		//stop the timer
		stop=std::clock();
		
		//print the time
		time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"Simulation loaded in "<<time<<" seconds.\n";
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
	else return sim;
}

}