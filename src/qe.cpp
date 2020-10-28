// c libraries
#include <ctime>
// c++ libraries
#include <iostream>
#include <string>
#include <stdexcept>
// eigen libraries
#include <Eigen/Dense>
// ann - ptable
#include "ptable.hpp"
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
	for(int i=0; i<strlist.size(); ++i){
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
//OUT format
//*****************************************************

namespace OUT{
	
void read(const char* file, const AtomType& atomT, Structure& struc){
	const char* func_name="read(FILE*,const AtomType&,Structure&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<func_name<<":\n";
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
	//structure
		int natomst=0;
		std::vector<int> natoms;
		int nspecies=0;
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
				for(int i=0; i<nspecies; ++i){
					fgets(input,string::M,reader);
					species.push_back(std::string(std::strtok(input,string::WS)));
				}
			} else if(std::strstr(input,str_posn)!=NULL){
				std::vector<std::string> strlist;
				for(int i=0; i<natomst; ++i){
					fgets(input,string::M,reader);
					string::split(input,string::WS,strlist);
					std::string name=strlist.at(1);
					for(int j=0; j<species.size(); ++j){
						if(name==species[j]){++natoms[j];break;}
					}
				}
			}
		}
		
		//print parameters
		if(QE_PRINT_STATUS>0){
			std::cout<<"ATOM    = "<<atomT<<"\n";
			std::cout<<"NATOMST = "<<natomst<<"\n";
			std::cout<<"SPECIES = "; for(int i=0; i<species.size(); ++i) std::cout<<species[i]<<" "; std::cout<<"\n";
			std::cout<<"NATOMS  = "; for(int i=0; i<natoms.size(); ++i) std::cout<<natoms[i]<<" "; std::cout<<"\n";
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
		if(std::fabs(lv.determinant())<math::constant::ZERO) throw std::runtime_error("Invalid lattice vector matrix.");
		
		//resize the structure
		if(QE_PRINT_STATUS>0) std::cout<<"resizing structure\n";
		struc.resize(natomst,atomT);
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
		
		//read atoms
		if(atomT.posn){
			if(QE_PRINT_STATUS>0) std::cout<<"reading atoms\n";
			std::rewind(reader);
			while(fgets(input,string::M,reader)!=NULL){
				if(std::strstr(input,str_posn)!=NULL){
					std::vector<std::string> strlist;
					for(int i=0; i<natomst; ++i){
						//read line
						fgets(input,string::M,reader);
						string::replace(input,'(',' ');
						string::replace(input,')',' ');
						string::split(input,string::WS,strlist);
						//read atom properties
						std::string name=strlist.at(1);
						const int an=ptable::an(name.c_str());
						Eigen::Vector3d posn;
						posn[0]=std::atof(strlist[5].c_str());
						posn[1]=std::atof(strlist[6].c_str());
						posn[2]=std::atof(strlist[7].c_str());
						//set data
						if(struc.atomType().name) struc.name(i)=name;
						if(struc.atomType().an) struc.an(i)=an;
						if(struc.atomType().mass) struc.mass(i)=ptable::mass(an);
						if(struc.atomType().posn) struc.posn(i)=posn;
					}
				}
			}
			//convert to cartesian coordinates
			for(int i=0; i<natomst; ++i) struc.posn(i)*=alat;
			//return to cell
			for(int i=0; i<natomst; ++i) Cell::returnToCell(struc.posn(i),struc.posn(i),struc.R(),struc.RInv());
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
					for(int i=0; i<natomst; ++i){
						fgets(input,string::M,reader);
						string::split(input,string::WS,strlist);
						Eigen::Vector3d force;
						force[0]=std::atof(strlist.at(6).c_str())*fac;
						force[1]=std::atof(strlist.at(7).c_str())*fac;
						force[2]=std::atof(strlist.at(8).c_str())*fac;
						if(struc.atomType().force) struc.force(i)=force;
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
	/*const char* func_name="read(const Format&,const Interval&,const AtomType&,Simulation&)";
	if(QE_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<func_name<<":\n";
	//file i/o
		FILE* reader=NULL;
	//simulation parameters
		Cell cell;
		int ts=0;
		int tsint=0;
		std::vector<std::string> species;
		std::vector<int> atomNumbers;
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
				IN::read_atoms(reader,species,atomNumbers);
				//read the timestep
				sim.timestep()=IN::read_timestep(reader);
				//close the "in" file
				fclose(reader); reader=NULL;
			}
			
			if(!format.filePos.empty()){
				//open the "pos" file
				reader=fopen(format.filePos.c_str(),"r");
				if(reader==NULL) throw std::runtime_error("I/O Error: Could not open \"pos\" file.");
				//read the number of timesteps
				ts=POS::read_timesteps(reader);
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
				POS::read_posns(reader,sim);
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
	else return sim;*/
}

}