// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <iostream>
#include <string>
#include <stdexcept>
// eigen libraries
#include <Eigen/Dense>
// ann - structure
#include "structure.hpp"
#include "sim.hpp"
// ann - strings
#include "string.hpp"
// ann - units
#include "units.hpp"
// ann - chemistry
#include "ptable.hpp"
// ann - vasp
#include "vasp.hpp"

namespace VASP{

//*****************************************************
//FORMAT struct
//*****************************************************

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-xdatcar"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-xdatcar\" option.");
			else format.xdatcar=strlist[i+1];
		} else if(strlist[i]=="-poscar"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-poscar\" option.");
			else format.poscar=strlist[i+1];
		} else if(strlist[i]=="-xml"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-poscar\" option.");
			else format.xml=strlist[i+1];
		} else if(strlist[i]=="-energy"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-energy\" option.");
			else format.energy=strlist[i+1];
		}
	}
	return format;
}

//*****************************************************
//POSCAR
//*****************************************************

namespace POSCAR{

void read(const char* file, const AtomType& atomT, Structure& struc){
	const char* funcName="read(const char*,const AtomType&,Structure&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<funcName<<":\n";
	//====  local function variables ==== 
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* str_name=new char[string::M];
		char* str_number=new char[string::M];
	//simulation flags
		bool direct;//whether the coordinates are in direct or Cartesian coordinates
	//cell
		double scale=0.0;
		Cell cell;
		Eigen::Matrix3d lv;
	//atom info
		int nAtomsT=0;
		std::vector<int> nAtoms;//the number of atoms in each species
		std::vector<std::string> names;//the names of each species
	//units
		double s_len=0.0;
		if(units::consts::system()==units::System::AU) s_len=units::BOHRpANG;
		else if(units::consts::system()==units::System::METAL) s_len=1.0;
		else throw std::runtime_error("Invalid units.");
	//misc
		bool error=false;
		
	try{
		//==== open file, clear simulation ====
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: ")+file);
		struc.clear();
		
		//==== read header ====
		if(DEBUG_VASP>1) std::cout<<"read lattice vectors\n";
		fgets(input,string::M,reader);//name
		std::sscanf(fgets(input,string::M,reader),"%lf",&scale);
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,0),&lv(1,0),&lv(2,0));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,1),&lv(1,1),&lv(2,1));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,2),&lv(1,2),&lv(2,2));
		lv*=s_len*scale;
		static_cast<Cell&>(struc).init(lv);
		
		//==== read species ====
		if(DEBUG_VASP>1) std::cout<<"read species\n";
		//read number of species
		fgets(str_name, string::M, reader);
		fgets(str_number, string::M, reader);
		const int nNames=string::substrN(str_name,string::WS);
		const int nNumbers=string::substrN(str_number,string::WS);
		const int nSpecies=string::substrN(str_name,string::WS);
		if(nNames!=nNumbers) throw std::runtime_error("Invalid number of species");
		if(nSpecies<=0) throw std::runtime_error("Invalid number of species");
		//read in the species names
		names.resize(nSpecies);
		names[0]=std::strtok(str_name,string::WS);
		for(int i=1; i<nSpecies; ++i) names[i]=std::strtok(NULL,string::WS);
		//read in the species numbers
		nAtoms.resize(nSpecies);
		nAtoms[0]=std::atoi(std::strtok(str_number,string::WS));
		for(int i=1; i<nSpecies; ++i) nAtoms[i]=std::atoi(std::strtok(NULL,string::WS));
		//compute the total number
		nAtomsT=0;
		for(int i=0; i<nSpecies; ++i) nAtomsT+=nAtoms[i];
		
		//==== read coord ====
		if(DEBUG_VASP>1) std::cout<<"read coord\n";
		fgets(input, string::M, reader);
		if(input[0]=='D') direct=true;
		else direct=false;
		
		//====  print data to screen ==== 
		if(DEBUG_VASP>1){
			std::cout<<"CELL    = \n"<<static_cast<Cell&>(struc)<<"\n";
			std::cout<<"DIRECT  = "<<(direct?"T":"F")<<"\n";
			std::cout<<"SPECIES = "; for(int i=0; i<names.size(); ++i) std::cout<<names[i]<<" "; std::cout<<"\n";
			std::cout<<"NUMBERS = "; for(int i=0; i<nAtoms.size(); ++i) std::cout<<nAtoms[i]<<" "; std::cout<<"\n";
		}
		
		//====  resize the simulation ==== 
		if(DEBUG_VASP>0) std::cout<<"allocating memory\n";
		AtomType atomTl=atomT;
		atomTl.frac=direct;
		struc.resize(nAtomsT,atomTl);
		
		//====  read positions ==== 
		if(DEBUG_VASP>1) std::cout<<"reading positions\n";
		for(int n=0; n<struc.nAtoms(); ++n){
			std::sscanf(
				fgets(input,string::M,reader),"%lf %lf %lf",
				&struc.posn(n)[0],&struc.posn(n)[1],&struc.posn(n)[2]
			);
		}
		
		//====  convert to cartesian coordinates (if necessary) ==== 
		if(DEBUG_VASP>1) std::cout<<"Converting to Cartesian coordinates\n";
		if(direct){
			for(int n=0; n<struc.nAtoms(); ++n){
				struc.posn(n)=struc.R()*struc.posn(n);
			}
		} else {
			for(int n=0; n<struc.nAtoms(); ++n){
				struc.posn(n)*=s_len;
			}
		}
		
		//==== set species ====
		if(atomT.name){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.name(count++)=names[n];
				}
			}
		}
		if(atomT.type){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.type(count++)=n;
				}
			}
		}
		if(atomT.an && atomT.name){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.an(count)=ptable::an(struc.name(count).c_str());
					++count;
				}
			}
		}
		if(atomT.mass && atomT.an){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.mass(count)=ptable::mass(struc.an(count));
					++count;
				}
			}
		} else if(atomT.mass && atomT.name){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					const int an=ptable::an(struc.name(count).c_str());
					struc.mass(count)=ptable::mass(struc.mass(an));
					++count;
				}
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
	delete[] input;
		
	if(error) throw std::runtime_error("I/O Exception: Could not read data.");
}

void write(const char* file, const AtomType& atomT, const Structure& struc){
	static const char* funcName="read<AtomT>(const char*,Structure&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//====  local function variables ==== 
	//file i/o
		FILE* writer=NULL;
	//misc
		bool error=false;
	
	try{
		//open the file
		if(DEBUG_VASP>1) std::cout<<"opening file\n";
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: ")+std::string(file));
		
		//find species names and numbers
		std::vector<std::string> names;
		std::vector<int> nAtoms;
		for(int i=0; i<struc.nAtoms(); ++i){
			int index=-1;
			for(int n=0; n<names.size(); ++n){
				if(struc.name(i)==names[n]){index=n;break;}
			}
			if(index<0){
				names.push_back(struc.name(i));
				nAtoms.push_back(1);
			} else ++nAtoms[index];
		}
		
		//writer header
		if(DEBUG_VASP>1) std::cout<<"writing header\n";
		fprintf(writer,"system\n");
		fprintf(writer,"%10.4f\n",1.0);
		for(int i=0; i<3; ++i){
			for(int j=0; j<3; ++j){
				fprintf(writer,"%11.6f ",struc.R()(j,i));
			}
			fprintf(writer,"\n");
		}
		for(int i=0; i<names.size(); ++i) fprintf(writer,"%s ",names[i].c_str()); fprintf(writer,"\n");
		for(int i=0; i<nAtoms.size(); ++i) fprintf(writer,"%i ",nAtoms[i]); fprintf(writer,"\n");
		
		//write positions
		if(DEBUG_VASP>1) std::cout<<"writing posns\n";
		if(!atomT.frac){
			fprintf(writer,"Cart\n");
			for(int n=0; n<struc.nAtoms(); ++n){
				fprintf(writer, "%.8f %.8f %.8f\n",
					struc.posn(n)[0],
					struc.posn(n)[1],
					struc.posn(n)[2]
				);
			}
		} else {
			fprintf(writer,"Direct\n");
			for(int n=0; n<struc.nAtoms(); ++n){
				Eigen::Vector3d posn=struc.RInv()*struc.posn(n);
				fprintf(writer, "%.8f %.8f %.8f\n",posn[0],posn[1],posn[2]);
			}
		}
		
		if(DEBUG_VASP>1) std::cout<<"closing file\n";
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	if(writer!=NULL) fclose(writer);
	
	if(error) throw std::runtime_error("I/O Exception: Could not write data.");
}

}

//*****************************************************
//XDATCAR
//*****************************************************

namespace XDATCAR{

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="read(const char*,const Interval&,Simulation&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* str_name=new char[string::M];
		char* str_number=new char[string::M];
		std::string str;
	//simulation flags
		bool direct;//whether the coordinates are in direct or Cartesian coordinates
	//time info
		int ts=0;//number of timesteps
		int tsint=0;//requested interval
	//cell info
		double scale=1;
		Eigen::Matrix3d lv;
	//atom info
		int nAtomsT=0;
		std::vector<int> nAtoms;//the number of atoms in each species
		std::vector<std::string> names;//the names of each species
	//units
		double s_len=0.0;
		if(units::consts::system()==units::System::AU) s_len=units::BOHRpANG;
		else if(units::consts::system()==units::System::METAL) s_len=1.0;
		else throw std::runtime_error("Invalid units.");
	//misc
		bool error=false;
		
	try{
		//start the timer
		const clock_t start=std::clock();
		
		//open the file
		if(DEBUG_VASP>0) std::cout<<"Opening file\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Unable to open file.");
		
		//==== clear the simulation ====
		if(DEBUG_VASP>0) std::cout<<"clearing simulation\n";
		sim.clear();
		
		//==== read in the system name ====
		if(DEBUG_VASP>0) std::cout<<"reading system name\n";
		sim.name()=string::trim(fgets(input,string::M,reader));
		
		//==== read the simulation cell ====
		if(DEBUG_VASP>0) std::cout<<"reading cell\n";
		std::sscanf(fgets(input,string::M,reader),"%lf",&scale);
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,0),&lv(1,0),&lv(2,0));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,1),&lv(1,1),&lv(2,1));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,2),&lv(1,2),&lv(2,2));
		lv*=s_len*scale;
		
		//==== read species ====
		if(DEBUG_VASP>1) std::cout<<"read species\n";
		//read number of species
		fgets(str_name, string::M, reader);
		fgets(str_number, string::M, reader);
		const int nNames=string::substrN(str_name,string::WS);
		const int nNumbers=string::substrN(str_number,string::WS);
		const int nSpecies=string::substrN(str_name,string::WS);
		if(nNames!=nNumbers) throw std::runtime_error("Invalid number of species");
		if(nSpecies<=0) throw std::runtime_error("Invalid number of species");
		//read in the species names
		names.resize(nSpecies);
		names[0]=std::strtok(str_name,string::WS);
		for(int i=1; i<nSpecies; ++i) names[i]=std::strtok(NULL,string::WS);
		//read in the species numbers
		nAtoms.resize(nSpecies);
		nAtoms[0]=std::atoi(std::strtok(str_number,string::WS));
		for(int i=1; i<nSpecies; ++i) nAtoms[i]=std::atoi(std::strtok(NULL,string::WS));
		//compute the total number
		nAtomsT=0;
		for(int i=0; i<nSpecies; ++i) nAtomsT+=nAtoms[i];
		
		//==== read coord ====
		if(DEBUG_VASP>0) std::cout<<"read coord\n";
		fgets(input, string::M, reader);
		if(input[0]=='D') direct=true;
		else direct=false;
		
		//==== check if the cell is variable or not ====
		if(DEBUG_VASP>0) std::cout<<"Checking whether cell is variable\n";
		for(int n=0; n<nAtomsT; ++n) fgets(input, string::M, reader);
		str=std::string(string::trim(fgets(input,string::M,reader)));
		if(str==sim.name()) sim.cell_fixed()=false;
		else sim.cell_fixed()=true;
		// reset the line position 
		if(!sim.cell_fixed()) for(int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader);
		
		//==== find the number of timesteps ====
		if(DEBUG_VASP>0) std::cout<<"reading the number of timesteps\n";
		std::rewind(reader);
		//find the total number of lines in the file
		int nLines=0;
		while(fgets(input, string::M, reader)!=NULL){++nLines;};
		if(sim.cell_fixed()) ts=std::floor((1.0*nLines-HEADER_SIZE)/(1.0*nAtomsT+1.0));
		else ts=std::floor((1.0*nLines)/(1.0*nAtomsT+1.0+HEADER_SIZE));
		
		//==== reset the line position ====
		if(DEBUG_VASP>0) std::cout<<"resetting the line position\n";
		std::rewind(reader);
		if(sim.cell_fixed()) for(int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader);
		
		//==== set the interval ====
		if(DEBUG_VASP>0) std::cout<<"setting the interval\n";
		if(interval.beg<0) throw std::invalid_argument("Invalid beginning timestep.");
		sim.beg()=interval.beg-1;
		if(interval.end<0){
			sim.end()=ts+interval.end;
		} else sim.end()=interval.end-1;
		tsint=sim.end()-sim.beg()+1;
		
		//==== print data to screen ====
		if(DEBUG_VASP>0){
			std::cout<<"NAME    = "<<sim.name()<<"\n";
			std::cout<<"ATOMT   = "<<atomT<<"\n";
			std::cout<<"DIRECT  = "<<(direct?"T":"F")<<"\n";
			std::cout<<"CELL    = \n"<<lv<<"\n";
			std::cout<<"SPECIES = "; for(int i=0; i<nSpecies; ++i) std::cout<<names[i]<<" "; std::cout<<"\n";
			std::cout<<"NUMBERS = "; for(int i=0; i<nSpecies; ++i) std::cout<<nAtoms[i]<<" "; std::cout<<"\n";
			std::cout<<"NATOMST = "<<nAtomsT<<"\n";
			std::cout<<"INTERVAL   = "<<sim.beg()<<":"<<sim.end()<<":"<<interval.stride<<" - "<<tsint<<"\n";
			std::cout<<"TIMESTEPS  = "<<ts<<"\n";
			std::cout<<"N_STEPS    = "<<tsint/interval.stride<<"\n";
		}
		
		//==== resize the simulation ====
		if(DEBUG_VASP>0) std::cout<<"allocating memory\n";
		sim.resize(tsint/interval.stride,nAtomsT,atomT);
		
		//==== read positions ====
		if(DEBUG_VASP>0) std::cout<<"reading positions\n";
		//skip timesteps until beg is reached
		for(int t=0; t<sim.beg(); ++t){
			if(!sim.cell_fixed()) for(int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader); //skip header
			fgets(input,string::M,reader);//skip single line
			for(int n=0; n<nAtomsT; ++n) fgets(input,string::M,reader);
		}
		//read the positions
		if(sim.cell_fixed()){
			for(int t=0; t<sim.timesteps(); ++t) static_cast<Cell&>(sim.frame(t)).init(lv);
			for(int t=0; t<sim.timesteps(); ++t){
				if(DEBUG_VASP>1) std::cout<<"T = "<<t<<"\n";
				else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
				fgets(input,string::M,reader);//skip line
				for(int n=0; n<nAtomsT; ++n){
					std::sscanf(
						fgets(input,string::M,reader),"%lf %lf %lf",
						&sim.frame(t).posn(n)[0],&sim.frame(t).posn(n)[1],&sim.frame(t).posn(n)[2]
					);
				}
				//skip "stride-1" steps
				for(int tt=0; tt<interval.stride-1; ++tt){
					fgets(input,string::M,reader);//skip line
					for(int n=0; n<nAtomsT; ++n){
						fgets(input,string::M,reader);
					}
				}
			}
		} else {
			for(int t=0; t<sim.timesteps(); ++t){
				if(DEBUG_VASP>1) std::cout<<"T = "<<t<<"\n";
				else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
				//read in lattice vectors
				fgets(input,string::M,reader);//name
				scale=std::atof(fgets(input,string::M,reader));//scale
				for(int i=0; i<3; ++i){
					fgets(input, string::M, reader);
					lv(0,i)=std::atof(std::strtok(input,string::WS));
					for(int j=1; j<3; ++j){
						lv(j,i)=std::atof(std::strtok(NULL,string::WS));
					}
				}
				static_cast<Cell&>(sim.frame(t)).init(s_len*scale*lv);
				fgets(input,string::M,reader);//skip line (atom names)
				fgets(input,string::M,reader);//skip line (atom numbers)
				fgets(input,string::M,reader);//skip line (Direct or Cart)
				for(int n=0; n<nAtomsT; ++n){
					std::sscanf(
						fgets(input,string::M,reader),"%lf %lf %lf",
						&sim.frame(t).posn(n)[0],&sim.frame(t).posn(n)[1],&sim.frame(t).posn(n)[2]
					);
				}
				//skip "stride-1" steps
				for(int tt=0; tt<interval.stride-1; ++tt){
					fgets(input,string::M,reader);//name
					fgets(input,string::M,reader);//scale
					for(int i=0; i<3; ++i){
						fgets(input,string::M,reader);//lv
					}
					fgets(input,string::M,reader);//skip line (atom names)
					fgets(input,string::M,reader);//skip line (atom numbers)
					fgets(input,string::M,reader);//skip line (Direct or Cart)
					for(int n=0; n<nAtomsT; ++n){
						fgets(input,string::M,reader);
					}
				}
			}
		}
		
		//==== convert to Cartesian coordinates if necessary ====
		if(direct){
			if(DEBUG_VASP>0) std::cout<<"converting to cartesian coordinates\n";
			for(int t=0; t<sim.timesteps(); ++t){
				for(int n=0; n<nAtomsT; ++n){
					sim.frame(t).posn(n)=sim.frame(t).R()*sim.frame(t).posn(n);
				}
			}
		} else if(s_len!=1.0){
			for(int t=0; t<sim.timesteps(); ++t){
				for(int n=0; n<nAtomsT; ++n){
					sim.frame(t).posn(n)*=s_len;
				}
			}
		}
		
		//==== set species ====
		if(DEBUG_VASP>0) std::cout<<"setting species\n";
		if(atomT.name){
			for(int t=0; t<sim.timesteps(); ++t){
				int count=0;
				for(int n=0; n<nSpecies; ++n){
					for(int m=0; m<nAtoms[n]; ++m){
						sim.frame(t).name(count++)=names[n];
					}
				}
			}
		}
		
		//==== close the file ====
		if(DEBUG_VASP>0) std::cout<<"closing the file\n";
		fclose(reader);
		reader=NULL;
		
		//==== stop the timer ====
		const clock_t stop=std::clock();
		
		//==== print the time ====
		const double time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"Simulation loaded in "<<time<<" seconds.\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free all local variables
	if(reader!=NULL) fclose(reader);
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Exception: Could not read data.");
}

}

//*****************************************************
//XML
//*****************************************************

namespace XML{

void read(const char* file, int t, const AtomType& atomT, Structure& struc){
	const char* funcName="read(const char*,Structure&,(int))";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
	//time info
		int ts=0;//number of timesteps
	//cell info
		Eigen::Matrix3d lv;
	//atom info
		int nSpecies=0;//the number of atomic species
		std::vector<int> nAtoms;//the number of atoms in each species
		std::vector<std::string> species;//the names of each species
		int N=0;
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
		int tt;
	
	try{
		
		//==== open the file ====
		if(DEBUG_VASP>0) std::cout<<"opening the file\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: ")+file);
		
		//==== read the number of timesteps ====
		if(DEBUG_VASP>0) std::cout<<"loading the timesteps\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"<calculation>")!=NULL) ++ts;
		}
		if(DEBUG_VASP>0) std::cout<<"ts = "<<ts<<"\n";
		if(t>ts) throw std::invalid_argument("Invalid timestep");
		
		//==== read in the atom info ====
		if(DEBUG_VASP>0) std::cout<<"reading in the atom info\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"atominfo")!=NULL){
				fgets(input,string::M,reader);
				std::strtok(input,">");
				N=std::atoi(std::strtok(NULL,"<"));
				for(int i=0; i<6; ++i) fgets(input,string::M,reader);
				for(int i=0; i<N; ++i){
					fgets(input,string::M,reader);
					std::strtok(input,">"); std::strtok(NULL,">");
					std::string name=std::strtok(NULL,"<");
					bool match=false;
					for(int j=0; j<species.size(); ++j){
						if(name==species[j]){
							++nAtoms[j];
							match=true; 
							break;
						}
					}
					if(!match){
						species.push_back(name);
						nAtoms.push_back(1);
					}
				}
				break;
			}
		}
		if(DEBUG_VASP>0){
			std::cout<<"ATOM_NAMES   = "; for(int i=0; i<species.size(); ++i) std::cout<<species[i]<<" "; std::cout<<"\n";
			std::cout<<"ATOM_NUMBERS = "; for(int i=0; i<nAtoms.size(); ++i) std::cout<<nAtoms[i]<<" "; std::cout<<"\n";
		}
		if(species.size()!=nAtoms.size()) throw std::runtime_error("Mismatch in atom names/numbers.");
		nSpecies=species.size();
		int nAtomsT=0;
		for(int i=0; i<nAtoms.size(); ++i) nAtomsT+=nAtoms[i];
		
		//==== resize the simulation ====
		if(DEBUG_VASP>0) std::cout<<"resizing the simulation\n";
		struc.resize(nAtomsT,atomT);
		
		//==== read the cells ====
		if(DEBUG_VASP>0) std::cout<<"reading the cells\n";
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
						lv(0,0)=std::atof(std::strtok(NULL,string::WS))*s_posn;
						lv(1,0)=std::atof(std::strtok(NULL,string::WS))*s_posn;
						lv(2,0)=std::atof(std::strtok(NULL,string::WS))*s_posn;
						fgets(input,string::M,reader);
						std::strtok(input,string::WS);
						lv(0,1)=std::atof(std::strtok(NULL,string::WS))*s_posn;
						lv(1,1)=std::atof(std::strtok(NULL,string::WS))*s_posn;
						lv(2,1)=std::atof(std::strtok(NULL,string::WS))*s_posn;
						fgets(input,string::M,reader);
						std::strtok(input,string::WS);
						lv(0,2)=std::atof(std::strtok(NULL,string::WS))*s_posn;
						lv(1,2)=std::atof(std::strtok(NULL,string::WS))*s_posn;
						lv(2,2)=std::atof(std::strtok(NULL,string::WS))*s_posn;
						static_cast<Cell&>(struc).init(lv);
						break;
					}
				}
			}
		}
		
		//==== read in the positions ====
		if(DEBUG_VASP>0) std::cout<<"reading in positions\n";
		if(atomT.posn){
			std::rewind(reader);
			tt=0;
			while(fgets(input,string::M,reader)!=NULL){
				if(std::strstr(input,"<calculation>")!=NULL){
					while(fgets(input,string::M,reader)!=NULL){
						if(std::strstr(input,"</calculation>")!=NULL){
							throw std::invalid_argument("No positions provided in calculation.");
						} else if(std::strstr(input,"positions")!=NULL){
							if(tt<t) continue;
							for(int n=0; n<struc.nAtoms(); ++n){
								std::strtok(fgets(input,string::M,reader),string::WS);
								struc.posn(n)[0]=std::atof(std::strtok(NULL,string::WS));
								struc.posn(n)[1]=std::atof(std::strtok(NULL,string::WS));
								struc.posn(n)[2]=std::atof(std::strtok(NULL,string::WS));
							}
							break;
						}
					}
				}
			}
		}
		
		//==== read in the energies ====
		if(DEBUG_VASP>0) std::cout<<"reading in the energies\n";
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
		if(DEBUG_VASP>0) std::cout<<"Reading in forces\n";
		if(atomT.force){
			std::rewind(reader);
			tt=0;
			while(fgets(input,string::M,reader)!=NULL){
				if(std::strstr(input,"<calculation>")!=NULL){
					while(fgets(input,string::M,reader)!=NULL){
						if(std::strstr(input,"</calculation>")!=NULL){
							throw std::invalid_argument("No forces provided in calculation.");
						} else if(std::strstr(input,"forces")!=NULL){
							if(tt<t) continue;
							const double fac=s_energy/s_posn;
							for(int n=0; n<struc.nAtoms(); ++n){
								std::strtok(fgets(input,string::M,reader),string::WS);
								struc.force(n)[0]=std::atof(std::strtok(NULL,string::WS))*fac;
								struc.force(n)[1]=std::atof(std::strtok(NULL,string::WS))*fac;
								struc.force(n)[2]=std::atof(std::strtok(NULL,string::WS))*fac;
							}
							++tt;
							break;
						}
					}
				}
			}
		}
		
		//==== set species ====
		if(atomT.name){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.name(count++)=species[n];
				}
			}
		}
		if(atomT.type){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.type(count++)=n;
				}
			}
		}
		if(atomT.an && atomT.name){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.an(count)=ptable::an(struc.name(count).c_str());
					++count;
				}
			}
		}
		if(atomT.mass && atomT.an){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.mass(count)=ptable::mass(struc.an(count));
					++count;
				}
			}
		} else if(atomT.mass && atomT.name){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					const int an=ptable::an(struc.name(count).c_str());
					struc.mass(count)=ptable::mass(struc.mass(an));
					++count;
				}
			}
		}
		
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//==== transform to Cartesian coordinates ====
	if(DEBUG_VASP>0) std::cout<<"transforming to Cartesian coordinates\n";
	for(int n=0; n<struc.nAtoms(); ++n){
		struc.posn(n)=struc.R()*struc.posn(n);
		Cell::returnToCell(struc.posn(n),struc.posn(n),struc.R(),struc.RInv());
	}
	
	//==== free all local variables ====
	if(reader!=NULL) fclose(reader);
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Exception: Could not read data.");
}

}

}