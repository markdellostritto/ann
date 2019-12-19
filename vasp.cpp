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
	for(unsigned int i=0; i<strlist.size(); ++i){
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
//reading
//*****************************************************

void read_cell(FILE* reader, Cell& cell){
	if(DEBUG_VASP>0) std::cout<<"read_cell(FILE*,Cell&):\n";
	double s=0.0;
	if(units::consts::system()==units::System::AU) s=units::BOHRpANG;
	else if(units::consts::system()==units::System::METAL) s=1.0;
	else throw std::runtime_error("Invalid units.");
	char* input=new char[string::M];
	double scale=std::atof(fgets(input, string::M, reader));
	if(scale<0) scale=std::pow(-1.0*scale,1.0/3.0);
	//read in the lattice vectors of the system
	/*
		Here we take the transpose of the lattice vector matrix.
		The reason is that VASP stores the lattice vector matrix such that the lattice vectors
		are the rows of the matrix.  However, it's more appropriate for operations on coordinates
		for the lattice vectors to be columns of the matrix.
	*/
	if(DEBUG_VASP>1) std::cout<<"Reading in the lattice vectors\n";
	Eigen::Matrix3d lv;
	for(unsigned int i=0; i<3; ++i){
		fgets(input, string::M, reader);
		lv(0,i)=std::atof(std::strtok(input,string::WS));
		for(unsigned int j=1; j<3; ++j){
			lv(j,i)=std::atof(std::strtok(NULL,string::WS));
		}
	}
	//initialize the cell
	cell.init(s*lv,scale);
	delete[] input;
}

void read_atoms(FILE* reader, std::vector<std::string>& names, std::vector<unsigned int>& natoms){
	if(DEBUG_VASP>0) std::cout<<"read_atoms(FILE*,std::vector<std::string>&,std::vector<unsigned int>&):\n";
	char* input=new char[string::M];
	//read in the number of species
	if(DEBUG_VASP>1) std::cout<<"Reading in the number of species\n";
	fgets(input, string::M, reader);
	const unsigned int nSpecies=string::substrN(input,string::WS);
	names.resize(nSpecies);
	natoms.resize(nSpecies);
	//read in the species names
	if(DEBUG_VASP>1) std::cout<<"Reading in the species names\n";
	names[0]=std::strtok(input,string::WS);
	for(unsigned int i=1; i<nSpecies; ++i){
		names[i]=std::strtok(NULL,string::WS);
	}
	//read in the species numbers
	if(DEBUG_VASP>1) std::cout<<"Reading in the species numbers\n";
	fgets(input, string::M, reader);
	natoms[0]=std::atoi(std::strtok(input,string::WS));
	for(unsigned int i=1; i<nSpecies; ++i){
		natoms[i]=std::atoi(std::strtok(NULL,string::WS));
	}
	delete[] input;
}

bool read_coord(FILE* reader){
	if(DEBUG_VASP>0) std::cout<<"read_coord(FILE*):\n";
	char* input=new char[string::M];
	bool direct=true;
	fgets(input, string::M, reader);
	if(input[0]=='D') direct=true;
	else direct=false;
	delete[] input;
	return direct;
}

//*****************************************************
//writing
//*****************************************************

void write_name(FILE* writer, const char* name){
	if(DEBUG_VASP>0) std::cout<<"write_name(FILE*,const char*):\n";
	fprintf(writer,"%s\n",name);
}

void write_cell(FILE* writer, const Cell& cell){
	if(DEBUG_VASP>0) std::cout<<"write_cell(FILE*,const Cell&):\n";
	fprintf(writer," %10.4f\n",cell.scale());
	for(unsigned int i=0; i<3; ++i){
		for(unsigned int j=0; j<3; ++j){
			fprintf(writer," %11.6f",cell.R()(j,i)/cell.scale());
		}
		fprintf(writer,"\n");
	}
}

void write_atoms(FILE* writer, const std::vector<std::string>& names, const std::vector<unsigned int>& natoms){
	if(DEBUG_VASP>0) std::cout<<"write_atoms(FILE*,const std::vector<std::string>&,const std::vector<unsigned int>&):\n";
	for(unsigned int i=0; i<names.size(); ++i){
		fprintf(writer," %4s",names[i].c_str());
	}
	fprintf(writer,"\n");
	for(unsigned int i=0; i<natoms.size(); ++i){
		fprintf(writer," %4i",natoms[i]);
	}
	fprintf(writer,"\n");
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
	//simulation flags
		bool direct;//whether the coordinates are in direct or Cartesian coordinates
	//cell
		Cell cell;
		Eigen::Matrix3d lv;
	//atom info
		std::vector<unsigned int> natoms;//the number of atoms in each species
		std::vector<std::string> names;//the names of each species
	//units
		double s=0.0;
		if(units::consts::system()==units::System::AU) s=units::BOHRpANG;
		else if(units::consts::system()==units::System::METAL) s=1.0;
		else throw std::runtime_error("Invalid units.");
	//misc
		bool error=false;
		
	try{
		//open the file
		if(DEBUG_VASP>1) std::cout<<"opening file\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: ")+file);
		
		//clear the simulation
		struc.clear();
		
		//====  read in the system name ==== 
		fgets(input,string::M,reader);
		std::string name=input;
		
		//====  read the simulation cell ==== 
		read_cell(reader,static_cast<Cell&>(struc));
		
		//====  read the atom info ==== 
		read_atoms(reader,names,natoms);
		
		//====  read the coordinate type ==== 
		direct=read_coord(reader);
		
		//====  print data to screen ==== 
		if(DEBUG_VASP>1){
			std::cout<<"NAME    = "<<name<<"\n";
			std::cout<<"DIRECT  = "<<(direct?"T":"F")<<"\n";
			std::cout<<"CELL    = \n"<<static_cast<Cell&>(struc)<<"\n";
			std::cout<<"SPECIES = "; for(unsigned int i=0; i<names.size(); ++i) std::cout<<names[i]<<" "; std::cout<<"\n";
			std::cout<<"NUMBERS = "; for(unsigned int i=0; i<natoms.size(); ++i) std::cout<<natoms[i]<<" "; std::cout<<"\n";
		}
		
		//====  resize the simulation ==== 
		if(DEBUG_VASP>0) std::cout<<"allocating memory\n";
		struc.resize(natoms,names,atomT);
		
		//====  read positions ==== 
		if(DEBUG_VASP>1) std::cout<<"reading positions\n";
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			fgets(input,string::M,reader);
			struc.posn(n)[0]=std::atof(std::strtok(input,string::WS));
			struc.posn(n)[1]=std::atof(std::strtok(NULL,string::WS));
			struc.posn(n)[2]=std::atof(std::strtok(NULL,string::WS));
		}
		
		//====  convert to cartesian coordinates (if necessary) ==== 
		if(DEBUG_VASP>1) std::cout<<"Converting to Cartesian coordinates\n";
		if(direct){
			for(unsigned int n=0; n<struc.nAtoms(); ++n){
				struc.posn(n)=struc.R()*struc.posn(n);
			}
		} else {
			for(unsigned int n=0; n<struc.nAtoms(); ++n){
				struc.posn(n)*=s;
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
		
		//print the system name
		if(DEBUG_VASP>1) std::cout<<"writing name\n";
		write_name(writer,"system");
		
		//print the simulation cell (transpose)
		if(DEBUG_VASP>1) std::cout<<"writing cell\n";
		write_cell(writer,static_cast<const Cell&>(struc));
		
		//print the atom info
		if(DEBUG_VASP>1) std::cout<<"writing atom info\n";
		write_atoms(writer,struc.species(),struc.nAtomsV());
		
		//print the positions
		if(DEBUG_VASP>1) std::cout<<"writing posns\n";
		if(!atomT.frac){
			fprintf(writer,"Cart\n");
			for(unsigned int n=0; n<struc.nAtoms(); ++n){
				fprintf(writer, "%.8f %.8f %.8f\n",
					struc.posn(n)[0],
					struc.posn(n)[1],
					struc.posn(n)[2]
				);
			}
		} else {
			fprintf(writer,"Direct\n");
			for(unsigned int n=0; n<struc.nAtoms(); ++n){
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

void read(const char* file, const Interval interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="read(const char*,Interval,Simulation&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//====  local function variables ==== 
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		std::string str;
	//simulation flags
		bool direct;//whether the coordinates are in direct or Cartesian coordinates
	//time info
		unsigned int ts=0;//number of timesteps
		unsigned int tsint=0;//requested interval
	//cell info
		double scale=1;
		Eigen::Matrix3d lv;
		Cell cell;
	//atom info
		unsigned int nSpecies=0;//the number of atomic species
		std::vector<unsigned int> natoms;//the number of atoms in each species
		std::vector<std::string> names;//the names of each species
		unsigned int N=0;
	//timing
		clock_t start,stop;
	//units
		double s=0.0;
		if(units::consts::system()==units::System::AU) s=units::BOHRpANG;
		else if(units::consts::system()==units::System::METAL) s=1.0;
		else throw std::runtime_error("Invalid units.");
	//misc
		bool error=false;
		
	try{
		//start the timer
		start=std::clock();
		
		//open the file
		if(DEBUG_VASP>1) std::cout<<"opening file\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Unable to open file.");
		
		//====  check the atom type ==== 
		if(!atomT.name || !atomT.type || !atomT.index || !atomT.posn)
			throw std::invalid_argument("Invalid atom type.");
		
		//====  clear the simulation ==== 
		sim.clear();
		
		//====  read in the system name ==== 
		sim.name()=std::string(string::trim(fgets(input,string::M,reader)));
		
		//====  read the simulation cell ==== 
		read_cell(reader,cell);
		
		//====  read the atom info ==== 
		read_atoms(reader,names,natoms);
		for(unsigned int i=0; i<natoms.size(); ++i) N+=natoms[i];
		
		//====  read the coordinate type ==== 
		direct=read_coord(reader);
		
		//====  check if the cell is variable or not ==== 
		if(DEBUG_VASP>0) std::cout<<"Checking whether cell is variable\n";
		for(unsigned int n=0; n<N; ++n) fgets(input, string::M, reader);
		str=std::string(string::trim(fgets(input,string::M,reader)));
		if(str==sim.name()) sim.cell_fixed()=false;
		else sim.cell_fixed()=true;
		//====  reset the line position ==== 
		if(!sim.cell_fixed()) for(unsigned int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader);
		
		//====  find the number of timesteps ==== 
		if(DEBUG_VASP>0) std::cout<<"Finding the number of timesteps\n";
		std::rewind(reader);
		//find the total number of lines in the file
		unsigned int nLines=0;
		for(unsigned int i=0; i<HEADER_SIZE; ++i) fgets(input, string::M, reader);
		while(fgets(input, string::M, reader)!=NULL){++nLines;};
		if(sim.cell_fixed()) ts=std::floor((1.0*nLines)/(1.0*N+1.0));
		else ts=std::floor((1.0*nLines)/(1.0*N+1.0+HEADER_SIZE));
		
		//====  reset the line position ==== 
		if(DEBUG_VASP>0) std::cout<<"resetting the line position\n";
		std::rewind(reader);
		if(sim.cell_fixed()) for(unsigned int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader);
		
		//set the interval
		if(DEBUG_VASP>0) std::cout<<"setting the interval\n";
		if(interval.beg<0) throw std::invalid_argument("Invalid beginning timestep.");
		sim.beg()=interval.beg;
		if(interval.end<0){
			sim.end()=ts+interval.end;
		} else sim.end()=interval.end;
		tsint=sim.end()-sim.beg()+1;
		
		//====  print data to screen ==== 
		if(DEBUG_VASP>1){
			std::cout<<"NAME       = "<<sim.name()<<"\n";
			std::cout<<"CELL       = \n"<<cell<<"\n";
			std::cout<<"CELL_FIXED = "<<(sim.cell_fixed()?"TRUE":"FALSE")<<"\n";
			std::cout<<"DIRECT     = "<<(direct?"TRUE":"FALSE")<<"\n";
			std::cout<<"SPECIES    = ";
			for(unsigned int i=0; i<names.size(); ++i){
				std::cout<<names[i]<<" ";
			}
			std::cout<<"\n";
			std::cout<<"NUMBERS    = ";
			for(unsigned int i=0; i<natoms.size(); ++i){
				std::cout<<natoms[i]<<" ";
			}
			std::cout<<"\n";
			std::cout<<"N_ATOMS    = "<<N<<"\n";
			std::cout<<"TIMESTEPS  = "<<ts<<"\n";
			std::cout<<"INTERVAL   = "<<sim.beg()<<":"<<sim.end()<<":"<<interval.stride<<" - "<<tsint<<"\n";
			std::cout<<"N_STEPS    = "<<tsint/interval.stride<<"\n";
		}
		
		//====  resize the simulation ==== 
		if(DEBUG_VASP>0) std::cout<<"allocating memory\n";
		sim.resize(tsint/interval.stride,natoms,names,sim.atomT());
		
		//====  set the atom info ==== 
		for(unsigned int t=0; t<sim.timesteps(); ++t){
			for(unsigned int n=0; n<sim.frame(t).nSpecies(); ++n){
				for(unsigned int m=0; m<sim.frame(t).nAtoms(n); ++m){
					sim.frame(t).name(n,m)=names[n];
					sim.frame(t).type(n,m)=n;
					sim.frame(t).index(n,m)=m;
					if(sim.atomT().an) sim.frame(t).an(n,m)=PTable::an(names[n].c_str());
				}
			}
		}
		
		//====  read positions ==== 
		if(DEBUG_VASP>0) std::cout<<"reading positions\n";
		//skip timesteps until beg is reached
		for(unsigned int t=0; t<sim.beg(); ++t){
			if(!sim.cell_fixed()) for(unsigned int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader); //skip header
			fgets(input,string::M,reader);//skip single line
			for(unsigned int n=0; n<N; ++n) fgets(input,string::M,reader);
		}
		//read the positions
		if(sim.cell_fixed()){
			for(unsigned int t=0; t<sim.timesteps(); ++t) static_cast<Cell&>(sim.frame(t))=cell;
			for(unsigned int t=0; t<sim.timesteps(); ++t){
				if(DEBUG_VASP>2) std::cout<<"T = "<<t<<"\n";
				else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
				fgets(input,string::M,reader);//skip line
				for(unsigned int n=0; n<N; ++n){
					fgets(input,string::M,reader);
					sim.frame(t).posn(n)[0]=std::atof(std::strtok(input,string::WS));
					sim.frame(t).posn(n)[1]=std::atof(std::strtok(NULL,string::WS));
					sim.frame(t).posn(n)[2]=std::atof(std::strtok(NULL,string::WS));
				}
				//skip "stride-1" steps
				for(unsigned int tt=0; tt<interval.stride-1; ++tt){
					fgets(input,string::M,reader);//skip line
					for(unsigned int n=0; n<N; ++n){
						fgets(input,string::M,reader);
					}
				}
			}
		} else {
			for(unsigned int t=0; t<sim.timesteps(); ++t){
				if(DEBUG_VASP>2) std::cout<<"T = "<<t<<"\n";
				else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
				//read in lattice vectors
				fgets(input,string::M,reader);//name
				scale=std::atof(fgets(input,string::M,reader));//scale
				for(unsigned int i=0; i<3; ++i){
					fgets(input, string::M, reader);
					lv(0,i)=std::atof(std::strtok(input,string::WS));
					for(unsigned int j=1; j<3; ++j){
						lv(j,i)=std::atof(std::strtok(NULL,string::WS));
					}
				}
				static_cast<Cell&>(sim.frame(t)).init(s*lv,scale);
				fgets(input,string::M,reader);//skip line (atom names)
				fgets(input,string::M,reader);//skip line (atom numbers)
				fgets(input,string::M,reader);//skip line (Direct or Cart)
				for(unsigned int n=0; n<N; ++n){
					fgets(input,string::M,reader);
					sim.frame(t).posn(n)[0]=std::atof(std::strtok(input,string::WS));
					sim.frame(t).posn(n)[1]=std::atof(std::strtok(NULL,string::WS));
					sim.frame(t).posn(n)[2]=std::atof(std::strtok(NULL,string::WS));
				}
				//skip "stride-1" steps
				for(unsigned int tt=0; tt<interval.stride-1; ++tt){
					fgets(input,string::M,reader);//name
					fgets(input,string::M,reader);//scale
					for(unsigned int i=0; i<3; ++i){
						fgets(input,string::M,reader);//lv
					}
					fgets(input,string::M,reader);//skip line (atom names)
					fgets(input,string::M,reader);//skip line (atom numbers)
					fgets(input,string::M,reader);//skip line (Direct or Cart)
					for(unsigned int n=0; n<N; ++n){
						fgets(input,string::M,reader);
					}
				}
			}
		}
		
		//====  convert to Cartesian coordinates if necessary ==== 
		if(direct){
			if(DEBUG_VASP>0) std::cout<<"Converting to Cartesian coordinates\n";
			for(unsigned int t=0; t<sim.timesteps(); ++t){
				for(unsigned int n=0; n<N; ++n){
					sim.frame(t).posn(n)=sim.frame(t).R()*sim.frame(t).posn(n);
				}
			}
		} else if(s!=1.0){
			for(unsigned int t=0; t<sim.timesteps(); ++t){
				for(unsigned int n=0; n<N; ++n){
					sim.frame(t).posn(n)*=s;
				}
			}
		}
		
		//close the file
		fclose(reader);
		reader=NULL;
		
		//stop the timer
		stop=std::clock();
		
		//print the time
		std::cout<<"Simulation loaded in "<<((double)(stop-start))/CLOCKS_PER_SEC<<" seconds.\n";
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

void write(const char* file, const Interval interval, const AtomType& atomT, const Simulation& sim){
	const char* funcName="write(const char*,const Interval,const AtomType&,Simulation&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//units
		double s=0.0;
		if(units::consts::system()==units::System::AU) s=units::ANGpBOHR;
		else if(units::consts::system()==units::System::METAL) s=1.0;
		else throw std::runtime_error("Invalid units.");
	//misc
		FILE* writer=NULL;
		bool error=true;
	
	try{
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: \"")+std::string(file)+std::string("\"\n"));
		
		//check simulation
		if(sim.timesteps()==0) throw std::invalid_argument("Invalid simulation.");
		
		//check timing info
		int beg=interval.beg;
		int end=interval.end;
		if(end<0) end=interval.end+sim.timesteps();
		if(beg<0) throw std::invalid_argument("Invalid beginning timestep.");
		if(end>=sim.timesteps()) throw std::invalid_argument("Invalid ending timestep.");
		if(end<beg) throw std::invalid_argument("Invalid timestep interval.");
		
		std::string coord;
		if(atomT.frac) coord=std::string("Direct");
		else coord=std::string("Cart");
		
		if(sim.cell_fixed()){
			fprintf(writer,"%s\n",sim.name().c_str());
			fprintf(writer,"1.0\n");
			for(unsigned int i=0; i<3; ++i){
				for(unsigned int j=0; j<3; ++j){
					fprintf(writer,"%f ",sim.frame(0).R()(j,i)*s);
				}
				fprintf(writer,"\n");
			}
			for(unsigned int i=0; i<sim.frame(0).nSpecies(); ++i) fprintf(writer,"%s ",sim.frame(0).species(i).c_str());
			for(unsigned int i=0; i<sim.frame(0).nSpecies(); ++i) fprintf(writer,"%i ",sim.frame(0).nAtoms(i));
			for(int t=beg; t<=end; ++t){
				fprintf(writer,"%s\n",coord.c_str());
				for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n){
					Eigen::Vector3d posn=sim.frame(t).RInv()*sim.frame(t).posn(n);
					fprintf(writer,"%f %f %f\n",posn[0],posn[1],posn[2]);
				}
			}
		} else {
			for(int t=beg; t<=end; ++t){
				fprintf(writer,"%s\n",sim.name().c_str());
				fprintf(writer,"1.0\n");
				for(unsigned int i=0; i<3; ++i){
					for(unsigned int j=0; j<3; ++j){
						fprintf(writer,"%f ",sim.frame(t).R()(j,i)*s);
					}
					fprintf(writer,"\n");
				}
				for(unsigned int i=0; i<sim.frame(t).nSpecies(); ++i) fprintf(writer,"%s ",sim.frame(t).species(i).c_str());
				for(unsigned int i=0; i<sim.frame(t).nSpecies(); ++i) fprintf(writer,"%i ",sim.frame(t).nAtoms(i));
				fprintf(writer,"%s\n",coord.c_str());
				for(unsigned int n=0; n<sim.frame(t).nAtoms(); ++n){
					Eigen::Vector3d posn=sim.frame(t).RInv()*sim.frame(t).posn(n);
					fprintf(writer,"%f %f %f\n",posn[0],posn[1],posn[2]);
				}
			}
		}
		
		fclose(writer);
		writer=NULL;
		
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Exception: Could not write data.");
}

}

//*****************************************************
//XML
//*****************************************************

namespace XML{

void read(const char* file, unsigned int t, const AtomType& atomT, Structure& struc){
	const char* funcName="read(const char*,Structure&,(int))";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
	//time info
		unsigned int ts=0;//number of timesteps
	//cell info
		Eigen::Matrix3d lv;
	//atom info
		unsigned int nSpecies=0;//the number of atomic species
		std::vector<unsigned int> nAtoms;//the number of atoms in each species
		std::vector<std::string> species;//the names of each species
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
				for(unsigned int i=0; i<6; ++i) fgets(input,string::M,reader);
				for(unsigned int i=0; i<N; ++i){
					fgets(input,string::M,reader);
					std::strtok(input,">"); std::strtok(NULL,">");
					std::string name=std::strtok(NULL,"<");
					bool match=false;
					for(unsigned int j=0; j<species.size(); ++j){
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
			std::cout<<"ATOM_NAMES   = "; for(unsigned int i=0; i<species.size(); ++i) std::cout<<species[i]<<" "; std::cout<<"\n";
			std::cout<<"ATOM_NUMBERS = "; for(unsigned int i=0; i<nAtoms.size(); ++i) std::cout<<nAtoms[i]<<" "; std::cout<<"\n";
		}
		if(species.size()!=nAtoms.size()) throw std::runtime_error("Mismatch in atom names/numbers.");
		nSpecies=species.size();
		
		//==== resize the simulation ====
		if(DEBUG_VASP>0) std::cout<<"resizing the simulation\n";
		struc.resize(nAtoms,species,atomT);
		
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
						for(unsigned int n=0; n<struc.nAtoms(); ++n){
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
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//==== transform to Cartesian coordinates ====
	if(DEBUG_VASP>0) std::cout<<"transforming to Cartesian coordinates\n";
	for(unsigned int n=0; n<struc.nAtoms(); ++n){
		struc.posn(n)=struc.R()*struc.posn(n);
		Cell::returnToCell(struc.posn(n),struc.posn(n),struc.R(),struc.RInv());
	}
	
	//==== free all local variables ====
	if(reader!=NULL) fclose(reader);
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Exception: Could not read data.");
}

}

Simulation& read(const Format& format, const Interval& interval, const AtomType& atomT, Simulation& sim){
	const char* func_name="read(const Format&,const Interval&,const AtomT&,Simulation&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<func_name<<":\n";
	//local variables
	clock_t start,stop;
	double time;
	bool error=false;
	
	try{		
		//start the timer
		start=std::clock();
		
		if(!format.poscar.empty()){
			Structure struc;
			POSCAR::read(format.poscar.c_str(),atomT,struc);
			sim.resize(1,struc.nAtomsV(),struc.species(),atomT);
			sim.frame(0)=struc;
			sim.timesteps()=1;
			sim.cell_fixed()=true;
			std::cout<<"TIMESTEPS = "<<sim.timesteps()<<"\n";
		} else if(!format.xdatcar.empty()){
			XDATCAR::read(format.xdatcar.c_str(),interval,atomT,sim);
		} else if(!format.xml.empty()){
			Structure struc;
			XML::read(format.xml.c_str(),interval.beg,atomT,struc);
			sim.resize(1,struc.nAtomsV(),struc.species(),atomT);
			sim.frame(0)=struc;
		}
		
		//stop the timer
		stop=std::clock();
		
		//print the time
		time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"Simulation read in "<<time<<" seconds.\n";
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
	else return sim;
}

const Simulation& write(const Format& format, const Interval& interval, const AtomType& atomT, const Simulation& sim){
	const char* func_name="write(const Format&,const Interval&,const AtomT&,const Simulation&)";
	if(DEBUG_VASP>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<func_name<<":\n";
	//local variables
	clock_t start,stop;
	double time;
	bool error=false;
	
	try{		
		//start the timer
		start=std::clock();
		
		if(!format.poscar.empty()){
			POSCAR::write(format.poscar.c_str(),atomT,sim.frame(interval.beg));
		} else if(!format.xdatcar.empty()){
			XDATCAR::write(format.xdatcar.c_str(),interval,atomT,sim);
		} 
		
		//stop the timer
		stop=std::clock();
		
		//print the time
		time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"Simulation read in "<<time<<" seconds.\n";
	}catch(std::exception& e){
		std::cout<<"Error in "<<NAMESPACE_GLOBAL<<"::"<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Error: Failed to read.");
	else return sim;
}

}