#include "ame.hpp"

namespace AME{

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(unsigned int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-ame"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-ame\" option.");
			else format.ame=strlist[i+1];
		}
	}
	return format;
}

void read(const char* file, const AtomType& atomT, Structure& struc){
	static const char* funcName="read(const char*,Structure&,const AtomType&)";
	if(DEBUG_AME_PRINT_FUNC>0) std::cout<<NAMESPACE<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* temp=new char[string::M];
	//simulation flags
		bool frac;//fractiona/Cartesian coordinates
	//time info
		unsigned int ts=0;//number of timesteps
		unsigned int interval=0;//requested interval
	//cell info
		Eigen::Matrix3d lv;
	//atom info
		unsigned int nspecies=0;//the number of atomic species
		std::vector<unsigned int> numbers;//the number of atoms in each species
		std::vector<std::string> names;//the names of each species
		unsigned int N=0;
	//units
		units::System::type unitsys;
	//misc
		bool error=false;
		unsigned int tt;
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
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading timestep...\n";
		fgets(input,string::M,reader);
		std::strtok(input,string::WS);
		double ts=std::atof(std::strtok(NULL,string::WS));
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"TIMESTEP = "<<ts<<"\n";
		
		//read in number of frames
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading number of frames...\n";
		fgets(input,string::M,reader);
		std::strtok(input,string::WS);
		unsigned int N=std::atof(std::strtok(NULL,string::WS));
		if(N==0) throw std::invalid_argument("Invalid number of frames.");
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"N        = "<<N<<"\n";
		
		//read in the cell
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading lattice vectors...\n";
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
		struc.cell().init(lv);
		
		//read in atom names
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading species names...\n";
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		nspecies=strlist.size()-1;
		if(nspecies==0) throw std::invalid_argument("Invalid number of species.");
		names.resize(nspecies);
		for(unsigned int i=0; i<nspecies; ++i){
			names[i]=strlist[i+1];
		}
		if(DEBUG_AME_PRINT_DATA>0){std::cout<<"NAMES = "; for(unsigned int i=0; i<nspecies; ++i) std::cout<<names[i]<<" "; std::cout<<"\n";}
		
		//read in atom numbers
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading species numbers...\n";
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		if(strlist.size()-1!=nspecies) throw std::invalid_argument("Invalid number of species.");
		numbers.resize(nspecies);
		for(unsigned int i=0; i<nspecies; ++i){
			numbers[i]=std::atoi(strlist[i+1].c_str());
		}
		if(DEBUG_AME_PRINT_DATA>0){std::cout<<"NUMBERS = "; for(unsigned int i=0; i<nspecies; ++i) std::cout<<numbers[i]<<" "; std::cout<<"\n";}
		
		//read in coordinate
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading coordinate...\n";
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		if(string::to_upper(strlist[1])=="FRAC") frac=true;
		else if(string::to_upper(strlist[1])=="CART") frac=false;
		else throw std::invalid_argument("Invalid coordinate.");
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"COORD = "<<frac<<"\n";
		
		//resize structure
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Resizing structure...\n";
		struc.resize(numbers,names,atomT);
		
		//read in energy
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading energy...\n";
		fgets(input,string::M,reader);
		std::strtok(input,string::WS);
		struc.energy()=std::atof(std::strtok(NULL,string::WS))*s_energy;
		
		//read in positions
		if(atomT.posn){
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading positions...\n";
			for(unsigned int i=0; i<struc.nAtoms(); ++i){
				fgets(input,string::M,reader);
				struc.posn(i)[0]=std::atof(std::strtok(input,string::WS));
				struc.posn(i)[1]=std::atof(std::strtok(NULL,string::WS));
				struc.posn(i)[2]=std::atof(std::strtok(NULL,string::WS));
				if(DEBUG_AME_PRINT_DATA>1) std::cout<<"posn["<<i<<"] = "<<struc.posn(i).transpose()<<"\n";
			}
			if(frac){
				for(unsigned int i=0; i<struc.nAtoms(); ++i){
					struc.posn(i)=struc.cell().R()*struc.posn(i);
				}
			} else {
				for(unsigned int i=0; i<struc.nAtoms(); ++i){
					struc.posn(i)*=s_posn;
				}
			}
		}
		
		//read in forces
		if(atomT.force){
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading forces...\n";
			for(unsigned int i=0; i<struc.nAtoms(); ++i){
				fgets(input,string::M,reader);
				struc.force(i)[0]=std::atof(std::strtok(input,string::WS));
				struc.force(i)[1]=std::atof(std::strtok(NULL,string::WS));
				struc.force(i)[2]=std::atof(std::strtok(NULL,string::WS));
				if(DEBUG_AME_PRINT_DATA>1) std::cout<<"force["<<i<<"] = "<<struc.force(i).transpose()<<"\n";
			}
			if(frac){
				for(unsigned int i=0; i<struc.nAtoms(); ++i){
					struc.force(i)=struc.cell().RInv()*struc.force(i)*s_energy;
				}
			} else {
				for(unsigned int i=0; i<struc.nAtoms(); ++i){
					struc.force(i)*=s_energy/s_posn;
				}
			}
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//==== free all local variables ====
	if(reader!=NULL) fclose(reader);
	delete[] input;
	delete[] temp;
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

void write(const char* file, const AtomType& atomT, const Structure& struc){
	static const char* funcName="read(const char*,const Structure&,const AtomType&)";
	if(DEBUG_AME_PRINT_FUNC>0) std::cout<<NAMESPACE<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* writer=NULL;
	//misc
		bool error=false;
		std::string str;
		std::stringstream buffer;
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
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing lattice vectors...\n";
		fprintf(writer,"CELL ");
		fprintf(writer,"%f ",struc.cell().R()(0,0)*s_posn);
		fprintf(writer,"%f ",struc.cell().R()(1,0)*s_posn);
		fprintf(writer,"%f ",struc.cell().R()(2,0)*s_posn);
		fprintf(writer,"%f ",struc.cell().R()(0,1)*s_posn);
		fprintf(writer,"%f ",struc.cell().R()(1,1)*s_posn);
		fprintf(writer,"%f ",struc.cell().R()(2,1)*s_posn);
		fprintf(writer,"%f ",struc.cell().R()(0,2)*s_posn);
		fprintf(writer,"%f ",struc.cell().R()(1,2)*s_posn);
		fprintf(writer,"%f ",struc.cell().R()(2,2)*s_posn);
		fprintf(writer,"\n");
		
		//write atom names
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing species names...\n";
		fprintf(writer,"ATOMS ");
		for(unsigned int i=0; i<struc.nSpecies(); ++i) fprintf(writer,"%s ",struc.atomNames(i).c_str());
		fprintf(writer,"\n");
		
		//write atom numbers
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing species numbers...\n";
		fprintf(writer,"NUMBERS ");
		for(unsigned int i=0; i<struc.nSpecies(); ++i) fprintf(writer,"%i ",struc.nAtoms(i));
		fprintf(writer,"\n");
		
		//write coordinates
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing coordinate...\n";
		fprintf(writer,"COORD FRAC\n");
		
		//write energy
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing energy...\n";
		fprintf(writer,"ENERGY %f\n",struc.energy()*s_energy);
		
		//write positions
		fprintf(writer,"POSNS\n");
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing positions...\n";
		if(atomT.posn){
			for(unsigned int i=0; i<struc.nAtoms(); ++i){
				Eigen::Vector3d posn=struc.cell().RInv()*struc.posn(i);
				fprintf(writer,"\t%f %f %f\n",posn[0],posn[1],posn[2]);
			}
		} else {
			for(unsigned int i=0; i<struc.nAtoms(); ++i){
				Eigen::Vector3d posn=struc.cell().RInv()*struc.posn(i);
				fprintf(writer,"\t0.0 0.0 0.0\n",posn[0],posn[1],posn[2]);
			}
		}
		
		//write forces
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing forces...\n";
		fprintf(writer,"FORCES\n");
		if(atomT.force){
			for(unsigned int i=0; i<struc.nAtoms(); ++i){
				Eigen::Vector3d force=struc.cell().R()*struc.force(i)*s_energy;
				fprintf(writer,"\t%f %f %f\n",force[0],force[1],force[2]);
			}
		} else {
			for(unsigned int i=0; i<struc.nAtoms(); ++i){
				Eigen::Vector3d posn=struc.cell().RInv()*struc.posn(i);
				fprintf(writer,"\t0.0 0.0 0.0\n",posn[0],posn[1],posn[2]);
			}
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//==== free all local variables ====
	if(writer!=NULL) fclose(writer);
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="read(const char*,const Interval&,const AtomType&,Simulation&):";
	if(DEBUG_AME_PRINT_FUNC>0) std::cout<<NAMESPACE<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* temp=new char[string::M];
	//simulation flags
		bool frac;//fractiona/Cartesian coordinates
	//cell info
		Eigen::Matrix3d lv;
		Cell cell;
	//atom info
		unsigned int nspecies=0;//the number of atomic species
		std::vector<unsigned int> numbers;//the number of atoms in each species
		std::vector<std::string> names;//the names of each species
		unsigned int N=0;
	//misc
		bool error=false;
		unsigned int tt;
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
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading timestep...\n";
		fgets(input,string::M,reader);
		std::strtok(input,string::WS);
		sim.timestep()=std::atof(std::strtok(NULL,string::WS));
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"TIMESTEP = "<<sim.timestep()<<"\n";
		
		//read in number of frames
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading number of frames...\n";
		fgets(input,string::M,reader);
		std::strtok(input,string::WS);
		unsigned int N=std::atof(std::strtok(NULL,string::WS));
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"N        = "<<N<<"\n";
		if(N==0) throw std::invalid_argument("Invalid number of frames.");
		
		//read in the cell
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading lattice vectors...\n";
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
		
		//read in atom names
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading species names...\n";
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		nspecies=strlist.size()-1;
		if(nspecies==0) throw std::invalid_argument("Invalid number of species.");
		names.resize(nspecies);
		for(unsigned int i=0; i<nspecies; ++i){
			names[i]=strlist[i+1];
		}
		if(DEBUG_AME_PRINT_DATA>0){std::cout<<"NAMES = "; for(unsigned int i=0; i<nspecies; ++i) std::cout<<names[i]<<" "; std::cout<<"\n";}
		
		//read in atom numbers
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading species numbers...\n";
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		if(strlist.size()-1!=nspecies) throw std::invalid_argument("Invalid number of species.");
		numbers.resize(nspecies);
		for(unsigned int i=0; i<nspecies; ++i){
			numbers[i]=std::atoi(strlist[i+1].c_str());
		}
		if(DEBUG_AME_PRINT_DATA>0){std::cout<<"NUMBERS = "; for(unsigned int i=0; i<nspecies; ++i) std::cout<<numbers[i]<<" "; std::cout<<"\n";}
		
		//read in coordinate
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading coordinate...\n";
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		if(string::to_upper(strlist[1])=="FRAC") frac=true;
		else if(string::to_upper(strlist[1])=="CART") frac=false;
		else throw std::invalid_argument("Invalid coordinate.");
		if(DEBUG_AME_PRINT_DATA>0) std::cout<<"COORD = "<<frac<<"\n";
		
		//resize structure
		if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Resizing structure...\n";
		sim.resize(N,numbers,names,atomT);
		
		//reset file pointer
		std::rewind(reader);
		//skip header
		fgets(input,string::M,reader);
		fgets(input,string::M,reader);
		fgets(input,string::M,reader);
		
		for(unsigned int t=0; t<interval.beg; ++t){
			//header
			for(unsigned int i=0; i<5; ++i) fgets(input,string::M,reader);
			//positions
			fgets(input,string::M,reader);
			for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i) fgets(input,string::M,reader);
			//forces
			fgets(input,string::M,reader);
			for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i) fgets(input,string::M,reader);
		}
		for(unsigned int t=0; t<sim.timesteps(); ++t){
			
			//print timestep
			if(DEBUG_AME_PRINT_DATA>0) std::cout<<"t = "<<t<<"\n";
			else if(t%1000==0) std::cout<<"t = "<<t<<"\n";
			
			//read in the cell
			if(DEBUG_AME_PRINT_STATUS>1) std::cout<<"Reading lattice vectors...\n";
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
			if(DEBUG_AME_PRINT_DATA>1) std::cout<<"LV = "<<lv<<"\n";
			sim.frame(t).cell().init(lv);
			
			//read in atom names
			if(DEBUG_AME_PRINT_STATUS>1) std::cout<<"Reading species names...\n";
			fgets(input,string::M,reader);
			
			//read in atom numbers
			if(DEBUG_AME_PRINT_STATUS>1) std::cout<<"Reading species numbers...\n";
			fgets(input,string::M,reader);
			
			//read in energy
			if(DEBUG_AME_PRINT_STATUS>1) std::cout<<"Reading energy...\n";
			fgets(input,string::M,reader);
			std::strtok(input,string::WS);
			sim.frame(t).energy()=s_energy*std::atof(std::strtok(NULL,string::WS));
			
			//read in positions
			if(atomT.posn){
				if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading positions...\n";
				fgets(input,string::M,reader);
				for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
					fgets(input,string::M,reader);
					sim.frame(t).posn(i)[0]=std::atof(std::strtok(input,string::WS));
					sim.frame(t).posn(i)[1]=std::atof(std::strtok(NULL,string::WS));
					sim.frame(t).posn(i)[2]=std::atof(std::strtok(NULL,string::WS));
					if(DEBUG_AME_PRINT_DATA>1) std::cout<<"posn["<<i<<"] = "<<sim.frame(t).posn(i).transpose()<<"\n";
				}
				if(frac){
					for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
						sim.frame(t).posn(i)=sim.frame(t).cell().R()*sim.frame(t).posn(i);
					}
				} else {
					for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
						sim.frame(t).posn(i)*=s_posn;
					}
				}
			} else {
				fgets(input,string::M,reader);
				for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i) fgets(input,string::M,reader);
			}
			
			//read in forces
			if(atomT.force){
				if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Reading forces...\n";
				fgets(input,string::M,reader);
				for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
					fgets(input,string::M,reader);
					sim.frame(t).force(i)[0]=std::atof(std::strtok(input,string::WS));
					sim.frame(t).force(i)[1]=std::atof(std::strtok(NULL,string::WS));
					sim.frame(t).force(i)[2]=std::atof(std::strtok(NULL,string::WS));
					if(DEBUG_AME_PRINT_DATA>1) std::cout<<"force["<<i<<"] = "<<sim.frame(t).force(i).transpose()<<"\n";
				}
				if(frac){
					for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
						sim.frame(t).force(i)=sim.frame(t).cell().RInv()*sim.frame(t).force(i)*s_energy;
					}
				} else {
					for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
						sim.frame(t).force(i)*=s_energy/s_posn;
					}
				}
			} else {
				fgets(input,string::M,reader);
				for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i) fgets(input,string::M,reader);
			}
		
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//==== free all local variables ====
	if(reader!=NULL) fclose(reader);
	delete[] input;
	delete[] temp;
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim){
	static const char* funcName="write(const char*,const Interval&,const AtomType&,const Simulation&)";
	if(DEBUG_AME_PRINT_FUNC>0) std::cout<<NAMESPACE<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* writer=NULL;
	//simulation flags
		bool frac;//fractiona/Cartesian coordinates
	//time
		unsigned int tsint=0;
	//misc
		bool error=false;
		std::string str;
		std::stringstream buffer;
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
		
		//check timing info
		int beg=interval.beg;
		int end=interval.end;
		if(end<0) end=interval.end+sim.timesteps();
		if(beg<0) throw std::invalid_argument("Invalid beginning timestep.");
		if(end>=sim.timesteps()) throw std::invalid_argument("Invalid ending timestep.");
		if(end<beg) throw std::invalid_argument("Invalid timestep interval.");
		tsint=end-beg+1;
		
		//write the timestep
		fprintf(writer,"TIMESTEP %i\n",sim.timestep());
		
		//write the number of timesteps
		fprintf(writer,"N %i\n",tsint);
		
		for(unsigned int t=beg; t<=end; ++t){
			//write the cell
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing lattice vectors...\n";
			fprintf(writer,"CELL ");
			fprintf(writer,"%f ",sim.frame(t).cell().R()(0,0)*s_posn);
			fprintf(writer,"%f ",sim.frame(t).cell().R()(1,0)*s_posn);
			fprintf(writer,"%f ",sim.frame(t).cell().R()(2,0)*s_posn);
			fprintf(writer,"%f ",sim.frame(t).cell().R()(0,1)*s_posn);
			fprintf(writer,"%f ",sim.frame(t).cell().R()(1,1)*s_posn);
			fprintf(writer,"%f ",sim.frame(t).cell().R()(2,1)*s_posn);
			fprintf(writer,"%f ",sim.frame(t).cell().R()(0,2)*s_posn);
			fprintf(writer,"%f ",sim.frame(t).cell().R()(1,2)*s_posn);
			fprintf(writer,"%f ",sim.frame(t).cell().R()(2,2)*s_posn);
			fprintf(writer,"\n");
			
			//write atom names
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing species names...\n";
			fprintf(writer,"ATOMS ");
			for(unsigned int i=0; i<sim.frame(t).nSpecies(); ++i) fprintf(writer,"%s ",sim.frame(t).atomNames(i).c_str());
			fprintf(writer,"\n");
			
			//write atom numbers
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing species numbers...\n";
			fprintf(writer,"NUMBERS ");
			for(unsigned int i=0; i<sim.frame(t).nSpecies(); ++i) fprintf(writer,"%i ",sim.frame(t).nAtoms(i));
			fprintf(writer,"\n");
			
			//write coordinates
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing coordinate...\n";
			fprintf(writer,"COORD FRAC\n");
			
			//write energy
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing coordinate...\n";
			fprintf(writer,"ENERGY %f\n",sim.frame(t).energy()*s_energy);
			
			//write positions
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing positions...\n";
			fprintf(writer,"POSNS\n");
			if(atomT.posn){
				for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
					Eigen::Vector3d posn=sim.frame(t).cell().RInv()*sim.frame(t).posn(i);
					fprintf(writer,"\t%f %f %f\n",posn[0],posn[1],posn[2]);
				}
			} else {
				for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
					fprintf(writer,"\t0.0 0.0 0.0\n");
				}
			}
			
			//write forces
			if(DEBUG_AME_PRINT_STATUS>0) std::cout<<"Writing forces...\n";
			fprintf(writer,"FORCES\n");
			if(atomT.force){
				for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
					Eigen::Vector3d force=sim.frame(t).cell().R()*sim.frame(t).force(i)*s_energy;
					fprintf(writer,"\t%f %f %f\n",force[0],force[1],force[2]);
				}
			} else {
				for(unsigned int i=0; i<sim.frame(t).nAtoms(); ++i){
					fprintf(writer,"\t0.0 0.0 0.0\n");
				}
			}
		}
		
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//==== free all local variables ====
	if(writer!=NULL) fclose(writer);
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

}