//c libraries
#include <ctime>
//string
#include "string.hpp"
//chem info
#include "ptable.hpp"
//list
#include "list.hpp"
//units
#include "units.hpp"
//structure
#include "structure.hpp"
//lammps
#include "lammps.hpp"

namespace LAMMPS{

//*****************************************************
//FORMAT struct
//*****************************************************

Format& Format::read(const std::vector<std::string>& strlist, Format& format){
	for(int i=0; i<strlist.size(); ++i){
		if(strlist[i]=="-in"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-in\" option.");
			else format.in=strlist[i+1];
		} else if(strlist[i]=="-data"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-data\" option.");
			else format.data=strlist[i+1];
		} else if(strlist[i]=="-dump"){
			if(i==strlist.size()-1) throw std::invalid_argument("No file specified for \"-dump\" option.");
			else format.dump=strlist[i+1];
		}
	}
	return format;
}

//*****************************************************
//STYLE_ATOM struct
//*****************************************************

STYLE_ATOM::type STYLE_ATOM::read(const char* str){
	if(std::strcmp(str,"FULL")==0) return STYLE_ATOM::FULL;
	else if(std::strcmp(str,"BOND")==0) return STYLE_ATOM::BOND;
	else if(std::strcmp(str,"ATOMIC")==0) return STYLE_ATOM::ATOMIC;
	else if(std::strcmp(str,"CHARGE")==0) return STYLE_ATOM::CHARGE;
	else return STYLE_ATOM::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, STYLE_ATOM::type& t){
	switch(t){
		case STYLE_ATOM::FULL: out<<"FULL"; break;
		case STYLE_ATOM::BOND: out<<"BOND"; break;
		case STYLE_ATOM::ATOMIC: out<<"ATOMIC"; break;
		case STYLE_ATOM::CHARGE: out<<"CHARGE"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

namespace DUMP{

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim, Format& format){
	static const char* funcName="read(const char*,const Interval&,AtomType&,Simulation&,Format&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//======== local function variables ========
	//==== file i/o ====
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* temp=new char[string::M];
	//==== time info ====
		int ts=0;//number of timesteps
	//==== cell info ====
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//==== atom info ====
		FORMAT_ATOM formatAtom;
	//==== misc ====
		bool error=false;
	//==== units ====
		double s=0.0;
		std::cout<<"format.units = \""<<format.units<<"\"\n";
		if(format.units=="metal" || format.units=="real"){
			if(units::consts::system()==units::System::AU) s=units::BOHRpANG;
			else if(units::consts::system()==units::System::METAL) s=1.0;
			else throw std::runtime_error("Invalid units.");
		}
	
	try{
		//==== start the timer ====
		const clock_t start=std::clock();
		
		//==== open the file ====
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Could not open file.");
		
		//==== find the number of timesteps ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"Loading number of timesteps\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"TIMESTEP")!=NULL) ++ts;
		}
		
		//==== read in the atom format ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading atom format\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"ITEM: ATOMS")!=NULL){
				const int nTokens=string::substrN(input,string::WS)-2;
				std::strtok(input,string::WS);
				std::strtok(NULL,string::WS);
				for(int i=0; i<nTokens; ++i){
					std::strcpy(temp,std::strtok(NULL,string::WS));
					if(std::strcmp(temp,"id")==0) formatAtom.atom=i;
					else if(std::strcmp(temp,"mol")==0) formatAtom.mol=i;
					else if(std::strcmp(temp,"type")==0) formatAtom.specie=i;
					else if(std::strcmp(temp,"q")==0) formatAtom.q=i;
					else if(std::strcmp(temp,"x")==0) formatAtom.x=i;
					else if(std::strcmp(temp,"y")==0) formatAtom.y=i;
					else if(std::strcmp(temp,"z")==0) formatAtom.z=i;
				}
				break;
			}
		}
		
		//==== set the timesteps ====
		sim.beg()=interval.beg-1;
		if(interval.end<0){
			sim.end()=ts+interval.end;
		} else sim.end()=interval.end-1;
		const int tsint=sim.end()-sim.beg()+1;
		if(LAMMPS_PRINT_DATA>0){
			std::cout<<"interval  = "<<interval<<"\n";
			std::cout<<"sim.beg() = "<<sim.beg()<<"\n";
			std::cout<<"sim.end() = "<<sim.end()<<"\n";
		}
		
		//==== resize the simulation ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"allocating memory\n";
		sim.resize(tsint/interval.stride,format.natoms,format.name,atomT);
		
		//==== read positions ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading positions\n";
		int timestep=0;
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"BOX")!=NULL){
				if(timestep<sim.beg()){continue;}
				if(timestep%interval.stride!=0){continue;}
				if(LAMMPS_PRINT_DATA>1) std::cout<<"Cell: "<<timestep<<"\n";
				lv.setZero();
				fgets(input,string::M,reader);
				lv(0,0)-=std::atof(std::strtok(input,string::WS));
				lv(0,0)+=std::atof(std::strtok(NULL,string::WS));
				fgets(input,string::M,reader);
				lv(1,1)-=std::atof(std::strtok(input,string::WS));
				lv(1,1)+=std::atof(std::strtok(NULL,string::WS));
				fgets(input,string::M,reader);
				lv(2,2)-=std::atof(std::strtok(input,string::WS));
				lv(2,2)+=std::atof(std::strtok(NULL,string::WS));
				sim.frame(timestep/interval.stride-sim.beg()).cell().init(lv*s);
			}
			if(std::strstr(input,"ITEM: ATOMS")!=NULL){
				if(LAMMPS_PRINT_DATA>1) std::cout<<"Atoms: "<<timestep<<"\n";
				if(timestep<sim.beg()){++timestep; continue;}
				if(timestep%interval.stride!=0){++timestep; continue;}
				DATA_ATOM dataAtom;
				std::vector<std::string> tokens;
				for(int i=0; i<format.N; ++i){
					//read in the next line
					fgets(input,string::M,reader);
					//split the line into tokens
					string::split(input,string::WS,tokens);
					//read in the data
					if(formatAtom.q>=0) dataAtom.q=std::atof(tokens[formatAtom.q].c_str());
					if(formatAtom.x>=0) dataAtom.posn[0]=std::atof(tokens[formatAtom.x].c_str());
					if(formatAtom.y>=0) dataAtom.posn[1]=std::atof(tokens[formatAtom.y].c_str());
					if(formatAtom.z>=0) dataAtom.posn[2]=std::atof(tokens[formatAtom.z].c_str());
					if(formatAtom.atom>=0) dataAtom.index=std::atoi(tokens[formatAtom.atom].c_str());
					if(formatAtom.specie>=0) dataAtom.specie=std::atoi(tokens[formatAtom.specie].c_str())-1;
					dataAtom.index=format.indexmap[dataAtom.specie][dataAtom.index];
					//set the simulation data
					if(dataAtom.specie>=sim.frame(timestep/interval.stride-sim.beg()).nSpecies()) throw std::invalid_argument("invliad specie");
					if(dataAtom.index>=sim.frame(timestep/interval.stride-sim.beg()).nAtoms(dataAtom.specie)) throw std::invalid_argument("invalid index");
					if(atomT.posn) sim.frame(timestep/interval.stride-sim.beg()).posn(dataAtom.specie,dataAtom.index)=dataAtom.posn*s;
					if(atomT.charge) sim.frame(timestep/interval.stride-sim.beg()).charge(dataAtom.specie,dataAtom.index)=dataAtom.q;
				}
				++timestep;
			}
			if(timestep>sim.end()) break;
		}
		
		//==== check if the cell is fixed ====
		sim.cell_fixed()=true;
		for(int t=1; t<sim.timesteps(); ++t){
			if(sim.frame(t).cell()!=sim.frame(0).cell()){sim.cell_fixed()=false;break;}
		}
		
		//==== stop the timer ====
		const clock_t stop=std::clock();
		
		//==== print the time ====
		const double time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"Positions loaded in "<<time<<" seconds.\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	if(reader!=NULL) fclose(reader);
	delete[] input;
	delete[] temp;
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

void write(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="write<AtomT>(const char*,const Interval&,const AtomType&,Simulation&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//local variables
	FILE* writer=NULL;
	bool error=false;
	
	try{
		//open the file
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error("Unable to open file.");
		
		//set the beginning and ending timesteps
		int lbeg=interval.beg-1;
		int lend=interval.end;
		if(lend<0) lend=sim.timesteps()+interval.end;
		if(lbeg<0 || lend>sim.timesteps() || lbeg>lend) throw std::invalid_argument("Invalid beginning and ending timesteps.");
		if(LAMMPS_PRINT_DATA>0) std::cout<<"Interval = ("<<lbeg<<","<<lend<<")\n";
		
		for(int t=lbeg; t<=lend; ++t){
			fprintf(writer,"ITEM: TIMESTEP\n");
			fprintf(writer,"%i\n",t);
			fprintf(writer,"ITEM: NUMBER OF ATOMS\n");
			fprintf(writer,"%i\n",sim.frame(t).nAtoms());
			fprintf(writer,"ITEM: BOX BOUNDS pp pp pp\n");
			fprintf(writer,"%f %f\n",0.0,sim.frame(t).cell().R()(0,0));
			fprintf(writer,"%f %f\n",0.0,sim.frame(t).cell().R()(1,1));
			fprintf(writer,"%f %f\n",0.0,sim.frame(t).cell().R()(2,2));
			fprintf(writer,"ITEM: ATOMS id type x y z\n");
			for(int n=0; n<sim.frame(t).nAtoms(); ++n){
				fprintf(writer,"%i %i %f %f %f\n",n,sim.frame(t).specie(n),
					sim.frame(t).posn(n)[0],sim.frame(t).posn(n)[1],sim.frame(t).posn(n)[2]
				);
			}
		}
		
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}
	
}

namespace DATA{

void read(const char* file, Format& format){
	static const char* funcName="read(const char*,STYLE_ATOM::type&,const AtomType&,Simulation&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//======== local function variables ========
	//==== file i/o ====
		FILE* reader;
		char* input=new char[string::M];
		std::vector<std::string> strlist;
	//==== miscellaneous ====
		bool error=false;
	
	try{
		//==== open the file ====
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Could not open file.");
		
		//==== read in the number of atoms, number of species, and cell ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading in natoms, nspecies, masses\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,"#");
			string::split(input,string::WS,strlist);
			if(strlist.size()==0) continue;
			else if(strlist.size()==1){
				if(strlist.at(0)=="Atoms"){
					break;
				} else if(strlist.at(0)=="Masses"){
					fgets(input,string::M,reader);
					for(int i=0; i<format.nSpecies; ++i){
						fgets(input,string::M,reader);
						string::trim_right(input,"#");
						std::strtok(input,string::WS);
						format.mass[i]=std::atof(std::strtok(NULL,string::WS));
					}
				}
			} else if(strlist.size()==2){
				if(strlist.at(1)=="atoms"){
					format.N=std::atoi(strlist.at(0).c_str());
				}
			} else if(strlist.size()==3){
				if(strlist.at(1)=="atom" && strlist.at(2)=="types"){
					format.nSpecies=std::atoi(strlist.at(0).c_str());
					format.name.resize(format.nSpecies,std::string("NULL"));
					format.mass.resize(format.nSpecies,0);
					format.natoms.resize(format.nSpecies,0);
					format.indexmap.resize(format.nSpecies);
				}
			}
		}
		
		//==== read in the atom numbers ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading atom numbers\n";
		std::rewind(reader);
		std::vector<int> indices(format.nSpecies,0);
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,"#");
			string::split(input,string::WS,strlist);
			if(strlist.size()==0) continue;
			if(strlist.at(0)=="Atoms"){
				//skip a line
				fgets(input,string::M,reader);
				//read through all the atoms
				if(format.styleAtom==STYLE_ATOM::FULL){
					for(int i=0; i<format.N; ++i){
						//read in the specie
						fgets(input,string::M,reader);
						const int index=std::atoi(std::strtok(input,string::WS));
						std::strtok(NULL,string::WS);
						const int specie=std::atoi(std::strtok(NULL,string::WS));
						++format.natoms[specie-1];
						format.indexmap[specie-1][index]=indices[specie-1]++;
					}
				} else if(format.styleAtom==STYLE_ATOM::ATOMIC){
					for(int i=0; i<format.N; ++i){
						//read in the specie
						fgets(input,string::M,reader);
						const int index=std::atoi(std::strtok(input,string::WS));
						const int specie=std::atoi(std::strtok(NULL,string::WS));
						++format.natoms[specie-1];
						format.indexmap[specie-1][index]=indices[specie-1]++;
					}
				} else if(format.styleAtom==STYLE_ATOM::CHARGE){
					for(int i=0; i<format.N; ++i){
						//read in the specie
						fgets(input,string::M,reader);
						const int index=std::atoi(std::strtok(input,string::WS));
						const int specie=std::atoi(std::strtok(NULL,string::WS));
						++format.natoms[specie-1];
						format.indexmap[specie-1][index]=indices[specie-1]++;
					}
				}
				break;
			}
		}
		
		//==== set the names ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"setting atom names\n";
		for(int i=0; i<format.nSpecies; ++i){
			format.name[i]=std::string(PTable::name(PTable::an(format.mass[i])));
		}
		
		/* print data to screen */
		if(LAMMPS_PRINT_DATA>0){
			std::cout<<"N         = "<<format.N<<"\n";
			std::cout<<"N_SPECIES = "<<format.nSpecies<<"\n";
			std::cout<<"SPECIES   = "; for(int i=0; i<format.name.size(); ++i) std::cout<<format.name[i]<<" "; std::cout<<"\n";
			std::cout<<"NUMBERS   = "; for(int i=0; i<format.natoms.size(); ++i) std::cout<<format.natoms[i]<<" "; std::cout<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
	}
	
	//free local variables
	if(reader!=NULL) fclose(reader);
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}
	
}

namespace IN{

void read(const char* file, Format& format, Simulation& sim){
	static const char* funcName="read(const char*,Format&,Simulation&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		std::vector<std::string> strlist;
	//miscellaneous
		bool error=false;
	
	try{
		//==== open the file ====
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Could not open file.");
		
		//==== read parameters ===
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading units, timestep, atom style\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,"#");
			if(string::split(input,string::WS,strlist)==0) continue;
			if(strlist.size()==2){
				if(strlist.at(0)=="units"){
					format.units=strlist.at(1);
				} else if(strlist.at(0)=="timestep"){
					sim.timestep()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="atom_style"){
					format.styleAtom=STYLE_ATOM::read(string::to_upper(strlist.at(1)).c_str());
				}
			}
		}
		
		//==== set the timestep ====
		if(format.units=="metal"){
			if(units::consts::system()==units::System::METAL) sim.timestep()*=1000;
			else if(units::consts::system()==units::System::AU) sim.timestep()*=1000.0/0.2418884326505;
		}
		
		//==== check the units ====
		if(format.styleAtom==STYLE_ATOM::UNKNOWN) throw std::runtime_error("Invalid atom style.");
		
		//==== print data ===
		if(LAMMPS_PRINT_DATA>0){
			std::cout<<"units      = "<<format.units<<"\n";
			std::cout<<"timstep    = "<<sim.timestep()<<"\n";
			std::cout<<"atom-style = "<<format.styleAtom<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
	}
	
	//==== free local variables ====
	if(reader!=NULL) fclose(reader);
	delete[] input;
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}
	
}

void read(Format& format, const Interval& interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="read<AtomT>(const Format&,const Interval&,const AtomType&,Simulation&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
	//======== local function variables ========
	//==== units ====
		double s=0.0;
		if(units::consts::system()==units::System::AU) s=units::BOHRpANG;
		else if(units::consts::system()==units::System::METAL) s=1.0;
		else throw std::runtime_error("Invalid units.");
	//==== miscellaneous ====
		bool error=false;
	//==== timing ====
		clock_t start,stop;
		double time;
		
	try{
		//==== start the timer ====
		start=std::clock();
		
		//==== load the atom style ====
		IN::read(format.in.c_str(),format,sim);
		
		//==== load the generic simulation data ====
		DATA::read(format.data.c_str(),format);
		
		//==== load the positions ====
		DUMP::read(format.dump.c_str(),interval,atomT,sim,format);
		
		//==== stop the timer ====
		stop=std::clock();
		
		//==== print the time ====
		time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"Simulation loaded in "<<time<<" seconds.\n";
	}catch(std::exception& e){
		std::cout<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
	}
}

}
