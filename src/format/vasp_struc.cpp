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
#include "struc/structure.hpp"
// ann - strings
#include "str/string.hpp"
// ann - chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// ann - vasp
#include "format/vasp_struc.hpp"

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
	if(VASP_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<funcName<<":\n";
	//====  local function variables ==== 
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* str_name=new char[string::M];
		char* str_number=new char[string::M];
		Token token;
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
		double s_len=0.0,s_mass=0.0;
		if(units::Consts::system()==units::System::LJ){
			s_len=1.0;
			s_mass=1.0;
		} else if(units::Consts::system()==units::System::AU){
			s_len=units::Ang2Bohr;
			s_mass=units::MPoME;
		} else if(units::Consts::system()==units::System::METAL){
			s_len=1.0;
			s_mass=1.0;
		} else throw std::runtime_error("Invalid units.");
	//misc
		bool error=false;
		
	try{
		//==== open file, clear simulation ====
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: ")+file);
		struc.clear();
		
		//==== read header ====
		if(VASP_PRINT_STATUS>0) std::cout<<"read lattice vectors\n";
		fgets(input,string::M,reader);//name
		std::sscanf(fgets(input,string::M,reader),"%lf",&scale);
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,0),&lv(1,0),&lv(2,0));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,1),&lv(1,1),&lv(2,1));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,2),&lv(1,2),&lv(2,2));
		lv*=s_len*scale;
		static_cast<Cell&>(struc).init(lv);
		
		//==== read species ====
		if(VASP_PRINT_STATUS>0) std::cout<<"read species\n";
		//read number of species
		fgets(str_name, string::M, reader);
		fgets(str_number, string::M, reader);
		//read the species names and numbers
		token.read(str_name,string::WS);
		while(!token.end()) names.push_back(token.next());
		token.read(str_number,string::WS);
		while(!token.end()) nAtoms.push_back(std::atof(token.next().c_str()));
		if(names.size()==0 || nAtoms.size()==0 || names.size()!=nAtoms.size()){
			throw std::runtime_error("Invalid number of species");
		}
		const int nSpecies=names.size();
		//compute the total number
		nAtomsT=0; for(int i=0; i<nSpecies; ++i) nAtomsT+=nAtoms[i];
		
		//==== read coord ====
		if(VASP_PRINT_STATUS>0) std::cout<<"read coord\n";
		fgets(input, string::M, reader);
		if(input[0]=='D') direct=true;
		else direct=false;
		
		//====  print data to screen ==== 
		if(VASP_PRINT_DATA>0){
			std::cout<<"CELL    = \n"<<static_cast<Cell&>(struc)<<"\n";
			std::cout<<"DIRECT  = "<<(direct?"T":"F")<<"\n";
			std::cout<<"SPECIES = "; for(int i=0; i<names.size(); ++i) std::cout<<names[i]<<" "; std::cout<<"\n";
			std::cout<<"NUMBERS = "; for(int i=0; i<nAtoms.size(); ++i) std::cout<<nAtoms[i]<<" "; std::cout<<"\n";
		}
		
		//====  resize the simulation ==== 
		if(VASP_PRINT_STATUS>0) std::cout<<"allocating memory\n";
		AtomType atomTl=atomT;
		atomTl.frac=direct;
		struc.resize(nAtomsT,atomTl);
		
		//====  read positions ==== 
		if(VASP_PRINT_STATUS>0) std::cout<<"reading positions\n";
		for(int n=0; n<struc.nAtoms(); ++n){
			std::sscanf(
				fgets(input,string::M,reader),"%lf %lf %lf",
				&struc.posn(n)[0],&struc.posn(n)[1],&struc.posn(n)[2]
			);
		}
		
		//====  convert to cartesian coordinates (if necessary) ==== 
		if(VASP_PRINT_STATUS>0) std::cout<<"Converting to Cartesian coordinates\n";
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
		
		//==== set type ====
		if(atomT.type){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.type(count++)=n;
				}
			}
		}
		
		//==== set an ====
		if(atomT.an && atomT.name){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.an(count)=ptable::an(struc.name(count).c_str());
					++count;
				}
			}
		}
		
		//==== set mass ====
		if(atomT.mass && atomT.an){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.mass(count)=ptable::mass(struc.an(count))*s_mass;
					++count;
				}
			}
		} else if(atomT.name && atomT.mass){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					const int an=ptable::an(struc.name(count).c_str());
					struc.mass(count)=ptable::mass(an)*s_mass;
					++count;
				}
			}
		}
		
		//==== set radius ====
		if(atomT.an && atomT.radius){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					struc.radius(count)=ptable::radius_covalent(struc.an(count));
					++count;
				}
			}
		} else if(atomT.name && atomT.radius){
			int count=0;
			for(int n=0; n<nSpecies; ++n){
				for(int m=0; m<nAtoms[n]; ++m){
					const int an=ptable::an(struc.name(count).c_str());
					struc.radius(count)=ptable::radius_covalent(an);
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
	if(VASP_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//====  local function variables ==== 
	//file i/o
		FILE* writer=NULL;
	//misc
		bool error=false;
	
	try{
		//open the file
		if(VASP_PRINT_STATUS>0) std::cout<<"opening file\n";
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
		if(VASP_PRINT_STATUS>0) std::cout<<"writing header\n";
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
		if(VASP_PRINT_STATUS>0) std::cout<<"writing posns\n";
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
		
		if(VASP_PRINT_STATUS>0) std::cout<<"closing file\n";
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
//XML
//*****************************************************

namespace XML{

void read(const char* file, int t, const AtomType& atomT, Structure& struc){
	const char* funcName="read(const char*,Structure&,(int))";
	if(VASP_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
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
		if(units::Consts::system()==units::System::LJ){
			s_posn=1.0;
			s_energy=1.0;
		} else if(units::Consts::system()==units::System::AU){
			s_posn=units::Ang2Bohr;
			s_energy=units::Ev2Eh;
		} else if(units::Consts::system()==units::System::METAL){
			s_posn=1.0;
			s_energy=1.0;
		}
		else throw std::runtime_error("Invalid units.");
	//misc
		bool error=false;
		int tt;
	
	try{
		
		//==== open the file ====
		if(VASP_PRINT_STATUS>0) std::cout<<"opening the file\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: ")+file);
		
		//==== read the number of timesteps ====
		if(VASP_PRINT_STATUS>0) std::cout<<"loading the timesteps\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"<calculation>")!=NULL) ++ts;
		}
		if(VASP_PRINT_STATUS>0) std::cout<<"ts = "<<ts<<"\n";
		if(t>ts) throw std::invalid_argument("Invalid timestep");
		
		//==== read in the atom info ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading in the atom info\n";
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
		if(VASP_PRINT_DATA>0){
			std::cout<<"ATOM_NAMES   = "; for(int i=0; i<species.size(); ++i) std::cout<<species[i]<<" "; std::cout<<"\n";
			std::cout<<"ATOM_NUMBERS = "; for(int i=0; i<nAtoms.size(); ++i) std::cout<<nAtoms[i]<<" "; std::cout<<"\n";
		}
		if(species.size()!=nAtoms.size()) throw std::runtime_error("Mismatch in atom names/numbers.");
		nSpecies=species.size();
		int nAtomsT=0;
		for(int i=0; i<nAtoms.size(); ++i) nAtomsT+=nAtoms[i];
		
		//==== resize the simulation ====
		if(VASP_PRINT_STATUS>0) std::cout<<"resizing the simulation\n";
		struc.resize(nAtomsT,atomT);
		
		//==== read the cells ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading the cells\n";
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
		if(VASP_PRINT_STATUS>0) std::cout<<"reading in positions\n";
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
		if(VASP_PRINT_STATUS>0) std::cout<<"reading in the energies\n";
		std::rewind(reader);
		tt=0;
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"<calculation>")!=NULL){
				if(tt<t) continue;
				double pe=0;
				while(fgets(input,string::M,reader)!=NULL){
					if(std::strstr(input,"</calculation>")!=NULL) break;
					if(std::strstr(input,"e_fr_energy")!=NULL){
						std::strtok(input,">");
						pe=std::atof(std::strtok(NULL,"<"));
					}
				}
				struc.pe()=pe*s_energy;
				break;
			}
		}
		
		//==== read in the forces ====
		if(VASP_PRINT_STATUS>0) std::cout<<"Reading in forces\n";
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
	if(VASP_PRINT_STATUS>0) std::cout<<"transforming to Cartesian coordinates\n";
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