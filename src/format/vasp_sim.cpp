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
#include "format/vasp_sim.hpp"

//*****************************************************
//XDATCAR
//*****************************************************

namespace VASP{
	
namespace XDATCAR{

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="read(const char*,const Interval&,Simulation&)";
	if(VASP_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		std::string str;
		Token token;
	//simulation flags
		bool direct;//whether the coordinates are in direct or Cartesian coordinates
	//time info
		int ts=0;//number of timesteps
	//cell info
		double scale=1;
		Eigen::Matrix3d lv;
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
		bool cell_fixed=false;
		
	try{
		//start the timer
		const clock_t start=std::clock();
		
		//open the file
		if(VASP_PRINT_STATUS>0) std::cout<<"Opening file\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Unable to open file.");
		
		//==== clear the simulation ====
		if(VASP_PRINT_STATUS>0) std::cout<<"clearing simulation\n";
		sim.clear();
		
		//==== read in the system name ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading system name\n";
		sim.name()=string::trim(fgets(input,string::M,reader));
		
		//==== read the simulation cell ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading cell\n";
		std::sscanf(fgets(input,string::M,reader),"%lf",&scale);
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,0),&lv(1,0),&lv(2,0));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,1),&lv(1,1),&lv(2,1));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,2),&lv(1,2),&lv(2,2));
		lv*=s_len*scale;
		
		//==== read species ====
		if(VASP_PRINT_STATUS>0) std::cout<<"read species\n";
		//read names
		fgets(input,string::M,reader);
		std::vector<std::string> names;
		token.read(input,string::WS);
		while(!token.end()) names.push_back(token.next());
		//read numbers
		fgets(input,string::M,reader);
		std::vector<int> natoms;
		token.read(input,string::WS);
		while(!token.end()) natoms.push_back(std::atoi(token.next().c_str()));
		//check the number of species
		if(names.size()==0 || natoms.size()==0 || names.size()!=natoms.size()) throw std::runtime_error("Invalid number of species");
		const int nspecies=names.size();
		int natomst=0;
		for(int i=0; i<natoms.size(); ++i) natomst+=natoms[i];
		
		//==== read coord ====
		if(VASP_PRINT_STATUS>0) std::cout<<"read coord\n";
		fgets(input,string::M,reader);
		if(input[0]=='D') direct=true;
		else direct=false;
		
		//==== check if the cell is variable or not ====
		if(VASP_PRINT_STATUS>0) std::cout<<"Checking whether cell is variable\n";
		for(int n=0; n<natomst; ++n) fgets(input, string::M, reader);
		str=std::string(string::trim(fgets(input,string::M,reader)));
		if(str==sim.name()) cell_fixed=false;
		else cell_fixed=true;
		
		//==== find the number of timesteps ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading the number of timesteps\n";
		std::rewind(reader);
		int nlines=0;
		while(fgets(input, string::M, reader)!=NULL){++nlines;};
		if(cell_fixed) ts=std::floor((1.0*nlines-HEADER_SIZE)/(1.0*natomst+1.0));
		else ts=std::floor((1.0*nlines)/(1.0*natomst+1.0+HEADER_SIZE));
		
		//==== reset the line position ====
		if(VASP_PRINT_STATUS>0) std::cout<<"resetting the line position\n";
		std::rewind(reader);
		
		//==== set the interval ====
		if(VASP_PRINT_STATUS>0) std::cout<<"setting the interval\n";
		const int ibeg=Interval::index(interval.beg(),ts);
		const int iend=Interval::index(interval.end(),ts);
		const int nsteps=(iend-ibeg)/interval.stride();
		
		//==== print data to screen ====
		if(VASP_PRINT_DATA>-1){
			std::cout<<"NAME    = "<<sim.name()<<"\n";
			std::cout<<"ATOMT   = "<<atomT<<"\n";
			std::cout<<"DIRECT  = "<<(direct?"T":"F")<<"\n";
			std::cout<<"CELL    = \n"<<lv<<"\n";
			std::cout<<"SPECIES = "; for(int i=0; i<names.size(); ++i) std::cout<<names[i]<<" "; std::cout<<"\n";
			std::cout<<"NUMBERS = "; for(int i=0; i<natoms.size(); ++i) std::cout<<natoms[i]<<" "; std::cout<<"\n";
			std::cout<<"NATOMST = "<<natomst<<"\n";
			std::cout<<"INTERVAL   = "<<interval<<"\n";
			std::cout<<"TIMESTEPS  = "<<ts<<"\n";
			std::cout<<"N_STEPS    = "<<nsteps<<"\n";
			std::cout<<"CELL_FIXED = "<<cell_fixed<<"\n";
		}
		
		//==== resize the simulation ====
		if(VASP_PRINT_STATUS>0) std::cout<<"allocating memory\n";
		sim.resize(nsteps,natomst,atomT);
		
		//==== read positions ====
		if(VASP_PRINT_STATUS>0) std::cout<<"reading positions\n";
		//read the positions
		if(cell_fixed){
			//skip header
			for(int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader);
			//skip timesteps until beg is reached
			for(int t=0; t<ibeg; ++t){
				fgets(input,string::M,reader);//coordinate line
				for(int n=0; n<natomst; ++n) fgets(input,string::M,reader);
			}
			//initialize lattice vectors
			for(int t=0; t<sim.timesteps(); ++t) static_cast<Cell&>(sim.frame(t)).init(lv);
			//read position
			for(int t=0; t<sim.timesteps(); ++t){
				if(VASP_PRINT_STATUS>1) std::cout<<"T = "<<t<<"\n";
				else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
				//coordinate line
				fgets(input,string::M,reader);
				//read positions
				for(int n=0; n<natomst; ++n){
					std::sscanf(
						fgets(input,string::M,reader),"%lf %lf %lf",
						&sim.frame(t).posn(n)[0],&sim.frame(t).posn(n)[1],&sim.frame(t).posn(n)[2]
					);
				}
				//skip "stride-1" steps
				for(int tt=0; tt<interval.stride()-1; ++tt){
					fgets(input,string::M,reader);//coordinate line
					for(int n=0; n<natomst; ++n) fgets(input,string::M,reader);
				}
			}
		} else {
			//skip timesteps until beg is reached
			for(int t=0; t<ibeg; ++t){
				//header
				for(int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader);
				//coordinate line
				fgets(input,string::M,reader);
				//positions
				for(int n=0; n<natomst; ++n) fgets(input,string::M,reader);
			}
			//read positions
			for(int t=0; t<sim.timesteps(); ++t){
				if(VASP_PRINT_STATUS>1) std::cout<<"T = "<<t<<"\n";
				else if(t%1000==0) std::cout<<"T = "<<t<<"\n";
				//read in lattice vectors
				fgets(input,string::M,reader);//name
				scale=std::atof(fgets(input,string::M,reader));//scale
				for(int i=0; i<3; ++i){
					token.read(fgets(input,string::M,reader),string::WS);
					for(int j=0; j<3; ++j){
						lv(j,i)=std::atof(token.next().c_str());
					}
				}
				//initialize lattice vectors
				static_cast<Cell&>(sim.frame(t)).init(s_len*scale*lv);
				//skip atom names, numbers, coordinate line
				fgets(input,string::M,reader);//skip line (atom names)
				fgets(input,string::M,reader);//skip line (atom numbers)
				fgets(input,string::M,reader);//skip line (coordinate)
				//read positions
				for(int n=0; n<natomst; ++n){
					std::sscanf(
						fgets(input,string::M,reader),"%lf %lf %lf",
						&sim.frame(t).posn(n)[0],&sim.frame(t).posn(n)[1],&sim.frame(t).posn(n)[2]
					);
				}
				//skip "stride-1" steps
				for(int tt=0; tt<interval.stride()-1; ++tt){
					//header
					for(int i=0; i<HEADER_SIZE; ++i) fgets(input,string::M,reader);
					//coordinate line
					fgets(input,string::M,reader);
					//positions
					for(int n=0; n<natomst; ++n) fgets(input,string::M,reader);
				}
			}
		}
		
		//==== convert to Cartesian coordinates if necessary ====
		if(direct){
			if(VASP_PRINT_STATUS>-1) std::cout<<"converting to cartesian coordinates\n";
			for(int t=0; t<sim.timesteps(); ++t){
				for(int n=0; n<natomst; ++n){
					sim.frame(t).posn(n)=sim.frame(t).R()*sim.frame(t).posn(n);
				}
			}
		} else if(s_len!=1.0){
			for(int t=0; t<sim.timesteps(); ++t){
				for(int n=0; n<natomst; ++n){
					sim.frame(t).posn(n)*=s_len;
				}
			}
		}
		
		//==== set the names ====
		if(atomT.name){
			for(int t=0; t<sim.timesteps(); ++t){
				int count=0;
				for(int n=0; n<nspecies; ++n){
					for(int m=0; m<natoms[n]; ++m){
						sim.frame(t).name(count++)=names[n];
					} 
				}
			}
		}
		
		//==== set type ====
		if(atomT.type){
			for(int t=0; t<sim.timesteps(); ++t){
				Structure& struc=sim.frame(t);
				int count=0;
				for(int n=0; n<nspecies; ++n){
					for(int m=0; m<natoms[n]; ++m){
						struc.type(count++)=n;
					}
				}
			}
		}
		
		//==== set an ====
		if(atomT.an && atomT.name){
			for(int t=0; t<sim.timesteps(); ++t){
				Structure& struc=sim.frame(t);
				int count=0;
				for(int n=0; n<nspecies; ++n){
					for(int m=0; m<natoms[n]; ++m){
						struc.an(count)=ptable::an(struc.name(count).c_str());
						++count;
					}
				}
			}
		}
		
		//==== set mass ====
		if(atomT.mass){
			if(atomT.an){
				for(int t=0; t<sim.timesteps(); ++t){
					Structure& struc=sim.frame(t);
					int count=0;
					for(int n=0; n<nspecies; ++n){
						for(int m=0; m<natoms[n]; ++m){
							struc.mass(count)=ptable::mass(struc.an(count))*s_mass;
							++count;
						}
					}
				}
			} else if(atomT.name){
				for(int t=0; t<sim.timesteps(); ++t){
					Structure& struc=sim.frame(t);
					int count=0;
					for(int n=0; n<nspecies; ++n){
						for(int m=0; m<natoms[n]; ++m){
							const int an=ptable::an(struc.name(count).c_str());
							struc.mass(count)=ptable::mass(an)*s_mass;
							++count;
						}
					}
				}
			}	
		}
		
		//==== close the file ====
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

void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim){
	const char* funcName="write(const char*,const Interval&,const AtomType&,Simulation&)";
	if(VASP_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<NAMESPACE_LOCAL<<"::"<<funcName<<":\n";
	//reading/writing
		FILE* writer=NULL;
		bool error=false;
		bool cell_fixed=false;
	//units
		double s_len=0.0;
		if(units::Consts::system()==units::System::LJ){
			s_len=1.0;
		} else if(units::Consts::system()==units::System::AU){
			s_len=units::Bohr2Ang;
		} else if(units::Consts::system()==units::System::METAL){
			s_len=1.0;
		} else throw std::runtime_error("Invalid units.");
	
	try{
		//==== open the file ====
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error(std::string("I/O Error: Could not open file: \"")+std::string(file)+std::string("\"\n"));
		
		//==== check simulation ====
		if(sim.timesteps()==0) throw std::invalid_argument("Invalid simulation.");
		
		//==== set the number of atoms ====
		const int nAtoms=sim.frame(0).nAtoms();
		
		//==== get the species ====
		std::vector<std::string> species;
		std::vector<int> natoms;
		int nspecies=0;
		std::vector<int> indices(nAtoms);
		for(int i=0; i<nAtoms; ++i){
			int index=-1;
			for(int j=0; j<species.size(); ++j){
				if(sim.frame(0).name(i)==species[j]){
					index=j; break;
				}
			}
			if(index<0){
				species.push_back(sim.frame(0).name(i));
				natoms.push_back(1);
				nspecies++;
				indices[i]=0;
			} else {
				indices[i]=natoms[index];
				natoms[index]++;
			}
		}
		std::vector<int> offsets(nspecies,0);
		for(int i=1; i<nspecies; ++i){
			for(int j=0; j<i; ++j){
				offsets[i]+=natoms[j];
			}
		}
		for(int i=0; i<nAtoms; ++i){
			for(int j=0; j<species.size(); ++j){
				if(sim.frame(0).name(i)==species[j]){
					indices[i]+=offsets[j]; break;
				}
			}
		}
		std::vector<int> indices2(nAtoms);
		for(int i=0; i<nAtoms; ++i){
			indices2[indices[i]]=i;
		}
		
		//==== check timing info ====
		const int ibeg=Interval::index(interval.beg(),sim.timesteps());
		const int iend=Interval::index(interval.end(),sim.timesteps());
		
		//==== write the coord ====
		std::string coord;
		if(atomT.frac) coord=std::string("Direct");
		else coord=std::string("Cart");
		
		if(cell_fixed){
			fprintf(writer,"%s\n",sim.name().c_str());
			fprintf(writer,"1.0\n");
			for(int i=0; i<3; ++i){
				for(int j=0; j<3; ++j){
					fprintf(writer,"%f ",sim.frame(0).R()(j,i)*s_len);
				}
				fprintf(writer,"\n");
			}
			for(int i=0; i<nspecies; ++i) fprintf(writer,"%s ",species[i].c_str());
			fprintf(writer,"\n");
			for(int i=0; i<nspecies; ++i) fprintf(writer,"%i ",natoms[i]);
			fprintf(writer,"\n");
			for(int t=ibeg; t<=iend; ++t){
				fprintf(writer,"%s\n",coord.c_str());
				for(int n=0; n<nAtoms; ++n){
					Eigen::Vector3d posn;
					if(atomT.frac) posn=sim.frame(t).RInv()*sim.frame(t).posn(indices2[n]);
					else posn=sim.frame(t).posn(indices2[n]);
					fprintf(writer,"%f %f %f\n",posn[0],posn[1],posn[2]);
				}
			}
		} else {
			for(int t=ibeg; t<=iend; ++t){
				fprintf(writer,"%s\n",sim.name().c_str());
				fprintf(writer,"1.0\n");
				for(int i=0; i<3; ++i){
					for(int j=0; j<3; ++j){
						fprintf(writer,"%f ",sim.frame(t).R()(j,i)*s_len);
					}
					fprintf(writer,"\n");
				}
				for(int i=0; i<nspecies; ++i) fprintf(writer,"%s ",species[i].c_str());
				fprintf(writer,"\n");
				for(int i=0; i<nspecies; ++i) fprintf(writer,"%i ",natoms[i]);
				fprintf(writer,"\n");
				fprintf(writer,"%s\n",coord.c_str());
				for(int n=0; n<nAtoms; ++n){
					Eigen::Vector3d posn;
					if(atomT.frac) posn=sim.frame(t).RInv()*sim.frame(t).posn(indices2[n]);
					else posn=sim.frame(t).posn(indices2[n]);
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

}
