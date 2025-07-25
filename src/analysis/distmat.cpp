// c libraries
#include <cstdlib>
#include <ctime>
//c++
#include <iostream>
#include <vector>
// structure
#include "struc/sim.hpp"
// format
#include "format/vasp_sim.hpp"
#include "format/lammps_sim.hpp"
#include "format/cp2k_sim.hpp"
#include "format/xyz_sim.hpp"
// str
#include "str/string.hpp"
#include "str/print.hpp"
// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"
#include "math/special.hpp"
//analysis
#include "analysis/group.hpp"

using math::special::mod;

//***********************************************************************
// Main
//***********************************************************************

int main(int argc, char* argv[]){
	//======== local function variables ========
	//==== file i/o ====
		FILE* reader=NULL;
		FILE_FORMAT::type fileFormat;
		char* paramfile=new char[string::M];
		char* input    =new char[string::M];
		char* simstr   =new char[string::M];
		char* strbuf   =new char[print::len_buf];
		std::vector<std::string> strlist;
	//==== simulation variables ====
		Simulation sim;
		Interval interval;
		Eigen::Vector3d box=Eigen::Vector3d::Zero();
		Eigen::Vector3d offset=Eigen::Vector3d::Zero();
		double ts=0;
		double T=0;
		double fwidth=0.0;
		Group group;
		double fpmin=0.0,fpmax=0.0;
	//==== atom type ====
		sim.atomT().name	=true;
		sim.atomT().type	=true;
		sim.atomT().posn	=true;
	//==== miscellaneous ====
		bool error=false;
		bool periodic=true;
	//==== units ====
		units::System unitsys;
	//==== misc ====
		int nprint=0;
		
	try{
		if(argc!=2) throw std::invalid_argument("Invalid number of command-line arguments.");
		
		//======== copy the parameter file ========
		std::cout<<"reading parameter file\n";
		std::strcpy(paramfile,argv[1]);
		
		//======== read in the general parameters ========
		reader=fopen(paramfile,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: could not open parameter file.");
		std::cout<<"reading general parameters\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);
			Token token(input,string::WS);
			if(token.end()) continue;
			const std::string tag=string::to_upper(token.next());
			if(tag=="INTERVAL"){
				interval=Interval::read(token.next().c_str(),interval);
			} else if(tag=="OFFSET"){
				offset[0]=std::atof(token.next().c_str());
				offset[1]=std::atof(token.next().c_str());
				offset[2]=std::atof(token.next().c_str());
			} else if(tag=="SIM"){
				std::strcpy(simstr,token.next().c_str());
			} else if(tag=="FORMAT"){
				fileFormat=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
			} else if(tag=="UNITS"){
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
			} else if(tag=="NPRINT"){
				nprint=std::atoi(token.next().c_str());
			} else if(tag=="BOX"){
				if(strlist.size()!=4) throw std::invalid_argument("Invalid box specification");
				box[0]=std::atof(token.next().c_str());
				box[1]=std::atof(token.next().c_str());
				box[2]=std::atof(token.next().c_str());
			} else if(tag=="GROUP"){
				group.read(token);
			} else if(tag=="PERIODIC"){
				periodic=string::boolean(token.next().c_str());
			}
		}
		//close the file
		fclose(reader);
		reader=NULL;
		
		//======== initialize the unit system ========
		std::cout<<"initializing the unit system\n";
		units::Consts::init(unitsys);
		
		//======== print the parameters ========
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::title("PARAMETERS",strbuf)<<"\n";
		std::cout<<"units    = "<<unitsys<<"\n";
		std::cout<<"atomt    = "<<sim.atomT()<<"\n";
		std::cout<<"sim      = \""<<simstr<<"\"\n";
		std::cout<<"format   = "<<fileFormat<<"\n";
		std::cout<<"interval = "<<interval<<"\n";
		std::cout<<"offset   = "<<offset.transpose()<<"\n";
		std::cout<<"box      = "<<box[0]<<" "<<box[1]<<" "<<box[2]<<"\n";
		std::cout<<"periodic = "<<periodic<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		
		//======== check the parameters ========
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		if(fileFormat==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
		
		//======== read the simulation ========
		std::cout<<"reading simulation\n";
		if(fileFormat==FILE_FORMAT::XDATCAR){
			VASP::XDATCAR::read(simstr,interval,sim.atomT(),sim);
		} else if(fileFormat==FILE_FORMAT::CP2K){
			CP2K::Format format;
			Token token(simstr,",");
			format.input=token.next();
			format.xyz=token.next();
			if(!token.end()) format.fxyz=token.next();
			CP2K::read(format,interval,sim.atomT(),sim);
		} else if(fileFormat==FILE_FORMAT::LAMMPS){
			LAMMPS::DUMP::read(simstr,interval,sim.atomT(),sim);
		} else if(fileFormat==FILE_FORMAT::XYZ){
			XYZ::read(simstr,interval,sim.atomT(),sim);
		} else throw std::invalid_argument("Invalid file format.");
		
		//======== print the data to screen ========
		std::cout<<"SIMULATION = \n"<<sim<<"\n";
		std::cout<<sim.frame(0)<<"\n";
		
		//======== build group ========
		std::cout<<"building group\n";
		group.build(sim.frame(0));
		
		//======== compute the requested properties ========
		std::cout<<"computing properties\n";
		clock_t start=std::clock();
		if(nprint==0) nprint=sim.timesteps()/10;
		if(nprint==0) nprint=1;
		
		const int nAtoms=group.size();
		Eigen::Vector3d dr;
		std::vector<double> dmat(nAtoms*(nAtoms-1)/2);
		FILE* writer=fopen("distmat.dat","w");
		if(writer==NULL) throw std::runtime_error("distmat::main(int,char[]*): Could not open \"distmat.dat\".");
		if(periodic && sim.frame(0).R().norm()>1.0e-6){
			for(int t=0; t<sim.timesteps(); ++t){
				//compute the distance matrix
				int c=0;
				for(int i=0; i<nAtoms; ++i){
					const int ii=group.atom(i);
					for(int j=i+1; j<nAtoms; ++j){
						const int jj=group.atom(j);
						dmat[c++]=sim.frame(t).dist(sim.frame(t).posn(ii),sim.frame(t).posn(jj),dr);
					}
				}
				//write the distance matrix
				c=0;
				fprintf(writer,"%i ",sim.frame(t).nAtoms());
				for(int i=0; i<nAtoms; ++i){
					for(int j=i+1; j<nAtoms; ++j){
						fprintf(writer,"%f ",dmat[c++]);
					}
				}
				fprintf(writer,"\n");
			}
		} else {
			for(int t=0; t<sim.timesteps(); ++t){
				//compute the distance matrix
				int c=0;
				for(int i=0; i<nAtoms; ++i){
					const int ii=group.atom(i);
					for(int j=i+1; j<nAtoms; ++j){
						const int jj=group.atom(j);
						dmat[c++]=(sim.frame(t).posn(ii)-sim.frame(t).posn(jj)).norm();
					}
				}
				//write the distance matrix
				c=0;
				fprintf(writer,"%i ",sim.frame(t).nAtoms());
				for(int i=0; i<nAtoms; ++i){
					for(int j=i+1; j<nAtoms; ++j){
						fprintf(writer,"%f ",dmat[c++]);
					}
				}
				fprintf(writer,"\n");
			}
		}
		
	}catch(std::exception& e){
		std::cout<<e.what()<<"\n";
		std::cout<<"ANALYSIS FAILED.\n";
		error=true;
	}
	
	std::cout<<"freeing local variables\n";
	delete[] paramfile;
	delete[] input;
	delete[] simstr;
	
	std::cout<<"exiting program\n";
	if(error) return 1;
	else return 0;
}