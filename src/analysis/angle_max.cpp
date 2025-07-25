// c libraries
#include <cstdlib>
#include <ctime>
//c++
#include <iostream>
#include <vector>
// signal
#include "signal/fft.hpp"
#include "signal/window.hpp"
// io
#include "str/string.hpp"
#include "str/print.hpp"
// chem
#include "chem/units.hpp"
// struc
#include "struc/sim.hpp"
#include "struc/pair.hpp"
// format
#include "format/vasp_sim.hpp"
#include "format/lammps_sim.hpp"
#include "format/cp2k_sim.hpp"
#include "format/xyz_sim.hpp"
// mem
#include "mem/map.hpp"
// math
#include "math/const.hpp"
#include "math/reduce.hpp"
// string
#include "str/token.hpp"
#include "str/string.hpp"
// analysis
#include "analysis/group.hpp"

//***********************************************************************
// Main
//***********************************************************************

int main(int argc, char* argv[]){
	//======== local function variables ========
		typedef fourier::FFT<1,fourier::DataT::COMPLEX,fourier::DataT::COMPLEX> FFT1D;
	//==== file i/o ====
		FILE* reader=NULL;
		char* pfile  =new char[string::M];
		char* sfile  =new char[string::M];
		char* input  =new char[string::M];
		char* strbuf =new char[print::len_buf];
		FILE_FORMAT::type fileFormat;
	//==== simulation ====
		Simulation sim;
		Interval interval;
		Eigen::Vector3d box=Eigen::Vector3d::Zero();
		Eigen::Vector3d offset=Eigen::Vector3d::Zero();
		Pair pair;
	//==== atom type ====
		sim.atomT().name	=true;
		sim.atomT().mass	=true;
		sim.atomT().type	=true;
		sim.atomT().posn	=true;
	//==== calculation ====
		double rcut=0.0;
		int nprint=-1;
		Group groupC;
		Group groupV;
	//==== miscellaneous ====
		bool error=false;
	//==== units ====
		units::System unitsys;
		
	try{
		if(argc!=2) throw std::invalid_argument("Invalid number of command-line arguments.");
		
		//======== copy the parameter file ========
		std::cout<<"reading parameter file\n";
		std::strcpy(pfile,argv[1]);
		
		//======== read in the general parameters ========
		reader=fopen(pfile,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: could not open parameter file.");
		std::cout<<"reading general parameters\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);
			Token token(input,string::WS);
			if(token.end()) continue;
			const std::string tag=string::to_upper(token.next());
			if(tag=="UNITS"){
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
			} else if(tag=="SIM"){
				std::strcpy(sfile,token.next().c_str());
			} else if(tag=="INTERVAL"){
				interval=Interval::read(token.next().c_str(),interval);
			} else if(tag=="OFFSET"){
				offset[0]=std::atof(token.next().c_str());
				offset[1]=std::atof(token.next().c_str());
				offset[2]=std::atof(token.next().c_str());
			} else if(tag=="FORMAT"){
				fileFormat=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
			} else if(tag=="BOX"){
				box[0]=std::atof(token.next().c_str());
				box[1]=std::atof(token.next().c_str());
				box[2]=std::atof(token.next().c_str());
			} else if(tag=="GROUPC"){
				groupC.read(token);
			} else if(tag=="GROUPV"){
				groupV.read(token);
			} else if(tag=="RCUT"){
				rcut=std::atof(token.next().c_str());
			} else if(tag=="PAIR"){
				pair.read(token);
			} 
		}
		//close the file
		fclose(reader);
		reader=NULL;
		
		//======== initialize the unit system ========
		std::cout<<"initializing the unit system\n";
		units::Consts::init(unitsys);
		const double hbar=units::Consts::hbar();
		const double kb=units::Consts::kb();
		
		//======== print the parameters ========
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("PARAMETERS",strbuf)<<"\n";
		std::cout<<"UNITS  = "<<unitsys<<"\n";
		std::cout<<"SIM    = "<<sfile<<"\n";
		std::cout<<"FORMAT = "<<fileFormat<<"\n";
		std::cout<<"OFFSET = "<<offset.transpose()<<"\n";
		std::cout<<"BOX    = "<<box[0]<<" "<<box[1]<<" "<<box[2]<<"\n";
		std::cout<<"NPRINT = "<<nprint<<"\n";
		std::cout<<"RCUT   = "<<rcut<<"\n";
		std::cout<<"GROUPC = "<<groupC<<"\n";
		std::cout<<"GROUPV = "<<groupV<<"\n";
		std::cout<<"PAIR   = "<<pair<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		
		//======== check the parameters ========
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		
		//======== read the simulation ========
		std::cout<<"reading simulation\n";
		if(fileFormat==FILE_FORMAT::XDATCAR){
			VASP::XDATCAR::read(sfile,interval,sim.atomT(),sim);
		} else if(fileFormat==FILE_FORMAT::CP2K){
			CP2K::Format format;
			Token token(sfile,",");
			format.input=token.next();
			format.xyz=token.next();
			if(!token.end()) format.fxyz=token.next();
			CP2K::read(format,interval,sim.atomT(),sim);
		} else if(fileFormat==FILE_FORMAT::LAMMPS){
			LAMMPS::DUMP::read(sfile,interval,sim.atomT(),sim);
		} else if(fileFormat==FILE_FORMAT::XYZ){
			XYZ::read(sfile,interval,sim.atomT(),sim);
		} else throw std::invalid_argument("Invalid file format.");
		if(nprint<0) nprint=sim.timesteps()/10;
		
		//======== print the data to screen ========
		std::cout<<"SIMULATION = \n"<<sim<<"\n";
		std::cout<<sim.frame(0)<<"\n";
		
		//======== compute the requested properties ========
		std::vector<double> min(sim.timesteps());
		std::vector<double> max(sim.timesteps());
		std::vector<double> avg(sim.timesteps());
		std::vector<double> dev(sim.timesteps());
		Eigen::Vector3d rCV1,rCV2;
		const double rcut2=rcut*rcut;
		for(int t=0; t<sim.timesteps(); ++t){
			Reduce<1> reduce;
			if(t%nprint==0) std::cout<<"T = "<<t<<"\n";
			if(t%pair.stride()==0) pair.build(sim.frame(t));
			for(int i=0; i<groupC.size(); ++i){
				const int atomC=groupC.atom(i);
				for(int j=0; j<pair.size(atomC); ++j){
					const int atomV1=pair.neigh(atomC,j);
					if(groupV.contains(atomV1)){
						sim.frame(t).diff(sim.frame(t).posn(atomV1),sim.frame(t).posn(atomC),rCV1);
						const double dr2CV1=rCV1.squaredNorm();
						if(dr2CV1<rcut2){
							for(int k=j+1; k<pair.size(atomC); ++k){
								const int atomV2=pair.neigh(atomC,k);
								if(groupV.contains(atomV2)){
									sim.frame(t).diff(sim.frame(t).posn(atomV2),sim.frame(t).posn(atomC),rCV2);
									const double dr2CV2=rCV2.squaredNorm();
									if(dr2CV2<rcut2){
										const double angle=std::acos(rCV1.dot(rCV2)/sqrt(dr2CV1*dr2CV2));
										reduce.push(angle);
									}
								}
							}
						}
					}
				}
			}
			min[t]=reduce.min();
			max[t]=reduce.max();
			avg[t]=reduce.avg();
			dev[t]=reduce.dev();
		}
		
		//==== write ====
		std::cout<<"writing\n";
		FILE* writer=fopen("angle_max.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#ts min max avg dev\n");
			for(int t=0; t<sim.timesteps(); ++t){
				fprintf(writer,"%i %12.8f %12.8f %12.8f %12.8f\n",t,min[t],max[t],avg[t],dev[t]);
			}
		}
	}catch(std::exception& e){
		std::cout<<e.what()<<"\n";
		std::cout<<"ANALYSIS FAILED.\n";
		error=true;
	}
	
	std::cout<<"freeing local variables\n";
	delete[] pfile;
	delete[] sfile;
	delete[] strbuf;
	delete[] input;

	std::cout<<"exiting program\n";
	if(error) return 1;
	else return 0;
}