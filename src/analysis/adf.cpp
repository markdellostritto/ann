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
#include "math/hist.hpp"
// signal
#include "signal/fft.hpp"
// string
#include "str/token.hpp"
#include "str/string.hpp"
//analysis
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
		Group groupC;
		Group groupV;
		Pair pair;
	//==== atom type ====
		sim.atomT().name	=true;
		sim.atomT().mass	=true;
		sim.atomT().type	=true;
		sim.atomT().posn	=true;
	//==== calculation ====
		int nprint=-1;
		Map<int,double> radii;
		std::vector<double> radiit;
		Eigen::Vector3d axis;
		Histogram adf;
		double amin=0;
		double amax=math::constant::PI;
		double da=0;
		double smw=0;
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
			} else if(tag=="RADIUS"){
				const int type=std::atof(token.next().c_str())-1;
				const double rad=std::atof(token.next().c_str());
				radii.add(type,rad);
			} else if(tag=="ANGLE"){
				amin=std::atof(token.next().c_str())/math::constant::RAD;
				amax=std::atof(token.next().c_str())/math::constant::RAD;
				da=std::atof(token.next().c_str())/math::constant::RAD;
			} else if(tag=="AXIS"){
				axis[0]=std::atof(token.next().c_str());
				axis[1]=std::atof(token.next().c_str());
				axis[2]=std::atof(token.next().c_str());
			} else if(tag=="PAIR"){
				pair.read(token);
			} else if(tag=="NPRINT"){
				nprint=std::atoi(token.next().c_str());
			} else if(tag=="SMW"){
				smw=std::atof(token.next().c_str());
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
		std::cout<<"SMW    = "<<smw<<"\n";
		std::cout<<"ANGLE  = "<<amin<<" "<<amax<<" "<<da<<"\n";
		std::cout<<"AXIS   = "<<axis.transpose()<<"\n";
		std::cout<<"GROUPC = "<<groupC<<"\n";
		std::cout<<"GROUPV = "<<groupV<<"\n";
		std::cout<<print::title("RADII",strbuf)<<"\n";
		for(int i=0; i<radii.size(); ++i){
			std::cout<<radii.key(i)<<" "<<radii.val(i)<<"\n";
		}
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
		
		//======== set the radii ========
		int ntypes=-1;
		for(int i=0; i<sim.frame(0).nAtoms(); ++i){
			if(sim.frame(0).type(i)>ntypes) ntypes=sim.frame(0).type(i);
		}
		ntypes++;
		radiit.resize(ntypes,0);
		for(int i=0; i<ntypes; ++i){
			if(radii.find(i)) radiit[i]=radii[i];
		}
		std::cout<<"radiit = "; for(int i=0; i<ntypes; ++i) std::cout<<radiit[i]<<" "; std::cout<<"\n";
		
		//======== build group ========
		std::cout<<"building groups\n";
		groupC.build(sim.frame(0));
		groupV.build(sim.frame(0));
		
		//======== compute the requested properties ========
		const int nbins=(amax-amin)/da;
		adf.init(amin,amax,nbins);
		Eigen::Vector3d dr;
		for(int t=0; t<sim.timesteps(); ++t){
			if(t%nprint==0) std::cout<<"T = "<<t<<"\n";
			if(t%pair.stride()==0) pair.build(sim.frame(t));
			for(int i=0; i<groupC.size(); ++i){
				const int atomC=groupC.atom(i);
				const double ri=radiit[sim.frame(t).type(atomC)];
				for(int j=0; j<pair.size(atomC); ++j){
					const int atomV=pair.neigh(atomC,j);
					const double rj=radiit[sim.frame(t).type(atomV)];
					if(groupV.contains(atomV)){
						const double dr2=sim.frame(t).dist2(sim.frame(t).posn(atomC),sim.frame(t).posn(atomV),dr);
						const double bl2=(ri+rj)*(ri+rj);
						if(dr2<bl2){
							const double angle=std::acos(dr.dot(axis)/(dr.norm()*axis.norm()));
							adf.push(angle);
						}
					}
				}
			}
		}
		
		//==== compute the adf ====
		std::cout<<"computing the adf\n";
		std::vector<double> adfx(adf.nbins(),0);
		std::vector<double> adfy(adf.nbins(),0);
		for(int i=0; i<adf.nbins(); ++i){
			adfx[i]=adf.abscissa(i)*math::constant::RAD;
			adfy[i]=adf.ordinate(i);
		}
		
		//==== smooth the adf ====
		std::cout<<"smoothing the adf\n";
		if(smw>0) signala::smooth(adfy,smw);
		
		//==== write the adf ====
		std::cout<<"writing the adf\n";
		FILE* writer=fopen("angle.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#angle(degrees) adf\n");
			for(int i=0; i<adf.nbins(); ++i){
				fprintf(writer,"%12.8f %12.8f\n",adfx[i],adfy[i]);
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