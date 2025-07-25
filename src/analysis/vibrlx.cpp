// c libraries
#include <cstdlib>
#include <ctime>
//c++
#include <iostream>
#include <vector>
// structure
#include "struc/sim.hpp"
// signal
#include "signal/fft.hpp"
#include "signal/window.hpp"
// format
#include "format/vasp_sim.hpp"
#include "format/lammps_sim.hpp"
#include "format/cp2k_sim.hpp"
#include "format/xyz_sim.hpp"
// io
#include "str/string.hpp"
// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"

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
		std::vector<std::string> strlist;
	//==== simulation variables ====
		Simulation sim;
		Interval interval;
		Eigen::Vector3d box=Eigen::Vector3d::Zero();
		Eigen::Vector3d offset=Eigen::Vector3d::Zero();
		double ts=0;
		double T=0;
		double fwidth=0.0;
	//==== atom type ====
		sim.atomT().name	=true;
		sim.atomT().an		=true;
		sim.atomT().mass	=true;
		sim.atomT().type	=true;
		sim.atomT().index	=true;
		sim.atomT().posn	=true;
		sim.atomT().force	=true;
	//==== miscellaneous ====
		bool error=false;
	//==== units ====
		units::System unitsys;
	//==== molecule =====
		int atom1=-1;
		int atom2=-1;
		double mass1=0;
		double mass2=0;
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
			} else if(tag=="ATOM1"){
				atom1=std::atoi(token.next().c_str());
			} else if(tag=="ATOM2"){
				atom2=std::atoi(token.next().c_str());
			} else if(tag=="MASS1"){
				mass1=std::atof(token.next().c_str());
			} else if(tag=="MASS2"){
				mass2=std::atof(token.next().c_str());
			} else if(tag=="TS"){
				ts=std::atof(token.next().c_str());
			} else if(tag=="TEMP"){
				T=std::atof(token.next().c_str());
			} else if(tag=="FWIDTH"){
				fwidth=std::atof(token.next().c_str());
			}
		}
		//close the file
		fclose(reader);
		reader=NULL;
		
		//======== initialize the unit system ========
		std::cout<<"initializing the unit system\n";
		units::consts::init(unitsys);
		const double hbar=units::consts::hbar();
		const double kb=units::consts::kb();
		
		//======== print the parameters ========
		std::cout<<"*********************************************************\n";
		std::cout<<"********************** PARAMETERS **********************\n";
		std::cout<<"UNITS    = "<<unitsys<<"\n";
		std::cout<<"N_PRINT  = "<<nprint<<"\n";
		std::cout<<"ATOM_T   = "<<sim.atomT()<<"\n";
		std::cout<<"SIM_STR  = \""<<simstr<<"\"\n";
		std::cout<<"FORMAT   = "<<fileFormat<<"\n";
		std::cout<<"INTERVAL = "<<interval<<"\n";
		std::cout<<"OFFSET   = "<<offset.transpose()<<"\n";
		std::cout<<"BOX      = "<<box[0]<<" "<<box[1]<<" "<<box[2]<<"\n";
		std::cout<<"ATOMS    = "<<atom1<<" "<<atom2<<"\n";
		std::cout<<"MASSES   = "<<mass1<<" "<<mass2<<"\n";
		std::cout<<"TIMESTEP = "<<ts<<"\n";
		std::cout<<"TEMP     = "<<T<<"\n";
		std::cout<<"HBAR     = "<<hbar<<"\n";
		std::cout<<"KB       = "<<kb<<"\n";
		std::cout<<"FWIDTH   = "<<fwidth<<"\n";
		std::cout<<"********************** PARAMETERS **********************\n";
		std::cout<<"*********************************************************\n";
		
		//======== check the parameters ========
		if(interval.end==0 || interval.beg==0 || (interval.end<interval.beg && interval.end>0)) throw std::invalid_argument("Invalid timestep interval.");
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		if(fileFormat==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
		if(atom1<0 || atom2<0) throw std::invalid_argument("Invalid atom indices.");
		if(mass1<=0 || mass2<=0) throw std::invalid_argument("Invalid atom masses.");
		if(ts<=0) throw std::invalid_argument("Invalid timestep.");
		if(T<=0) throw std::invalid_argument("Invalid temperature.");
		
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
		if(interval.end<0) interval.end+=sim.timesteps()+1;
		
		//======== print the data to screen ========
		std::cout<<"SIMULATION = \n"<<sim<<"\n";
		std::cout<<sim.frame(0)<<"\n";
		
		//======== compute the requested properties ========
		std::cout<<"computing properties\n";
		clock_t start=std::clock();
		if(nprint==0) nprint=sim.timesteps()/10;
		if(nprint==0) nprint=1;
		const double mu=mass1*mass2/(mass1+mass2);
		std::vector<double> force(sim.timesteps(),0);
		Eigen::Vector3d dr;
		double davg=0;
		for(int t=0; t<sim.timesteps(); ++t){
			if(t%nprint==0) std::cout<<"ts "<<t<<"\n";
			const Eigen::Vector3d r=sim.frame(t).diff(sim.frame(t).posn(atom1),sim.frame(t).posn(atom2),dr);
			davg+=r.norm();
			const Eigen::Vector3d rn=r/r.norm();
			force[t]=mu*(sim.frame(t).force(atom1)/mass1-sim.frame(t).force(atom2)/mass2).dot(rn);
		}
		davg/=sim.timesteps();
		clock_t stop=std::clock();
		std::cout<<"time = "<<((double)(stop-start))/CLOCKS_PER_SEC<<"\n";
		
		const int N=sim.timesteps();
		const double df=1000.0/(ts*2.0*N);
		typedef fourier::FFT<1,fourier::DataT::COMPLEX,fourier::DataT::COMPLEX> FFT1D;
		FFT1D fftf(2*N,FFTW_FORWARD); fftf.init();
		FFT1D fftb(2*N,FFTW_BACKWARD); fftb.init();
		std::function<double(double)> window=window::BlackmanHarris(N);
		
		//load the forces
		for(int t=0; t<N; ++t){
			fftf.in(t)[0]=force[t];
			fftf.in(t)[1]=0;
		}
		//pad with zeros
		for(int t=N; t<2*N; ++t){
			fftf.in(t)[0]=0;
			fftf.in(t)[1]=0;
		}
		//tranform the forces
		fftf.transform();
		//record the Fourier transform of the correlation function
		for(int t=0; t<2*N; ++t){
			fftb.in(t)[0]=(fftf.out(t)[0]*fftf.out(t)[0]+fftf.out(t)[1]*fftf.out(t)[1])/(2.0*N);
			fftb.in(t)[1]=0;
		}
		//perform the reverse transform
		fftb.transform();
		//normalize, shift, and window the correlation function
		for(int t=0; t<N; ++t){
			const double fac=window(t)/(t+1);
			fftf.in(t)[0]=fftb.out(t+N)[0]*fac;
			fftf.in(t)[1]=fftb.out(t+N)[1]*fac;
		}
		for(int t=N; t<2*N; ++t){
			const double fac=window(t-N)/(2*N-t);
			fftf.in(t)[0]=fftb.out(t-N)[0]*fac;
			fftf.in(t)[1]=fftb.out(t-N)[1]*fac;
		}
		//perform a forward transform 
		fftf.transform();
		
		const double norm=1000*davg*davg*2.0/(2.0*N*hbar*hbar);
		//print the correlation function
		std::cout<<"time space\n";
		for(int t=0; t<2*N; ++t) std::cout<<t-N<<" "<<fftb.out(t)[0]<<" "<<fftb.out(t)[1]<<"\n";
		//print the frequency
		std::cout<<"freq space\n";
		for(int t=0; t<2*N; ++t) std::cout<<t*df<<" "<<fftf.out(t)[0]<<" "<<fftf.out(t)[1]<<"\n";
		//write the frequency
		std::cout<<"power spectrum\n";
		std::vector<double> freq(2*N);
		std::vector<double> ps(2*N);
		for(int t=0; t<2*N; ++t) freq[t]=t*df;
		for(int t=0; t<2*N; ++t) ps[t]=norm*sqrt(fftf.out(t)[0]*fftf.out(t)[0]+fftf.out(t)[1]*fftf.out(t)[1])/(1.0+exp(-hbar*freq[t]/(kb*T)));
		if(fwidth>0.0) signala::smooth(ps,fwidth);
		FILE* writer=fopen("vibrlx.dat","w");
		if(writer!=NULL){
			fprintf(writer,"#freq(THz) ps\n");
			for(int t=0; t<2*N; ++t){
				fprintf(writer,"%12.8f %12.8f\n",freq[t],ps[t]);
			}
		}
		std::cout<<"ANALYSIS COMPLETED\n";
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