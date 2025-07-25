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
// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"
// string
#include "str/token.hpp"
#include "str/string.hpp"

//***********************************************************************
// Main
//***********************************************************************

int main(int argc, char* argv[]){
	//======== local function variables ========
		typedef fourier::FFT<1,fourier::DataT::COMPLEX,fourier::DataT::COMPLEX> FFT1D;
	//==== file i/o ====
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input    =new char[string::M];
		char* fvacf    =new char[string::M];
		char* fout     =new char[string::M];
	//==== simulation variables ====
		double ts=0;
		double T=0;
		double fwidth=0.0;
		double fpmin=0.0,fpmax=0.0;
	//==== miscellaneous ====
		bool error=false;
	//==== units ====
		units::System unitsys;
		
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
			if(tag=="UNITS"){
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
			} else if(tag=="TS"){
				ts=std::atof(token.next().c_str());
			} else if(tag=="TEMP"){
				T=std::atof(token.next().c_str());
			} else if(tag=="FWIDTH"){
				fwidth=std::atof(token.next().c_str());
			} else if(tag=="FINT"){
				fpmin=std::atof(token.next().c_str());
				fpmax=std::atof(token.next().c_str());
			} else if(tag=="VACF"){
				std::strcpy(fvacf,token.next().c_str());
			} else if(tag=="OUT"){
				std::strcpy(fout,token.next().c_str());
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
		std::cout<<"*********************************************************\n";
		std::cout<<"********************** PARAMETERS **********************\n";
		std::cout<<"UNITS    = "<<unitsys<<"\n";
		std::cout<<"TIMESTEP = "<<ts<<"\n";
		std::cout<<"TEMP     = "<<T<<"\n";
		std::cout<<"HBAR     = "<<hbar<<"\n";
		std::cout<<"KB       = "<<kb<<"\n";
		std::cout<<"FWIDTH   = "<<fwidth<<"\n";
		std::cout<<"FINT     = "<<fpmin<<" "<<fpmax<<"\n";
		std::cout<<"FVACF    = "<<fvacf<<"\n";
		std::cout<<"FOUT     = "<<fout<<"\n";
		std::cout<<"********************** PARAMETERS **********************\n";
		std::cout<<"*********************************************************\n";
		
		//======== check the parameters ========
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		if(ts<=0) throw std::invalid_argument("Invalid timestep.");
		
		//======== read vacf ========
		std::cout<<"reading vacf\n";
		reader=fopen(fvacf,"r");
		std::vector<int> step;
		std::vector<std::vector<double> > vacf(4);
		if(reader!=NULL){
			while(fgets(input,string::M,reader)!=NULL){
				if(input[0]=='#') continue;
				Token token(input,string::WS);
				step.push_back(std::atoi(token.next().c_str()));
				vacf[0].push_back(std::atof(token.next().c_str()));
				vacf[1].push_back(std::atof(token.next().c_str()));
				vacf[2].push_back(std::atof(token.next().c_str()));
				vacf[3].push_back(std::atof(token.next().c_str()));
			}
			fclose(reader);
			reader=NULL;
		}
		
		//======== compute the requested properties ========
		std::cout<<"computing properties\n";
		clock_t start=std::clock();
		
		const int N=step.size();
		std::cout<<"N = "<<N<<"\n";
		const double fmin=1000.0/(ts*N);//min freq
		const double fmax=1000.0/ts;//min freq
		const double df=fmin;
		const int imin=(int)std::floor(fpmin/df);
		const int imax=(int)std::ceil(fpmax/df);
		
		/*FFT1D fft(2*N,FFTW_FORWARD); fft.init();
		std::function<double(double)> window=window::BlackmanHarris(2*N);
		std::vector<std::vector<double> > vdosr(4,std::vector<double>(2*N));
		std::vector<std::vector<double> > vdosi(4,std::vector<double>(2*N));
		std::vector<std::vector<double> > vdost(4,std::vector<double>(2*N));
		std::vector<double> error(4,0.0);
		
		for(int n=0; n<4; ++n){
			for(int t=0; t<N; ++t){
				const double fac=window(t);
				fft.in(t)[0]=vacf[n][N-1-t]*fac;
				fft.in(t)[1]=0.0;
			}
			for(int t=N; t<2*N; ++t){
				const double fac=window(t);
				fft.in(t)[0]=vacf[n][t-N]*fac;
				fft.in(t)[1]=0.0;
			}
			fft.transform();
			const double norm=1.0/sqrt(2.0*N);
			for(int t=0; t<2*N; ++t){
				vdosr[n][t]=fft.out(t)[0]*norm;
				vdosi[n][t]=fft.out(t)[1]*norm;
				vdost[n][t]=sqrt(fft.out(t)[0]*fft.out(t)[0]+fft.out(t)[1]*fft.out(t)[1])*norm;
			}
			//signala::smooth(vdosr[n],fwidth);
			//signala::smooth(vdosi[n],fwidth);
			signala::smooth(vdost[n],fwidth);
		}*/
		
		FFT1D fft(N,FFTW_FORWARD); fft.init();
		std::function<double(double)> window=window::BlackmanHarris(N);
		std::vector<std::vector<double> > vdosr(4,std::vector<double>(N));
		std::vector<std::vector<double> > vdosi(4,std::vector<double>(N));
		std::vector<std::vector<double> > vdost(4,std::vector<double>(N));
		std::vector<double> error(4,0.0);
		
		for(int n=0; n<4; ++n){
			for(int t=0; t<N; ++t){
				const double fac=window(t);
				fft.in(t)[0]=vacf[n][t]*fac;
				fft.in(t)[1]=0.0;
			}
			fft.transform();
			const double norm=1.0/sqrt(N);
			for(int t=0; t<N; ++t){
				vdosr[n][t]=fft.out(t)[0]*norm;
				vdosi[n][t]=fft.out(t)[1]*norm;
				vdost[n][t]=sqrt(fft.out(t)[0]*fft.out(t)[0]+fft.out(t)[1]*fft.out(t)[1])*norm;
			}
			//signala::smooth(vdosr[n],fwidth);
			//signala::smooth(vdosi[n],fwidth);
			signala::smooth(vdost[n],fwidth);
		}
		
		//write the vdos
		FILE* writer=fopen(fout,"w");
		if(writer!=NULL){
			fprintf(writer,"#freq(THz) vdosx vdosy vdosz vdos\n");
			for(int t=imin; t<imax; ++t){
				fprintf(writer,"%12.8f %12.8f %12.8f %12.8f %12.8f\n",t*df,vdost[0][t],vdost[1][t],vdost[2][t],vdost[3][t]);
			}
			fclose(writer);
			writer=NULL;
		}
		
		/*FILE* writer2=fopen("test.dat","w");
		if(writer2!=NULL){
			fprintf(writer2,"#freq(THz) vdosr vdosi\n");
			for(int t=imin; t<imax; ++t){
				fprintf(writer2,"%12.8f %12.8f %12.8f\n",t*df,vdosr[3][t],vdosi[3][t]);
			}
			fclose(writer2);
			writer2=NULL;
		}*/
		
		
	}catch(std::exception& e){
		std::cout<<e.what()<<"\n";
		std::cout<<"ANALYSIS FAILED.\n";
		error=true;
	}
	
	std::cout<<"freeing local variables\n";
	delete[] paramfile;
	delete[] input;
	delete[] fvacf;
	delete[] fout;
	
	std::cout<<"exiting program\n";
	if(error) return 1;
	else return 0;
}