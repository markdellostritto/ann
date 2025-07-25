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
		Group group;
		Pair pair;
	//==== atom type ====
		sim.atomT().name	=true;
		sim.atomT().mass	=true;
		sim.atomT().type	=true;
		sim.atomT().posn	=true;
	//==== calculation ====
		int nprint=-1;
		std::vector<int> ncomp_;//number of connected components
		std::vector<int> cdist_;//cluster distribution
		double rmax=0;
		double rmax2=0;
		Eigen::VectorXf degree_;
		Eigen::MatrixXf adj_;
		Eigen::MatrixXf laplacian_;
		bool norm=true;
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
			} else if(tag=="GROUP"){
				group.read(token);
			} else if(tag=="RADIUS"){
				rmax=std::atof(token.next().c_str());
				rmax2=rmax*rmax;
			} else if(tag=="PAIR"){
				pair.read(token);
			} else if(tag=="NPRINT"){
				nprint=std::atoi(token.next().c_str());
			} else if(tag=="NORM"){
				norm=string::boolean(token.next().c_str());
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
		std::cout<<"RADIUS = "<<rmax<<"\n";
		std::cout<<"GROUP  = "<<group<<"\n";
		std::cout<<"NORM   = "<<norm<<"\n";
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
		const int nAtoms=sim.frame(0).nAtoms();
		const int size=group.size();
		degree_.resize(size);
		adj_.resize(size,size);
		laplacian_.resize(size,size);
		pair.build(sim.frame(0));
		cdist_.resize(size);
		int count=0;
		for(int t=0; t<sim.timesteps(); ++t){
			const Structure& struc=sim.frame(t);
			Eigen::Vector3d diff;
			if(t%nprint==0) std::cout<<"T = "<<t<<"\n";
			//reset pair
			if(t%pair.stride()==0) pair.build(sim.frame(t));
			//reset matrices
			degree_.setZero();
			adj_.setZero();
			laplacian_.setZero();
			//set degree and adj matrices
			for(int n=0; n<size; ++n){
				for(int m=n+1; m<size; ++m){
					const double dist2=struc.dist2(struc.posn(group.atom(n)),struc.posn(group.atom(m)),diff);
					if(dist2<rmax2){
						++degree_[n];
						++degree_[m];
						++adj_(n,m);
					}
				}
			}
			//symmetrize the adjacency matrix
			laplacian_.noalias()=adj_.transpose();
			adj_.noalias()+=laplacian_;
			//compute the laplacian matrix
			laplacian_=degree_.asDiagonal();
			laplacian_.noalias()-=adj_;
			//compute the eigenvalues and eigenvectors
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(laplacian_);
			//find the number of connected components
			int nzero=0;
			const Eigen::VectorXf& evalues=eigensolver.eigenvalues();
			std::vector<int> zindices;
			for(int i=0; i<size; ++i){
				if(evalues[i]<math::constant::ZERO){
					++nzero;
					zindices.push_back(i);
				}
			}
			//find the size of each connected component 
			const Eigen::MatrixXf& evectors=eigensolver.eigenvectors();
			std::vector<int> csize(nzero,0);
			for(int i=0; i<nzero; ++i){
				for(int j=0; j<size; ++j){
					if(std::fabs(evectors(j,zindices[i]))>math::constant::ZERO) ++csize[i];
				}
			}
			//bin the sizes
			for(int i=0; i<nzero; ++i){
				++cdist_[csize[i]-1];
				++count;
			}
		}
		
		//==== write cluster ====
		std::cout<<"writing cluster\n";
		FILE* writer=fopen("cluster.dat","w");
		if(writer==NULL) throw std::runtime_error("Could not open output file.");
		fprintf(writer,"#CLUSTER_SIZE FREQ\n");
		if(norm){
			for(int i=0; i<group.size(); ++i){
				fprintf(writer,"%i %f\n",i+1,cdist_[i]*1.0/(1.0*count));
			}
		} else {
			for(int i=0; i<group.size(); ++i){
				fprintf(writer,"%i %f\n",i+1,cdist_[i]*1.0);
			}
		}
		//close the output file
		fclose(writer);
		writer=NULL;
		
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