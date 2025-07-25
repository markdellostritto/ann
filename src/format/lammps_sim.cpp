// c libraries
#include <ctime>
// c++ libraries
#include <iostream>
// str
#include "str/string.hpp"
#include "str/token.hpp"
// chem
#include "chem/ptable.hpp"
#include "chem/units.hpp"
// structure
#include "format/lammps_sim.hpp"

namespace LAMMPS{

namespace DUMP{

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim){
	static const char* funcName="read(const char*,const Interval&,AtomType&,Simulation&,Format&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
	//======== local function variables ========
	//==== file i/o ====
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* temp=new char[string::M];
		Token token;
	//==== time info ====
		int ts=0;//number of timesteps
	//==== cell info ====
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
	//==== atom info ====
		int natoms=0;
		DATA_ATOM dataAtom;
		FORMAT_ATOM formatAtom;
		int minindex=-1;
	//==== timing ====
		clock_t start,stop;
		double time;
	//==== misc ====
		bool error=false;
	//==== units ====
		double s_len=1.0,s_energy=1.0;
		
	try{
		//==== start the timer ====
		start=std::clock();
		
		//==== open the file ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"opening file\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error("I/O Error: Could not open file.");
		
		//==== find the number of timesteps ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading number of timesteps\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"TIMESTEP")!=NULL) ++ts;
		}
		std::rewind(reader);
		
		//==== read in the atom format ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading atom format\n";
		while(fgets(input,string::M,reader)!=NULL){
			/*if(std::strstr(input,"ITEM: ATOMS")!=NULL){
				int nTokens=string::substrN(input,string::WS)-2;
				std::strtok(input,string::WS);
				std::strtok(NULL,string::WS);
				for(int i=0; i<nTokens; ++i){
					std::strcpy(temp,std::strtok(NULL,string::WS));
					if(std::strcmp(temp,"id")==0) formatAtom.index=i;
					else if(std::strcmp(temp,"mol")==0) formatAtom.mol=i;
					else if(std::strcmp(temp,"type")==0) formatAtom.type=i;
					else if(std::strcmp(temp,"q")==0) formatAtom.q=i;
					else if(std::strcmp(temp,"mass")==0) formatAtom.m=i;
					else if(std::strcmp(temp,"x")==0) formatAtom.x=i;
					else if(std::strcmp(temp,"y")==0) formatAtom.y=i;
					else if(std::strcmp(temp,"z")==0) formatAtom.z=i;
					else if(std::strcmp(temp,"xu")==0) formatAtom.xu=i;
					else if(std::strcmp(temp,"yu")==0) formatAtom.yu=i;
					else if(std::strcmp(temp,"zu")==0) formatAtom.zu=i;
					else if(std::strcmp(temp,"vx")==0) formatAtom.vx=i;
					else if(std::strcmp(temp,"vy")==0) formatAtom.vy=i;
					else if(std::strcmp(temp,"vz")==0) formatAtom.vz=i;
					else if(std::strcmp(temp,"fx")==0) formatAtom.fx=i;
					else if(std::strcmp(temp,"fy")==0) formatAtom.fy=i;
					else if(std::strcmp(temp,"fz")==0) formatAtom.fz=i;
				}
				break;
			}*/
			if(std::strstr(input,"ITEM: ATOMS")!=NULL){
				token.read(input,string::WS).next(2);
				int i=0;
				while(!token.end()){
					const std::string tmp=token.next();
					if(tmp=="id") formatAtom.index=i;
					else if(tmp=="mol") formatAtom.mol=i;
					else if(tmp=="type") formatAtom.type=i;
					else if(tmp=="q") formatAtom.q=i;
					else if(tmp=="mass") formatAtom.m=i;
					else if(tmp=="x") formatAtom.x=i;
					else if(tmp=="y") formatAtom.y=i;
					else if(tmp=="z") formatAtom.z=i;
					else if(tmp=="xu") formatAtom.xu=i;
					else if(tmp=="yu") formatAtom.yu=i;
					else if(tmp=="zu") formatAtom.zu=i;
					else if(tmp=="vx") formatAtom.vx=i;
					else if(tmp=="vy") formatAtom.vy=i;
					else if(tmp=="vz") formatAtom.vz=i;
					else if(tmp=="fx") formatAtom.fx=i;
					else if(tmp=="fy") formatAtom.fy=i;
					else if(tmp=="fz") formatAtom.fz=i;
					i++;
				}
				break;
			}
		}
		if(LAMMPS_PRINT_DATA>0) std::cout<<"formatAtom = \n"<<formatAtom<<"\n";
		std::rewind(reader);
		
		//==== read in the number of atoms ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading number of atoms\n";
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"NUMBER OF ATOMS")!=NULL){
				natoms=std::atoi(fgets(input,string::M,reader));
				break;
			}
		}
		
		//==== find the min index ===
		/*while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"ITEM: ATOMS")!=NULL){
				minindex=natoms;
				for(int i=0; i<1; ++i){
					//read in the next line
					fgets(input,string::M,reader);
					//split the line into tokens
					std::vector<std::string> tokens;
					string::split(input,string::WS,tokens);
					//find index
					int index=-1;
					if(formatAtom.index>=0) index=std::atoi(tokens[formatAtom.index].c_str())-1;
					minindex=index;
				}
				for(int i=1; i<natoms; ++i){
					//read in the next line
					fgets(input,string::M,reader);
					//split the line into tokens
					std::vector<std::string> tokens;
					string::split(input,string::WS,tokens);
					//find index
					int index=-1;
					if(formatAtom.index>=0) index=std::atoi(tokens[formatAtom.index].c_str())-1;
					if(index<minindex) minindex=index;
				}
				break;
			}
		}*/
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"ITEM: ATOMS")!=NULL){
				minindex=natoms;
				for(int i=0; i<1; ++i){
					token.read(fgets(input,string::M,reader),string::WS).next(formatAtom.index);
					int index=-1;
					if(formatAtom.index>=0) index=std::atoi(token.next().c_str())-1;
					minindex=index;
				}
				for(int i=1; i<natoms; ++i){
					token.read(fgets(input,string::M,reader),string::WS).next(formatAtom.index);
					//find index
					int index=-1;
					if(formatAtom.index>=0) index=std::atoi(token.next().c_str())-1;
					if(index<minindex) minindex=index;
				}
				break;
			}
		}
		
		std::rewind(reader);
		
		//==== set the timesteps ====
		const int ibeg=Interval::index(interval.beg(),ts);
		const int iend=Interval::index(interval.end(),ts);
		const int tsint=iend-ibeg+1;
		const int nsteps=tsint/interval.stride();
		if(LAMMPS_PRINT_DATA>1){
			std::cout<<"interval  = "<<interval<<"\n";
			std::cout<<"ts        = "<<ts<<"\n";
			std::cout<<"(beg,end) = ("<<ibeg<<","<<iend<<")\n";
			std::cout<<"tsint     = "<<tsint<<"\n";
			std::cout<<"nsteps    = "<<nsteps<<"\n";
		}
		
		//==== resize the simulation ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"allocating memory\n";
		sim.resize(nsteps,natoms,atomT);
		
		//==== read atoms ====
		if(LAMMPS_PRINT_STATUS>0) std::cout<<"reading atoms\n";
		int timestep=0;
		while(fgets(input,string::M,reader)!=NULL){
			if(std::strstr(input,"BOX")!=NULL){
				//check timestep
				if(timestep<ibeg){continue;}
				if(timestep%interval.stride()!=0){continue;}
				if(LAMMPS_PRINT_DATA>1) std::cout<<"Cell: "<<timestep<<"\n";
				//local variables
				Token token;
				lv.setZero();
				//x
				token.read(fgets(input,string::M,reader),string::WS);
				const double xlob=std::atof(token.next().c_str());//xlo
				const double xhib=std::atof(token.next().c_str());//xhi
				double xy=0;
				if(!token.end()) xy=std::atof(token.next().c_str());//xy
				//y
				token.read(fgets(input,string::M,reader),string::WS);
				const double ylob=std::atof(token.next().c_str());//ylo
				const double yhib=std::atof(token.next().c_str());//yhi
				double xz=0;
				if(!token.end()) xz=std::atof(token.next().c_str());//xz
				//z
				token.read(fgets(input,string::M,reader),string::WS);
				const double zlob=std::atof(token.next().c_str());//zlo
				const double zhib=std::atof(token.next().c_str());//zhi
				double yz=0;
				if(!token.end()) yz=std::atof(token.next().c_str());//yz
				//set xlo,xhi,ylo,yhi,zlo,zhi
				const double xlo=xlob-std::min(0.0,std::min(xy,std::min(xz,xy+xz)));
				const double xhi=xhib-std::max(0.0,std::max(xy,std::max(xz,xy+xz)));
				const double ylo=ylob-std::min(0.0,yz);
				const double yhi=yhib-std::max(0.0,yz);
				const double zlo=zlob;
				const double zhi=zhib;
				//set lv
				lv(0,0)=xhi-xlo;
				lv(1,1)=yhi-ylo;
				lv(2,2)=zhi-zlo;
				lv(0,1)=xy;
				lv(0,2)=xz;
				lv(1,2)=yz;
				//set cell
				static_cast<Cell&>(sim.frame(timestep/interval.stride()-ibeg)).init(lv*s_len);
			}
			if(std::strstr(input,"ITEM: ATOMS")!=NULL){
				if(LAMMPS_PRINT_DATA>1) std::cout<<"Atoms: "<<timestep<<"\n";
				if(timestep<ibeg){++timestep; continue;}
				if(timestep%interval.stride()!=0){++timestep; continue;}
				const int lts=timestep/interval.stride()-ibeg;//local time step
				for(int i=0; i<natoms; ++i){
					std::vector<std::string> tokens;
					token.read(fgets(input,string::M,reader),string::WS);
					while(!token.end()) tokens.push_back(token.next());
					//read in the data
					if(formatAtom.q>=0) dataAtom.q=std::atof(tokens[formatAtom.q].c_str());
					if(formatAtom.m>=0) dataAtom.m=std::atof(tokens[formatAtom.m].c_str());
					if(formatAtom.x>=0) dataAtom.posn[0]=std::atof(tokens[formatAtom.x].c_str());
					if(formatAtom.y>=0) dataAtom.posn[1]=std::atof(tokens[formatAtom.y].c_str());
					if(formatAtom.z>=0) dataAtom.posn[2]=std::atof(tokens[formatAtom.z].c_str());
					if(formatAtom.xu>=0) dataAtom.posn[0]=std::atof(tokens[formatAtom.xu].c_str());
					if(formatAtom.yu>=0) dataAtom.posn[1]=std::atof(tokens[formatAtom.yu].c_str());
					if(formatAtom.zu>=0) dataAtom.posn[2]=std::atof(tokens[formatAtom.zu].c_str());
					if(formatAtom.vx>=0) dataAtom.vel[0]=std::atof(tokens[formatAtom.vx].c_str());
					if(formatAtom.vy>=0) dataAtom.vel[1]=std::atof(tokens[formatAtom.vy].c_str());
					if(formatAtom.vz>=0) dataAtom.vel[2]=std::atof(tokens[formatAtom.vz].c_str());
					if(formatAtom.fx>=0) dataAtom.force[0]=std::atof(tokens[formatAtom.fx].c_str());
					if(formatAtom.fy>=0) dataAtom.force[1]=std::atof(tokens[formatAtom.fy].c_str());
					if(formatAtom.fz>=0) dataAtom.force[2]=std::atof(tokens[formatAtom.fz].c_str());
					if(formatAtom.index>=0) dataAtom.index=std::atoi(tokens[formatAtom.index].c_str())-1;
					if(formatAtom.type>=0) dataAtom.type=std::atoi(tokens[formatAtom.type].c_str())-1;
					//set the simulation data
					const int index=dataAtom.index-minindex;
					if(atomT.type) sim.frame(lts).type(index)=dataAtom.type;
					if(atomT.index) sim.frame(lts).index(index)=dataAtom.index;
					if(atomT.posn) sim.frame(lts).posn(index)=dataAtom.posn*s_len;
					if(atomT.charge) sim.frame(lts).charge(index)=dataAtom.q;
					if(atomT.mass) sim.frame(lts).mass(index)=dataAtom.m;
					if(atomT.vel) sim.frame(lts).vel(index)=dataAtom.vel*s_len;
					if(atomT.force) sim.frame(lts).force(index)=dataAtom.force*s_energy/s_len;
					if(atomT.name) sim.frame(lts).name(index)=std::string("X")+std::to_string(dataAtom.type);
				}
				++timestep;
			}
			if(timestep>iend) break;
		}
		
		//==== move the atoms within the cell ====
		for(int t=0; t<sim.timesteps(); ++t){
			for(int n=0; n<sim.frame(t).nAtoms(); ++n){
				sim.frame(t).modv(sim.frame(t).posn(n),sim.frame(t).posn(n));
			}
		}
		
		//==== stop the timer ====
		stop=std::clock();
		
		//==== print the time ====
		time=((double)(stop-start))/CLOCKS_PER_SEC;
		std::cout<<"positions read in "<<time<<" seconds\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free local variables
	if(reader!=NULL) fclose(reader);
	delete[] input;
	delete[] temp;
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim){
	static const char* funcName="write<AtomT>(const char*,const Interval&,const AtomType&,Simulation&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
	//local variables
	FILE* writer=NULL;
	bool error=false;
	
	try{
		//open the file
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error("Unable to open file.");
		
		//set the beginning and ending timesteps
		const int ibeg=Interval::index(interval.beg(),sim.timesteps());
		const int iend=Interval::index(interval.end(),sim.timesteps());
		
		for(int t=ibeg; t<=iend; ++t){
			fprintf(writer,"ITEM: TIMESTEP\n");
			fprintf(writer,"%i\n",t);
			fprintf(writer,"ITEM: NUMBER OF ATOMS\n");
			fprintf(writer,"%i\n",sim.frame(t).nAtoms());
			fprintf(writer,"ITEM: BOX BOUNDS pp pp pp\n");
			const Eigen::Vector3d ra=sim.frame(t).R().col(0);
			const Eigen::Vector3d rb=sim.frame(t).R().col(1);
			const Eigen::Vector3d rc=sim.frame(t).R().col(2);
			const double A=ra.norm();
			const double B=rb.norm();
			const double C=rc.norm();
			const double cosA=rb.dot(rc)/(B*C);
			const double cosB=ra.dot(rc)/(A*C);
			const double cosC=ra.dot(rb)/(A*B);
			const double lx=A;
			const double xy=B*cosC;
			const double xz=C*cosB;
			const double ly=sqrt(B*B-xy*xy);
			const double yz=(B*C*cosA-xy*xz)/ly;
			const double lz=sqrt(C*C-xz*xz-yz*yz);
			fprintf(writer,"%f %f %f\n",0.0,lx,xy);
			fprintf(writer,"%f %f %f\n",0.0,ly,xz);
			fprintf(writer,"%f %f %f\n",0.0,lz,yz);
			fprintf(writer,"ITEM: ATOMS id type x y z\n");
			for(int n=0; n<sim.frame(t).nAtoms(); ++n){
				fprintf(writer,"%i %i %f %f %f\n",n+1,sim.frame(t).type(n),
					sim.frame(t).posn(n)[0],sim.frame(t).posn(n)[1],sim.frame(t).posn(n)[2]
				);
			}
		}
		
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}
	
}

}
