// c libraries
#include <ctime>
// c++ libraries
#include <iostream>
#include <string>
#include <stdexcept>
// eigen libraries
#include <Eigen/Dense>
// ann - structure
#include "structure.hpp"
// ann - strings
#include "string.hpp"
// ann - units
#include "units.hpp"
// ann - math
#include "math_const.hpp"
// ann - ann
#include "ann.hpp"

namespace ANN{
	
void read(const char* file, const AtomType& atomT, Structure& struc){
	const char* func_name="ANN::read(FILE*,const AtomType&,Structure&)";
	if(ANN_PRINT_FUNC>0) std::cout<<func_name<<":\n";
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		std::vector<std::string> strlist;
	//structure
		int natoms=0;
		Eigen::Matrix3d lv;
		Format f;
	//misc
		bool error=false;
	
	try{
		//open the file
		if(ANN_PRINT_STATUS>0) std::cout<<"opening file: "<<file<<"\n";
		reader=fopen(file,"r");
		if(reader==NULL) throw std::runtime_error(std::string("ERROR: Could not open file: \"")+std::string(file));
		
		//read name
		fgets(input,string::M,reader);
		//read energy
		struc.energy()=std::atof(fgets(input,string::M,reader));
		//read natoms
		natoms=std::atoi(fgets(input,string::M,reader));
		
		//read cell
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,0),&lv(1,0),&lv(2,0));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,1),&lv(1,1),&lv(2,1));
		std::sscanf(fgets(input,string::M,reader),"%lf %lf %lf",&lv(0,2),&lv(1,2),&lv(2,2));
		static_cast<Cell&>(struc).init(lv);
		
		//resize struc
		struc.resize(natoms,atomT);
		
		//read format
		fgets(input,string::M,reader);
		string::split(input,string::WS,strlist);
		for(int i=0; i<strlist.size(); ++i){
			if(strlist[i]=="name") f.name=i;
			else if(strlist[i]=="mass") f.mass=i;
			else if(strlist[i]=="charge") f.charge=i;
			else if(strlist[i]=="x") f.x=i;
			else if(strlist[i]=="y") f.y=i;
			else if(strlist[i]=="z") f.z=i;
			else if(strlist[i]=="vx") f.vx=i;
			else if(strlist[i]=="vy") f.vy=i;
			else if(strlist[i]=="vz") f.vz=i;
			else if(strlist[i]=="fx") f.fx=i;
			else if(strlist[i]=="fy") f.fy=i;
			else if(strlist[i]=="fz") f.fz=i;
			else if(strlist[i]=="symm") f.symm=i;
		}
		//check format - symm must be at the back
		if(f.symm>=0){
			if(f.symm<f.name || f.symm<f.mass || f.symm<f.charge || f.symm<f.x || f.symm<f.y || f.symm<f.z ||
				f.symm<f.vx || f.symm<f.vy || f.symm<f.vz || f.symm<f.fx || f.symm<f.fy || f.symm<f.fz)
				throw std::invalid_argument("Invalid format: symm must be at back");
		}
		
		//read atoms
		for(int i=0; i<natoms; ++i){
			fgets(input,string::M,reader);
			string::split(input,string::WS,strlist);
			if(atomT.name && f.name>=0) struc.name(i)=strlist[f.name];
			if(atomT.mass && f.mass>=0) struc.mass(i)=std::atof(strlist[f.mass].c_str());
			if(atomT.charge && f.charge>=0) struc.charge(i)=std::atof(strlist[f.charge].c_str());
			if(atomT.posn && f.x>=0 && f.y>=0 && f.z>=0) struc.posn(i)<<std::atof(strlist[f.x].c_str()),std::atof(strlist[f.y].c_str()),std::atof(strlist[f.z].c_str());
			if(atomT.vel && f.vx>=0 && f.vy>=0 && f.vz>=0) struc.vel(i)<<std::atof(strlist[f.vx].c_str()),std::atof(strlist[f.vy].c_str()),std::atof(strlist[f.vz].c_str());
			if(atomT.force && f.fx>=0 && f.fy>=0 && f.fz>=0) struc.vel(i)<<std::atof(strlist[f.fx].c_str()),std::atof(strlist[f.fy].c_str()),std::atof(strlist[f.fz].c_str());
			if(atomT.symm && f.symm>=0){
				const int size=strlist.size()-f.symm;
				struc.symm(i).resize(size);
				for(int j=0; j<size; ++j) struc.symm(i)[j]=std::atof(strlist[f.symm+j].c_str());
			}
		}
		
		//close file
		fclose(reader);
		reader=NULL;
	}catch(std::exception& e){
		std::cout<<"Error in "<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] input;
}

void write(const char* file, const AtomType& atomT, const Structure& struc){
	const char* func_name="ANN::write(FILE*,const AtomType&,const Structure&)";
	if(ANN_PRINT_FUNC>0) std::cout<<func_name<<":\n";
	//file i/o
		FILE* writer=NULL;
	//structure
		Format f;
	//misc
		bool error=false;
		
	try{
		//open the file
		if(ANN_PRINT_STATUS>0) std::cout<<"opening file: "<<file<<"\n";
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error(std::string("ERROR: Could not open file: \"")+std::string(file));
		
		//write name
		fprintf(writer,"STRUCTURE\n");
		//write energy
		fprintf(writer,"%lf\n",struc.energy());
		//write natoms
		fprintf(writer,"%i\n",struc.nAtoms());
		
		//write cell
		const Eigen::Matrix3d lv=struc.R();
		fprintf(writer,"%lf %lf %lf\n",lv(0,0),lv(1,0),lv(2,0));
		fprintf(writer,"%lf %lf %lf\n",lv(0,1),lv(1,1),lv(2,1));
		fprintf(writer,"%lf %lf %lf\n",lv(0,2),lv(1,2),lv(2,2));
		
		//set format
		int count=0;
		if(atomT.name) f.name=count++;
		if(atomT.mass) f.mass=count++;
		if(atomT.charge) f.charge=count++;
		if(atomT.posn){f.x=count++; f.y=count++; f.z=count++;}
		if(atomT.vel){f.vx=count++; f.vy=count++; f.vz=count++;}
		if(atomT.force){f.fx=count++; f.fy=count++; f.fz=count++;}
		if(atomT.symm) f.symm=count++;
		//write format
		if(f.name>=0) fprintf(writer,"name ");
		if(f.charge>=0) fprintf(writer,"charge ");
		if(f.mass>=0) fprintf(writer,"mass ");
		if(f.x>=0) fprintf(writer,"x ");
		if(f.y>=0) fprintf(writer,"y ");
		if(f.z>=0) fprintf(writer,"z ");
		if(f.vx>=0) fprintf(writer,"vx ");
		if(f.vy>=0) fprintf(writer,"vy ");
		if(f.vz>=0) fprintf(writer,"vz ");
		if(f.fx>=0) fprintf(writer,"fx ");
		if(f.fy>=0) fprintf(writer,"fy ");
		if(f.fz>=0) fprintf(writer,"fz ");
		if(f.symm>=0) fprintf(writer,"symm ");
		fprintf(writer,"\n");
		
		//read atoms
		for(int i=0; i<struc.nAtoms(); ++i){
			if(atomT.name) fprintf(writer,"%s ",struc.name(i).c_str());
			if(atomT.mass) fprintf(writer,"%f ",struc.mass(i));
			if(atomT.charge) fprintf(writer,"%11.6f ",struc.charge(i));
			if(atomT.posn) fprintf(writer,"%11.6f %11.6f %11.6f ",struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2]);
			if(atomT.vel) fprintf(writer,"%11.6f %11.6f %11.6f ",struc.vel(i)[0],struc.vel(i)[1],struc.vel(i)[2]);
			if(atomT.force) fprintf(writer,"%11.6f %11.6f %11.6f ",struc.force(i)[0],struc.force(i)[1],struc.force(i)[2]);
			if(atomT.symm){
				for(int j=0; j<struc.symm(i).size(); ++j) fprintf(writer,"%11.6e ",struc.symm(i)[j]);
			}
			fprintf(writer,"\n");
		}
		
		//close file
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"Error in "<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
}

}