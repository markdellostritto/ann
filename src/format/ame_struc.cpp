// c libraries
#include <cstdio>
#include <ctime>
// c++ libraries
#include <string>
#include <stdexcept>
#include <iostream>
// ann - structure
#include "struc/structure.hpp"
// ann - strings
#include "str/string.hpp"
#include "str/token.hpp"
// ann - chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// ann - format
#include "format/ame_struc.hpp"

namespace AME{

//*****************************************************
//reading
//*****************************************************

void read(const char* file, const AtomType& atomT, Structure& struc){
	if(AME_PRINT_FUNC>0) std::cout<<"AME::read(const char*,const AtomType&,Structure&):\n";
	//==== local function variables ====
	//file i/o
		FILE* reader=NULL;
		char* input=new char[string::M];
		char* name=new char[string::M];
		Token token;
	//atom info
		int natoms=0;
		Eigen::Vector3d posn;
		double pe=0;
		Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
		FORMAT_ATOM format;
	//atom data
		std::vector<Eigen::Vector3d> r_;
		std::vector<Eigen::Vector3d> f_;
		std::vector<double> q_;
		std::vector<double> m_;
		std::vector<double> c6_;
		std::vector<std::string> name_;
	//units
		units::System units;
		
	//open file
	if(AME_PRINT_STATUS>0) std::cout<<"opening file\n";
	reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error(std::string("ERROR in AME::read(const char*,const AtomType&,Structure&): Could not open file: ")+std::string(file));
	
	//read lines
	while(fgets(input,string::M,reader)!=NULL){
		token.read(input,string::WS);
		const std::string& label=token.next();
		if(label=="units"){
			units=units::System::read(token.next().c_str());
		} else if(label=="energy"){
			pe=std::atof(token.next().c_str());
		} else if(label=="cell"){
			std::cout<<"reading cell\n";
			lv(0,0)=std::atof(token.next().c_str());
			lv(1,0)=std::atof(token.next().c_str());
			lv(2,0)=std::atof(token.next().c_str());
			lv(0,1)=std::atof(token.next().c_str());
			lv(1,1)=std::atof(token.next().c_str());
			lv(2,1)=std::atof(token.next().c_str());
			lv(0,2)=std::atof(token.next().c_str());
			lv(1,2)=std::atof(token.next().c_str());
			lv(2,2)=std::atof(token.next().c_str());
		} else if(label=="atoms"){
			std::vector<std::string> strlist;
			natoms=std::atoi(token.next().c_str());
			//read format
			string::split(fgets(input,string::M,reader),string::WS,strlist);
			for(int i=0; i<strlist.size(); ++i){
				const char* tmpstr=strlist.at(i).c_str();
				if(std::strcmp(tmpstr,"name")==0) format.name=i;
				else if(std::strcmp(tmpstr,"q")==0) format.q=i;
				else if(std::strcmp(tmpstr,"mass")==0) format.m=i;
				else if(std::strcmp(tmpstr,"rx")==0) format.rx=i;
				else if(std::strcmp(tmpstr,"ry")==0) format.ry=i;
				else if(std::strcmp(tmpstr,"rz")==0) format.rz=i;
				else if(std::strcmp(tmpstr,"fx")==0) format.fx=i;
				else if(std::strcmp(tmpstr,"fy")==0) format.fy=i;
				else if(std::strcmp(tmpstr,"fz")==0) format.fz=i;
			}
			//read atom data
			r_.resize(natoms,Eigen::Vector3d::Zero());
			f_.resize(natoms,Eigen::Vector3d::Zero());
			q_.resize(natoms,0.0);
			name_.resize(natoms,"NULL");
			for(int i=0; i<natoms; ++i){
				string::split(fgets(input,string::M,reader),string::WS,strlist);
				if(format.name>=0) name_[i]=strlist[format.name].c_str();
				if(format.q>=0) q_[i]=std::atof(strlist[format.q].c_str());
				if(format.m>=0) m_[i]=std::atof(strlist[format.m].c_str());
				if(format.c6>=0) c6_[i]=std::atof(strlist[format.c6].c_str());
				if(format.rx>=0) r_[i][0]=std::atof(strlist[format.rx].c_str());
				if(format.ry>=0) r_[i][1]=std::atof(strlist[format.ry].c_str());
				if(format.rz>=0) r_[i][2]=std::atof(strlist[format.rz].c_str());
				if(format.fx>=0) f_[i][0]=std::atof(strlist[format.fx].c_str());
				if(format.fy>=0) f_[i][1]=std::atof(strlist[format.fy].c_str());
				if(format.fz>=0) f_[i][2]=std::atof(strlist[format.fz].c_str());
			}
		}
	}
	//close the file
	if(AME_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(reader);
	reader=NULL;
	
	//resize the structure
	if(AME_PRINT_STATUS>0) std::cout<<"resizing structure\n";
	struc.resize(natoms,atomT);
	static_cast<Cell&>(struc).init(lv);
	
	//set the atom data
	if(AME_PRINT_STATUS>0) std::cout<<"setting atom data\n";
	struc.pe()=pe;
	for(int i=0; i<natoms; ++i){
		if(atomT.posn) struc.posn(i)=r_[i];
		if(atomT.force) struc.force(i)=f_[i];
		if(atomT.charge) struc.charge(i)=q_[i];
		if(atomT.c6) struc.c6(i)=c6_[i];
		if(atomT.name) struc.name(i)=name_[i];
		if(atomT.name && atomT.an) struc.an(i)=ptable::an(struc.name(i).c_str());
	}
	
	//free memory
	delete[] input;
	delete[] name;
}

//*****************************************************
//writing
//*****************************************************

void write(const char* file, const AtomType& atomT, const Structure& struc){
	if(AME_PRINT_FUNC>0) std::cout<<"write(const char*,const AtomType&,const Structure&):\n";
	//==== local function variables ====
	//file i/o
		FILE* writer=NULL;
	//atom info
		FORMAT_ATOM format;
	//atom data
		std::vector<Eigen::Vector3d> r_,f_;
		std::vector<double> q_,eta_,chi_;
		std::vector<std::string> name_;
	//units
		units::System units;
		
	//open file
	if(AME_PRINT_STATUS>0) std::cout<<"opening file\n";
	writer=fopen(file,"w");
	if(writer==NULL) throw std::runtime_error(std::string("ERROR in AME::read(const char*,const AtomType&,Structure&): Could not open file: ")+std::string(file));
	
	//write units
	fprintf(writer,"units %s\n",units::System::name(units::Consts::system()));
	//fprintf(writer,"units METAL\n");
	//write energy
	fprintf(writer,"energy %19.10f\n",struc.pe());
	//write cell
	fprintf(writer,"cell ");
	for(int i=0; i<3; ++i){
		for(int j=0; j<3; ++j){
			fprintf(writer,"%8.6f ",struc.R()(j,i));
		}
	}
	fprintf(writer,"\n");
	//write atoms
	fprintf(writer,"atoms %i\n",struc.nAtoms());
	int index=0;
	if(atomT.name) fprintf(writer,"name ");
	if(atomT.charge) fprintf(writer,"q ");
	if(atomT.mass) fprintf(writer,"m ");
	if(atomT.posn) fprintf(writer,"rx ry rz ");
	if(atomT.force) fprintf(writer,"fx fy fz ");
	fprintf(writer,"\n");
	
	for(int i=0; i<struc.nAtoms(); ++i){
		if(atomT.name) fprintf(writer,"%-2s ",struc.name(i).c_str());
		if(atomT.charge) fprintf(writer,"%8.6f ",struc.charge(i));
		if(atomT.mass) fprintf(writer,"%8.6f ",struc.mass(i));
		if(atomT.posn) fprintf(writer,"%19.10f %19.10f %19.10f ",struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2]);
		if(atomT.force) fprintf(writer,"%19.10f %19.10f %19.10f",struc.force(i)[0],struc.force(i)[1],struc.force(i)[2]);
		fprintf(writer,"\n");
	}
	
	//close the file
	if(AME_PRINT_STATUS>0) std::cout<<"closing file\n";
	fclose(writer);
	writer=NULL;
	
}

	
}