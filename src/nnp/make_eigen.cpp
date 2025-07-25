// c++
#include <iostream>
#include <stdexcept>
// string
#include "str/string.hpp"
#include "str/token.hpp"
#include "str/print.hpp"
// structure
#include "struc/structure.hpp"
// format
#include "format/file_struc.hpp"
#include "format/format.hpp"

int main(int argc, char* argv[]){
	//file i/o
		FILE* reader=NULL;
		FILE_FORMAT::type format;
		char* pfile=new char[string::M];
		char* sfile=new char[string::M];
		char* efile=new char[string::M];
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		Token token;
	// structure
		Structure struc;
	// atom type 
		AtomType atomT;
		atomT.name	=true;
		atomT.an		=true;
		atomT.mass	=true;
		atomT.type	=true;
		atomT.index	=true;
		atomT.posn	=true;
		atomT.force	=true;
		atomT.frac	=true;
	//misc
		bool error=false;
	
	try{
		//======== print title ========
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::title("MAKE EIGEN",strbuf,' ')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		
		//======== check the arguments ========
		if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
		
		//======== load the parameter file ========
		std::cout<<"reading parameter file\n";
		std::strcpy(pfile,argv[1]);
		
		//======== open the parameter file ========
		std::cout<<"opening parameter file\n";
		reader=fopen(pfile,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+pfile);
		
		//======== read in the parameters ========
		std::cout<<"reading parameters\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);
			Token token(input,string::WS);
			if(token.end()) continue;
			const std::string tag=string::to_upper(token.next());
			if(tag=="STRUC"){
				std::strcpy(sfile,token.next().c_str());
			} else if(tag=="EIGEN"){
				std::strcpy(efile,token.next().c_str());
			} else if(tag=="FORMAT"){
				format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
			} 
		}
		fclose(reader);
		reader=NULL;
		
		//read structure
		std::cout<<"reading the structure\n";
		read_struc(sfile,format,atomT,struc);
		
		//read eigen
		std::cout<<"reading eigen\n";
		reader=fopen(efile,"r");
		if(reader==NULL) throw std::runtime_error("Could not open eigen file.");
		int nlines=0;
		while(fgets(input,string::M,reader)!=NULL) ++nlines;
		const int nModes=nlines/(struc.nAtoms()+1);
		std::rewind(reader);
		std::vector<double> dr(nModes,0.0);
		std::vector<double> fr(nModes,0.0);
		std::vector<int> nr(nModes,0);
		std::vector<std::vector<Eigen::Vector3d> > modes(nModes,std::vector<Eigen::Vector3d>(struc.nAtoms(),Eigen::Vector3d::Zero()));
		for(int i=0; i<nModes; ++i){
			token.read(fgets(input,string::M,reader),string::WS);
			dr[i]=std::atof(token.next().c_str());
			nr[i]=std::atoi(token.next().c_str());
			fr[i]=std::atof(token.next().c_str());
			for(int j=0; j<struc.nAtoms(); ++j){
				token.read(fgets(input,string::M,reader),string::WS);
				modes[i][j][0]=std::atof(token.next().c_str());
				modes[i][j][1]=std::atof(token.next().c_str());
				modes[i][j][2]=std::atof(token.next().c_str());
			}
		}
		for(int i=0; i<nModes; ++i){
			for(int j=0; j<struc.nAtoms(); ++j){
				const double norm=modes[i][j].norm();
				modes[i][j]*=1.0/norm;
			}
		}
		
		for(int i=0; i<nModes; ++i){
			std::cout<<"dr["<<i<<"] = "<<0.5/fr[i]<<"\n";
			//dr[i]=0.5/fr[i];
		}
		for(int i=0; i<nModes; ++i){
			std::cout<<"lim = "<<dr[i]*nr[i]*-1.0<<" "<<dr[i]*nr[i]*1.0<<"\n";
		}
		
		int count=0;
		for(int i=0; i<nModes; ++i){
			for(int j=nr[i]; j>=1; --j){
				Structure tmp=struc;
				for(int k=0; k<struc.nAtoms(); ++k){
					tmp.posn(k).noalias()+=-1.0*j*dr[i]*modes[i][k];
				}
				std::string file_out="struc_n"+std::to_string(count)+".vasp";
				write_struc(file_out.c_str(),format,atomT,tmp);
				count++;
			}
			for(int j=1; j<nr[i]; ++j){
				Structure tmp=struc;
				for(int k=0; k<struc.nAtoms(); ++k){
					tmp.posn(k).noalias()+=1.0*j*dr[i]*modes[i][k];
				}
				std::string file_out="struc_n"+std::to_string(count)+".vasp";
				write_struc(file_out.c_str(),format,atomT,tmp);
				count++;
			}
		}
		
	}catch(std::exception& e){
		std::cout<<"ERROR in make_basis(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] pfile;
	delete[] sfile;
	delete[] efile;
	delete[] input;
	delete[] strbuf;
	
	if(error) return 1;
	return 0;
}