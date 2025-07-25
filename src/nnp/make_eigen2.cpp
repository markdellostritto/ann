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
		Eigen::Vector3d box=Eigen::Vector3d::Zero();
		bool frac=true;
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
	// eigen
		double dr=0;
		double decay=0;
		int nr=0;
		int index=-1;
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
			} else if(tag=="DR"){
				dr=std::atof(token.next().c_str());
			} else if(tag=="NR"){
				nr=std::atoi(token.next().c_str());
			} else if(tag=="INDEX"){
				index=std::atoi(token.next().c_str());
			} else if(tag=="DECAY"){
				decay=std::atof(token.next().c_str());
			} else if(tag=="BOX"){
				box[0]=std::atof(token.next().c_str());
				box[1]=std::atof(token.next().c_str());
				box[2]=std::atof(token.next().c_str());
			} else if(tag=="FRAC"){
				frac=string::boolean(token.next().c_str());
			}
		}
		fclose(reader);
		reader=NULL;
		
		std::cout<<"STRUC  = "<<sfile<<"\n";
		std::cout<<"EIGEN  = "<<efile<<"\n";
		std::cout<<"FORMAT = "<<format<<"\n";
		std::cout<<"DR     = "<<dr<<"\n";
		std::cout<<"INDEX  = "<<index<<"\n";
		std::cout<<"DECAY  = "<<decay<<"\n";
		std::cout<<"FRAC   = "<<frac<<"\n";
		std::cout<<"BOX    = "<<box<<"\n";
		atomT.frac=frac;
		
		//read structure
		std::cout<<"reading the structure\n";
		read_struc(sfile,format,atomT,struc);
		if(box.norm()>0){
			Eigen::MatrixXd lv=box.asDiagonal();
			struc.init(lv);
		}
		
		//read eigen
		std::cout<<"reading eigen\n";
		reader=fopen(efile,"r");
		if(reader==NULL) throw std::runtime_error("Could not open eigen file.");
		std::vector<Eigen::Vector3d> mode(struc.nAtoms(),Eigen::Vector3d::Zero());
		for(int j=0; j<struc.nAtoms(); ++j){
			token.read(fgets(input,string::M,reader),string::WS);
			mode[j][0]=std::atof(token.next().c_str());
			mode[j][1]=std::atof(token.next().c_str());
			mode[j][2]=std::atof(token.next().c_str());
		}
		/*
		for(int j=0; j<struc.nAtoms(); ++j){
			const double norm=mode[j].norm();
			if(norm>1e-16) mode[j]*=1.0/norm;
		}
		*/
		std::cout<<"lim = "<<dr*nr*-1.0<<" "<<dr*nr*1.0<<"\n";
		
		std::vector<double> drv(nr,dr);
		for(int j=1; j<nr; ++j){
			drv[j]=(1.0-decay)*drv[j-1];
		}
		std::cout<<"drv = "; for(int j=0; j<nr; ++j) std::cout<<drv[j]<<" "; std::cout<<"\n";
		std::vector<double> drx(nr,dr);
		for(int j=1; j<nr; ++j){
			drx[j]=drv[j]+drx[j-1];
		}
		std::cout<<"drx = "; for(int j=0; j<nr; ++j) std::cout<<drx[j]<<" "; std::cout<<"\n";
		for(int j=nr-1; j>=0; --j){
			Structure tmp=struc;
			for(int k=0; k<struc.nAtoms(); ++k){
				tmp.posn(k).noalias()+=-1.0*drx[j]*mode[k];
			}
			std::string file_out="struc_q"+std::to_string(index)+"_m"+std::to_string(j+1)+".vasp";
			write_struc(file_out.c_str(),format,atomT,tmp);
		}
		for(int j=0; j<nr; ++j){
			Structure tmp=struc;
			for(int k=0; k<struc.nAtoms(); ++k){
				tmp.posn(k).noalias()+=1.0*drx[j]*mode[k];
			}
			std::string file_out="struc_q"+std::to_string(index)+"_p"+std::to_string(j+1)+".vasp";
			write_struc(file_out.c_str(),format,atomT,tmp);
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