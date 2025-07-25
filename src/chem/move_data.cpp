// c++
#include <iostream>
#include <stdexcept>
// chem
#include "chem/moldata.hpp"
// string
#include "str/parse.hpp"
#include "str/token.hpp"

int main(int argc, char* argv[]){
	//file i/o
		std::string file_in;
		std::string file_out;
	//moldata
		MolData moldata;
	//offset 
		Eigen::Vector3d offset=Eigen::Vector3d::Zero();
	//arguments
		std::vector<input::Arg> args;
	//misc
		bool error=false;
		
	try{
		//check the number of arguments
		if(argc==1) throw std::invalid_argument("No arguments provided.");
		
		//parse the arguments
		std::cout<<"parsing arguments\n";
		input::parse(argc,argv,args);
		for(int i=0; i<args.size(); ++i){
			if(args[i].key()=="fin"){
				file_in=args[i].val(0);
			} else if(args[i].key()=="fout"){
				file_out=args[i].val(0);
			} else if(args[i].key()=="offset"){
				Token token;
				token.read(args[i].val(0),":");
				offset[0]=std::atof(token.next().c_str());
				offset[1]=std::atof(token.next().c_str());
				offset[2]=std::atof(token.next().c_str());
			}
		}
		
		std::cout<<"found "<<file_in.size()<<" input files\n";
		if(file_in.size()==0) throw std::invalid_argument("Invalid number of input files.\n");
		
		//read
		std::cout<<"reading data\n";
		std::cout<<file_in<<"\n";
		MolData::read(file_in.c_str(),moldata);
		std::cout<<"reading completed\n";
		
		//add offset
		std::cout<<"setting offset\n";
		for(int i=0; i<moldata.nAtoms; ++i){
			moldata.atoms[i].posn.noalias()+=offset;
		}
		moldata.xlim[0]+=offset[0];
		moldata.ylim[0]+=offset[1];
		moldata.zlim[0]+=offset[2];
		moldata.xlim[1]+=offset[0];
		moldata.ylim[1]+=offset[1];
		moldata.zlim[1]+=offset[2];
		
		//write
		std::cout<<"writing data\n";
		std::cout<<"file_out = "<<file_out<<"\n";
		MolData::write(file_out.c_str(),moldata);
		std::cout<<"writing completed\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in convert_sim::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) return 1;
	else return 0;
}