// c++
#include <iostream>
#include <stdexcept>
// chem
#include "chem/moldata.hpp"
// string
#include "str/parse.hpp"

int main(int argc, char* argv[]){
	//file i/o
		std::vector<std::string> file_in;
		std::string file_out;
	//moldata
		std::vector<MolData> moldata;
		MolData moltot;
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
				file_in=args[i].vals();
			} else if(args[i].key()=="fout"){
				file_out=args[i].val(0);
			}
		}
		
		std::cout<<"found "<<file_in.size()<<" input files\n";
		if(file_in.size()==0) throw std::invalid_argument("Invalid number of input files.\n");
		
		//read
		std::cout<<"reading data\n";
		moldata.resize(file_in.size());
		for(int i=0; i<file_in.size(); ++i){
			std::cout<<file_in[i]<<"\n";
			MolData::read(file_in[i].c_str(),moldata[i]);
		}
		std::cout<<"reading completed\n";
		
		//concatenate
		std::cout<<"concatenating\n";
		for(int i=0; i<file_in.size(); ++i){
			moltot+=moldata[i];
		}
		
		//write
		std::cout<<"writing data\n";
		std::cout<<"file_out = "<<file_out<<"\n";
		MolData::write(file_out.c_str(),moltot);
		std::cout<<"writing completed\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in convert_sim::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) return 1;
	else return 0;
}