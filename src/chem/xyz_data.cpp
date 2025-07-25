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
	//names
		std::vector<std::string> names;
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
			} else if(args[i].key()=="names"){
				names=args[i].vals();
			}
		}
		
		std::cout<<"found "<<file_in.size()<<" input files\n";
		if(file_in.size()==0) throw std::invalid_argument("Invalid number of input files.\n");
		
		//print names
		std::cout<<"names = "; for(int i=0; i<names.size(); ++i) std::cout<<names[i]<<" "; std::cout<<"\n";
		
		//read
		std::cout<<"reading data\n";
		std::cout<<file_in<<"\n";
		MolData::read(file_in.c_str(),moldata);
		std::cout<<"reading completed\n";
		
		//write
		std::cout<<"writing data\n";
		std::cout<<"file_out = "<<file_out<<"\n";
		FILE* writer=fopen(file_out.c_str(),"w");
		if(writer!=NULL){
			fprintf(writer,"%i\n",moldata.nAtoms);
			fprintf(writer,"data\n");
			for(int i=0; i<moldata.nAtoms; ++i){
				const int type=moldata.atoms[i].type-1;
				Eigen::Vector3d& x=moldata.atoms[i].posn;
				fprintf(writer,"%s %f %f %f\n",names[type].c_str(),x[0],x[1],x[2]);
			}
			fclose(writer);
			writer=NULL;
		} else throw std::runtime_error("Could not open output file.");
		std::cout<<"writing completed\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in convert_sim::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) return 1;
	else return 0;
}