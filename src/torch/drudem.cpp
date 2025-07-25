// mem
#include "mem/map.hpp"
// format
#include "format/file_struc.hpp"
// struc
#include "struc/structure.hpp"
// string
#include "str/string.hpp"
#include "str/print.hpp"
// torch
#include "torch/drude.hpp"
#include "torch/dump.hpp"
// math
#include "math/const.hpp"
#include "math/reduce.hpp"
// nnp
#include "nnp/type.hpp"
// util
#include "util/time.hpp"

int main(int argc, char* argv[]){
	//file parameter
		char* fparam=new char[string::M];
		char* fstruc=new char[string::M];
		char* input=new char[string::M];
		char* str=new char[string::M];
		FILE* reader=NULL;
		std::vector<std::string> strlist;
	//structure
		AtomType atomT;
		atomT.name=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=true; atomT.vel=true;
		atomT.mass=true; atomT.charge=true; atomT.alpha=true;
		atomT.radius=true;
		Structure struc;
		FILE_FORMAT::type format;//format of training data
	//drude
		int ts=0;
		Drude drude;
		std::vector<Type> types;
	//units
		units::System unitsys=units::System::UNKNOWN;
	//misc
		bool error=true;
		
	try{
		
		//==== check the arguments ====
		if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
		
		//==== load the parameter file ====
		std::cout<<"reading parameter file\n";
		std::strcpy(fparam,argv[1]);
		
		//==== open the parameter file ====
		std::cout<<"opening parameter file\n";
		FILE* reader=fopen(fparam,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+fparam);
		
		//==== read in the parameters ====
		std::cout<<"reading parameters\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);//trim comments
			Token token(input,string::WS); //split line into tokens
			if(token.end()) continue; //skip empty lines
			std::string tag=string::to_upper(token.next());
			if(tag=="UNITS"){//units
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
			} else if(tag=="STRUC"){
				std::strcpy(fstruc,token.next().c_str());
			} else if(tag=="FORMAT"){
				format=FILE_FORMAT::read(token.next().c_str());
			} else if(tag=="DT"){
				drude.dt()=std::atof(token.next().c_str());
			} else if(tag=="TS"){
				ts=std::atof(token.next().c_str());
			} else if(tag=="TYPE"){
				Type type;
				Type::read(type,token);
				types.push_back(type);
			}
		}
		
		//==== close the parameter file ====
		std::cout<<"closing parameter file\n";
		fclose(reader);
		reader=NULL;
		
		//==== check the parameters ====
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		
		//==== set the unit system ====
		std::cout<<"setting the unit system\n";
		units::Consts::init(unitsys);
		
		//==== print the parameters ====
		std::cout<<print::buf(str)<<"\n";
		std::cout<<print::title("GENERAL",str)<<"\n";
		std::cout<<"units  = "<<unitsys<<"\n";
		std::cout<<"fstruc = "<<fstruc<<"\n";
		std::cout<<"ke     = "<<units::Consts::ke()<<"\n";
		std::cout<<print::buf(str)<<"\n";
		std::cout<<print::title("TYPES",str)<<"\n";
		for(int i=0; i<types.size(); ++i){
			std::cout<<types[i]<<"\n";
		}
		std::cout<<print::buf(str)<<"\n";
		
		//==== read the structure ====
		std::cout<<"reading the structure\n";
		read_struc(fstruc,format,atomT,struc);
		std::cout<<struc<<"\n";
		
		//==== set atom data ====
		std::cout<<"setting atom data\n";
		//type
		std::cout<<"type\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			for(int j=0; j<types.size(); ++j){
				if(struc.name(i)==types[j].name()){struc.type(i)=j; break;}
			}
		}
		//charge
		std::cout<<"charge\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.charge(i)=types[struc.type(i)].charge().val();
		}
		//alpha
		std::cout<<"alpha\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.alpha(i)=types[struc.type(i)].alpha().val();
		}
		//radius
		std::cout<<"radius\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.radius(i)=types[struc.type(i)].rcov().val();
		}
		//print
		for(int i=0; i<struc.nAtoms(); ++i){
			std::cout<<struc.name(i)<<" "<<struc.type(i)<<" "<<struc.charge(i)<<" "<<struc.alpha(i)<<" "<<struc.radius(i)<<"\n";
		}
		
		//==== minimizing ====
		std::cout<<"computing drude positions\n";
		drude.load(6.0,struc);
		FILE* writer=fopen("drude.dump","w");
		if(writer==NULL) throw std::invalid_argument("Could not open dump file.");
		std::cout<<"t ke pe eS eR eK\n";
		for(int t=0; t<ts; ++t){
			drude.quickmin();
			printf("%i %f %f %f %f %f\n",
				t,drude.struc().ke(),drude.struc().pe(),
				drude.energyS(),drude.energyR(),drude.energyK()
			);
			Structure tmp=drude.struc();
			Dump::write(tmp,writer);
		}
		fclose(writer);
		writer=NULL;
		
	}catch(std::exception& e){
		std::cout<<"ERROR in QEQM::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] fparam;
	delete[] fstruc;
	delete[] input;
	delete[] str;
	
	if(error) return 1;
	else return 0;
}