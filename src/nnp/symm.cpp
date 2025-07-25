// c libraries
#include <cstdio>
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
#include <cmath>
#elif (defined __ICC || defined __INTEL_COMPILER)
#include <mathimf.h> //intel math library
#else
#include <cmath>
#endif
// c++ libraries
#include <iostream>
#include <exception>
#include <algorithm>
#include <random>
#include <chrono>
// structure
#include "struc/structure.hpp"
#include "struc/neighbor.hpp"
// math
#include "math/const.hpp"
// format
#include "format/file_struc.hpp"
#include "format/format.hpp"
// string
#include "str/string.hpp"
#include "str/token.hpp"
#include "str/print.hpp"
// chem
#include "chem/units.hpp"
// util
#include "util/compiler.hpp"
#include "util/time.hpp"
// nnp
#include "nnp/nnp.hpp"

int main(int argc, char* argv[]){
	//======== global variables ========
	//units
		units::System unitsys=units::System::UNKNOWN;
	//atom format
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=false; atomT.symm=true; atomT.charge=false;
	//structure
		Structure struc;
		FILE_FORMAT::type format;//format of training data
		std::string fstruc;
	//NNP
		NNP nnp;
		std::vector<Type> types;//unique atomic species
		std::vector<std::vector<int> > nh;//hidden layer configuration
	//timing
		Clock clock;
	//file i/o
		FILE* reader=NULL;
		char* paramfile=new char[string::M];
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		std::vector<std::string> files_basis;//file - stores basis
	//string
		Token token;
	
	try{
		//************************************************************************************
		// LOADING/INITIALIZATION
		//************************************************************************************
		
		//======== print title ========
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::title("SYMM",strbuf,' ')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		std::cout<<print::buf(strbuf,'*')<<"\n";
		
		//======== print compiler information ========
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("COMPILER",strbuf)<<"\n";
		std::cout<<"date     = "<<compiler::date()<<"\n";
		std::cout<<"time     = "<<compiler::time()<<"\n";
		std::cout<<"compiler = "<<compiler::name()<<"\n";
		std::cout<<"version  = "<<compiler::version()<<"\n";
		std::cout<<"standard = "<<compiler::standard()<<"\n";
		std::cout<<"arch     = "<<compiler::arch()<<"\n";
		std::cout<<"instr    = "<<compiler::instr()<<"\n";
		std::cout<<"os       = "<<compiler::os()<<"\n";
		std::cout<<"omp      = "<<compiler::omp()<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		
		//======== print mathematical constants ========
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("MATHEMATICAL CONSTANTS",strbuf)<<"\n";
		std::printf("PI     = %.15f\n",math::constant::PI);
		std::printf("RadPI  = %.15f\n",math::constant::RadPI);
		std::printf("Rad2   = %.15f\n",math::constant::Rad2);
		std::printf("Log2   = %.15f\n",math::constant::LOG2);
		std::printf("Eps<D> = %.15e\n",std::numeric_limits<double>::epsilon());
		std::printf("Min<D> = %.15e\n",std::numeric_limits<double>::min());
		std::printf("Max<D> = %.15e\n",std::numeric_limits<double>::max());
		std::printf("Min<I> = %i\n",std::numeric_limits<int>::min());
		std::printf("Max<I> = %i\n",std::numeric_limits<int>::max());
		std::cout<<print::buf(strbuf)<<"\n";
		
		//======== print physical constants ========
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
		std::printf("alpha        = %.12f\n",units::ALPHA);
		std::printf("p/e mass     = %.12f\n",units::MPoME);
		std::printf("bohr-r  (A)  = %.12f\n",units::Bohr2Ang);
		std::printf("hartree (eV) = %.12f\n",units::Eh2Ev);
		std::cout<<print::buf(strbuf)<<"\n";
		
		//======== check the arguments ========
		if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
		
		//======== load the parameter file ========
		std::cout<<"reading parameter file\n";
		std::strcpy(paramfile,argv[1]);
		
		//======== open the parameter file ========
		std::cout<<"opening parameter file\n";
		reader=fopen(paramfile,"r");
		if(reader==NULL) throw std::runtime_error(std::string("I/O Error: Could not open parameter file: ")+paramfile);
		
		//======== read in the parameters ========
		std::cout<<"reading parameters\n";
		while(fgets(input,string::M,reader)!=NULL){
			token.read(string::trim_right(input,string::COMMENT),string::WS);
			if(token.end()) continue;//skip empty line
			const std::string tag=string::to_upper(token.next());
			//general
			if(tag=="UNITS"){//units
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
			} else if(tag=="FORMAT"){//simulation format
				format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
			} else if(tag=="R_CUT"){
				nnp.rc()=std::atof(token.next().c_str());
			} else if(tag=="STRUC"){
				fstruc=token.next();
			}
			//atom
			if(tag=="ATOM"){//atom - name/mass/energy
				//process the string
				const std::string name=token.next();
				const std::string atomtag=string::to_upper(token.next());
				const int id=string::hash(name);
				//look for the atom name in the existing list of atom names
				int index=-1;
				for(int i=0; i<types.size(); ++i){
					if(name==types[i].name()){index=i;break;}
				}
				//if atom is not found, add it
				if(index<0){
					index=types.size();
					types.push_back(Type());
					types.back().name()=name;
					types.back().id()=id;
					files_basis.resize(files_basis.size()+1);
					nh.resize(nh.size()+1);
				}
				//set tag value
				if(atomtag=="MASS"){
					types[index].mass().flag()=true;
					types[index].mass().val()=std::atof(token.next().c_str());
				} else if(atomtag=="CHARGE"){
					types[index].charge().flag()=true;
					types[index].charge().val()=std::atof(token.next().c_str());
					atomT.charge=true;
				} else if(atomtag=="CHI"){
					types[index].chi().flag()=true;
					types[index].chi().val()=std::atof(token.next().c_str());
					atomT.chi=true;
				} else if(atomtag=="ETA"){
					types[index].eta().flag()=true;
					types[index].eta().val()=std::atof(token.next().c_str());
					atomT.eta=true;
				} else if(atomtag=="ENERGY"){
					types[index].energy().flag()=true;
					types[index].energy().val()=std::atof(token.next().c_str());
				} else if(atomtag=="RVDW"){
					types[index].rvdw().flag()=true;
					types[index].rvdw().val()=std::atof(token.next().c_str());
				} else if(atomtag=="RCOV"){
					types[index].rcov().flag()=true;
					types[index].rcov().val()=std::atof(token.next().c_str());
				} else if(atomtag=="C6"){
					types[index].c6().flag()=true;
					types[index].c6().val()=std::atof(token.next().c_str());
				} else if(atomtag=="BASIS"){
					files_basis[index]=token.next();
				} else if(atomtag=="NH"){
					nh[index].clear();
					while(!token.end()) nh[index].push_back(std::atoi(token.next().c_str()));
				}
			} 
		}
		
		//======== close parameter file ========
		std::cout<<"closing parameter file\n";
		fclose(reader);
		reader=NULL;
		
		//==== read the potential ====
		std::cout<<"resizing potential\n";
		nnp.resize(types);
		//read basis files
		std::cout<<"reading basis files\n";
		if(files_basis.size()!=nnp.ntypes()) throw std::runtime_error("main(int,char**): invalid number of basis files.");
		for(int i=0; i<nnp.ntypes(); ++i){
			const char* atomName=types[i].name().c_str();
			NNP::read_basis(fstruc.c_str(),nnp,atomName);
		}
		//initialize the neural network hamiltonians
		std::cout<<"initializing neural network hamiltonians\n";
		for(int i=0; i<nnp.ntypes(); ++i){
			nnp.nnh(i).type()=types[i];
		}
		
		//======== print parameters ========
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
		std::cout<<"ATOM_T     = "<<atomT<<"\n";
		std::cout<<"FORMAT     = "<<format<<"\n";
		std::cout<<"UNITS      = "<<unitsys<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TYPES",strbuf)<<"\n";
		for(int i=0; i<types.size(); ++i){
			std::cout<<types[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<nnp<<"\n";
		
		//========= check the data =========
		if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid file format.");
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		if(types.size()==0) throw std::invalid_argument("Invalid number of types.");
		
		//======== set the unit system ========
		std::cout<<"setting the unit system\n";
		units::Consts::init(unitsys);
		
		//************************************************************************************
		// Read Structure
		//************************************************************************************
		
		//======== read the structures ========
		std::cout<<"reading structures\n";
		read_struc(fstruc.c_str(),format,atomT,struc);
		
		//************************************************************************************
		// ATOM PROPERTIES
		//************************************************************************************
		
		//======== set atom properties ========
		std::cout<<"setting atomic properties\n";
		
		//======== set the indices ========
		std::cout<<"setting the indices\n";
		for(int j=0; j<struc.nAtoms(); ++j){
			struc.index(j)=j;
		}
		
		//======== set the types ========
		std::cout<<"setting the types\n";
		for(int j=0; j<struc.nAtoms(); ++j){
			struc.type(j)=nnp.index(struc.name(j));
		}

		//************************************************************************************
		// SET INPUTS
		//************************************************************************************
		
		//======== initialize the symmetry functions ========
		std::cout<<"initializing symmetry functions\n";
		NNP::init(nnp,struc);
		
		//======== compute the symmetry functions ========
		std::cout<<"setting the inputs (symmetry functions)\n";
		NeighborList nlist(struc,nnp.rc());
		NNP::symm(nnp,struc,nlist);
		
		for(int i=0; i<struc.nAtoms(); ++i){
			std::cout<<i<<" "<<struc.type(i)<<" "<<struc.name(i)<<" "<<struc.posn(i).transpose()<<" "<<struc.symm(i).transpose()<<"\n";
		}
		
	}catch(std::exception& e){
		std::cout<<"ERROR in symm::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//======== free local variables ========
	delete[] paramfile;
	delete[] input;
	delete[] strbuf;
	
	return 0;
}
