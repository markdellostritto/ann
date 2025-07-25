//c++
#include <iostream>
#include <utility>
//structure
#include "struc/structure.hpp"
// format
#include "format/vasp_struc.hpp"
#include "format/xyz_struc.hpp"
//chem
#include "chem/molkit2.hpp"
#include "chem/alias.hpp"
#include "chem/units.hpp"
#include "chem/ptable.hpp"
//math
#include "math/graph.hpp"
//string
#include "str/string.hpp"
#include "str/print.hpp"
#include "str/token.hpp"

int main(int argc, char* argv[]){
    //==== global variables ====
	//file i/o
		FILE* reader=NULL;
		FILE_FORMAT::type format;
		char* paramfile = new char[string::M];
		char* input     = new char[string::M];
		char* simstr    = new char[string::M];
		char* strbuf    = new char[print::len_buf];
	//structure
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true;
		atomT.posn=true; atomT.mass=true; atomT.charge=true; atomT.radius=true;
		Structure struc;
		Eigen::Vector3d box=Eigen::Vector3d::Zero();
	//molecule
		Graph molgraph;
		std::vector<std::pair<std::string,double> > radii;
	//units
        units::System unitsys;
	//misc
		bool error=false;
		int assign_error=0;
		
    try{
        if(argc!=2) throw std::invalid_argument("Invalid number of command-line arguments.");
        
        //==== copy the parameter file ====
        std::cout<<"reading parameter file\n";
        std::strcpy(paramfile,argv[1]);
        
        //==== read in the general parameters ====
        reader=fopen(paramfile,"r");
        if(reader==NULL) throw std::runtime_error("I/O Error: could not open parameter file.");
        std::cout<<"reading general parameters\n";
        while(fgets(input,string::M,reader)!=NULL){
            string::trim_right(input,string::COMMENT);
            Token token(input,string::WS);
            if(token.end()) continue;
            const std::string tag=string::to_upper(token.next());
            //read structure
            if(tag=="STRUC"){
                std::strcpy(simstr,token.next().c_str());
            } else if(tag=="FORMAT"){
                format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
            } else if(tag=="UNITS"){
                unitsys=units::System::read(string::to_upper(token.next()).c_str());
            } else if(tag=="BOX"){
                box[0]=std::atof(token.next().c_str());
                box[1]=std::atof(token.next().c_str());
                box[2]=std::atof(token.next().c_str());
            } 
            //elements
            if(tag=="RADIUS"){
                const std::string name=token.next();
                double rad=std::atof(token.next().c_str());
                radii.push_back(std::pair<std::string,double>(name,rad));
            } 
        }
        //close the file
        fclose(reader);
        reader=NULL;

        //==== print the parameters ====
		//**** general parmaeters ****
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
		std::cout<<"UNITS     = "<<unitsys<<"\n";
		std::cout<<"ATOM_T    = "<<atomT<<"\n";
		std::cout<<"SIM_STR   = \""<<simstr<<"\"\n";
		std::cout<<"FORMAT    = "<<format<<"\n";
		std::cout<<"BOX       = "<<box.transpose()<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		
        //==== read the simulation ====
		std::cout<<"reading simulation\n";
		if(format==FILE_FORMAT::POSCAR){
			VASP::POSCAR::read(simstr,atomT,struc);
		} else if(format==FILE_FORMAT::XYZ){
			XYZ::read(simstr,atomT,struc);
			Eigen::Matrix3d lv=Eigen::Matrix3d::Zero();
			lv(0,0)=box(0); lv(1,1)=box(1); lv(2,2)=box(2);
			struc.init(lv);
		} else throw std::invalid_argument("Invalid format.");
		std::cout<<struc<<"\n";
		
		//==== assign radii ====
		for(int i=0; i<struc.nAtoms(); ++i){
			bool match=false;
			for(int j=0; j<radii.size(); ++j){
				if(radii[j].first==struc.name(i)){
					struc.radius(i)=radii[j].second;
					match=true;
					break;
				}
			}
			if(!match){
				struc.radius(i)=ptable::radius_covalent(struc.an(i));
				std::cout<<"WARNING: found no radius for "<<struc.name(i)<<", using covalent radius.\n";
			}
		}

        //==== make the graph ====
		std::cout<<"making the graph\n";
		molkit::make_graph(struc,molgraph);
		
        //==== color the graph ====
		std::cout<<"coloring the graph\n";
		const int nmol=Graph::color_cc(molgraph);
        
        //==== write the molecule ====
        FILE* writer=fopen("mol.xyz","w");
        if(writer==NULL) throw std::runtime_error("Could not open output file.");
        fprintf(writer,"%i\n",struc.nAtoms());
        fprintf(writer,"test\n");
        for(int i=0; i<struc.nAtoms(); ++i){
            fprintf(writer,"%-2s %19.10f %19.10f %19.10f %10i\n",struc.name(i).c_str(),
                struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2],molgraph.node(i).color()
            );
        }
    }catch(std::exception& e){
        std::cout<<"ERROR in ffa::main(int,char**):\n";
        std::cout<<e.what()<<"\n";
        error=true;
    }
    
    delete[] paramfile;
    delete[] input;
    delete[] simstr;
    delete[] strbuf;
    
    if(error) return 1;
    else return 0;          
    
}