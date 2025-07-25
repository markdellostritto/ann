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
		std::vector<std::vector<std::string> > lsmiles;
		std::vector<std::pair<std::string,double> > radii;
	//flags
		bool print=false;
		bool print_alias=false;
		bool norm_chg=false;
	//types
		int dmax=0;
		std::vector<molkit::Type> types;
		std::vector<Alias> aliases;
	//charges
		std::vector<std::pair<std::string,double> > chargeD1;
		std::vector<std::pair<std::string,double> > chargeD2;
	//labels
		std::vector<molkit::Label> charge_labels;
		std::vector<molkit::Label> mass_labels;
		std::vector<molkit::Label> pair_labels;
		std::vector<molkit::Label> bond_labels;
		std::vector<molkit::Label> angle_labels;
		std::vector<molkit::Label> dihedral_labels;
		std::vector<molkit::Label> improper_labels;
	//coefficients
		std::vector<molkit::Coeff> charge_coeffs;
		std::vector<molkit::Coeff> mass_coeffs;
		std::vector<molkit::Coeff> pair_coeffs;
		std::vector<molkit::Coeff> bond_coeffs;
		std::vector<molkit::Coeff> angle_coeffs;
		std::vector<molkit::Coeff> dihedral_coeffs;
		std::vector<molkit::Coeff> improper_coeffs;
	//bonded groups
		std::vector<std::array<int,3> > bond_list;
		std::vector<std::array<int,4> > angle_list;
		std::vector<std::array<int,5> > dihedral_list;
		std::vector<std::array<int,5> > improper_list;
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
			} else if(tag=="PRINT_ALIAS"){
				print_alias=string::boolean(token.next().c_str());
			} 
			//elements
			if(tag=="RADIUS"){
				const std::string name=token.next();
				double rad=std::atof(token.next().c_str());
				radii.push_back(std::pair<std::string,double>(name,rad));
			} 
			//read types
			if(tag=="DMAX"){
				dmax=std::atoi(token.next().c_str());
			} else if(tag=="TYPE"){
				types.push_back(molkit::Type());
				molkit::Type::read(token,types.back());
			} else if(tag=="ALIAS"){
				aliases.push_back(Alias());
				Alias::read(token,aliases.back());
			} 
			//charges
			if(tag=="CHARGE_D1"){
				const std::string name=token.next();
				const double chg=std::atof(token.next().c_str());
				chargeD1.push_back(std::pair<std::string,double>(name,chg));
			}
			//read labels
			if(tag=="MASS_LABEL"){
				mass_labels.push_back(molkit::Label());
				mass_labels.back().index()=std::atoi(token.next().c_str());
				for(int i=0; i<1; ++i) mass_labels.back().types().push_back(molkit::Type(token.next()));
			} else if(tag=="CHARGE_LABEL"){
				charge_labels.push_back(molkit::Label());
				charge_labels.back().index()=std::atoi(token.next().c_str());
				for(int i=0; i<1; ++i) charge_labels.back().types().push_back(molkit::Type(token.next()));
			} else if(tag=="PAIR_LABEL"){
				pair_labels.push_back(molkit::Label());
				pair_labels.back().index()=std::atoi(token.next().c_str());
				for(int i=0; i<1; ++i) pair_labels.back().types().push_back(molkit::Type(token.next()));
			} else if(tag=="BOND_LABEL"){
				bond_labels.push_back(molkit::Label());
				bond_labels.back().index()=std::atoi(token.next().c_str());
				for(int i=0; i<2; ++i) bond_labels.back().types().push_back(molkit::Type(token.next()));
			} else if(tag=="ANGLE_LABEL"){
				angle_labels.push_back(molkit::Label());
				angle_labels.back().index()=std::atoi(token.next().c_str());
				for(int i=0; i<3; ++i) angle_labels.back().types().push_back(molkit::Type(token.next()));
			} else if(tag=="DIHEDRAL_LABEL"){
				dihedral_labels.push_back(molkit::Label());
				dihedral_labels.back().index()=std::atoi(token.next().c_str());
				for(int i=0; i<4; ++i) dihedral_labels.back().types().push_back(molkit::Type(token.next()));
			} else if(tag=="IMPROPER_LABEL"){
				improper_labels.push_back(molkit::Label());
				improper_labels.back().index()=std::atoi(token.next().c_str());
				for(int i=0; i<4; ++i) improper_labels.back().types().push_back(molkit::Type(token.next()));
			}
			//read coefficients
			if(tag=="MASS_COEFF"){
				mass_coeffs.push_back(molkit::Coeff());
				mass_coeffs.back().index()=std::atoi(token.next().c_str());
				while(!token.end()) mass_coeffs.back().params().push_back(std::atof(token.next().c_str()));
			} else if(tag=="CHARGE_COEFF"){
				charge_coeffs.push_back(molkit::Coeff());
				charge_coeffs.back().index()=std::atoi(token.next().c_str());
				while(!token.end()) charge_coeffs.back().params().push_back(std::atof(token.next().c_str()));
			} else if(tag=="PAIR_COEFF"){
				pair_coeffs.push_back(molkit::Coeff());
				pair_coeffs.back().index()=std::atoi(token.next().c_str());
				while(!token.end()) pair_coeffs.back().params().push_back(std::atof(token.next().c_str()));
			} else if(tag=="BOND_COEFF"){
				bond_coeffs.push_back(molkit::Coeff());
				bond_coeffs.back().index()=std::atoi(token.next().c_str());
				while(!token.end()) bond_coeffs.back().params().push_back(std::atof(token.next().c_str()));
			} else if(tag=="ANGLE_COEFF"){
				angle_coeffs.push_back(molkit::Coeff());
				angle_coeffs.back().index()=std::atoi(token.next().c_str());
				while(!token.end()) angle_coeffs.back().params().push_back(std::atof(token.next().c_str()));
			} else if(tag=="DIHEDRAL_COEFF"){
				dihedral_coeffs.push_back(molkit::Coeff());
				dihedral_coeffs.back().index()=std::atoi(token.next().c_str());
				while(!token.end()) dihedral_coeffs.back().params().push_back(std::atof(token.next().c_str()));
			} else if(tag=="IMPROPER_COEFF"){
				improper_coeffs.push_back(molkit::Coeff());
				improper_coeffs.back().index()=std::atoi(token.next().c_str());
				while(!token.end()) improper_coeffs.back().params().push_back(std::atof(token.next().c_str()));
			}
			//read flags
			if(tag=="PRINT"){
				print=string::boolean(token.next().c_str());
			} else if(tag=="NORM_CHG"){
				norm_chg=string::boolean(token.next().c_str());
			} 
		}
		//close the file
		fclose(reader);
		reader=NULL;
		
		if(dmax<1) throw std::invalid_argument("DMAX must be at least 1.");
		
		//==== print the parameters ====
		//**** general parmaeters ****
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
		std::cout<<"UNITS     = "<<unitsys<<"\n";
		std::cout<<"ATOM_T    = "<<atomT<<"\n";
		std::cout<<"SIM_STR   = \""<<simstr<<"\"\n";
		std::cout<<"FORMAT    = "<<format<<"\n";
		std::cout<<"BOX       = "<<box.transpose()<<"\n";
		std::cout<<"PRINT     = "<<print<<"\n";
		std::cout<<"PRINT_ALIAS = "<<print_alias<<"\n";
		std::cout<<"NORM_CHG   = "<<norm_chg<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		//**** charges ****
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("CHARGES - D1",strbuf)<<"\n";
		for(int i=0; i<chargeD1.size(); ++i){
			std::cout<<chargeD1[i].first<<" "<<chargeD1[i].second<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		//**** types ****
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TYPES",strbuf)<<"\n";
		for(int i=0; i<types.size(); ++i){
			std::cout<<"types["<<i<<"] = "<<types[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("ALIASES",strbuf)<<"\n";
		for(int i=0; i<aliases.size(); ++i){
			std::cout<<aliases[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		//**** coeffs ****
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("MASS",strbuf)<<"\n";
		for(int i=0; i<mass_labels.size(); ++i){
			std::cout<<"mass_labels["<<i<<"] = "<<mass_labels[i]<<"\n";
		}
		for(int i=0; i<mass_coeffs.size(); ++i){
			std::cout<<"mass_coeffs["<<i<<"] = "<<mass_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("CHARGES",strbuf)<<"\n";
		for(int i=0; i<charge_labels.size(); ++i){
			std::cout<<"charge_label["<<i<<"] = "<<charge_labels[i]<<"\n";
		}
		for(int i=0; i<charge_coeffs.size(); ++i){
			std::cout<<"charge_coeffs["<<i<<"] = "<<charge_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("PAIRS",strbuf)<<"\n";
		for(int i=0; i<pair_labels.size(); ++i){
			std::cout<<"pair_label["<<i<<"] = "<<pair_labels[i]<<"\n";
		}
		for(int i=0; i<pair_coeffs.size(); ++i){
			std::cout<<"pair_coeffs["<<i<<"] = "<<pair_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("BONDS",strbuf)<<"\n";
		for(int i=0; i<bond_labels.size(); ++i){
			std::cout<<"bond_label["<<i<<"] = "<<bond_labels[i]<<"\n";
		}
		for(int i=0; i<bond_coeffs.size(); ++i){
			std::cout<<"bond_coeffs["<<i<<"] = "<<bond_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("ANGLES",strbuf)<<"\n";
		for(int i=0; i<angle_labels.size(); ++i){
			std::cout<<"angle_label["<<i<<"] = "<<angle_labels[i]<<"\n";
		}
		for(int i=0; i<angle_coeffs.size(); ++i){
			std::cout<<"angle_coeffs["<<i<<"] = "<<angle_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("DIHEDRALS",strbuf)<<"\n";
		for(int i=0; i<dihedral_labels.size(); ++i){
			std::cout<<"dihedral_label["<<i<<"] = "<<dihedral_labels[i]<<"\n";
		}
		for(int i=0; i<dihedral_coeffs.size(); ++i){
			std::cout<<"dihedral_coeffs["<<i<<"] = "<<dihedral_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("IMPROPERS",strbuf)<<"\n";
		for(int i=0; i<improper_labels.size(); ++i){
			std::cout<<"improper_label["<<i<<"] = "<<improper_labels[i]<<"\n";
		}
		for(int i=0; i<improper_coeffs.size(); ++i){
			std::cout<<"improper_coeffs["<<i<<"] = "<<improper_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		
		//==== set aliases ====
		std::cout<<"setting aliases\n";
		for(int i=0; i<aliases.size(); ++i){
			//charges
			for(int n=0; n<1; ++n){
				for(int j=0; j<charge_labels.size(); ++j){
					if(charge_labels[j].type(n).name()==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Label tmp=charge_labels[j];
							tmp.type(n).name()=aliases[i].labels()[k];
							charge_labels.push_back(tmp);
						}
						charge_labels.erase(charge_labels.begin()+j);
						--j;
					}
				}
			}
			//pairs
			for(int n=0; n<1; ++n){
				for(int j=0; j<pair_labels.size(); ++j){
					if(pair_labels[j].type(n).name()==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Label tmp=pair_labels[j];
							tmp.type(n).name()=aliases[i].labels()[k];
							pair_labels.push_back(tmp);
						}
						pair_labels.erase(pair_labels.begin()+j);
						--j;
					}
				}
			}
			//bonds
			for(int n=0; n<2; ++n){
				for(int j=0; j<bond_labels.size(); ++j){
					if(bond_labels[j].type(n).name()==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Label tmp=bond_labels[j];
							tmp.type(n).name()=aliases[i].labels()[k];
							bond_labels.push_back(tmp);
						}
						bond_labels.erase(bond_labels.begin()+j);
						--j;
					}
				}
			}
			//angles
			for(int n=0; n<3; ++n){
				for(int j=0; j<angle_labels.size(); ++j){
					if(angle_labels[j].type(n).name()==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Label tmp=angle_labels[j];
							tmp.type(n).name()=aliases[i].labels()[k];
							angle_labels.push_back(tmp);
						}
						angle_labels.erase(angle_labels.begin()+j);
						--j;
					}
				}
			}
			//dihedrals
			for(int n=0; n<4; ++n){
				for(int j=0; j<dihedral_labels.size(); ++j){
					if(dihedral_labels[j].type(n).name()==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Label tmp=dihedral_labels[j];
							tmp.type(n).name()=aliases[i].labels()[k];
							dihedral_labels.push_back(tmp);
						}
						dihedral_labels.erase(dihedral_labels.begin()+j);
						--j;
					}
				}
			}
			//impropers
			for(int n=0; n<4; ++n){
				for(int j=0; j<improper_labels.size(); ++j){
					if(improper_labels[j].type(n).name()==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Label tmp=improper_labels[j];
							tmp.type(n).name()=aliases[i].labels()[k];
							improper_labels.push_back(tmp);
						}
						improper_labels.erase(improper_labels.begin()+j);
						--j;
					}
				}
			}
		}
		
		//==== set the types for the labels ====
		std::cout<<"setting the types for the labels\n";
		for(int i=0; i<charge_labels.size(); ++i){
			for(int j=0; j<charge_labels[i].types().size(); ++j){
				bool match=false;
				for(int k=0; k<types.size(); ++k){
					if(charge_labels[i].type(j).name()==types[k].name()){
						charge_labels[i].type(j)=types[k];
						match=true; break;
					}
				}
				if(!match){
					std::cout<<"WARNING: Could not find type for charge label: "<<charge_labels[i]<<"\n";
				}
			}
		}
		for(int i=0; i<pair_labels.size(); ++i){
			for(int j=0; j<pair_labels[i].types().size(); ++j){
				bool match=false;
				for(int k=0; k<types.size(); ++k){
					if(pair_labels[i].type(j).name()==types[k].name()){
						pair_labels[i].type(j)=types[k];
						match=true; break;
					}
				}
				if(!match){
					std::cout<<"WARNING: Could not find type for pair label: "<<pair_labels[i]<<"\n";
				}
			}
		}
		for(int i=0; i<bond_labels.size(); ++i){
			for(int j=0; j<bond_labels[i].types().size(); ++j){
				bool match=false;
				for(int k=0; k<types.size(); ++k){
					if(bond_labels[i].type(j).name()==types[k].name()){
						bond_labels[i].type(j)=types[k];
						match=true; break;
					}
				}
				if(!match){
					std::cout<<"WARNING: Could not find type for bond label: "<<bond_labels[i]<<"\n";
				}
			}
		}
		for(int i=0; i<angle_labels.size(); ++i){
			for(int j=0; j<angle_labels[i].types().size(); ++j){
				bool match=false;
				for(int k=0; k<types.size(); ++k){
					if(angle_labels[i].type(j).name()==types[k].name()){
						angle_labels[i].type(j)=types[k];
						match=true; break;
					}
				}
				if(!match){
					std::cout<<"WARNING: Could not find type for angle label: "<<angle_labels[i]<<"\n";
				}
			}
		}
		for(int i=0; i<dihedral_labels.size(); ++i){
			for(int j=0; j<dihedral_labels[i].types().size(); ++j){
				bool match=false;
				for(int k=0; k<types.size(); ++k){
					if(dihedral_labels[i].type(j).name()==types[k].name()){
						dihedral_labels[i].type(j)=types[k];
						match=true; break;
					}
				}
				if(!match){
					std::cout<<"WARNING: Could not find type for dihedral label: "<<dihedral_labels[i]<<"\n";
				}
			}
		}
		for(int i=0; i<improper_labels.size(); ++i){
			for(int j=0; j<improper_labels[i].types().size(); ++j){
				bool match=false;
				for(int k=0; k<types.size(); ++k){
					if(improper_labels[i].type(j).name()==types[k].name()){
						improper_labels[i].type(j)=types[k];
						match=true; break;
					}
				}
				if(!match){
					std::cout<<"WARNING: Could not find type for improper label: "<<improper_labels[i]<<"\n";
				}
			}
		}
		
		if(print_alias){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("CHARGES",strbuf)<<"\n";
			for(int i=0; i<charge_labels.size(); ++i){
				std::cout<<"charge_labels["<<i<<"] = "<<charge_labels[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("PAIRS",strbuf)<<"\n";
			for(int i=0; i<pair_labels.size(); ++i){
				std::cout<<"pair_label["<<i<<"] = "<<pair_labels[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("BONDS",strbuf)<<"\n";
			for(int i=0; i<bond_labels.size(); ++i){
				std::cout<<"bond_label["<<i<<"] = "<<bond_labels[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("ANGLES",strbuf)<<"\n";
			for(int i=0; i<angle_labels.size(); ++i){
				std::cout<<"angle_label["<<i<<"] = "<<angle_labels[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DIHEDRALS",strbuf)<<"\n";
			for(int i=0; i<dihedral_labels.size(); ++i){
				std::cout<<"dihedral_label["<<i<<"] = "<<dihedral_labels[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("IMPROPERS",strbuf)<<"\n";
			for(int i=0; i<improper_labels.size(); ++i){
				std::cout<<"improper_label["<<i<<"] = "<<improper_labels[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
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
		
		//==== make lsmiles - ====
		std::cout<<"making lsmiles\n";
		lsmiles.resize(dmax,std::vector<std::string>(struc.nAtoms()));
		for(int d=0; d<dmax; ++d){
			std::cout<<"depth "<<d+1<<"\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				std::vector<int> lsmilesv;
				//make lsmiles vector
				molkit::make_lsmiles(struc,molgraph,i,d+1,lsmilesv);
				//pack vector
				for(int j=0; j<lsmilesv.size(); ++j){
					lsmiles[d][i]+=ptable::name(lsmilesv[j]);
				}
				const std::string tmp=struc.name(i)+":"+lsmiles[d][i];
				lsmiles[d][i]=tmp;
			}
			std::cout<<print::buf(strbuf)<<"\n";
			const std::string title=std::string("LSMILES - DEPTH ")+std::to_string(d+1);
			std::cout<<print::title(title.c_str(),strbuf)<<"\n";
			if(print){
				for(int i=0; i<struc.nAtoms(); ++i){
					std::cout<<"atom "<<i+1<<" "<<struc.name(i)<<" "<<lsmiles[d][i]<<"\n";
				}
			}
			std::cout<<print::buf(strbuf)<<"\n";
		
		}
		
		//==== assign types ====
		std::cout<<"assigning types\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			for(int n=0; n<pair_labels.size(); ++n){
				if(lsmiles[pair_labels[n].type(0).depth()-1][i]==pair_labels[n].type(0).lsmiles()){
					struc.type(i)=pair_labels[n].index();
				}
			}
			if(struc.type(i)<0){
				std::cout<<"WARNING: Could not find match for: ";
				std::cout<<struc.name(i);
				for(int j=0; j<dmax; ++j) std::cout<<" ls("<<j+1<<") "<<lsmiles[j][i];
				std::cout<<"\n";
			}
		}
		
		//==== assign charges ====
		std::cout<<"assigning charges\n";
		double qtot=0;
		for(int i=0; i<struc.nAtoms(); ++i){
			for(int n=0; n<charge_labels.size(); ++n){
				if(charge_labels[n].type(0).lsmiles()==lsmiles[charge_labels[n].type(0).depth()-1][i]){
					struc.charge(i)=charge_coeffs[charge_labels[n].index()-1].param(0);
				}
			}
		}
		bool zero_charge=false;
		for(int i=0; i<struc.nAtoms(); ++i){
			if(fabs(struc.charge(i))<1.0e-16) zero_charge=true;
		}
		if(zero_charge){
			std::cout<<"WARNING: FOUND ZERO CHARGE.\n";
		}
		
		//==== compute total charge ====
		std::cout<<"computing total charge\n";
		//compute total charge
		qtot=0;
		for(int i=0; i<struc.nAtoms(); ++i){
			qtot+=struc.charge(i);
		}
		std::cout<<"qtot - beg = "<<qtot<<"\n";
		//correct total charge
		if(norm_chg){
			const double dq=1.0*qtot/struc.nAtoms();
			for(int i=0; i<struc.nAtoms(); ++i){
				struc.charge(i)-=dq;
			}
		}
		//compute new total charge
		qtot=0;
		for(int i=0; i<struc.nAtoms(); ++i){
			qtot+=struc.charge(i);
		}
		std::cout<<"qtot - end = "<<qtot<<"\n";
		
		
		//==== compute bonds ====
		if(bond_labels.size()>0){
			std::cout<<"computing bonds\n";
			//find bonds
			for(int i=0; i<struc.nAtoms(); ++i){
				const int ii=i;
				for(int j=0; j<molgraph.edges(i).size(); ++j){
					const int jj=molgraph.edge(i,j).end();
					for(int n=0; n<bond_labels.size(); ++n){
						if(
							bond_labels[n].type(0).lsmiles()==lsmiles[bond_labels[n].type(0).depth()-1][ii] &&
							bond_labels[n].type(1).lsmiles()==lsmiles[bond_labels[n].type(1).depth()-1][jj] 
						){
							std::array<int,3> arr={bond_labels[n].index(),ii+1,jj+1};
							bond_list.push_back(arr);
						}
					}
				}
			}
			//sort bonds
			for(int i=0; i<bond_list.size(); ++i){
				if(bond_list[i][1]>bond_list[i][2]){
					const int tmp=bond_list[i][1];
					bond_list[i][1]=bond_list[i][2];
					bond_list[i][2]=tmp;
				}
			}
			//remove duplicates
			for(int i=0; i<bond_list.size(); ++i){
				for(int j=i+1; j<bond_list.size(); ++j){
					if(bond_list[i]==bond_list[j]){
						bond_list.erase(bond_list.begin()+j);
						j--;
					}
				}
			}
		}
		
		//==== compute angles ====
		if(angle_labels.size()>0){
			std::cout<<"computing angles\n";
			//find angles
			for(int i=0; i<struc.nAtoms(); ++i){
				const int ii=i;
				for(int j=0; j<molgraph.edges(i).size(); ++j){
					const int jj=molgraph.edge(i,j).end();
					for(int k=j+1; k<molgraph.edges(i).size(); ++k){
						const int kk=molgraph.edge(i,k).end();
						for(int n=0; n<angle_labels.size(); ++n){
							if(
								(
								angle_labels[n].type(0).lsmiles()==lsmiles[angle_labels[n].type(0).depth()-1][jj] &&
								angle_labels[n].type(1).lsmiles()==lsmiles[angle_labels[n].type(1).depth()-1][ii] &&
								angle_labels[n].type(2).lsmiles()==lsmiles[angle_labels[n].type(2).depth()-1][kk]
								) || (
								angle_labels[n].type(0).lsmiles()==lsmiles[angle_labels[n].type(0).depth()-1][kk] &&
								angle_labels[n].type(1).lsmiles()==lsmiles[angle_labels[n].type(1).depth()-1][ii] &&
								angle_labels[n].type(2).lsmiles()==lsmiles[angle_labels[n].type(2).depth()-1][jj]
								)
							){
								std::array<int,4> arr={angle_labels[n].index(),jj+1,ii+1,kk+1};
								angle_list.push_back(arr);
							}
						}
					}
				}
			}
			//sort angles
			for(int i=0; i<angle_list.size(); ++i){
				if(angle_list[i][1]>angle_list[i][3]){
					const int tmp=angle_list[i][1];
					angle_list[i][1]=angle_list[i][3];
					angle_list[i][3]=tmp;
				}
			}
			//remove duplicates
			for(int i=0; i<angle_list.size(); ++i){
				for(int j=i+1; j<angle_list.size(); ++j){
					if(angle_list[i]==angle_list[j]){
						angle_list.erase(angle_list.begin()+j);
						j--;
					}
				}
			}
		}
		
		//==== compute dihedrals ====
		if(dihedral_labels.size()>0){
			for(int i1=0; i1<struc.nAtoms(); ++i1){
				const int ii1=i1;
				Graph::color_path(molgraph,ii1,3);
				for(int i0=0; i0<molgraph.edges(ii1).size(); ++i0){
					const int ii0=molgraph.edge(ii1,i0).end();
					for(int i2=0; i2<molgraph.edges(ii1).size(); ++i2){
						if(i2==i0) continue;
						const int ii2=molgraph.edge(ii1,i2).end();
						for(int i3=0; i3<molgraph.edges(ii2).size(); ++i3){
							const int ii3=molgraph.edge(ii2,i3).end();
							if(molgraph.node(ii3).color()<=molgraph.node(ii2).color()) continue;
							for(int n=0; n<dihedral_labels.size(); ++n){
								if(
									dihedral_labels[n].type(0).lsmiles()==lsmiles[dihedral_labels[n].type(0).depth()-1][ii0] &&
									dihedral_labels[n].type(1).lsmiles()==lsmiles[dihedral_labels[n].type(1).depth()-1][ii1] &&
									dihedral_labels[n].type(2).lsmiles()==lsmiles[dihedral_labels[n].type(2).depth()-1][ii2] &&
									dihedral_labels[n].type(3).lsmiles()==lsmiles[dihedral_labels[n].type(3).depth()-1][ii3]
								){
									std::array<int,5> arr={dihedral_labels[n].index(),ii0+1,ii1+1,ii2+1,ii3+1};
									dihedral_list.push_back(arr);
								}
							}
						}
					}
				}
			}
			//remove duplicates
			for(int i=0; i<dihedral_list.size(); ++i){
				for(int j=i+1; j<dihedral_list.size(); ++j){
					if(
						dihedral_list[i][0]==dihedral_list[j][0] &&
						dihedral_list[i][1]==dihedral_list[j][4] &&
						dihedral_list[i][2]==dihedral_list[j][3] &&
						dihedral_list[i][3]==dihedral_list[j][2] &&
						dihedral_list[i][4]==dihedral_list[j][1] 
					){
						dihedral_list.erase(dihedral_list.begin()+j);
						j--;
					}
				}
			}
		}
		
		//==== compute impropers ====
		if(improper_labels.size()>0){
			for(int i=0; i<struc.nAtoms(); ++i){
				const int ii=i;
				for(int j0=0; j0<molgraph.edges(ii).size(); ++j0){
					const int jj0=molgraph.edge(ii,j0).end();
					for(int j1=0; j1<molgraph.edges(ii).size(); ++j1){
						const int jj1=molgraph.edge(ii,j1).end();
						if(jj1==jj0) continue;
						for(int j2=0; j2<molgraph.edges(ii).size(); ++j2){
							const int jj2=molgraph.edge(ii,j2).end();
							if(jj2==jj1 || jj2==jj0) continue;
							for(int n=0; n<improper_labels.size(); ++n){
								if(
									improper_labels[n].type(0).lsmiles()==lsmiles[improper_labels[n].type(0).depth()-1][ii] &&
									improper_labels[n].type(1).lsmiles()==lsmiles[improper_labels[n].type(1).depth()-1][jj0] &&
									improper_labels[n].type(2).lsmiles()==lsmiles[improper_labels[n].type(2).depth()-1][jj1] &&
									improper_labels[n].type(3).lsmiles()==lsmiles[improper_labels[n].type(3).depth()-1][jj2]
								){
									std::array<int,5> arr={improper_labels[n].index(),ii+1,jj0+1,jj1+1,jj2+1};
									improper_list.push_back(arr);
								}
							}
						}
					}
				}
			}
			/*
			//rotate impropers
			for(int i=0; i<improper_list.size(); ++i){
				int minpos=-1;
				if(improper_list[i][2]<improper_list[i][3] && improper_list[i][2]<improper_list[i][4]) minpos=2;
				if(improper_list[i][3]<improper_list[i][2] && improper_list[i][3]<improper_list[i][4]) minpos=3;
				if(improper_list[i][4]<improper_list[i][2] && improper_list[i][4]<improper_list[i][3]) minpos=4;
				if(minpos==3){//rotate left
					double tmp=improper_list[i][2];
					improper_list[i][2]=improper_list[i][3];
					improper_list[i][3]=improper_list[i][4];
					improper_list[i][4]=tmp;
				}
				if(minpos==4){//rotate right
					double tmp=improper_list[i][4];
					improper_list[i][4]=improper_list[i][3];
					improper_list[i][3]=improper_list[i][2];
					improper_list[i][2]=tmp;
				}
			}
			//remove duplicates
			for(int i=0; i<improper_list.size(); ++i){
				if(improper_list[i][3]>improper_list[i][4]){
					improper_list.erase(improper_list.begin()+i);
					i--;
				}
			}
			*/
			for(int i=0; i<improper_list.size(); ++i){
				for(int j=i+1; j<improper_list.size(); ++j){
					if(
						(
							improper_list[i][0]==improper_list[j][0] &&
							improper_list[i][1]==improper_list[j][1] &&
							improper_list[i][2]==improper_list[j][3] &&
							improper_list[i][3]==improper_list[j][2] &&
							improper_list[i][4]==improper_list[j][4] 
						) || (
							improper_list[i][0]==improper_list[j][0] &&
							improper_list[i][1]==improper_list[j][1] &&
							improper_list[i][2]==improper_list[j][2] &&
							improper_list[i][3]==improper_list[j][4] &&
							improper_list[i][4]==improper_list[j][3] 
						) 
					){
						improper_list.erase(improper_list.begin()+j);
						j--;
					}
				}
			}
			
		}
		
		//==== label the graph by connected component (molkit) ====
		std::cout<<"labelling molecules (connected components)\n";
		const int ncc=Graph::color_cc(molgraph);
		std::cout<<"nmol (ncc's) = "<<ncc<<"\n";
		
		//==== write data file ====
		{
			std::cout<<"writing data file\n";
			FILE* writer=fopen("lmp.data","w");
			if(writer==NULL) throw std::runtime_error("Could not open data file.");
			fprintf(writer,"# lammps data file - made with ffa\n");
			fprintf(writer,"\n");
			fprintf(writer," %6i atoms\n",struc.nAtoms());
			if(bond_list.size()>0) fprintf(writer," %6i bonds\n",bond_list.size());
			if(angle_list.size()>0) fprintf(writer," %6i angles\n",angle_list.size());
			if(dihedral_list.size()>0) fprintf(writer," %6i dihedrals\n",dihedral_list.size());
			if(improper_list.size()>0) fprintf(writer," %6i impropers\n",improper_list.size());
			fprintf(writer,"\n");
			fprintf(writer," %6i atom types\n",pair_coeffs.size());
			if(bond_coeffs.size()>0) fprintf(writer," %6i bond types\n",bond_coeffs.size());
			if(angle_coeffs.size()>0) fprintf(writer," %6i angle types\n",angle_coeffs.size());
			if(dihedral_coeffs.size()>0) fprintf(writer," %6i dihedral types\n",dihedral_coeffs.size());
			if(improper_coeffs.size()>0) fprintf(writer," %6i improper types\n",improper_coeffs.size());
			fprintf(writer,"\n");
			fprintf(writer," %12.6f %12.6f xlo xhi \n",0.0,struc.R()(0,0));
			fprintf(writer," %12.6f %12.6f ylo yhi\n",0.0,struc.R()(1,1));
			fprintf(writer," %12.6f %12.6f zlo zhi\n",0.0,struc.R()(2,2));
			fprintf(writer,"\n");
			fprintf(writer," Masses\n");
			fprintf(writer,"\n");
			for(int i=0; i<mass_coeffs.size(); ++i){
				fprintf(writer," %i %12.6f\n",i+1,mass_coeffs[i].param(0));
			}
			fprintf(writer,"\n");
			if(pair_coeffs.size()>0){
				fprintf(writer," Pair Coeffs\n");
				fprintf(writer,"\n");
				for(int i=0; i<pair_coeffs.size(); ++i){
					fprintf(writer," %i ",pair_coeffs[i].index());
					for(int j=0; j<pair_coeffs[i].params().size(); ++j){
						fprintf(writer,"%12.6e ",pair_coeffs[i].param(j));
					}
					fprintf(writer,"\n");
				}
				fprintf(writer,"\n");
			}
			if(bond_coeffs.size()>0){
				fprintf(writer," Bond Coeffs\n");
				fprintf(writer,"\n");
				for(int i=0; i<bond_coeffs.size(); ++i){
					fprintf(writer," %i ",bond_coeffs[i].index());
					for(int j=0; j<bond_coeffs[i].params().size(); ++j){
						fprintf(writer,"%12.6f ",bond_coeffs[i].param(j));
					}
					fprintf(writer,"\n");
				}
				fprintf(writer,"\n");
			}
			if(angle_coeffs.size()>0){
				fprintf(writer," Angle Coeffs\n");
				fprintf(writer,"\n");
				for(int i=0; i<angle_coeffs.size(); ++i){
					fprintf(writer," %i ",angle_coeffs[i].index());
					for(int j=0; j<angle_coeffs[i].params().size(); ++j){
						fprintf(writer,"%12.6f ",angle_coeffs[i].param(j));
					}
					fprintf(writer,"\n");
				}
				fprintf(writer,"\n");
			}
			if(dihedral_coeffs.size()>0){
				fprintf(writer," Dihedral Coeffs\n");
				fprintf(writer,"\n");
				for(int i=0; i<dihedral_coeffs.size(); ++i){
					fprintf(writer," %i ",dihedral_coeffs[i].index());
					for(int j=0; j<dihedral_coeffs[i].params().size(); ++j){
						fprintf(writer,"%12.6f ",dihedral_coeffs[i].param(j));
					}
					fprintf(writer,"\n");
				}
				fprintf(writer,"\n");
			}
			if(improper_coeffs.size()>0){
				fprintf(writer," Improper Coeffs\n");
				fprintf(writer,"\n");
				for(int i=0; i<improper_coeffs.size(); ++i){
					fprintf(writer," %i ",improper_coeffs[i].index());
					fprintf(writer,"%12.6f ",improper_coeffs[i].param(0));
					fprintf(writer,"%6i ",static_cast<int>(improper_coeffs[i].param(1)));
					fprintf(writer,"%6i ",static_cast<int>(improper_coeffs[i].param(2)));
					fprintf(writer,"\n");
				}
				fprintf(writer,"\n");
			}
			fprintf(writer," Atoms # full\n\n");
			for(int i=0; i<struc.nAtoms(); ++i){
				//atom-ID molkit-ID atom-type q x y z
				fprintf(writer," %6i %6i %6i % 12.6f % 12.6f % 12.6f % 12.6f\n",
					i+1,molgraph.node(i).color(),struc.type(i),
					struc.charge(i),struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2]
				);
			}
			fprintf(writer,"\n");
			if(bond_list.size()>0){
				fprintf(writer," Bonds\n\n");
				//bond-ID bond-type atom1 atom2
				for(int i=0; i<bond_list.size(); ++i){
					fprintf(writer," %6i %6i %6i %6i\n",i+1,
						bond_list[i][0],bond_list[i][1],bond_list[i][2]
					);
				}
				fprintf(writer,"\n");
			}
			if(angle_list.size()>0){
				fprintf(writer," Angles\n\n");
				//angle-ID angle-type atom1 atom2 atom3
				for(int i=0; i<angle_list.size(); ++i){
					fprintf(writer," %6i %6i %6i %6i %6i\n",i+1,
						angle_list[i][0],angle_list[i][1],angle_list[i][2],angle_list[i][3]
					);
				}
				fprintf(writer,"\n");
			}
			if(dihedral_list.size()>0){
				fprintf(writer," Dihedrals\n\n");
				//dihedral-ID dihedral-type atom1 atom2 atom3
				for(int i=0; i<dihedral_list.size(); ++i){
					fprintf(writer," %6i %6i %6i %6i %6i %6i\n",i+1,
						dihedral_list[i][0],dihedral_list[i][1],dihedral_list[i][2],dihedral_list[i][3],dihedral_list[i][4]
					);
				}
				fprintf(writer,"\n");
			}
			if(improper_list.size()>0){
				fprintf(writer," Impropers\n\n");
				//improper-ID improper-type atom1 atom2 atom3
				for(int i=0; i<improper_list.size(); ++i){
					fprintf(writer," %6i %6i %6i %6i %6i %6i\n",i+1,
						improper_list[i][0],improper_list[i][1],improper_list[i][2],improper_list[i][3],improper_list[i][4]
					);
				}
				fprintf(writer,"\n");
			}
			fclose(writer);
			writer=NULL;
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