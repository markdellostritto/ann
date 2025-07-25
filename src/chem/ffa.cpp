//c++
#include <iostream>
#include <utility>
//structure
#include "struc/structure.hpp"
// format
#include "format/vasp_struc.hpp"
#include "format/xyz_struc.hpp"
//chem
#include "chem/molkit.hpp"
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
		char* paramfile=new char[string::M];
		char* input    =new char[string::M];
		char* simstr   =new char[string::M];
		char* strbuf=new char[print::len_buf];
	//structure
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true;
		atomT.posn=true; atomT.mass=true; atomT.charge=true; atomT.radius=true;
		Structure struc;
		std::vector<molkit::Type> atom_types;
		Eigen::Vector3d box=Eigen::Vector3d::Zero();
	//molkit
		bool print=false;
		bool print_alias=false;
		bool norm_chg=false;
		Graph molgraph;
		std::vector<double> mass;
		std::vector<double> radius;
		std::vector<molkit::Type> types;
		std::vector<molkit::Alias> aliases;
		std::vector<std::pair<std::string,double> > charges;
		std::vector<std::pair<std::string,double> > radii;
		std::vector<molkit::Coeff> pair_coeffs;
		std::vector<molkit::Coeff> bond_coeffs;
		std::vector<molkit::Coeff> angle_coeffs;
		std::vector<molkit::Coeff> dihedral_coeffs;
		std::vector<molkit::Coeff> improper_coeffs;
		std::vector<molkit::Link> pairs;
		std::vector<molkit::Link> bonds;
		std::vector<molkit::Link> angles;
		std::vector<molkit::Link> dihedrals;
		std::vector<molkit::Link> impropers;
		std::vector<std::array<int,3> > bond_list;
		std::vector<std::array<int,4> > angle_list;
		std::vector<std::array<int,5> > dihedral_list;
		std::vector<std::array<int,5> > improper_list;
	//units
		units::System unitsys;
	//misc
		bool error=false;
	
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
			} else if(tag=="TYPE"){
				types.push_back(molkit::Type());
				molkit::Type::read(token,types.back());
			} else if(tag=="ALIAS"){
				aliases.push_back(molkit::Alias());
				molkit::Alias::read(token,aliases.back());
			} else if(tag=="CHARGE"){
				const std::string name=token.next();
				double chg=std::atof(token.next().c_str());
				charges.push_back(std::pair<std::string,double>(name,chg));
			} else if(tag=="RADIUS"){
				const std::string name=token.next();
				double rad=std::atof(token.next().c_str());
				radii.push_back(std::pair<std::string,double>(name,rad));
			} else if(tag=="PAIR_COEFF"){
				pair_coeffs.push_back(molkit::Coeff());
				molkit::Coeff::read(token,pair_coeffs.back());
			} else if(tag=="BOND_COEFF"){
				bond_coeffs.push_back(molkit::Coeff());
				molkit::Coeff::read(token,bond_coeffs.back());
			} else if(tag=="ANGLE_COEFF"){
				angle_coeffs.push_back(molkit::Coeff());
				molkit::Coeff::read(token,angle_coeffs.back());
			} else if(tag=="DIHEDRAL_COEFF"){
				dihedral_coeffs.push_back(molkit::Coeff());
				molkit::Coeff::read(token,dihedral_coeffs.back());
			} else if(tag=="IMPROPER_COEFF"){
				improper_coeffs.push_back(molkit::Coeff());
				molkit::Coeff::read(token,improper_coeffs.back());
			} else if(tag=="PAIR"){
				pairs.push_back(molkit::Link());
				molkit::Link::read(token,pairs.back());
			} else if(tag=="BOND"){
				bonds.push_back(molkit::Link());
				molkit::Link::read(token,bonds.back());
			} else if(tag=="ANGLE"){
				angles.push_back(molkit::Link());
				molkit::Link::read(token,angles.back());
			} else if(tag=="DIHEDRAL"){
				dihedrals.push_back(molkit::Link());
				molkit::Link::read(token,dihedrals.back());
			} else if(tag=="IMPROPER"){
				impropers.push_back(molkit::Link());
				molkit::Link::read(token,impropers.back());
			} else if(tag=="PRINT"){
				print=string::boolean(token.next().c_str());
			} else if(tag=="PRINT_ALIAS"){
				print_alias=string::boolean(token.next().c_str());
			} else if(tag=="NORM_CHG"){
				norm_chg=string::boolean(token.next().c_str());
			} else if(tag=="MASS"){
				const int index=std::atoi(token.next().c_str())-1;
				const double m=std::atof(token.next().c_str());
				if(index>=mass.size()) mass.resize(index+1);
				mass[index]=m;
			} 
		}
		//close the file
		fclose(reader);
		reader=NULL;
		
		//==== print the parameters ====
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
		std::cout<<print::title("RADII",strbuf)<<"\n";
		for(int i=0; i<radii.size(); ++i){
			std::cout<<radii[i].first<<" "<<radii[i].second<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("MASS",strbuf)<<"\n";
		for(int i=0; i<mass.size(); ++i){
			std::cout<<mass[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TYPES",strbuf)<<"\n";
		for(int i=0; i<types.size(); ++i){
			std::cout<<types[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("ALIASES",strbuf)<<"\n";
		for(int i=0; i<aliases.size(); ++i){
			std::cout<<aliases[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("PAIR_COEFFS",strbuf)<<"\n";
		for(int i=0; i<pair_coeffs.size(); ++i){
			std::cout<<pair_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("BOND_COEFFS",strbuf)<<"\n";
		for(int i=0; i<bond_coeffs.size(); ++i){
			std::cout<<bond_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("ANGLE_COEFFS",strbuf)<<"\n";
		for(int i=0; i<angle_coeffs.size(); ++i){
			std::cout<<angle_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("DIHEDRAL_COEFFS",strbuf)<<"\n";
		for(int i=0; i<dihedral_coeffs.size(); ++i){
			std::cout<<dihedral_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("IMPROPER_COEFFS",strbuf)<<"\n";
		for(int i=0; i<improper_coeffs.size(); ++i){
			std::cout<<improper_coeffs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("CHARGES",strbuf)<<"\n";
		for(int i=0; i<charges.size(); ++i){
			std::cout<<charges[i].first<<" "<<charges[i].second<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("PAIRS",strbuf)<<"\n";
		for(int i=0; i<pairs.size(); ++i){
			std::cout<<pairs[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("BONDS",strbuf)<<"\n";
		for(int i=0; i<bonds.size(); ++i){
			std::cout<<bonds[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("ANGLES",strbuf)<<"\n";
		for(int i=0; i<angles.size(); ++i){
			std::cout<<angles[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("DIHEDRALS",strbuf)<<"\n";
		for(int i=0; i<dihedrals.size(); ++i){
			std::cout<<dihedrals[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("IMPROPERS",strbuf)<<"\n";
		for(int i=0; i<impropers.size(); ++i){
			std::cout<<impropers[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		
		//==== set aliases ====
		std::cout<<"setting aliases\n";
		for(int i=0; i<aliases.size(); ++i){
			//charge
			for(int j=0; j<charges.size(); ++j){
				if(charges[j].first==aliases[i].alias()){
					for(int k=0; k<aliases[i].labels().size(); ++k){
						charges.push_back(
							std::pair<std::string,double>(
								aliases[i].labels()[k],charges[j].second
							)
						);
					}
					charges.erase(charges.begin()+j);
					--j;
				}
			}
			//pairs
			for(int j=0; j<pairs.size(); ++j){
				if(pairs[j].labels()[0]==aliases[i].alias()){
					for(int k=0; k<aliases[i].labels().size(); ++k){
						molkit::Link tmp=pairs[j];
						tmp.labels()[0]=aliases[i].labels()[k];
						pairs.push_back(tmp);
					}
					pairs.erase(pairs.begin()+j);
					--j;
				}
			}
			//bonds
			for(int n=0; n<2; ++n){
				for(int j=0; j<bonds.size(); ++j){
					if(bonds[j].labels()[n]==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Link tmp=bonds[j];
							tmp.labels()[n]=aliases[i].labels()[k];
							bonds.push_back(tmp);
						}
						bonds.erase(bonds.begin()+j);
						--j;
					}
				}
			}
			//angles
			for(int n=0; n<3; ++n){
				for(int j=0; j<angles.size(); ++j){
					if(angles[j].labels()[n]==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Link tmp=angles[j];
							tmp.labels()[n]=aliases[i].labels()[k];
							angles.push_back(tmp);
						}
						angles.erase(angles.begin()+j);
						--j;
					}
				}
			}
			//dihedrals
			for(int n=0; n<4; ++n){
				for(int j=0; j<dihedrals.size(); ++j){
					if(dihedrals[j].labels()[n]==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Link tmp=dihedrals[j];
							tmp.labels()[n]=aliases[i].labels()[k];
							dihedrals.push_back(tmp);
						}
						dihedrals.erase(dihedrals.begin()+j);
						--j;
					}
				}
			}
			//impropers
			for(int n=0; n<4; ++n){
				for(int j=0; j<impropers.size(); ++j){
					if(impropers[j].labels()[n]==aliases[i].alias()){
						for(int k=0; k<aliases[i].labels().size(); ++k){
							molkit::Link tmp=impropers[j];
							tmp.labels()[n]=aliases[i].labels()[k];
							impropers.push_back(tmp);
						}
						impropers.erase(impropers.begin()+j);
						--j;
					}
				}
			}
		}
			
		if(print_alias){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("CHARGES",strbuf)<<"\n";
			for(int i=0; i<charges.size(); ++i){
				std::cout<<charges[i].first<<" "<<charges[i].second<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("PAIRS",strbuf)<<"\n";
			for(int i=0; i<pairs.size(); ++i){
				std::cout<<pairs[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("BONDS",strbuf)<<"\n";
			for(int i=0; i<bonds.size(); ++i){
				std::cout<<bonds[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("ANGLES",strbuf)<<"\n";
			for(int i=0; i<angles.size(); ++i){
				std::cout<<angles[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("DIHEDRALS",strbuf)<<"\n";
			for(int i=0; i<dihedrals.size(); ++i){
				std::cout<<dihedrals[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("IMPROPERS",strbuf)<<"\n";
			for(int i=0; i<impropers.size(); ++i){
				std::cout<<impropers[i]<<"\n";
			}
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//==== assign the charges ====
		if(types.size()!=charges.size()) std::cout<<"WARNING: mismatch between types and charges.\n";
		for(int i=0; i<types.size(); ++i){
			for(int j=0; j<charges.size(); ++j){
				if(charges[j].first==types[i].label()){
					types[i].charge()=charges[j].second;
					break;
				}
			}
		}
		for(int i=0; i<types.size(); ++i){
			if(types[i].charge()==0) std::cout<<"WARNING: type "<<types[i].label()<<" has zero charge\n";
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
		
		//==== make lsmiles ====
		std::cout<<"making lsmiles\n";
		atom_types.resize(struc.nAtoms());
		std::vector<std::string> lsmiles_name(struc.nAtoms());
		for(int i=0; i<struc.nAtoms(); ++i){
			std::vector<int> lsmilesv;
			//make lsmiles vector
			molkit::make_lsmiles(struc,molgraph,i,2,lsmilesv);
			//pack vector
			for(int j=0; j<lsmilesv.size(); ++j){
				lsmiles_name[i]+=ptable::name(lsmilesv[j]);
			}
			const std::string tmp=struc.name(i)+":"+lsmiles_name[i];
			lsmiles_name[i]=tmp;
			//hash vector
			const int hash=string::hash(lsmiles_name[i].c_str());
			//set type
			for(int j=0; j<types.size(); ++j){
				if(hash==types[j].lsmiles_hash()){
					atom_types[i]=types[j];
				}
			}
			
		}
		if(print){
			for(int i=0; i<struc.nAtoms(); ++i){
				std::cout<<"atom "<<i+1<<" "<<struc.name(i)<<" "<<lsmiles_name[i]<<"\n";
			}
		}
		
		//==== check the lsmiles for each atom ====
		if(types.size()>0){
			std::cout<<"checking lsmiles\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				bool match=false;
				for(int j=0; j<types.size(); ++j){
					if(atom_types[i].lsmiles_hash()==types[j].lsmiles_hash()){
						atom_types[i]=types[j];
						match=true; break;
					}
				}
				if(!match) std::cout<<"WARNING: No match for "<<struc.name(i)<<" "<<atom_types[i].lsmiles_name()<<"\n";
			}
		
		
			//==== assign charges ====
			std::cout<<"assigning charges\n";
			double qtot=0;
			for(int i=0; i<struc.nAtoms(); ++i){
				for(int j=0; j<types.size(); ++j){
					if(atom_types[i].lsmiles_hash()==types[j].lsmiles_hash()){
						struc.charge(i)=types[j].charge();
						qtot+=struc.charge(i);
						break;
					}
				}
			}
			std::cout<<"qtot = "<<qtot<<"\n";
			if(norm_chg){
				const double dq=1.0*qtot/struc.nAtoms();
				for(int i=0; i<struc.nAtoms(); ++i){
					struc.charge(i)-=dq;
				}
			}
			
			//==== assign types ====
			std::cout<<"assigning types\n";
			for(int i=0; i<struc.nAtoms(); ++i){
				for(int j=0; j<pairs.size(); ++j){
					if(pairs[j].labels()[0]==atom_types[i].label()){
						struc.type(i)=pairs[j].type();
					}
				}
			}
			
		}
		
		//==== compute bonds ====
		if(bonds.size()>0){
			std::cout<<"computing bonds\n";
			//find bonds
			for(int i=0; i<struc.nAtoms(); ++i){
				const int ii=i;
				for(int j=0; j<molgraph.edges(i).size(); ++j){
					const int jj=molgraph.edge(i,j).end();
					for(int n=0; n<bonds.size(); ++n){
						if(
							atom_types[ii].label()==bonds[n].labels()[0] &&
							atom_types[jj].label()==bonds[n].labels()[1]
						){
							std::array<int,3> arr={bonds[n].type(),ii+1,jj+1};
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
		if(angles.size()>0){
			std::cout<<"computing angles\n";
			//find angles
			for(int i=0; i<struc.nAtoms(); ++i){
				const int ii=i;
				for(int j=0; j<molgraph.edges(i).size(); ++j){
					const int jj=molgraph.edge(i,j).end();
					for(int k=j+1; k<molgraph.edges(i).size(); ++k){
						const int kk=molgraph.edge(i,k).end();
						for(int n=0; n<angles.size(); ++n){
							if(
								(
								atom_types[jj].label()==angles[n].labels()[0] &&
								atom_types[ii].label()==angles[n].labels()[1] &&
								atom_types[kk].label()==angles[n].labels()[2]
								) || (
								atom_types[kk].label()==angles[n].labels()[0] &&
								atom_types[ii].label()==angles[n].labels()[1] &&
								atom_types[jj].label()==angles[n].labels()[2]
								)
							){
								std::array<int,4> arr={angles[n].type(),jj+1,ii+1,kk+1};
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
		if(dihedrals.size()>0){
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
							for(int n=0; n<dihedrals.size(); ++n){
								if(
									atom_types[ii0].label()==dihedrals[n].labels()[0] &&
									atom_types[ii1].label()==dihedrals[n].labels()[1] &&
									atom_types[ii2].label()==dihedrals[n].labels()[2] &&
									atom_types[ii3].label()==dihedrals[n].labels()[3]
								){
									std::array<int,5> arr={dihedrals[n].type(),ii0+1,ii1+1,ii2+1,ii3+1};
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
		if(impropers.size()>0){
			std::cout<<"computing impropers\n";
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
							for(int n=0; n<impropers.size(); ++n){
								if(
									atom_types[ii].label()==impropers[n].labels()[0] &&
									atom_types[jj0].label()==impropers[n].labels()[1] &&
									atom_types[jj1].label()==impropers[n].labels()[2] &&
									atom_types[jj2].label()==impropers[n].labels()[3] 
								){
									std::array<int,5> arr={impropers[n].type(),ii+1,jj0+1,jj1+1,jj2+1};
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
		for(int i=0; i<mass.size(); ++i){
			fprintf(writer," %i %12.6f\n",i+1,mass[i]);
		}
		fprintf(writer,"\n");
		if(pair_coeffs.size()>0){
			fprintf(writer," Pair Coeffs\n");
			fprintf(writer,"\n");
			for(int i=0; i<pair_coeffs.size(); ++i){
				fprintf(writer," %i ",pair_coeffs[i].type());
				for(int j=0; j<pair_coeffs[i].params().size(); ++j){
					fprintf(writer,"%12.6e ",pair_coeffs[i].params()[j]);
				}
				fprintf(writer,"\n");
			}
			fprintf(writer,"\n");
		}
		if(bond_coeffs.size()>0){
			fprintf(writer," Bond Coeffs\n");
			fprintf(writer,"\n");
			for(int i=0; i<bond_coeffs.size(); ++i){
				fprintf(writer," %i ",bond_coeffs[i].type());
				for(int j=0; j<bond_coeffs[i].params().size(); ++j){
					fprintf(writer,"%12.6f ",bond_coeffs[i].params()[j]);
				}
				fprintf(writer,"\n");
			}
			fprintf(writer,"\n");
		}
		if(angle_coeffs.size()>0){
			fprintf(writer," Angle Coeffs\n");
			fprintf(writer,"\n");
			for(int i=0; i<angle_coeffs.size(); ++i){
				fprintf(writer," %i ",angle_coeffs[i].type());
				for(int j=0; j<angle_coeffs[i].params().size(); ++j){
					fprintf(writer,"%12.6f ",angle_coeffs[i].params()[j]);
				}
				fprintf(writer,"\n");
			}
			fprintf(writer,"\n");
		}
		if(dihedral_coeffs.size()>0){
			fprintf(writer," Dihedral Coeffs\n");
			fprintf(writer,"\n");
			for(int i=0; i<dihedral_coeffs.size(); ++i){
				fprintf(writer," %i ",dihedral_coeffs[i].type());
				for(int j=0; j<dihedral_coeffs[i].params().size(); ++j){
					fprintf(writer,"%12.6f ",dihedral_coeffs[i].params()[j]);
				}
				fprintf(writer,"\n");
			}
			fprintf(writer,"\n");
		}
		if(improper_coeffs.size()>0){
			fprintf(writer," Improper Coeffs\n");
			fprintf(writer,"\n");
			for(int i=0; i<improper_coeffs.size(); ++i){
				fprintf(writer," %i ",improper_coeffs[i].type());
				fprintf(writer,"%12.6f ",improper_coeffs[i].params()[0]);
				fprintf(writer,"%6i ",static_cast<int>(improper_coeffs[i].params()[1]));
				fprintf(writer,"%6i ",static_cast<int>(improper_coeffs[i].params()[2]));
				fprintf(writer,"\n");
			}
			fprintf(writer,"\n");
		}
		fprintf(writer," Atoms # full\n\n");
		for(int i=0; i<struc.nAtoms(); ++i){
			//atom-ID molkit-ID atom-type q x y z
			fprintf(writer," %6i %6i %6i % 8.6f % 12.6f % 12.6f % 12.6f\n",
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