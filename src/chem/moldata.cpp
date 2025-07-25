// chem
#include "chem/moldata.hpp"
// str
#include "str/token.hpp"
#include "str/string.hpp"

void Atom::clear(){
	index=0;
	type=0;
	mol=0;
	charge=0;
	posn=Eigen::Vector3d::Zero();
	vel=Eigen::Vector3d::Zero();
}

//==== operators ====

MolData& MolData::operator+=(const MolData& moldata){
	
	//count
	std::cout<<"summing count\n";
	const int nAtomsT=nAtoms+moldata.nAtoms;
	const int nBondsT=nBonds+moldata.nBonds;
	const int nAnglesT=nAngles+moldata.nAngles;
	const int nDihedralsT=nDihedrals+moldata.nDihedrals;
	const int nImpropersT=nImpropers+moldata.nImpropers;
	
	//limits
	std::cout<<"setting limits\n";
	xlim[0]=std::min(xlim[0],moldata.xlim[0]);
	ylim[0]=std::min(ylim[0],moldata.ylim[0]);
	zlim[0]=std::min(zlim[0],moldata.zlim[0]);
	xlim[1]=std::max(xlim[1],moldata.xlim[1]);
	ylim[1]=std::max(ylim[1],moldata.ylim[1]);
	zlim[1]=std::max(zlim[1],moldata.zlim[1]);
	
	//types
	std::cout<<"summing types\n";
	const int nTypesT=nTypes+moldata.nTypes;
	const int nBondTypesT=nBondTypes+moldata.nBondTypes;
	const int nAngleTypesT=nAngleTypes+moldata.nAngleTypes;
	const int nDihedralTypesT=nDihedralTypes+moldata.nDihedralTypes;
	const int nImproperTypesT=nImproperTypes+moldata.nImproperTypes;
	mass.resize(nTypesT);
	for(int i=nTypes; i<nTypesT; ++i){
		mass[i]=moldata.mass[i-nTypes];
	}
	pair.resize(nTypesT);
	for(int i=nTypes; i<nTypesT; ++i){
		pair[i]=moldata.pair[i-nTypes];
	}
	
	//coeffs
	std::cout<<"combining coeffs\n";
	cbond.resize(nBondTypesT);
	for(int i=nBondTypes; i<nBondTypesT; ++i){
		cbond[i]=moldata.cbond[i-nBondTypes];
	}
	cangle.resize(nAngleTypesT);
	for(int i=nAngleTypes; i<nAngleTypesT; ++i){
		cangle[i]=moldata.cangle[i-nAngleTypes];
	}
	cdihedral.resize(nDihedralTypesT);
	for(int i=nDihedralTypes; i<nDihedralTypesT; ++i){
		cdihedral[i]=moldata.cdihedral[i-nDihedralTypes];
	}
	cimproper.resize(nImproperTypesT);
	for(int i=nImproperTypes; i<nImproperTypesT; ++i){
		cimproper[i]=moldata.cimproper[i-nImproperTypes];
	}
	
	//links
	std::cout<<"combining links\n";
	bonds.resize(nBondsT);
	for(int i=nBonds; i<nBondsT; ++i){
		bonds[i]=moldata.bonds[i-nBonds];
		bonds[i][0]+=nBondTypes;
		bonds[i][1]+=nAtoms;
		bonds[i][2]+=nAtoms;
	}
	angles.resize(nAnglesT);
	for(int i=nAngles; i<nAnglesT; ++i){
		angles[i]=moldata.angles[i-nAngles];
		angles[i][0]+=nAngleTypes;
		angles[i][1]+=nAtoms;
		angles[i][2]+=nAtoms;
		angles[i][3]+=nAtoms;
	}
	dihedrals.resize(nDihedralsT);
	for(int i=nDihedrals; i<nDihedralsT; ++i){
		dihedrals[i]=moldata.dihedrals[i-nDihedrals];
		dihedrals[i][0]+=nDihedralTypes;
		dihedrals[i][1]+=nAtoms;
		dihedrals[i][2]+=nAtoms;
		dihedrals[i][3]+=nAtoms;
		dihedrals[i][4]+=nAtoms;
	}
	impropers.resize(nImpropersT);
	for(int i=nImpropers; i<nImpropersT; ++i){
		impropers[i]=moldata.impropers[i-nImpropers];
		impropers[i][0]+=nImproperTypes;
		impropers[i][1]+=nAtoms;
		impropers[i][2]+=nAtoms;
		impropers[i][3]+=nAtoms;
		impropers[i][4]+=nAtoms;
	}
	
	//atoms
	std::cout<<"combining atoms\n";
	atoms.resize(nAtomsT);
	for(int i=nAtoms; i<nAtomsT; ++i){
		atoms[i]=moldata.atoms[i-nAtoms];
		atoms[i].index+=nAtoms;
		atoms[i].type+=nTypes;
	}
	
	//set total count
	std::cout<<"setting total count\n";
	nBonds=nBondsT;
	nAngles=nAnglesT;
	nDihedrals=nDihedralsT;
	nImpropers=nImpropersT;
	nAtoms=nAtomsT;
	
	//set total types
	std::cout<<"setting total types\n";
	nTypes=nTypesT;
	nBondTypes=nBondTypesT;
	nAngleTypes=nAngleTypesT;
	nDihedralTypes=nDihedralTypesT;
	nImproperTypes=nImproperTypesT;
	
	return *this;
}
	
//==== member functions ====

void MolData::clear(){
	//count
	nAtoms=0;
	nBonds=0;
	nAngles=0;
	nDihedrals=0;
	nImpropers=0;
	//types
	nTypes=0;
	nBondTypes=0;
	nAngleTypes=0;
	nDihedralTypes=0;
	nImproperTypes=0;
	mass.clear();
	pair.clear();
	//coeffs
	cbond.clear();
	cangle.clear();
	cdihedral.clear();
	cimproper.clear();
	//links
	bonds.clear();
	angles.clear();
	dihedrals.clear();
	impropers.clear();
	//atoms
	atoms.clear();
}

//==== static functions ====

MolData& MolData::read(const char* file, MolData& moldata){
	FILE* reader=fopen(file,"r");
	if(reader==NULL) throw std::runtime_error("Could not open data file.");
	char* input=new char[string::M];
	Token token;
	
	//read in number of atoms/bonds/...
	std::cout<<"reading in atoms/bonds/...\n";
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"atoms")!=NULL){
			token.read(input,string::WS);
			moldata.nAtoms=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"atom types")!=NULL){
			token.read(input,string::WS);
			moldata.nTypes=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"bonds")!=NULL){
			token.read(input,string::WS);
			moldata.nBonds=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"bond types")!=NULL){
			token.read(input,string::WS);
			moldata.nBondTypes=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"angles")!=NULL){
			token.read(input,string::WS);
			moldata.nAngles=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"angle types")!=NULL){
			token.read(input,string::WS);
			moldata.nAngleTypes=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"dihedrals")!=NULL){
			token.read(input,string::WS);
			moldata.nDihedrals=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"dihedral types")!=NULL){
			token.read(input,string::WS);
			moldata.nDihedralTypes=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"impropers")!=NULL){
			token.read(input,string::WS);
			moldata.nImpropers=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"improper types")!=NULL){
			token.read(input,string::WS);
			moldata.nImproperTypes=std::atoi(token.next().c_str());
		} else if(std::strstr(input,"Atoms")!=NULL){
			break;
		}
	}
	std::rewind(reader);
	
	//check
	if(moldata.nTypes<=0) throw std::invalid_argument("Invalid number of types.\n");
	if(moldata.nBondTypes<0) throw std::invalid_argument("Invalid number of bond types.\n");
	if(moldata.nAngleTypes<0) throw std::invalid_argument("Invalid number of angle types.\n");
	if(moldata.nDihedralTypes<0) throw std::invalid_argument("Invalid number of dihedral types.\n");
	if(moldata.nImproperTypes<0) throw std::invalid_argument("Invalid number of improper types.\n");
	if(moldata.nBonds<0 || (moldata.nBondTypes>0 && moldata.nBonds==0)) throw std::invalid_argument("Invalid number of bonds.\n");
	if(moldata.nAngles<0 || (moldata.nAngleTypes>0 && moldata.nAngles==0)) throw std::invalid_argument("Invalid number of angles.\n");
	if(moldata.nDihedrals<0 || (moldata.nDihedralTypes>0 && moldata.nDihedrals==0)) throw std::invalid_argument("Invalid number of dihedrals.\n");
	if(moldata.nImpropers<0 || (moldata.nImproperTypes>0 && moldata.nImpropers==0)) throw std::invalid_argument("Invalid number of impropers.\n");
	
	//resize
	std::cout<<"resizing\n";
	//types
	moldata.mass.resize(moldata.nTypes);
	moldata.pair.resize(moldata.nTypes);
	//coeffs
	moldata.cbond.resize(moldata.nBondTypes);
	moldata.cangle.resize(moldata.nAngleTypes);
	moldata.cdihedral.resize(moldata.nDihedralTypes);
	moldata.cimproper.resize(moldata.nImproperTypes);
	//links
	moldata.bonds.resize(moldata.nBonds);
	moldata.angles.resize(moldata.nAngles);
	moldata.dihedrals.resize(moldata.nDihedrals);
	moldata.impropers.resize(moldata.nImpropers);
	//atoms
	moldata.atoms.resize(moldata.nAtoms);
	
	//read limits
	std::cout<<"reading limits\n";
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"xlo")!=NULL){
			token.read(input,string::WS);
			moldata.xlim[0]=std::atof(token.next().c_str());
			moldata.xlim[1]=std::atof(token.next().c_str());
		} else if(std::strstr(input,"ylo")!=NULL){
			token.read(input,string::WS);
			moldata.ylim[0]=std::atof(token.next().c_str());
			moldata.ylim[1]=std::atof(token.next().c_str());
		} else if(std::strstr(input,"zlo")!=NULL){
			token.read(input,string::WS);
			moldata.zlim[0]=std::atof(token.next().c_str());
			moldata.zlim[1]=std::atof(token.next().c_str());
		} else if(std::strstr(input,"Atoms")!=NULL){
			break;
		}
	}
	std::rewind(reader);
	std::cout<<"xlim = "<<moldata.xlim[0]<<" "<<moldata.xlim[1]<<"\n";
	std::cout<<"ylim = "<<moldata.ylim[0]<<" "<<moldata.ylim[1]<<"\n";
	std::cout<<"zlim = "<<moldata.zlim[0]<<" "<<moldata.zlim[1]<<"\n";
	
	//read data
	std::cout<<"reading coefficients\n";
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"Masses")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nTypes; ++i){
				token.read(fgets(input,string::M,reader),string::WS); token.next();
				moldata.mass[i]=std::atof(token.next().c_str());
			}
		} else if(std::strstr(input,"Pair Coeffs")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nTypes; ++i){
				token.read(fgets(input,string::M,reader),string::WS); token.next();
				while(!token.end()){
					moldata.pair[i].push_back(std::atof(token.next().c_str()));
				}
			}
		} else if(std::strstr(input,"Bond Coeffs")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nBondTypes; ++i){
				token.read(fgets(input,string::M,reader),string::WS); token.next();
				while(!token.end()){
					moldata.cbond[i].push_back(std::atof(token.next().c_str()));
				}
			}
		} else if(std::strstr(input,"Angle Coeffs")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nAngleTypes; ++i){
				token.read(fgets(input,string::M,reader),string::WS); token.next();
				while(!token.end()){
					moldata.cangle[i].push_back(std::atof(token.next().c_str()));
				}
			}
		} else if(std::strstr(input,"Dihedral Coeffs")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nDihedralTypes; ++i){
				token.read(fgets(input,string::M,reader),string::WS); token.next();
				while(!token.end()){
					moldata.cdihedral[i].push_back(std::atof(token.next().c_str()));
				}
			}
		} else if(std::strstr(input,"Improper Coeffs")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nImproperTypes; ++i){
				token.read(fgets(input,string::M,reader),string::WS); token.next();
				while(!token.end()){
					moldata.cimproper[i].push_back(std::atof(token.next().c_str()));
				}
			}
		} else if(std::strstr(input,"Atoms")!=NULL){
			break;
		}
	}
	std::rewind(reader);
	
	//read atoms
	std::cout<<"reading atoms\n";
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"Atoms")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nAtoms; ++i){
				token.read(fgets(input,string::M,reader),string::WS);
				const int id=std::atoi(token.next().c_str())-1;
				moldata.atoms[id].index=id+1;
				moldata.atoms[id].mol=std::atoi(token.next().c_str());
				moldata.atoms[id].type=std::atoi(token.next().c_str());
				moldata.atoms[id].charge=std::atof(token.next().c_str());
				moldata.atoms[id].posn[0]=std::atof(token.next().c_str());
				moldata.atoms[id].posn[1]=std::atof(token.next().c_str());
				moldata.atoms[id].posn[2]=std::atof(token.next().c_str());
			}
			break;
		}
	}
	std::rewind(reader);
	
	//read velocities
	std::cout<<"reading velocities\n";
	moldata.atoms.resize(moldata.nAtoms);
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"Velocities")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nAtoms; ++i){
				token.read(fgets(input,string::M,reader),string::WS);
				const int id=std::atoi(token.next().c_str())-1;
				moldata.atoms[id].vel[0]=std::atof(token.next().c_str());
				moldata.atoms[id].vel[1]=std::atof(token.next().c_str());
				moldata.atoms[id].vel[2]=std::atof(token.next().c_str());
			}
			break;
		}
	}
	std::rewind(reader);
	
	//read bonds
	std::cout<<"reading bonds\n";
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"Bonds")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nBonds; ++i){
				token.read(fgets(input,string::M,reader),string::WS);
				const int id=std::atoi(token.next().c_str())-1;
				const int type=std::atoi(token.next().c_str());
				const int a1=std::atoi(token.next().c_str());
				const int a2=std::atoi(token.next().c_str());
				moldata.bonds[id][0]=type;
				moldata.bonds[id][1]=a1;
				moldata.bonds[id][2]=a2;
			}
			break;
		}
	}
	std::rewind(reader);
	
	//read angles
	std::cout<<"reading angles\n";
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"Angles")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nAngles; ++i){
				token.read(fgets(input,string::M,reader),string::WS);
				const int id=std::atoi(token.next().c_str())-1;
				const int type=std::atoi(token.next().c_str());
				const int a1=std::atoi(token.next().c_str());
				const int a2=std::atoi(token.next().c_str());
				const int a3=std::atoi(token.next().c_str());
				moldata.angles[id][0]=type;
				moldata.angles[id][1]=a1;
				moldata.angles[id][2]=a2;
				moldata.angles[id][3]=a3;
			}
			break;
		}
	}
	std::rewind(reader);
	
	//read dihedrals
	std::cout<<"reading dihedrals\n";
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"Dihedrals")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nDihedrals; ++i){
				token.read(fgets(input,string::M,reader),string::WS);
				const int id=std::atoi(token.next().c_str())-1;
				const int type=std::atoi(token.next().c_str());
				const int a1=std::atoi(token.next().c_str());
				const int a2=std::atoi(token.next().c_str());
				const int a3=std::atoi(token.next().c_str());
				const int a4=std::atoi(token.next().c_str());
				moldata.dihedrals[id][0]=type;
				moldata.dihedrals[id][1]=a1;
				moldata.dihedrals[id][2]=a2;
				moldata.dihedrals[id][3]=a3;
				moldata.dihedrals[id][4]=a4;
			}
			break;
		}
	}
	std::rewind(reader);
	
	//read impropers
	std::cout<<"reading impropers\n";
	while(fgets(input,string::M,reader)!=NULL){
		if(std::strstr(input,"Impropers")!=NULL){
			fgets(input,string::M,reader);
			for(int i=0; i<moldata.nImpropers; ++i){
				token.read(fgets(input,string::M,reader),string::WS);
				const int id=std::atoi(token.next().c_str())-1;
				const int type=std::atoi(token.next().c_str());
				const int a1=std::atoi(token.next().c_str());
				const int a2=std::atoi(token.next().c_str());
				const int a3=std::atoi(token.next().c_str());
				const int a4=std::atoi(token.next().c_str());
				moldata.impropers[id][0]=type;
				moldata.impropers[id][1]=a1;
				moldata.impropers[id][2]=a2;
				moldata.impropers[id][3]=a3;
				moldata.impropers[id][4]=a4;
			}
			break;
		}
	}
	std::rewind(reader);
	
	delete[] input;
	fclose(reader);
	reader=NULL;
	
	return moldata;
}

const MolData& MolData::write(const char* file, const MolData& moldata){
	FILE* writer=fopen(file,"w");
	if(writer==NULL) throw std::runtime_error("Could not open data file.");
	
	fprintf(writer,"LAMMPS data file\n\n");
	
	fprintf(writer,"%i atoms\n",moldata.nAtoms);
	fprintf(writer,"%i atom types\n",moldata.nTypes);
	fprintf(writer,"%i bonds\n",moldata.nBonds);
	fprintf(writer,"%i bond types\n",moldata.nBondTypes);
	fprintf(writer,"%i angles\n",moldata.nAngles);
	fprintf(writer,"%i angle types\n",moldata.nAngleTypes);
	fprintf(writer,"%i dihedrals\n",moldata.nDihedrals);
	fprintf(writer,"%i dihedral types\n",moldata.nDihedralTypes);
	fprintf(writer,"%i impropers\n",moldata.nImpropers);
	fprintf(writer,"%i improper types\n",moldata.nImproperTypes);
	fprintf(writer,"\n");
	
	fprintf(writer,"%f %f xlo xhi\n",moldata.xlim[0],moldata.xlim[1]);
	fprintf(writer,"%f %f ylo yhi\n",moldata.ylim[0],moldata.ylim[1]);
	fprintf(writer,"%f %f zlo zhi\n",moldata.zlim[0],moldata.zlim[1]);
	fprintf(writer,"\n");
	
	fprintf(writer,"Masses\n\n");
	for(int i=0; i<moldata.nTypes; ++i){
		fprintf(writer,"%i %.3f\n",i+1,moldata.mass[i]);
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Pair Coeffs\n\n");
	for(int i=0; i<moldata.nTypes; ++i){
		fprintf(writer,"%i ",i+1);
		for(int j=0; j<moldata.pair[i].size(); ++j){
			fprintf(writer,"%f ",moldata.pair[i][j]);
		}
		fprintf(writer,"\n");
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Bond Coeffs\n\n");
	for(int i=0; i<moldata.nBondTypes; ++i){
		fprintf(writer,"%i ",i+1);
		for(int j=0; j<moldata.cbond[i].size(); ++j){
			fprintf(writer,"%f ",moldata.cbond[i][j]);
		}
		fprintf(writer,"\n");
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Angle Coeffs\n\n");
	for(int i=0; i<moldata.nAngleTypes; ++i){
		fprintf(writer,"%i ",i+1);
		for(int j=0; j<moldata.cangle[i].size(); ++j){
			fprintf(writer,"%f ",moldata.cangle[i][j]);
		}
		fprintf(writer,"\n");
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Dihedral Coeffs\n\n");
	for(int i=0; i<moldata.nDihedralTypes; ++i){
		fprintf(writer,"%i ",i+1);
		for(int j=0; j<moldata.cdihedral[i].size(); ++j){
			fprintf(writer,"%f ",moldata.cdihedral[i][j]);
		}
		fprintf(writer,"\n");
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Improper Coeffs\n\n");
	for(int i=0; i<moldata.nImproperTypes; ++i){
		fprintf(writer,"%i ",i+1);
		for(int j=0; j<moldata.cimproper[i].size(); ++j){
			fprintf(writer,"%f ",moldata.cimproper[i][j]);
		}
		fprintf(writer,"\n");
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Atoms #full\n\n");
	for(int i=0; i<moldata.nAtoms; ++i){
		const Atom atom=moldata.atoms[i];
		fprintf(writer,"%i %i %i %f %f %f %f\n",
			atom.index,atom.mol,atom.type,atom.charge,
			atom.posn[0],atom.posn[1],atom.posn[2]
		);
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Velocities\n\n");
	for(int i=0; i<moldata.nAtoms; ++i){
		const Atom atom=moldata.atoms[i];
		fprintf(writer,"%i %f %f %f\n",
			atom.index,atom.vel[0],atom.vel[1],atom.vel[2]
		);
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Bonds\n\n");
	for(int i=0; i<moldata.nBonds; ++i){
		fprintf(writer,"%i %i %i %i\n",
			i+1,moldata.bonds[i][0],moldata.bonds[i][1],moldata.bonds[i][2]
		);
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Angles\n\n");
	for(int i=0; i<moldata.nAngles; ++i){
		fprintf(writer,"%i %i %i %i %i\n",
			i+1,moldata.angles[i][0],moldata.angles[i][1],moldata.angles[i][2],moldata.angles[i][3]
		);
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Dihedrals\n\n");
	for(int i=0; i<moldata.nDihedrals; ++i){
		fprintf(writer,"%i %i %i %i %i %i\n",
			i+1,moldata.dihedrals[i][0],moldata.dihedrals[i][1],moldata.dihedrals[i][2],moldata.dihedrals[i][3],moldata.dihedrals[i][4]
		);
	}
	fprintf(writer,"\n");
	
	fprintf(writer,"Impropers\n\n");
	for(int i=0; i<moldata.nImpropers; ++i){
		fprintf(writer,"%i %i %i %i %i %i\n",
			i+1,moldata.impropers[i][0],moldata.impropers[i][1],moldata.impropers[i][2],moldata.impropers[i][3],moldata.impropers[i][4]
		);
	}
	fprintf(writer,"\n");
	
	fclose(writer);
	writer=NULL;
	
	return moldata;
}