#include "structure.hpp"

//**********************************************************************************************
//AtomType
//**********************************************************************************************

std::ostream& operator<<(std::ostream& out, const AtomType& atomT){
	if(atomT.name)	out<<"name ";
	if(atomT.an)	out<<"an ";
	if(atomT.type)	out<<"type ";
	if(atomT.index)	out<<"index ";
	if(atomT.mass)	out<<"mass ";
	if(atomT.charge)out<<"charge ";
	if(atomT.posn)	out<<"posn ";
	if(atomT.force)	out<<"force ";
	if(atomT.symm)	out<<"symm ";
	if(atomT.frac)	out<<"frac ";
	return out;
}

void AtomType::defaults(){
	name	=false;
	an		=false;
	type	=false;
	index	=false;
	mass	=false;
	charge	=false;
	posn	=false;
	force	=false;
	symm	=false;
	frac	=false;
}

unsigned int AtomType::nbytes()const{
	unsigned int nbytes=0;
	nbytes+=sizeof(char)*2;
	nbytes+=sizeof(unsigned int)*an;
	nbytes+=sizeof(unsigned int)*type;
	nbytes+=sizeof(unsigned int)*index;
	nbytes+=sizeof(double)*mass;
	nbytes+=sizeof(double)*charge;
	nbytes+=sizeof(double)*3*posn;
	nbytes+=sizeof(double)*3*force;
	nbytes+=sizeof(unsigned int)*8*symm;
}

//**********************************************************************************************
//Interval
//**********************************************************************************************

std::ostream& operator<<(std::ostream& out, const Interval& i){
	return out<<i.beg<<":"<<i.end<<":"<<i.stride;
}

Interval Interval::read(const char* str){
	std::vector<std::string> strlist;
	string::split(str,":",strlist);
	if(strlist.size()!=2 && strlist.size()!=3) throw std::invalid_argument("Invalid interval format.");
	Interval interval;
	if(string::to_upper(strlist.at(0))=="BEG") interval.beg=1;
	else interval.beg=std::atoi(strlist.at(0).c_str());
	if(string::to_upper(strlist.at(1))=="END") interval.end=-1;
	else interval.end=std::atoi(strlist.at(1).c_str());
	if(strlist.size()==3){
		interval.stride=std::atoi(strlist.at(2).c_str());
	}
	return interval;
}

//**********************************************************************************************
//FILE_FORMAT struct
//**********************************************************************************************

FILE_FORMAT::type FILE_FORMAT::read(const std::string& str){
	if(str=="XDATCAR") return FILE_FORMAT::XDATCAR;
	else if(str=="POSCAR") return FILE_FORMAT::POSCAR;
	else if(str=="OUTCAR") return FILE_FORMAT::OUTCAR;
	else if(str=="VASP_XML") return FILE_FORMAT::VASP_XML;
	else if(str=="XYZ") return FILE_FORMAT::XYZ;
	else if(str=="LAMMPS") return FILE_FORMAT::LAMMPS;
	else if(str=="QE") return FILE_FORMAT::QE;
	else if(str=="AME") return FILE_FORMAT::AME;
	else return FILE_FORMAT::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, FILE_FORMAT::type& format){
	if(format==FILE_FORMAT::UNKNOWN) out<<"UNKNOWN";
	else if(format==FILE_FORMAT::XDATCAR) out<<"XDATCAR";
	else if(format==FILE_FORMAT::POSCAR) out<<"POSCAR";
	else if(format==FILE_FORMAT::OUTCAR) out<<"OUTCAR";
	else if(format==FILE_FORMAT::VASP_XML) out<<"VASP_XML";
	else if(format==FILE_FORMAT::XYZ) out<<"XYZ";
	else if(format==FILE_FORMAT::LAMMPS) out<<"LAMMPS";
	else if(format==FILE_FORMAT::QE) out<<"QE";
	else if(format==FILE_FORMAT::AME) out<<"AME";
	return out;
}

//**********************************************************************************************
//AtomList
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const AtomList& obj){
	for(uint i=0; i<obj.nSpecies_; ++i){
		out<<obj.species_[i]<<" ";
	}
	out<<"\n";
	for(uint i=0; i<obj.nSpecies_; ++i){
		out<<obj.nAtoms_[i]<<" ";
	}
	return out;
}

//==== member functions ====

void AtomList::defaults(){
	if(DEBUG_STRUCTURE>0) std::cout<<"AtomList::defaults():\n";
	nSpecies_=0;
	nAtomsT_=0;
	nAtoms_.clear();
	offsets_.clear();
	species_.clear();
}

void AtomList::resize(const std::vector<uint>& nAtoms, const std::vector<std::string>& atomNames){
	if(DEBUG_STRUCTURE>0) std::cout<<"AtomList::resize(const std::vector<uint>&,const std::vector<std::string>&):\n";
	if(nAtoms.size()!=atomNames.size()) throw std::invalid_argument("Array size mismatch.");
	nSpecies_=nAtoms.size();
	nAtoms_=nAtoms;
	species_=atomNames;
	nAtomsT_=0;
	offsets_.resize(nSpecies_,0);
	for(uint i=0; i<nSpecies_; ++i) nAtomsT_+=nAtoms_[i];
	for(uint i=1; i<nSpecies_; ++i) offsets_[i]=offsets_[i-1]+nAtoms_[i-1];
}

//==== static functions ====

int AtomList::speciesIndex(const std::string& str, const std::vector<std::string>& atomNames){
	if(DEBUG_STRUCTURE>0) std::cout<<"AtomList::speciesIndex(const std::string&, const std::vector<std::string>&):\n";
	for(uint i=0; i<atomNames.size(); ++i) if(str==atomNames[i]) return i;
	return -1;
}

int AtomList::speciesIndex(const char* str, const std::vector<std::string>& atomNames){
	if(DEBUG_STRUCTURE>0) std::cout<<"AtomList::speciesIndex(const char*,const std::vector<std::string>&):\n";
	for(uint i=0; i<atomNames.size(); ++i) if(std::strcmp(str,atomNames[i].c_str())==0) return i;
	return -1;
}

std::vector<uint>& AtomList::read_atoms(const char* str, std::vector<uint>& ids, const AtomList& strucI){
	std::cout<<"read_atoms\n";
	//local function variables
	uint nAtoms=0;
	std::vector<int> atomIndices;
	std::vector<std::string> atomNames;
	std::vector<int> nameIndices;
	
	//load the number of atoms, atom indices, and atom names
	nAtoms=AtomList::read_natoms(str);
	AtomList::read_indices(str,atomIndices);
	AtomList::read_names(str,atomNames);
	nameIndices.resize(atomNames.size());
	for(uint n=0; n<nameIndices.size(); ++n){
		nameIndices[n]=strucI.speciesIndex(atomNames[n]);
		if(nameIndices[n]<0) throw std::invalid_argument("ERROR: Invalid atom name in atom string.");
	}
	
	//set the atom ids
	ids.resize(nAtoms,0);
	for(uint n=0; n<nAtoms; ++n) ids[n]=strucI.id(nameIndices[n],atomIndices[n]);
	
	return ids;
}

uint AtomList::read_natoms(const char* str){
	const char* func_name="AtomList::read_natoms(const char*)";
	if(DEBUG_STRUCTURE>0) std::cout<<func_name<<":\n";
	//local function variables
	char* strtemp=new char[string::M];
	char* substr=new char[string::M];
	char* temp=new char[string::M];
	std::vector<std::string> substrs;
	uint nStrs=0,nAtoms=0;
	bool error=false;
	
	try{
		//copy the string
		std::strcpy(strtemp,str);
		//find the number of substrings
		nStrs=string::substrN(strtemp,",");
		substrs.resize(nStrs);
		substrs[0]=std::string(std::strcpy(temp,std::strtok(strtemp,",")));
		for(uint i=1; i<nStrs; ++i){
			substrs[i]=std::string(std::strcpy(temp,std::strtok(NULL,",")));
		}
		
		//parse the line by commas
		for(uint i=0; i<nStrs; ++i){
			std::strcpy(substr,substrs[i].c_str());
			//find out if this substring has only one atom, or a set of atoms
			if(std::strpbrk(substr,":")==NULL) ++nAtoms; //single atom
			else {
				//set of atoms
				int beg, end;
				//find the beginning index
				std::strcpy(temp,std::strtok(substr,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else beg=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//find the ending index
				std::strcpy(temp,std::strtok(NULL,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else end=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//check the indices
				if(beg<0 || end<0 || end<beg) throw std::invalid_argument("Invalid atomic indices.");
				else nAtoms+=end-beg+1;
			}
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] strtemp;
	delete[] substr;
	delete[] temp;
	
	if(error) throw std::invalid_argument("Invalid atom string.");
	else return nAtoms;
}
	
std::vector<int>& AtomList::read_indices(const char* str, std::vector<int>& indices){
	const char* func_name="AtomList::read_indices(const char*,std::vector<int>&)";
	if(DEBUG_STRUCTURE>0) std::cout<<func_name<<":\n";
	//local function variables
	char* strtemp=new char[string::M];
	char* substr=new char[string::M];
	char* temp=new char[string::M];
	std::vector<std::string> substrs;
	uint nStrs=0;
	bool error=false;
	
	try{
		//clear the vector
		indices.clear();
		//copy the string
		std::strcpy(strtemp,str);
		//find the number of substrings
		nStrs=string::substrN(strtemp,",");
		substrs.resize(nStrs);
		substrs[0]=std::string(std::strcpy(temp,std::strtok(strtemp,",")));
		for(uint i=1; i<nStrs; ++i){
			substrs[i]=std::string(std::strcpy(temp,std::strtok(NULL,",")));
		}
		
		//parse the line by commas
		for(uint i=0; i<nStrs; ++i){
			std::strcpy(substr,substrs[i].c_str());
			//find out if this substring has only one atom, or a set of atoms
			if(std::strpbrk(substr,":")==NULL){
				//single atom
				if(DEBUG_STRUCTURE>1) std::cout<<"Single Atom\n";
				if(std::strpbrk(substr,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else indices.push_back(std::atoi(std::strpbrk(substr,string::DIGITS))-1);
			} else {
				//set of atoms
				if(DEBUG_STRUCTURE>1) std::cout<<"Set of AtomData\n";
				int beg, end;
				//find the beginning index
				std::strcpy(temp,std::strtok(substr,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else beg=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//find the ending index
				std::strcpy(temp,std::strtok(NULL,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else end=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//check the indices
				if(beg<0 || end<0 || end<beg) throw std::invalid_argument("Invalid atomic indices.");
				for(int j=0; j<end-beg+1; j++) indices.push_back(beg+j);
			}
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] strtemp;
	delete[] substr;
	delete[] temp;
	
	if(error){
		indices.clear();
		throw std::invalid_argument("Invalid atom string.");
	} else return indices;
}

std::vector<std::string>& AtomList::read_names(const char* str, std::vector<std::string>& names){
	const char* func_name="AtomList::read_names(const char*,std::vector<std::string>&)";
	if(DEBUG_STRUCTURE>0) std::cout<<func_name<<":\n";
	//local function variables
	char* strtemp=new char[string::M];
	char* substr=new char[string::M];
	char* temp=new char[string::M];
	char* atomName=new char[string::M];
	std::vector<std::string> substrs;
	uint nStrs=0;
	bool error=false;
	
	try{
		//clear the vector
		names.clear();
		//copy the string
		std::strcpy(strtemp,str);
		//find the number of substrings
		nStrs=string::substrN(strtemp,",");
		substrs.resize(nStrs);
		substrs[0]=std::string(std::strcpy(temp,std::strtok(strtemp,",")));
		for(uint i=1; i<nStrs; ++i){
			substrs[i]=std::string(std::strcpy(temp,std::strtok(NULL,",")));
		}
		
		//parse the line by commas
		for(uint i=0; i<nStrs; ++i){
			std::strcpy(substr,substrs[i].c_str());
			//find out if this substring has only one atom, or a set of atoms
			if(std::strpbrk(substr,":")==NULL){
				//single atom
				if(std::strpbrk(substr,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else names.push_back(std::string(string::trim_right(substr,string::DIGITS)));
			} else {
				//set of atoms
				int beg, end;
				//find the beginning index
				std::strcpy(temp,std::strtok(substr,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else beg=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//find the ending index
				std::strcpy(temp,std::strtok(NULL,":"));
				if(std::strpbrk(temp,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else end=std::atoi(std::strpbrk(temp,string::DIGITS))-1;
				//check the indices
				if(beg<0 || end<0 || end<beg) throw std::invalid_argument("Invalid atomic indices");
				else {
					std::strcpy(atomName,string::trim_right(temp,string::DIGITS));
					for(int j=0; j<end-beg+1; ++j) names.push_back(std::string(atomName));
				}
			}
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<func_name<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] strtemp;
	delete[] substr;
	delete[] temp;
	delete[] atomName;
	
	if(error){
		names.clear();
		throw std::invalid_argument("Invalid atom string.");
	} else return names;
}

//**********************************************************************************************
//Thermo
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Thermo& obj){
	out<<"energy = "<<obj.energy_<<"\n";
	out<<"temp   = "<<obj.temp_<<"\n";
	return out;
}

//**********************************************************************************************
//AtomData
//**********************************************************************************************

//member functions

void AtomData::clear(){
	if(DEBUG_STRUCTURE>0) std::cout<<"AtomData::clear():\n";
	name_.clear();
	an_.clear();
	type_.clear();
	index_.clear();
	mass_.clear();
	charge_.clear();
	posn_.clear();
	force_.clear();
	symm_.clear();
}

AtomType AtomData::atomType()const{
	AtomType atomT;
	if(name_.size()>0)	atomT.name=true;
	if(an_.size()>0)	atomT.an=true;
	if(type_.size()>0)	atomT.type=true;
	if(index_.size()>0)	atomT.index=true;
	if(mass_.size()>0)	atomT.mass=true;
	if(charge_.size()>0)atomT.charge=true;
	if(posn_.size()>0)	atomT.posn=true;
	if(force_.size()>0)	atomT.force=true;
	if(symm_.size()>0)	atomT.symm=true;
	return atomT;
}

//resizing

void AtomData::resize(uint nAtoms, const AtomType& atomT){
	if(DEBUG_STRUCTURE>0) std::cout<<"AtomData::resize(uint nAtoms,const AtomType&):\n";
	if(atomT.name)	name_.resize(nAtoms);
	if(atomT.an)	an_.resize(nAtoms);
	if(atomT.type)	type_.resize(nAtoms);
	if(atomT.index)	index_.resize(nAtoms);
	if(atomT.mass)	mass_.resize(nAtoms);
	if(atomT.charge)charge_.resize(nAtoms);
	if(atomT.posn)	posn_.resize(nAtoms);
	if(atomT.force)	force_.resize(nAtoms);
	if(atomT.symm)	symm_.resize(nAtoms);
}

//**********************************************************************************************
//Structure
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Structure& struc){
	out<<static_cast<const Cell&>(struc)<<"\n";
	out<<static_cast<const AtomList&>(struc)<<"\n";
	out<<static_cast<const Thermo&>(struc)<<"\n";
	return out;
}

//==== member functions ====

void Structure::clear(){
	if(DEBUG_STRUCTURE>0) std::cout<<"Structure::clear():\n";
	AtomList::clear();
	AtomData::clear();
}

void Structure::resize(const std::vector<uint>& nAtoms, const std::vector<std::string>& atomNames, const AtomType& atomT){
	if(DEBUG_STRUCTURE>0) std::cout<<"Structure::resize(const std::vector<uint>&,const std::vector<std::string>&):\n";
	unsigned int nAtomsT=0;
	for(unsigned int i=0; i<nAtoms.size(); ++i) nAtomsT+=nAtoms[i];
	AtomList::resize(nAtoms,atomNames);
	AtomData::resize(nAtomsT,atomT);
	//set the names, species, and indices
	uint count=0;
	for(uint n=0; n<this->nSpecies_; ++n){
		for(uint m=0; m<this->nAtoms_[n]; ++m){
			if(this->name_.size()>0) this->name_[count]=this->species_[n];
			if(this->an_.size()>0) this->an_[count]=PTable::an(this->species_[n].c_str());
			if(this->type_.size()>0) this->type_[count]=n;
			if(this->index_.size()>0) this->index_[count]=m;
			++count;
		}
	}
}

//**********************************************
// Simulation
//**********************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Simulation& sim){
	out<<"****************************************\n";
	out<<"************** SIMULATION **************\n";
	out<<"SIM   = "<<sim.name_<<"\n";
	out<<"TS    = "<<sim.timestep_<<"\n";
	out<<"T     = "<<sim.timesteps_<<"\n";
	out<<"CF    = "<<sim.cell_fixed_<<"\n";
	out<<"AtomT = "<<sim.atomT_<<"\n";
	out<<"************** SIMULATION **************\n";
	out<<"****************************************";
	return out;
}

//==== member functions ====

void Simulation::defaults(){
	if(DEBUG_STRUCTURE>0) std::cout<<"Simulation::defaults():\n";
	name_=std::string("SYSTEM");
	timestep_=0;
	timesteps_=0;
}

void Simulation::clear(){
	if(DEBUG_STRUCTURE>0) std::cout<<"Simulation::clear():\n";
	frames_.clear();
	defaults();
}

void Simulation::resize(uint ts, const std::vector<uint>& nAtoms, const std::vector<std::string>& names, const AtomType& atomT){
	if(DEBUG_STRUCTURE>0) std::cout<<"Simulation::resize(uint,const std::vector<uint>&,const std::vector<std::string>&,const AtomType&):\n";
	timesteps_=ts;
	atomT_=atomT;
	frames_.resize(timesteps_,Structure(nAtoms,names,atomT));
}

void Simulation::resize(uint ts){
	if(DEBUG_STRUCTURE>0) std::cout<<"Simulation::resize(uint):\n";
	timesteps_=ts;
	frames_.resize(timesteps_);
}
