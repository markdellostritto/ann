//c++ libraries
#include <iostream>
//c libraries
#include <stdexcept>
// ann - strings
#include "string.hpp"
// ann - chemistry
#include "ptable.hpp"
// ann - eigen
#include "eigen.hpp"
// ann - print
#include "print.hpp"
// ann - structure
#include "structure.hpp"
// ann - math
#include "math_const.hpp"

//**********************************************************************************************
//AtomType
//**********************************************************************************************

std::ostream& operator<<(std::ostream& out, const AtomType& atomT){
	//basic properties
	if(atomT.name)		out<<"name ";
	if(atomT.an)		out<<"an ";
	if(atomT.type)		out<<"type ";
	if(atomT.index)	out<<"index ";
	//serial properties
	if(atomT.mass)		out<<"mass ";
	if(atomT.charge)	out<<"charge ";
	if(atomT.spin)		out<<"spin ";
	//vector properties
	if(atomT.posn)		out<<"posn ";
	if(atomT.vel)		out<<"vel ";
	if(atomT.force)	out<<"force ";
	//nnp
	if(atomT.symm)		out<<"symm ";
	//neigh
	if(atomT.neigh)	out<<"neigh ";
	return out;
}

void AtomType::defaults(){
	//basic properties
	name		=false;
	an		=false;
	type		=false;
	index	=false;
	//serial properties
	mass		=false;
	charge	=false;
	spin		=false;
	//vector properties
	posn		=false;
	vel		=false;
	force	=false;
	//nnp
	symm		=false;
	//neigh
	neigh	=false;
}

//**********************************************************************************************
//Thermo
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Thermo& obj){
	out<<"energy = "<<obj.energy_<<"\n";
	out<<"ewald  = "<<obj.ewald_<<"\n";
	out<<"temp   = "<<obj.temp_<<"\n";
	out<<"press  = "<<obj.press_;
	return out;
}

//==== member functions ====

void Thermo::clear(){
	energy_=0;
	ewald_=0;
	temp_=0;
	press_=0;
}

//==== static functions ====

Thermo& Thermo::make_super(const Eigen::Vector3i& s, const Thermo& thermo1, Thermo& thermo2){
	if(&thermo1==&thermo2) throw std::runtime_error("Thermo::make_super(const Eigen::Vector3i&,const Thermo&,Thermo&): identical references.\n");
	if(s[0]<=0 || s[1]<=0 || s[2]<=0) throw std::runtime_error("Thermo::make_super(const Eigen::Vector3i&,const Thermo&,Thermo&): invalid supercell vector.\n");
	const int st=s[0]*s[1]*s[2];
	thermo2=thermo1;
	thermo2.energy()*=st;
	thermo2.ewald()*=st;
	return thermo2;
}

//**********************************************************************************************
//Neighbor
//**********************************************************************************************

void Neighbor::clear(){
	r_.setZero();
	dr_=0.0;
	type_=-1;
	index_=-1;
}

//**********************************************************************************************
//AtomData
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const AtomData& obj){
	out<<"n_atoms   = "<<obj.nAtoms_<<"\n";
	out<<"atom_type = "<<obj.atomType_;
	return out;
}

//==== member functions ====

void AtomData::clear(){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomData::clear():\n";
	//basic properties
	name_.clear();
	an_.clear();
	type_.clear();
	index_.clear();
	//serial properties
	mass_.clear();
	charge_.clear();
	spin_.clear();
	//vector properties
	posn_.clear();
	vel_.clear();
	force_.clear();
	//nnp
	symm_.clear();
	//neigh
	neigh_.clear();
}

//==== resizing ====

void AtomData::resize(int nAtoms, const AtomType& atomT){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomData::resize(int,const AtomType&):\n";
	//check arguments
	if(nAtoms<=0) throw std::runtime_error("AtomData::resize(int,const AtomType&): invalid number of atoms");
	//set atom info
	atomType_=atomT;
	nAtoms_=nAtoms;
	//basic properties
	if(atomT.name)	name_.resize(nAtoms);
	if(atomT.an)	an_.resize(nAtoms,0);
	if(atomT.type)	type_.resize(nAtoms,-1);
	if(atomT.index)	index_.resize(nAtoms,-1);
	//serial properties
	if(atomT.mass)	mass_.resize(nAtoms,0.0);
	if(atomT.charge)charge_.resize(nAtoms,0.0);
	if(atomT.spin)	spin_.resize(nAtoms,0.0);
	//vector properties
	if(atomT.posn)	posn_.resize(nAtoms,Eigen::Vector3d::Zero());
	if(atomT.vel)	vel_.resize(nAtoms,Eigen::Vector3d::Zero());
	if(atomT.force)	force_.resize(nAtoms,Eigen::Vector3d::Zero());
	//nnp
	if(atomT.symm)	symm_.resize(nAtoms);
	//neigh
	if(atomT.neigh)	neigh_.resize(nAtoms);
}

//==== static functions ====

AtomData& AtomData::make_super(const Eigen::Vector3i& s, const AtomData& ad1, AtomData& ad2){
	if(&ad1==&ad2) throw std::runtime_error("AtomData::make_super(const Eigen::Vector3i&,const AtomData&,AtomData&): identical references.\n");
	if(s[0]<=0 || s[1]<=0 || s[2]<=0) throw std::runtime_error("AtomData::make_super(const Eigen::Vector3i&,const AtomData&,AtomData&): invalid supercell vector.\n");
	const int nAtoms2=ad1.nAtoms()*s[0]*s[1]*s[2];
	const AtomType& atomT=ad1.atomType();
	ad2.resize(nAtoms2,atomT);
	int count=0;
	for(int n=0; n<ad1.nAtoms(); ++n){
		for(int i=0; i<s[0]; ++i){
			for(int j=0; j<s[1]; ++j){
				for(int k=0; k<s[2]; ++k){
					//basic properties
					if(atomT.name)	ad2.name(count)=ad1.name(n);
					if(atomT.an)	ad2.an(count)=ad1.an(n);
					if(atomT.type)	ad2.type(count)=ad1.type(n);
					if(atomT.index)	ad2.index(count)=ad1.index(n);
					//serial properties
					if(atomT.mass)	ad2.mass(count)=ad1.mass(n);
					if(atomT.charge)ad2.charge(count)=ad1.charge(n);
					if(atomT.spin)	ad2.spin(count)=ad1.spin(n);
					//vector properties
					if(atomT.posn)	ad2.posn(count)=ad1.posn(n);
					if(atomT.vel)	ad2.vel(count)=ad1.vel(n);
					if(atomT.force)	ad2.force(count)=ad1.force(n);
					//nnp
					if(atomT.symm)	ad2.symm(count)=ad1.symm(n);
					//neigh
					if(atomT.neigh)	ad2.neigh(count)=ad1.neigh(n);
					//increment
					++count;
				}
			}
		}
	}
	return ad2;
}

//**********************************************************************************************
//Structure
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Structure& struc){
	out<<static_cast<const AtomData&>(struc)<<"\n";
	out<<static_cast<const Cell&>(struc)<<"\n";
	out<<static_cast<const Thermo&>(struc)<<"\n";
	return out;
}

//==== member functions ====

void Structure::clear(){
	if(STRUC_PRINT_FUNC>0) std::cout<<"Structure::clear():\n";
	AtomData::clear();
	Cell::clear();
	Thermo::clear();
}

//==== static functions ====

void Structure::write_binary(const Structure& struc, const char* file){
	if(STRUC_PRINT_FUNC>0) std::cout<<"Structure::write_binary(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* writer=NULL;
	bool error=false;
	int nWrite=-1;
	try{
		//open file
		writer=fopen(file,"wb");
		if(writer==NULL) throw std::runtime_error(std::string("write_binary(Structure&,const char*): Could not open file: ")+std::string(file));
		//allocate buffer
		const int nBytes=serialize::nbytes(struc);
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("write_binary(Structure&,const char*): Could not allocate memory.");
		//write to buffer
		serialize::pack(struc,arr);
		//write to file
		nWrite=fwrite(&nBytes,sizeof(int),1,writer);
		if(nWrite!=1) throw std::runtime_error("write_binary(Structure&,const char*): Write error.");
		nWrite=fwrite(arr,sizeof(char),nBytes,writer);
		if(nWrite!=nBytes) throw std::runtime_error("write_binary(Structure&,const char*): Write error.");
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(writer); writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in write_binary(Structure& struc,const char*):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(writer!=NULL) fclose(writer);
	if(error) throw std::runtime_error("Failed to write");
}

void Structure::read_binary(Structure& struc, const char* file){
	if(STRUC_PRINT_FUNC>0) std::cout<<"Structure::read_binary(const char*):\n";
	//local variables
	char* arr=NULL;
	FILE* reader=NULL;
	bool error=false;
	int nRead=-1;
	try{
		//open file
		reader=fopen(file,"rb");
		if(reader==NULL) throw std::runtime_error(std::string("read_binary(Structure&,const char*): Could not open file: ")+std::string(file));
		//find size
		int nBytes=0;
		nRead=fread(&nBytes,sizeof(int),1,reader);
		if(nRead!=1) throw std::runtime_error("read_binary(Structure&,const char*): Read error.");
		//allocate buffer
		arr=new char[nBytes];
		if(arr==NULL) throw std::runtime_error("read_binary(Structure&,const char*): Could not allocate memory.");
		//read from file
		nRead=fread(arr,sizeof(char),nBytes,reader);
		if(nRead!=nBytes) throw std::runtime_error("read_binary(Structure&,const char*): Read error.");
		//read from buffer
		serialize::unpack(struc,arr);
		//close the file, free memory
		delete[] arr; arr=NULL;
		fclose(reader); reader=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in read_binary(Structure& struc,const char*):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	//free local variables
	if(arr!=NULL) delete[] arr;
	if(reader!=NULL) fclose(reader);
	if(error) throw std::runtime_error("Failed to read");
}

Structure& Structure::make_super(const Eigen::Vector3i& s, const Structure& struc1, Structure& struc2){
	if(&struc1==&struc2) throw std::runtime_error("Structure::make_super(const Eigen::Vector3i&,const Structure&,Structure&): identical references.\n");
	if(s[0]<=0 || s[1]<=0 || s[2]<=0) throw std::runtime_error("Structure::make_super(const Eigen::Vector3i&,const Structure&,Structure&): invalid supercell vector.\n");
	Cell::make_super(s,static_cast<const Cell&>(struc1),static_cast<Cell&>(struc2));
	Thermo::make_super(s,static_cast<const Thermo&>(struc1),static_cast<Thermo&>(struc2));
	AtomData::make_super(s,static_cast<const AtomData&>(struc1),static_cast<AtomData&>(struc2));
	const AtomType& atomT=struc1.atomType();
	int count=0;
	const Eigen::Matrix3d& R=struc1.R();
	for(int n=0; n<struc1.nAtoms(); ++n){
		for(int i=0; i<s[0]; ++i){
			for(int j=0; j<s[1]; ++j){
				for(int k=0; k<s[2]; ++k){
					if(atomT.posn) struc2.posn(count++).noalias()+=i*R.col(0)+j*R.col(1)+k*R.col(2);
				}
			}
		}
	}
	return struc2;
}

void Structure::neigh_list(Structure& struc, double rc){
	//local variables
	Eigen::Vector3d tmp;
	const double rc2=rc*rc;
	//lattice vector shifts
	const int shellx=floor(2.0*rc/struc.R().row(0).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the x-dir.
	const int shelly=floor(2.0*rc/struc.R().row(1).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the y-dir.
	const int shellz=floor(2.0*rc/struc.R().row(2).lpNorm<Eigen::Infinity>());//number of repeated unit cells needed in the z-dir.
	const int Rsize=(2*shellx+1)*(2*shelly+1)*(2*shellz+1);
	if(STRUC_PRINT_DATA>0) std::cout<<"Rsize = "<<Rsize<<"\n";
	if(STRUC_PRINT_DATA>0) std::cout<<"shell = ("<<shellx<<","<<shelly<<","<<shellz<<") = "<<(2*shellx+1)*(2*shelly+1)*(2*shellz+1)<<"\n";
	std::vector<Eigen::Vector3d> R_(Rsize);
	int count=0;
	for(int ix=-shellx; ix<=shellx; ++ix){
		for(int iy=-shelly; iy<=shelly; ++iy){
			for(int iz=-shellz; iz<=shellz; ++iz){
				R_[count++].noalias()=ix*struc.R().col(0)+iy*struc.R().col(1)+iz*struc.R().col(2);
			}
		}
	}
	//loop over all atoms
	if(STRUC_PRINT_DATA>0) std::cout<<"computing neighbor list\n";
	for(int i=0; i<struc.nAtoms(); ++i){
		//clear the neighbor list
		struc.neigh(i).clear();
		//loop over all atoms
		for(int j=0; j<struc.nAtoms(); ++j){
			const Eigen::Vector3d rIJ_=struc.diff(struc.posn(i),struc.posn(j),tmp);
			//loop over lattice vector shifts - atom j
			for(int n=0; n<Rsize; ++n){
				//alter the rIJ_ distance by a lattice vector shift
				const Eigen::Vector3d rIJt_=rIJ_-R_[n];
				const double dIJ2=rIJt_.squaredNorm();
				if(math::constant::ZERO<dIJ2 && dIJ2<=rc2){
					struc.neigh(i).push_back(Neighbor());
					struc.neigh(i).back().r()=rIJt_;
					struc.neigh(i).back().dr()=std::sqrt(dIJ2);
					struc.neigh(i).back().type()=struc.type(j);
					const Eigen::Vector3d rIJf_=struc.RInv()*rIJt_;
					if(
						-0.5<=rIJf_[0] && rIJf_[0]<=0.5 &&
						-0.5<=rIJf_[1] && rIJf_[1]<=0.5 &&
						-0.5<=rIJf_[2] && rIJf_[2]<=0.5
					) struc.neigh(i).back().index()=j;
					//if(struc.neigh(i).back().index()>=0) std::cout<<"neigh index "<<struc.neigh(i).back().index()<<"\n";
				}
			}
		}
	}
}

//**********************************************************************************************
//AtomSpecies
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const AtomSpecies& obj){
	const int ns=obj.species_.size();
	for(int i=0; i<ns; ++i) out<<obj.species_[i]<<" "<<obj.nAtoms_[i]<<" ";
	return out;
}

//==== member functions ====

int AtomSpecies::nTot()const{
	int ntot=0;
	for(int i=0; i<nAtoms_.size(); ++i) ntot+=nAtoms_[i];
	return ntot;
}

void AtomSpecies::defaults(){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomSpecies::defaults():\n";
	species_.clear();
	nAtoms_.clear();
	offsets_.clear();
}

void AtomSpecies::resize(const std::vector<std::string>& names, const std::vector<int>& nAtoms){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomSpecies::resize(const std::vector<int>&,const std::vector<std::string>&):\n";
	if(nAtoms.size()!=names.size()) throw std::invalid_argument("Array size mismatch.");
	nSpecies_=names.size();
	species_=names;
	nAtoms_=nAtoms;
	offsets_.resize(nSpecies_,0);
	for(int i=1; i<nSpecies_; ++i) offsets_[i]=offsets_[i-1]+nAtoms_[i-1];
}

void AtomSpecies::resize(const Structure& struc){
	if(!struc.atomType().name) throw std::runtime_error("AtomSpecies::resize(const Structure&): cannot initialize without atom names");
	std::vector<std::string> names;
	std::vector<int> nAtoms;
	for(int i=0; i<struc.nAtoms(); ++i){
		int index=-1;
		for(int j=0; j<names.size(); ++j){
			if(names[j]==struc.name(i)){index=j;break;}
		}
		if(index<0){
			names.push_back(struc.name(i));
			nAtoms.push_back(1);
		} else ++nAtoms[index];
	}
	nSpecies_=names.size();
	species_=names;
	nAtoms_=nAtoms;
	offsets_.resize(nSpecies_,0);
	for(int i=1; i<nSpecies_; ++i) offsets_[i]=offsets_[i-1]+nAtoms_[i-1];
}

//==== static functions ====

int AtomSpecies::index_species(const std::string& str, const std::vector<std::string>& names){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomSpecies::index_species(const std::string&, const std::vector<std::string>&):\n";
	for(int i=0; i<names.size(); ++i) if(str==names[i]) return i;
	return -1;
}

int AtomSpecies::index_species(const char* str, const std::vector<std::string>& names){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomSpecies::index_species(const char*,const std::vector<std::string>&):\n";
	for(int i=0; i<names.size(); ++i) if(std::strcmp(str,names[i].c_str())==0) return i;
	return -1;
}

std::vector<int>& AtomSpecies::read_atoms(const AtomSpecies& as, const char* str, std::vector<int>& ids){
	//local function variables
	int nAtoms=0;
	std::vector<int> atomIndices;
	std::vector<std::string> atomNames;
	std::vector<int> nameIndices;
	
	//load the number of atoms, atom indices, and atom names
	nAtoms=AtomSpecies::read_natoms(str);
	AtomSpecies::read_indices(str,atomIndices);
	AtomSpecies::read_names(str,atomNames);
	nameIndices.resize(atomNames.size());
	for(int n=0; n<nameIndices.size(); ++n){
		nameIndices[n]=as.index_species(atomNames[n]);
		if(nameIndices[n]<0) throw std::invalid_argument("ERROR: Invalid atom name in atom string.");
	}
	
	//set the atom ids
	ids.resize(nAtoms,0);
	for(int n=0; n<nAtoms; ++n) ids[n]=as.index(nameIndices[n],atomIndices[n]);
	
	return ids;
}

int AtomSpecies::read_natoms(const char* str){
	const char* func_name="AtomSpecies::read_natoms(const char*)";
	if(STRUC_PRINT_FUNC>0) std::cout<<func_name<<":\n";
	//local function variables
	char* strtemp=new char[string::M];
	char* substr=new char[string::M];
	char* temp=new char[string::M];
	std::vector<std::string> substrs;
	int nStrs=0,nAtoms=0;
	bool error=false;
	
	try{
		//copy the string
		std::strcpy(strtemp,str);
		//find the number of substrings
		nStrs=string::substrN(strtemp,",");
		substrs.resize(nStrs);
		substrs[0]=std::string(std::strcpy(temp,std::strtok(strtemp,",")));
		for(int i=1; i<nStrs; ++i){
			substrs[i]=std::string(std::strcpy(temp,std::strtok(NULL,",")));
		}
		
		//parse the line by commas
		for(int i=0; i<nStrs; ++i){
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
	
std::vector<int>& AtomSpecies::read_indices(const char* str, std::vector<int>& indices){
	const char* func_name="AtomSpecies::read_indices(const char*,std::vector<int>&)";
	if(STRUC_PRINT_FUNC>0) std::cout<<func_name<<":\n";
	//local function variables
	char* strtemp=new char[string::M];
	char* substr=new char[string::M];
	char* temp=new char[string::M];
	std::vector<std::string> substrs;
	int nStrs=0;
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
		for(int i=1; i<nStrs; ++i){
			substrs[i]=std::string(std::strcpy(temp,std::strtok(NULL,",")));
		}
		
		//parse the line by commas
		for(int i=0; i<nStrs; ++i){
			std::strcpy(substr,substrs[i].c_str());
			//find out if this substring has only one atom, or a set of atoms
			if(std::strpbrk(substr,":")==NULL){
				//single atom
				if(STRUC_PRINT_STATUS>0) std::cout<<"Single Atom\n";
				if(std::strpbrk(substr,string::DIGITS)==NULL) throw std::invalid_argument("Invalid atom specification: no index.");
				else indices.push_back(std::atoi(std::strpbrk(substr,string::DIGITS))-1);
			} else {
				//set of atoms
				if(STRUC_PRINT_STATUS>0) std::cout<<"Set of AtomData\n";
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

std::vector<std::string>& AtomSpecies::read_names(const char* str, std::vector<std::string>& names){
	const char* func_name="AtomSpecies::read_names(const char*,std::vector<std::string>&)";
	if(STRUC_PRINT_FUNC>0) std::cout<<func_name<<":\n";
	//local function variables
	char* strtemp=new char[string::M];
	char* substr=new char[string::M];
	char* temp=new char[string::M];
	char* atomName=new char[string::M];
	std::vector<std::string> substrs;
	int nStrs=0;
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
		for(int i=1; i<nStrs; ++i){
			substrs[i]=std::string(std::strcpy(temp,std::strtok(NULL,",")));
		}
		
		//parse the line by commas
		for(int i=0; i<nStrs; ++i){
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

void AtomSpecies::set_species(const AtomSpecies& as, Structure& struc){
	//find number of atoms, species
	int nAtomsT=0;
	const int nSpecies=as.species().size();
	for(int i=0; i<as.nAtoms().size(); ++i) nAtomsT+=as.nAtoms(i);
	if(nAtomsT!=struc.nAtoms()) throw std::runtime_error("AtomSpecies::set_species(const AtomSpecies&,Structure&): Mismatch in number of atoms.");
	//set species data
	if(struc.atomType().name){
		for(int i=0; i<nSpecies; ++i){
			for(int j=0; j<as.nAtoms(i); ++j){
				struc.name(as.index(i,j))=as.species(i);
			}
		}
	}
	if(struc.atomType().type){
		for(int i=0; i<nSpecies; ++i){
			for(int j=0; j<as.nAtoms(i); ++j){
				struc.type(as.index(i,j))=i;
			}
		}
	}
	if(struc.atomType().index){
		for(int i=0; i<nSpecies; ++i){
			for(int j=0; j<as.nAtoms(i); ++j){
				struc.index(as.index(i,j))=j;
			}
		}
	}
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const AtomType& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const AtomType&)\n";
		int size=0;
		//basic properties
		size+=sizeof(obj.name);
		size+=sizeof(obj.an);
		size+=sizeof(obj.type);
		size+=sizeof(obj.index);
		//serial properties
		size+=sizeof(obj.mass);
		size+=sizeof(obj.charge);
		size+=sizeof(obj.spin);
		//vector properties
		size+=sizeof(obj.posn);
		size+=sizeof(obj.vel);
		size+=sizeof(obj.force);
		//nnp
		size+=sizeof(obj.symm);
		//neigh
		size+=sizeof(obj.neigh);
		return size;
	}
	template <> int nbytes(const Thermo& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const Thermo&)\n";
		int size=0;
		size+=sizeof(obj.energy());
		size+=sizeof(obj.ewald());
		size+=sizeof(obj.temp());
		size+=sizeof(obj.press());
		return size;
	}
	template <> int nbytes(const Neighbor& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const Neighbor&)\n";
		int size=0;
		size+=sizeof(double)*3;//r_
		size+=sizeof(double);//dr_
		size+=sizeof(int);//type_
		size+=sizeof(int);//index_
		return size;
	}
	template <> int nbytes(const AtomData& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const AtomData&)\n";
		int size=0;
		//atom type
		size+=nbytes(obj.atomType());
		//number of atoms
		size+=sizeof(obj.nAtoms());
		//basic properties
		if(obj.atomType().name)  size+=nbytes(obj.name());
		if(obj.atomType().an)    size+=nbytes(obj.an());
		if(obj.atomType().type)  size+=nbytes(obj.type());
		if(obj.atomType().index) size+=nbytes(obj.index());
		//serial properties
		if(obj.atomType().mass)   size+=nbytes(obj.mass());
		if(obj.atomType().charge) size+=nbytes(obj.charge());
		if(obj.atomType().spin)   size+=nbytes(obj.spin());
		//vector properties
		if(obj.atomType().posn)   size+=nbytes(obj.posn());
		if(obj.atomType().vel)    size+=nbytes(obj.vel());
		if(obj.atomType().force)  size+=nbytes(obj.force());
		//nnp
		if(obj.atomType().symm) size+=nbytes(obj.symm());
		//neigh
		if(obj.atomType().neigh){
			for(int i=0; i<obj.nAtoms(); ++i){
				size+=sizeof(int);
				for(int j=0; j<obj.neigh(i).size(); ++j){
					size+=nbytes(obj.neigh(i)[j]);
				}
			}
		}
		//return
		return size;
	}
	template <> int nbytes(const Structure& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const Structure&)\n";
		int size=0;
		size+=nbytes(static_cast<const Cell&>(obj));
		size+=nbytes(static_cast<const Thermo&>(obj));
		size+=nbytes(static_cast<const AtomData&>(obj));
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const AtomType& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const AtomType&,char*):\n";
		int pos=0;
		//basic properties
		std::memcpy(arr+pos,&obj.name,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.an,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.type,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.index,sizeof(bool)); pos+=sizeof(bool);
		//serial properties
		std::memcpy(arr+pos,&obj.mass,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.charge,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.spin,sizeof(bool)); pos+=sizeof(bool);
		//vector properties
		std::memcpy(arr+pos,&obj.posn,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.vel,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.force,sizeof(bool)); pos+=sizeof(bool);
		//nnp
		std::memcpy(arr+pos,&obj.symm,sizeof(bool)); pos+=sizeof(bool);
		//neigh
		std::memcpy(arr+pos,&obj.neigh,sizeof(bool)); pos+=sizeof(bool);
		return pos;
	}
	template <> int pack(const Thermo& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const Thermo&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.energy(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.ewald(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.temp(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.press(),sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	template <> int pack(const Neighbor& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const Neighbor&,char*):\n";
		int pos=0;
		pos+=pack(obj.r(),arr+pos);
		std::memcpy(arr+pos,&obj.dr(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.type(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.index(),sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	template <> int pack(const AtomData& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const AtomData&,char*):\n";
		int pos=0;
		//atom type
		pos+=pack(obj.atomType(),arr+pos);
		//natoms
		std::memcpy(arr+pos,&obj.nAtoms(),sizeof(int)); pos+=sizeof(int);
		//basic properties
		if(obj.atomType().name)  pos+=pack(obj.name(),arr+pos);
		if(obj.atomType().an)    pos+=pack(obj.an(),arr+pos);
		if(obj.atomType().type)  pos+=pack(obj.type(),arr+pos);
		if(obj.atomType().index) pos+=pack(obj.index(),arr+pos);
		//serial properties
		if(obj.atomType().mass)   pos+=pack(obj.mass(),arr+pos);
		if(obj.atomType().charge) pos+=pack(obj.charge(),arr+pos);
		if(obj.atomType().spin)   pos+=pack(obj.spin(),arr+pos);
		//vector properties
		if(obj.atomType().posn)   pos+=pack(obj.posn(),arr+pos);
		if(obj.atomType().vel)    pos+=pack(obj.vel(),arr+pos);
		if(obj.atomType().force)  pos+=pack(obj.force(),arr+pos);
		//nnp
		if(obj.atomType().symm) pos+=pack(obj.symm(),arr+pos);
		//neigh
		if(obj.atomType().neigh){
			for(int i=0; i<obj.nAtoms(); ++i){
				const int s=obj.neigh(i).size();
				std::memcpy(arr+pos,&s,sizeof(int)); pos+=sizeof(int);
				for(int j=0; j<s; ++j){
					pos+=pack(obj.neigh(i)[j],arr+pos);
				}
			}
		}
		//return
		return pos;
	}
	template <> int pack(const Structure& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const Structure&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const Cell&>(obj),arr+pos);
		pos+=pack(static_cast<const Thermo&>(obj),arr+pos);
		pos+=pack(static_cast<const AtomData&>(obj),arr+pos);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(AtomType& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(AtomType&,const char*):\n";
		int pos=0;
		//basic properties
		std::memcpy(&obj.name,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.an,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.type,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.index,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//serial properties
		std::memcpy(&obj.mass,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.charge,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.spin,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//vector properties
		std::memcpy(&obj.posn,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.vel,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.force,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//nnp
		std::memcpy(&obj.symm,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//neigh
		std::memcpy(&obj.neigh,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		return pos;
	}
	template <> int unpack(Thermo& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(Thermo&,const char*):\n";
		int pos=0;
		std::memcpy(&obj.energy(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.ewald(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.temp(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.press(),arr+pos,sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	template <> int unpack(Neighbor& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(Neighbor&,const char*):\n";
		int pos=0;
		pos+=unpack(obj.r(),arr+pos);
		std::memcpy(&obj.dr(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.type(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.index(),arr+pos,sizeof(int)); pos+=sizeof(int);
		return pos;
	}
	template <> int unpack(AtomData& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(AtomData&,const char*):\n";
		int pos=0;
		//atom type
		AtomType atomT;
		pos+=unpack(atomT,arr+pos);
		//natoms
		int nAtoms=0;
		std::memcpy(&nAtoms,arr+pos,sizeof(int)); pos+=sizeof(int);
		//resize
		obj.resize(nAtoms,atomT);
		//basic properties
		if(obj.atomType().name)  pos+=unpack(obj.name(),arr+pos);
		if(obj.atomType().an)    pos+=unpack(obj.an(),arr+pos);
		if(obj.atomType().type)  pos+=unpack(obj.type(),arr+pos);
		if(obj.atomType().index) pos+=unpack(obj.index(),arr+pos);
		//serial properties
		if(obj.atomType().mass)   pos+=unpack(obj.mass(),arr+pos);
		if(obj.atomType().charge) pos+=unpack(obj.charge(),arr+pos);
		if(obj.atomType().spin)   pos+=unpack(obj.spin(),arr+pos);
		//vector properties
		if(obj.atomType().posn)   pos+=unpack(obj.posn(),arr+pos);
		if(obj.atomType().vel)    pos+=unpack(obj.vel(),arr+pos);
		if(obj.atomType().force)  pos+=unpack(obj.force(),arr+pos);
		//nnp
		if(obj.atomType().symm) pos+=unpack(obj.symm(),arr+pos);
		//neigh
		if(obj.atomType().neigh){
			for(int i=0; i<obj.nAtoms(); ++i){
				int s=-1;
				std::memcpy(&s,arr+pos,sizeof(int)); pos+=sizeof(int);
				if(s<0) throw std::runtime_error("Invalid number of neighbors\n");
				obj.neigh(i).clear();
				if(s>0){
					obj.neigh(i).resize(s);
					for(int j=0; j<s; ++j){
						pos+=unpack(obj.neigh(i)[j],arr+pos);
					}
				}
			}
		}
		//return
		return pos;
	}
	template <> int unpack(Structure& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(Structure&,const char*):\n";
		int pos=0;
		pos+=unpack(static_cast<Cell&>(obj),arr+pos);
		pos+=unpack(static_cast<Thermo&>(obj),arr+pos);
		pos+=unpack(static_cast<AtomData&>(obj),arr+pos);
		return pos;
	}
	
}
