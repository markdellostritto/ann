#include "structure.hpp"

//**********************************************************************************************
//FILE_FORMAT struct
//**********************************************************************************************

FILE_FORMAT::type FILE_FORMAT::load(const std::string& str){
	if(str=="XDATCAR") return FILE_FORMAT::XDATCAR;
	else if(str=="POSCAR") return FILE_FORMAT::POSCAR;
	else if(str=="OUTCAR") return FILE_FORMAT::OUTCAR;
	else if(str=="VASP_XML") return FILE_FORMAT::VASP_XML;
	else if(str=="GAUSSIAN") return FILE_FORMAT::GAUSSIAN;
	else if(str=="DFTB") return FILE_FORMAT::DFTB;
	else if(str=="XYZ") return FILE_FORMAT::XYZ;
	else if(str=="CAR") return FILE_FORMAT::CAR;
	else if(str=="LAMMPS") return FILE_FORMAT::LAMMPS;
	else if(str=="GROMACS") return FILE_FORMAT::GROMACS;
	else if(str=="QE") return FILE_FORMAT::QE;
	else if(str=="PROPHET") return FILE_FORMAT::PROPHET;
	else if(str=="XSF") return FILE_FORMAT::XSF;
	else return FILE_FORMAT::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, FILE_FORMAT::type& format){
	if(format==FILE_FORMAT::UNKNOWN) out<<"UNKNOWN";
	else if(format==FILE_FORMAT::XDATCAR) out<<"XDATCAR";
	else if(format==FILE_FORMAT::POSCAR) out<<"POSCAR";
	else if(format==FILE_FORMAT::OUTCAR) out<<"OUTCAR";
	else if(format==FILE_FORMAT::VASP_XML) out<<"VASP_XML";
	else if(format==FILE_FORMAT::GAUSSIAN) out<<"GAUSSIAN";
	else if(format==FILE_FORMAT::DFTB) out<<"DFTB";
	else if(format==FILE_FORMAT::XYZ) out<<"XYZ";
	else if(format==FILE_FORMAT::CAR) out<<"CAR";
	else if(format==FILE_FORMAT::LAMMPS) out<<"LAMMPS";
	else if(format==FILE_FORMAT::GROMACS) out<<"GROMACS";
	else if(format==FILE_FORMAT::QE) out<<"QE";
	else if(format==FILE_FORMAT::PROPHET) out<<"PROPHET";
	else if(format==FILE_FORMAT::XSF) out<<"XSF";
	return out;
}

//**********************************************************************************************
//Sim Interface
//**********************************************************************************************

//constructors/destructors

StructureI::StructureI(const StructureI& struc){
	//cells
	cell_=struc.cell();
	cellFixed_=struc.cellFixed();
	//simulation info
	system_=struc.system();
	timestep_=struc.timestep();
	periodic_=struc.periodic();
	energy_=struc.energy();
	//atoms
	nSpecies_=struc.nSpecies();
	nAtomsT_=struc.nAtoms();
	nAtoms_.resize(nSpecies_);
	offsets_.resize(nSpecies_);
	for(unsigned int i=0; i<nSpecies_; ++i){
		nAtoms_[i]=struc.nAtoms(i);
		offsets_[i]=struc.offset(i);
	}
	atomNames_=struc.atomNames();
}

//operators

StructureI& StructureI::operator=(const StructureI& struc){
	//cells
	cell_=struc.cell();
	cellFixed_=struc.cellFixed();
	//simulation info
	timestep_=struc.timestep();
	periodic_=struc.periodic();
	energy_=struc.energy();
	//atoms
	nSpecies_=struc.nSpecies();
	nAtomsT_=struc.nAtoms();
	nAtoms_.resize(nSpecies_);
	offsets_.resize(nSpecies_);
	for(unsigned int i=0; i<nSpecies_; ++i){
		nAtoms_[i]=struc.nAtoms(i);
		offsets_[i]=struc.offset(i);
	}
	atomNames_=struc.atomNames();
	return *this;
}

std::ostream& operator<<(std::ostream& out, const StructureI& struc){
	out<<"SYSTEM = "<<struc.system_<<"\n";
	out<<"TIMESTEP = "<<struc.timestep_<<"\n";
	out<<"PERIODIC = "<<struc.periodic_<<"\n";
	out<<"ENERGY = "<<struc.energy_<<"\n";
	out<<"TEMP = "<<struc.temp_<<"\n";
	for(unsigned int i=0; i<struc.atomNames_.size(); ++i) out<<struc.atomNames_[i]<<" "; out<<"\n";
	for(unsigned int i=0; i<struc.nAtoms_.size(); ++i) out<<struc.nAtoms_[i]<<" ";
	out<<struc.cell_<<"\n";
	return out;
}

//member functions

void StructureI::defaults(){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureI::defaults():\n";
	//cell
	cellFixed_=true;
	cell_.clear();
	//simulation data
	system_=std::string("Simulation");
	timestep_=0.5;
	periodic_=true;
	energy_=0;
	temp_=0;
	//atoms
	nSpecies_=0;
	nAtomsT_=0;
	nAtoms_.clear();
	offsets_.clear();
	atomNames_.clear();
}

void StructureI::resize(const std::vector<unsigned int>& nAtoms, const std::vector<std::string>& atomNames){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureI::resize(const std::vector<unsigned int>&,const std::vector<std::string>&):\n";
	if(nAtoms.size()!=atomNames.size()) throw std::invalid_argument("Array size mismatch.");
	nSpecies_=nAtoms.size();
	nAtoms_=nAtoms;
	atomNames_=atomNames;
	nAtomsT_=0;
	offsets_.resize(nSpecies_,0);
	for(unsigned int i=0; i<nSpecies_; ++i) nAtomsT_+=nAtoms_[i];
	for(unsigned int i=1; i<nSpecies_; ++i) offsets_[i]=offsets_[i-1]+nAtoms_[i-1];
}

//static functions

int StructureI::speciesIndex(const std::string& str, const std::vector<std::string>& atomNames){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureI::speciesIndex(const std::string&, const std::vector<std::string>&):\n";
	for(unsigned int i=0; i<atomNames.size(); ++i) if(str==atomNames[i]) return i;
	return -1;
}

int StructureI::speciesIndex(const char* str, const std::vector<std::string>& atomNames){
	if(DEBUG_STRUCTURE>0) std::cout<<"StructureI::speciesIndex(const char*,const std::vector<std::string>&):\n";
	for(unsigned int i=0; i<atomNames.size(); ++i) if(std::strcmp(str,atomNames[i].c_str())==0) return i;
	return -1;
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	//ListData
	
	template <> unsigned int nbytes(const ListData<Name>& obj){unsigned int size=0;for(unsigned int i=0; i<obj.N(); ++i)size+=nbytes(obj[i]);return size;};
	template <> unsigned int nbytes(const ListData<AN>& obj){return obj.N()?0:obj.N()*nbytes(obj[0]);};
	template <> unsigned int nbytes(const ListData<Species>& obj){return obj.N()?0:obj.N()*nbytes(obj[0]);};
	template <> unsigned int nbytes(const ListData<Index>& obj){return obj.N()?0:obj.N()*nbytes(obj[0]);};
	template <> unsigned int nbytes(const ListData<Mass>& obj){return obj.N()?0:obj.N()*nbytes(obj[0]);};
	template <> unsigned int nbytes(const ListData<Charge>& obj){return obj.N()?0:obj.N()*nbytes(obj[0]);};
	template <> unsigned int nbytes(const ListData<Position>& obj){return obj.N()?0:obj.N()*nbytes(obj[0]);};
	template <> unsigned int nbytes(const ListData<Velocity>& obj){return obj.N()?0:obj.N()*nbytes(obj[0]);};
	template <> unsigned int nbytes(const ListData<Force>& obj){return obj.N()?0:obj.N()*nbytes(obj[0]);};
	template <> unsigned int nbytes(const ListData<Symm>& obj){return obj.N()?0:obj.N()*nbytes(obj[0]);};
	
	//StructureI
	
	template <> unsigned int nbytes(const StructureI& obj){
		unsigned int size=0;
		size+=2*sizeof(bool);
		size+=3*sizeof(double);
		size+=2*sizeof(unsigned int);
		
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	//ListData
	
	template <> void pack(const ListData<Name>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	template <> void pack(const ListData<AN>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	template <> void pack(const ListData<Species>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	template <> void pack(const ListData<Index>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	template <> void pack(const ListData<Mass>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	template <> void pack(const ListData<Charge>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	template <> void pack(const ListData<Position>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	template <> void pack(const ListData<Velocity>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	template <> void pack(const ListData<Force>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	template <> void pack(const ListData<Symm>& obj, char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)pack(obj[i],arr+i*n);};
	
	//**********************************************
	// unpacking
	//**********************************************
	
	//ListData
	
	template <> void unpack(ListData<Name>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	template <> void unpack(ListData<AN>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	template <> void unpack(ListData<Species>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	template <> void unpack(ListData<Index>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	template <> void unpack(ListData<Mass>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	template <> void unpack(ListData<Charge>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	template <> void unpack(ListData<Position>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	template <> void unpack(ListData<Velocity>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	template <> void unpack(ListData<Force>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	template <> void unpack(ListData<Symm>& obj, const char* arr){unsigned int n=obj.N()?0:nbytes(obj[0]);for(unsigned int i=0; i<obj.N(); ++i)unpack(obj[i],arr+i*n);};
	
}