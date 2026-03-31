//c++ libraries
#include <iostream>
//c libraries
#include <stdexcept>
// ann - strings
#include "str/string.hpp"
// ann - chemistry
#include "chem/ptable.hpp"
// ann - eigen
#include "math/eigen.hpp"
#include "math/const.hpp"
// ann - print
#include "str/print.hpp"
// ann - structure
#include "struc/structure.hpp"

//**********************************************************************************************
//AtomData
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const AtomData& obj){
	out<<"natoms = "<<obj.nAtoms_<<"\n";
	out<<"type   = "<<obj.atomType_;
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
	radius_.clear();
	entropy_.clear();
	chi_.clear();
	eta_.clear();
	c6_.clear();
	js_.clear();
	alpha_.clear();
	weight_.clear();
	drudeQ_.clear();
	drudeM_.clear();
	drudeW_.clear();
	drudeN_.clear();
	//vector properties
	image_.clear();
	posn_.clear();
	vel_.clear();
	force_.clear();
	spin_.clear();
	drudeR_.clear();
	//nnp
	symm_.clear();
}

//==== resizing ====

void AtomData::resize(int nAtoms, const AtomType& atomT){
	if(STRUC_PRINT_FUNC>0) std::cout<<"AtomData::resize(int,const AtomType&):\n";
	//check arguments
	if(nAtoms<0) throw std::runtime_error("AtomData::resize(int,const AtomType&): invalid number of atoms");
	//set atom info
	atomType_=atomT;
	nAtoms_=nAtoms;
	if(nAtoms_>0){
		//basic properties
		if(atomT.name)   name_.resize(nAtoms_);
		if(atomT.an)     an_.resize(nAtoms_,0);
		if(atomT.type)   type_.resize(nAtoms_,-1);
		if(atomT.index)  index_.resize(nAtoms_,-1);
		//serial properties
		if(atomT.mass)   mass_.resize(nAtoms_,0.0);
		if(atomT.charge) charge_.resize(nAtoms_,0.0);
		if(atomT.radius) radius_.resize(nAtoms_,0.0);
		if(atomT.entropy)entropy_.resize(nAtoms_,0.0);
		if(atomT.chi)    chi_.resize(nAtoms_,0.0);
		if(atomT.eta)    eta_.resize(nAtoms_,0.0);
		if(atomT.c6)     c6_.resize(nAtoms_,0.0);
		if(atomT.js)     js_.resize(nAtoms_,0.0);
		if(atomT.alpha)  alpha_.resize(nAtoms_,0.0);
		if(atomT.weight) weight_.resize(nAtoms_,0.0);
		if(atomT.drudeQ) drudeQ_.resize(nAtoms_,0.0);
		if(atomT.drudeM) drudeM_.resize(nAtoms_,0.0);
		if(atomT.drudeW) drudeW_.resize(nAtoms_,0.0);
		if(atomT.drudeN) drudeN_.resize(nAtoms_,0.0);
		//vector properties
		if(atomT.image)  image_.resize(nAtoms_,Eigen::Vector3i::Zero());
		if(atomT.posn)   posn_.resize(nAtoms_,Eigen::Vector3d::Zero());
		if(atomT.vel)    vel_.resize(nAtoms_,Eigen::Vector3d::Zero());
		if(atomT.force)  force_.resize(nAtoms_,Eigen::Vector3d::Zero());
		if(atomT.spin)   spin_.resize(nAtoms_,Eigen::Vector3d::Zero());
		if(atomT.drudeR) drudeR_.resize(nAtoms_,Eigen::Vector3d::Zero());
		//nnp
		if(atomT.symm)	symm_.resize(nAtoms_);
		//set index
		if(atomT.index) for(int i=0; i<nAtoms_; ++i) index_[i]=i;
	}
}

//**********************************************************************************************
//Structure
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Structure& struc){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("STRUCTURE",str)<<"\n";
	out<<static_cast<const AtomData&>(struc)<<"\n";
	out<<static_cast<const Cell&>(struc)<<"\n";
	out<<static_cast<const State&>(struc)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

void Structure::clear(){
	if(STRUC_PRINT_FUNC>0) std::cout<<"Structure::clear():\n";
	AtomData::clear();
	Cell::clear();
	State::clear();
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

Structure& Structure::super(const Structure& struc, Structure& superc, const Eigen::Vector3i nlat){
	if(nlat[0]<=0 || nlat[1]<=0 || nlat[2]<=0) throw std::invalid_argument("Invalid lattice.");
	const int np=nlat.prod();
	const int nAtomsT=struc.nAtoms()*np;
	superc.resize(nAtomsT,struc.atomType());
	//set the atomic properties
	int c=0;
	const AtomType& atomT=struc.atomType();
	for(int i=0; i<nlat[0]; ++i){
		for(int j=0; j<nlat[1]; ++j){
			for(int k=0; k<nlat[2]; ++k){
				const Eigen::Vector3d R=i*struc.R().col(0)+j*struc.R().col(1)+k*struc.R().col(2);
				for(int n=0; n<struc.nAtoms(); ++n){
					//set map
					Eigen::Vector3i index; index<<i,j,k;
					//basic properties
					if(atomT.name)		superc.name(c)=struc.name(n);
					if(atomT.an)		superc.an(c)=struc.an(n);
					if(atomT.type)		superc.type(c)=struc.type(n);
					if(atomT.index)	superc.index(c)=struc.index(n);
					//serial properties
					if(atomT.mass)		superc.mass(c)=struc.mass(n);
					if(atomT.charge)	superc.charge(c)=struc.charge(n);
					if(atomT.radius)	superc.radius(c)=struc.radius(n);
					if(atomT.entropy)	superc.entropy(c)=struc.entropy(n);
					if(atomT.chi)		superc.chi(c)=struc.chi(n);
					if(atomT.eta)		superc.eta(c)=struc.eta(n);
					if(atomT.c6)		superc.c6(c)=struc.c6(n);
					if(atomT.js)		superc.js(c)=struc.js(n);
					if(atomT.alpha)	superc.alpha(c)=struc.alpha(n);
					if(atomT.weight)	superc.weight(c)=struc.weight(n);
					if(atomT.drudeQ)	superc.drudeQ(c)=struc.drudeQ(n);
					if(atomT.drudeM)	superc.drudeM(c)=struc.drudeM(n);
					if(atomT.drudeW)	superc.drudeW(c)=struc.drudeW(n);
					if(atomT.drudeN)	superc.drudeN(c)=struc.drudeN(n);
					//vector properties
					if(atomT.image)		superc.image(c)=struc.image(n);
					if(atomT.posn)		superc.posn(c)=struc.posn(n)+R;
					if(atomT.vel) 		superc.vel(c)=struc.vel(n);
					if(atomT.force) 	superc.force(c)=struc.force(n);
					if(atomT.spin) 	superc.spin(c)=struc.spin(n);
					if(atomT.drudeR)	superc.drudeR(c)=struc.drudeR(n);
					//nnp
					if(atomT.symm) 	superc.symm(c)=struc.symm(n);
					//increment
					c++;
				}
			}
		}
	}
	Eigen::MatrixXd Rnew=struc.R();
	Rnew.col(0)*=nlat[0];
	Rnew.col(1)*=nlat[1];
	Rnew.col(2)*=nlat[2];
	static_cast<Cell&>(superc).init(Rnew);
	return superc;
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const AtomData& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const AtomData&)\n";
		int size=0;
		//atom type
		size+=nbytes(obj.atomType());
		//number of atoms
		size+=sizeof(obj.nAtoms());
		//basic properties
		if(obj.atomType().name)   size+=nbytes(obj.name());
		if(obj.atomType().an)     size+=nbytes(obj.an());
		if(obj.atomType().type)   size+=nbytes(obj.type());
		if(obj.atomType().index)  size+=nbytes(obj.index());
		//serial properties
		if(obj.atomType().mass)   size+=nbytes(obj.mass());
		if(obj.atomType().charge) size+=nbytes(obj.charge());
		if(obj.atomType().radius) size+=nbytes(obj.radius());
		if(obj.atomType().entropy)size+=nbytes(obj.entropy());
		if(obj.atomType().chi)    size+=nbytes(obj.chi());
		if(obj.atomType().eta)    size+=nbytes(obj.eta());
		if(obj.atomType().c6)     size+=nbytes(obj.c6());
		if(obj.atomType().js)     size+=nbytes(obj.js());
		if(obj.atomType().alpha)  size+=nbytes(obj.alpha());
		if(obj.atomType().weight) size+=nbytes(obj.weight());
		if(obj.atomType().drudeQ) size+=nbytes(obj.drudeQ());
		if(obj.atomType().drudeM) size+=nbytes(obj.drudeM());
		if(obj.atomType().drudeW) size+=nbytes(obj.drudeW());
		if(obj.atomType().drudeN) size+=nbytes(obj.drudeN());
		//vector properties
		if(obj.atomType().image)  size+=nbytes(obj.image());
		if(obj.atomType().posn)   size+=nbytes(obj.posn());
		if(obj.atomType().vel)    size+=nbytes(obj.vel());
		if(obj.atomType().force)  size+=nbytes(obj.force());
		if(obj.atomType().spin)   size+=nbytes(obj.spin());
		if(obj.atomType().drudeR) size+=nbytes(obj.drudeR());
		//nnp
		if(obj.atomType().symm)   size+=nbytes(obj.symm());
		//return
		return size;
	}
	template <> int nbytes(const Structure& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const Structure&)\n";
		int size=0;
		size+=nbytes(static_cast<const Cell&>(obj));
		size+=nbytes(static_cast<const State&>(obj));
		size+=nbytes(static_cast<const AtomData&>(obj));
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const AtomData& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const AtomData&,char*):\n";
		int pos=0;
		//atom type
		pos+=pack(obj.atomType(),arr+pos);
		//natoms
		std::memcpy(arr+pos,&obj.nAtoms(),sizeof(int)); pos+=sizeof(int);
		//basic properties
		if(obj.atomType().name)   pos+=pack(obj.name(),arr+pos);
		if(obj.atomType().an)     pos+=pack(obj.an(),arr+pos);
		if(obj.atomType().type)   pos+=pack(obj.type(),arr+pos);
		if(obj.atomType().index)  pos+=pack(obj.index(),arr+pos);
		//serial properties
		if(obj.atomType().mass)   pos+=pack(obj.mass(),arr+pos);
		if(obj.atomType().charge) pos+=pack(obj.charge(),arr+pos);
		if(obj.atomType().radius) pos+=pack(obj.radius(),arr+pos);
		if(obj.atomType().entropy)pos+=pack(obj.entropy(),arr+pos);
		if(obj.atomType().chi)    pos+=pack(obj.chi(),arr+pos);
		if(obj.atomType().eta)    pos+=pack(obj.eta(),arr+pos);
		if(obj.atomType().c6)     pos+=pack(obj.c6(),arr+pos);
		if(obj.atomType().js)     pos+=pack(obj.js(),arr+pos);
		if(obj.atomType().alpha)  pos+=pack(obj.alpha(),arr+pos);
		if(obj.atomType().weight) pos+=pack(obj.weight(),arr+pos);
		if(obj.atomType().drudeQ) pos+=pack(obj.drudeQ(),arr+pos);
		if(obj.atomType().drudeM) pos+=pack(obj.drudeM(),arr+pos);
		if(obj.atomType().drudeW) pos+=pack(obj.drudeW(),arr+pos);
		if(obj.atomType().drudeN) pos+=pack(obj.drudeN(),arr+pos);
		//vector properties
		if(obj.atomType().image)  pos+=pack(obj.image(),arr+pos);
		if(obj.atomType().posn)   pos+=pack(obj.posn(),arr+pos);
		if(obj.atomType().vel)    pos+=pack(obj.vel(),arr+pos);
		if(obj.atomType().force)  pos+=pack(obj.force(),arr+pos);
		if(obj.atomType().spin)   pos+=pack(obj.spin(),arr+pos);
		if(obj.atomType().drudeR) pos+=pack(obj.drudeR(),arr+pos);
		//nnp
		if(obj.atomType().symm)   pos+=pack(obj.symm(),arr+pos);
		//return
		return pos;
	}
	template <> int pack(const Structure& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const Structure&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const Cell&>(obj),arr+pos);
		pos+=pack(static_cast<const State&>(obj),arr+pos);
		pos+=pack(static_cast<const AtomData&>(obj),arr+pos);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
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
		if(obj.atomType().name)   pos+=unpack(obj.name(),arr+pos);
		if(obj.atomType().an)     pos+=unpack(obj.an(),arr+pos);
		if(obj.atomType().type)   pos+=unpack(obj.type(),arr+pos);
		if(obj.atomType().index)  pos+=unpack(obj.index(),arr+pos);
		//serial properties
		if(obj.atomType().mass)   pos+=unpack(obj.mass(),arr+pos);
		if(obj.atomType().charge) pos+=unpack(obj.charge(),arr+pos);
		if(obj.atomType().radius) pos+=unpack(obj.radius(),arr+pos);
		if(obj.atomType().entropy)pos+=unpack(obj.entropy(),arr+pos);
		if(obj.atomType().chi)    pos+=unpack(obj.chi(),arr+pos);
		if(obj.atomType().eta)    pos+=unpack(obj.eta(),arr+pos);
		if(obj.atomType().c6)     pos+=unpack(obj.c6(),arr+pos);
		if(obj.atomType().js)     pos+=unpack(obj.js(),arr+pos);
		if(obj.atomType().alpha)  pos+=unpack(obj.alpha(),arr+pos);
		if(obj.atomType().weight) pos+=unpack(obj.weight(),arr+pos);
		if(obj.atomType().drudeQ) pos+=unpack(obj.drudeQ(),arr+pos);
		if(obj.atomType().drudeM) pos+=unpack(obj.drudeM(),arr+pos);
		if(obj.atomType().drudeW) pos+=unpack(obj.drudeW(),arr+pos);
		if(obj.atomType().drudeN) pos+=unpack(obj.drudeN(),arr+pos);
		//vector properties
		if(obj.atomType().image)  pos+=unpack(obj.image(),arr+pos);
		if(obj.atomType().posn)   pos+=unpack(obj.posn(),arr+pos);
		if(obj.atomType().vel)    pos+=unpack(obj.vel(),arr+pos);
		if(obj.atomType().force)  pos+=unpack(obj.force(),arr+pos);
		if(obj.atomType().spin)   pos+=unpack(obj.spin(),arr+pos);
		if(obj.atomType().drudeR) pos+=unpack(obj.drudeR(),arr+pos);
		//nnp
		if(obj.atomType().symm)   pos+=unpack(obj.symm(),arr+pos);
		//return
		return pos;
	}
	template <> int unpack(Structure& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(Structure&,const char*):\n";
		int pos=0;
		pos+=unpack(static_cast<Cell&>(obj),arr+pos);
		pos+=unpack(static_cast<State&>(obj),arr+pos);
		pos+=unpack(static_cast<AtomData&>(obj),arr+pos);
		return pos;
	}
	
}
