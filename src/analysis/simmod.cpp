// c++
#include <memory>
// structure
#include "struc/structure.hpp"
//format
#include "format/format.hpp"
#include "format/file_sim.hpp"
// io
#include "str/string.hpp"
#include "str/print.hpp"
#include "str/token.hpp"
// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"
// smod
#include "analysis/simmod.hpp"

//***********************************************************************
// OPERATION
//***********************************************************************

std::ostream& operator<<(std::ostream& out, const Operation& op){
	switch(op){
		case Operation::SORT: out<<"SORT"; break;
		case Operation::ROTATE: out<<"ROTATE"; break;
		case Operation::TRANSLATE: out<<"TRANSLATE"; break;
		case Operation::DILATE: out<<"DILATE"; break;
		case Operation::REPLICATE: out<<"REPLICATE"; break;
		case Operation::BOX: out<<"BOX"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Operation::name(const Operation& op){
	switch(op){
		case Operation::SORT: return "SORT";
		case Operation::ROTATE: return "ROTATE";
		case Operation::TRANSLATE: return "TRANSLATE";
		case Operation::DILATE: return "DILATE";
		case Operation::REPLICATE: return "REPLICATE";
		case Operation::BOX: return "BOX";
		default: return "UNKNOWN";
	}
}

Operation Operation::read(const char* str){
	if(std::strcmp(str,"SORT")==0) return Operation::SORT;
	else if(std::strcmp(str,"ROTATE")==0) return Operation::ROTATE;
	else if(std::strcmp(str,"TRANSLATE")==0) return Operation::TRANSLATE;
	else if(std::strcmp(str,"DILATE")==0) return Operation::DILATE;
	else if(std::strcmp(str,"REPLICATE")==0) return Operation::REPLICATE;
	else if(std::strcmp(str,"BOX")==0) return Operation::BOX;
	else return Operation::UNKNOWN;
}

//***********************************************************************
// SORT
//***********************************************************************

const char* Sort::name(const Sort& t){
	switch(t){
		case Sort::MOLECULE: return "MOLECULE";
		case Sort::TYPE: return "TYPE";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Sort& t){
	switch(t){
		case Sort::MOLECULE: out<<"MOLECULE"; break;
		case Sort::TYPE: out<<"TYPE"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

Sort Sort::read(const char* str){
	if(std::strcmp(str,"MOLECULE")==0) return Sort::MOLECULE;
	else if(std::strcmp(str,"TYPE")==0) return Sort::TYPE;
	else return Sort::UNKNOWN;
}

//***********************************************************************
// Translation
//***********************************************************************

Structure& Translate::mod(Structure& struc)const{
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.posn(i).noalias()+=dr_;
	}
	if(struc.R().squaredNorm()>0.0){
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.modv(struc.posn(i),struc.posn(i));
		}
	}
	return struc;
}

//***********************************************************************
// Rotation
//***********************************************************************

Structure& Rotate::mod(Structure& struc)const{
	Eigen::Matrix3d rmat;
	const double ux=axis_[0];
	const double uy=axis_[1];
	const double uz=axis_[2];
	const double c=cos(angle_);
	const double s=sin(angle_);
	rmat(0,0)=ux*ux*(1.0-c)+c;
	rmat(1,0)=uy*ux*(1.0-c)+uz*s;
	rmat(2,0)=uz*ux*(1.0-c)-uy*s;
	rmat(0,1)=ux*uy*(1.0-c)-uz*s;
	rmat(1,1)=uy*uy*(1.0-c)+c;
	rmat(2,1)=uz*uy*(1.0-c)+ux*s;
	rmat(0,2)=ux*uz*(1.0-c)+uy*s;
	rmat(1,2)=uy*uz*(1.0-c)-ux*s;
	rmat(2,2)=uz*uz*(1.0-c)+c;
	//shift to origin
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.posn(i).noalias()-=origin_;
	}
	//rotate
	for(int i=0; i<struc.nAtoms(); ++i){
		const Eigen::Vector3d r=rmat*struc.posn(i);
		struc.posn(i)=r;
	}
	//shift from origin
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.posn(i).noalias()+=origin_;
	}
	//return to unit cell
	if(struc.R().squaredNorm()>0.0){
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.modv(struc.posn(i),struc.posn(i));
		}
	}
	return struc;
}

//***********************************************************************
// Dilation
//***********************************************************************

Structure& Dilate::mod(Structure& struc)const{
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.posn(i)*=s_;
	}
	const Eigen::Matrix3d R=struc.R()*s_;
	struc.init(R);
	return struc;
}

//***********************************************************************
// Replication
//***********************************************************************

Structure& Replicate::mod(Structure& struc)const{
	if(s_[0]<=0 || s_[1]<=0 || s_[2]<=0) throw std::invalid_argument("Invalid lattice.");
	const int np=s_.prod();
	const int nAtomsT=struc.nAtoms()*np;
	Structure super(nAtomsT,struc.atomType());
	//set the atomic properties
	int c=0;
	const AtomType& atomT=struc.atomType();
	for(int i=0; i<s_[0]; ++i){
		for(int j=0; j<s_[1]; ++j){
			for(int k=0; k<s_[2]; ++k){
				const Eigen::Vector3d R=i*struc.R().col(0)+j*struc.R().col(1)+k*struc.R().col(2);
				for(int n=0; n<struc.nAtoms(); ++n){
					//set map
					Eigen::Vector3i index; index<<i,j,k;
					//basic properties
					if(atomT.name)		super.name(c)=struc.name(n);
					if(atomT.an)		super.an(c)=struc.an(n);
					if(atomT.type)		super.type(c)=struc.type(n);
					if(atomT.index)	super.index(c)=struc.index(n);
					//serial properties
					if(atomT.mass)		super.mass(c)=struc.mass(n);
					if(atomT.charge)	super.charge(c)=struc.charge(n);
					if(atomT.chi)		super.chi(c)=struc.chi(n);
					if(atomT.eta)		super.eta(c)=struc.eta(n);
					if(atomT.c6)		super.c6(c)=struc.c6(n);
					if(atomT.js)		super.js(c)=struc.js(n);
					//vector properties
					if(atomT.posn)		super.posn(c)=struc.posn(n)+R;
					if(atomT.vel) 		super.vel(c)=struc.vel(n);
					if(atomT.force) 	super.force(c)=struc.force(n);
					if(atomT.spin) 	super.spin(c)=struc.spin(n);
					//nnp
					if(atomT.symm) 	super.symm(c)=struc.symm(n);
					//increment
					c++;
				}
			}
		}
	}
	Eigen::MatrixXd Rnew=struc.R();
	Rnew.col(0)*=s_[0];
	Rnew.col(1)*=s_[1];
	Rnew.col(2)*=s_[2];
	static_cast<Cell&>(super).init(Rnew);
	struc=super;
	return struc;
}

//***********************************************************************
// Box
//***********************************************************************

Structure& Box::mod(Structure& struc)const{
	Structure subset;
	std::vector<int> indices;
	for(int i=0; i<struc.nAtoms(); ++i){
		if(
			struc.posn(i)[0]>=c1_[0] && struc.posn(i)[0]<c2_[0] &&
			struc.posn(i)[1]>=c1_[1] && struc.posn(i)[1]<c2_[0] &&
			struc.posn(i)[2]>=c1_[2] && struc.posn(i)[2]<c2_[0]
		) indices.push_back(i);
	}
	if(indices.size()>0){
		const int nsub=indices.size();
		subset.resize(nsub,struc.atomType());
		const AtomType atomT=struc.atomType();
		int c=0;
		for(int i=0; i<nsub; ++i){
			//basic properties
			if(atomT.name)		subset.name(c)=struc.name(indices[i]);
			if(atomT.an)		subset.an(c)=struc.an(indices[i]);
			if(atomT.type)		subset.type(c)=struc.type(indices[i]);
			if(atomT.index)	subset.index(c)=struc.index(indices[i]);
			//serial properties
			if(atomT.mass)		subset.mass(c)=struc.mass(indices[i]);
			if(atomT.charge)	subset.charge(c)=struc.charge(indices[i]);
			if(atomT.chi)		subset.chi(c)=struc.chi(indices[i]);
			if(atomT.eta)		subset.eta(c)=struc.eta(indices[i]);
			if(atomT.c6)		subset.c6(c)=struc.c6(indices[i]);
			if(atomT.js)		subset.js(c)=struc.js(indices[i]);
			//vector properties
			if(atomT.posn)		subset.posn(c)=struc.posn(indices[i]);
			if(atomT.vel) 		subset.vel(c)=struc.vel(indices[i]);
			if(atomT.force) 	subset.force(c)=struc.force(indices[i]);
			if(atomT.spin) 	subset.spin(c)=struc.spin(indices[i]);
			//nnp
			if(atomT.symm) 	subset.symm(c)=struc.symm(indices[i]);
			//increment
			c++;
		}
		Eigen::Matrix3d R=Eigen::Matrix3d::Zero();
		R(0,0)=c2_[0]-c1_[0];
		R(1,1)=c2_[1]-c1_[1];
		R(2,2)=c2_[2]-c1_[2];
		static_cast<Cell&>(subset).init(R);
		struc=subset;
		for(int i=0; i<nsub; ++i){
			//struc.modv(struc.posn(i),struc.posn(i));
		}
	} else struc.clear();
	return struc;
}

//***********************************************************************
// Main
//***********************************************************************

int main(int argc, char* argv[]){
	//======== local function variables ========
	//==== file i/o ====
		FILE* reader=NULL;
		FILE_FORMAT::type iformat;
		FILE_FORMAT::type oformat;
		char* input  =new char[string::M];
		char* strbuf =new char[print::len_buf];
		char* istruc =new char[string::M];
		char* ostruc =new char[string::M];
	//==== simulation variables ====
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.charge=true; atomT.mass=true;
		atomT.posn=true; atomT.force=true; atomT.vel=true;
		Simulation sim;
		Interval interval;
	//==== operations ====
		std::vector<Operation> ops;
		std::vector<std::shared_ptr<Modify> > mods;
	//==== miscellaneous ====
		bool error=false;
	//==== units ====
		units::System unitsys;
		
	try{
		if(argc!=1) throw std::invalid_argument("Invalid number of command-line arguments.");
		
		//======== parameters from stdin ========
		std::cout<<"reading general parameters\n";
		while(fgets(input,string::M,stdin)!=NULL){
			string::trim_right(input,string::COMMENT);
			Token token(input,string::WS);
			if(token.end()) continue;
			const std::string tag=string::to_upper(token.next());
			if(tag=="OP"){
				const std::string opstr=string::to_upper(token.next());
				Operation op=Operation::read(opstr.c_str());
				ops.push_back(op);
				switch(op){
					case Operation::SORT:{
						throw std::invalid_argument("Invalid operation.");
					} break;
					case Operation::TRANSLATE:{
						Translate opf;
						opf.dr()[0]=std::atof(token.next().c_str());
						opf.dr()[1]=std::atof(token.next().c_str());
						opf.dr()[2]=std::atof(token.next().c_str());
						std::shared_ptr<Modify> ptr(new Translate(opf));
						mods.push_back(ptr);
					} break;
					case Operation::ROTATE:{
						Rotate opf;
						opf.origin()[0]=std::atof(token.next().c_str());
						opf.origin()[1]=std::atof(token.next().c_str());
						opf.origin()[2]=std::atof(token.next().c_str());
						opf.axis()[0]=std::atof(token.next().c_str());
						opf.axis()[1]=std::atof(token.next().c_str());
						opf.axis()[2]=std::atof(token.next().c_str());
						opf.angle()=std::atof(token.next().c_str());
						const double norm=opf.axis().norm();
						opf.axis()/=norm;
						opf.angle()*=math::constant::PI/180.0;
						std::shared_ptr<Modify> ptr(new Rotate(opf));
						mods.push_back(ptr);
					} break;
					case Operation::DILATE:{
						Dilate opf;
						opf.s()=std::atof(token.next().c_str());
						std::shared_ptr<Modify> ptr(new Dilate(opf));
						mods.push_back(ptr);
					} break;
					case Operation::REPLICATE:{
						Replicate opf;
						opf.s()[0]=std::atoi(token.next().c_str());
						opf.s()[1]=std::atoi(token.next().c_str());
						opf.s()[2]=std::atoi(token.next().c_str());
						std::shared_ptr<Modify> ptr(new Replicate(opf));
						mods.push_back(ptr);
					} break;
					case Operation::BOX:{
						Box opf;
						opf.c1()[0]=std::atof(token.next().c_str());
						opf.c1()[1]=std::atof(token.next().c_str());
						opf.c1()[2]=std::atof(token.next().c_str());
						opf.c2()[0]=std::atof(token.next().c_str());
						opf.c2()[1]=std::atof(token.next().c_str());
						opf.c2()[2]=std::atof(token.next().c_str());
						std::shared_ptr<Modify> ptr(new Box(opf));
						mods.push_back(ptr);
					} break;
					default:
						throw std::invalid_argument("Invalid operation.");
					break;
				}
			} else if(tag=="IN"){
				std::strcpy(istruc,token.next().c_str());
			} else if(tag=="OUT"){
				std::strcpy(ostruc,token.next().c_str());
			} else if(tag=="FIN"){
				iformat=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
			} else if(tag=="FOUT"){
				oformat=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
			} else if(tag=="UNITS"){
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
			} else if(tag=="INTERVAL"){
				interval=Interval::read(token.next().c_str(),interval);
			} 
		}
		
		//======== initialize the unit system ========
		std::cout<<"initializing the unit system\n";
		units::Consts::init(unitsys);
		
		//======== print the parameters ========
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("GENERAL PARAMETERS",strbuf)<<"\n";
		std::cout<<"UNITS    = "<<unitsys<<"\n";
		std::cout<<"ATOM_T   = "<<atomT<<"\n";
		std::cout<<"ISTRUC   = \""<<istruc<<"\"\n";
		std::cout<<"OSTRUC   = \""<<ostruc<<"\"\n";
		std::cout<<"INTERVAL = "<<interval<<"\n";
		std::cout<<"IFORMAT  = "<<iformat<<"\n";
		std::cout<<"OFORMAT  = "<<oformat<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("OPERATIONS",strbuf)<<"\n";
		for(int i=0; i<ops.size(); ++i){
			std::cout<<ops[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		
		//======== check the parameters ========
		if(unitsys==units::System::UNKNOWN) throw std::invalid_argument("Invalid unit system.");
		if(iformat==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid input file format.");
		if(oformat==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid output file format.");
		if(std::strlen(istruc)==0) throw std::invalid_argument("Empty input structure file.");
		if(std::strlen(ostruc)==0) throw std::invalid_argument("Empty output structure file.");
		
		//read
		std::cout<<"reading simulation\n";
		read_sim(istruc,iformat,interval,atomT,sim);
		
		//operations
		std::cout<<"performing operations\n";
		for(int t=0; t<sim.timesteps(); ++t){
			for(int i=0; i<mods.size(); ++i){
				mods[i]->mod(sim.frame(t));
			}
		}
		
		//write
		std::cout<<"writing simulation\n";
		write_sim(ostruc,oformat,interval,atomT,sim);
		
	}catch(std::exception& e){
		std::cout<<e.what()<<"\n";
		std::cout<<"ANALYSIS FAILED.\n";
		error=true;
	}
	
	std::cout<<"freeing local variables\n";
	delete[] input;
	delete[] strbuf;
	delete[] istruc;
	delete[] ostruc;
	
	std::cout<<"exiting program\n";
	if(error) return 1;
	else return 0;
}