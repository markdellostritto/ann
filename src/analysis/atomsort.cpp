// c libraries
#include <cstdlib>
#include <ctime>
//c++
#include <iostream>
#include <stack>
// chem
#include "chem/ptable.hpp"
// str
#include "str/parse.hpp"
// structure
#include "struc/structure.hpp"
// analysis
#include "analysis/atomsort.hpp"
// format
#include "format/file_struc.hpp"
#include "format/vasp_struc.hpp"
#include "format/lammps_struc.hpp"
#include "format/xyz_struc.hpp"

Sort Sort::read(const char* str){
	if(std::strcmp(str,"MOLECULE")==0) return Sort::MOLECULE;
	else return Sort::UNKNOWN;
}

const char* Sort::name(const Sort& t){
	switch(t){
		case Sort::MOLECULE: return "MOLECULE";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, const Sort& t){
	switch(t){
		case Sort::MOLECULE: out<<"MOLECULE"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

struct Atom{
	int mol;
	int index;
	int an;
};

bool compare_index(const Atom& i, const Atom& j){
	if(i.mol==j.mol){
		return i.index<j.index;
	} else {
		return i.mol<j.mol;
	}
}

bool compare_an(const Atom& i, const Atom& j){
	if(i.mol==j.mol){
		return i.an>j.an;
	} else {
		return i.mol<j.mol;
	}
}

Structure& sort_mol(Structure& struc){
	const int nAtoms=struc.nAtoms();
	//compute distances
	Eigen::MatrixXd dMat=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	if(struc.R().norm()>0){
		for(int i=0; i<nAtoms; ++i){
			Eigen::Vector3d dr;
			for(int j=i+1; j<nAtoms; ++j){
				dMat(i,j)=struc.dist(struc.posn(i),struc.posn(j),dr);
				dMat(j,i)=dMat(i,j);
			}
		}
	} else {
		for(int i=0; i<nAtoms; ++i){
			for(int j=i+1; j<nAtoms; ++j){
				dMat(i,j)=(struc.posn(i)-struc.posn(j)).norm();
				dMat(j,i)=dMat(i,j);
			}
		}
	}
	//assign covalent radius
	std::vector<double> rcov(nAtoms,0);
	for(int i=0; i<nAtoms; ++i){
		rcov[i]=ptable::radius_covalent(struc.an(i));
	}
	//assign molecules
	std::stack<int> molstack;
	std::vector<Atom> atoms(struc.nAtoms());
	for(int i=0; i<nAtoms; ++i){
		atoms[i].index=i;
		atoms[i].mol=-1;
		atoms[i].an=struc.an(i);
	}
	int moli=0;
	for(int i=0; i<nAtoms; ++i){
		if(atoms[i].mol<0){
			atoms[i].mol=moli;
			molstack.push(i);
			while(!molstack.empty()){
				const int atom=molstack.top();
				molstack.pop();
				for(int j=0; j<nAtoms; ++j){
					if(atoms[j].mol<0 && dMat(atom,j)<1.10*(rcov[atom]+rcov[j])){
						atoms[j].mol=moli;
						molstack.push(j);
					}
				}
			}
			++moli;
		}
	}
	//sort molecules
	std::sort(atoms.begin(),atoms.end(),compare_an);
	//move data
	Structure snew=struc;
	for(int i=0; i<nAtoms; ++i){
		snew.name(i)=struc.name(atoms[i].index);
		snew.type(i)=struc.type(atoms[i].index);
		snew.an(i)=struc.an(atoms[i].index);
		snew.posn(i)=struc.posn(atoms[i].index);
		snew.force(i)=struc.force(atoms[i].index);
	}
	struc=snew;
	return struc;
}

int main(int argc, char* argv[]){
	//simulation
		std::string fstruc;
		Structure struc;
		FILE_FORMAT::type format;
	//atom type
		AtomType atomT;
		atomT.name=true; atomT.an=true; atomT.type=true; atomT.index=true;
		atomT.posn=true; atomT.force=true;
	//arguments
		parse::Arg arg;
		std::vector<parse::Arg> args;
	//sorting	
		Sort sort;
		
	//parse args
	std::cout<<"parsing args\n";
	parse::read(argc,argv,args);
	if(parse::farg(args,"format",arg) && arg.nvals()==1) format=FILE_FORMAT::read(arg.val(0).c_str());
	else throw std::invalid_argument("Invalid input specification.");
	if(parse::farg(args,"struc",arg) && arg.nvals()==1) fstruc=arg.val(0);
	else throw std::invalid_argument("Invalid input specification.");
	if(parse::farg(args,"sort",arg) && arg.nvals()==1) sort=Sort::read(arg.val(0).c_str());
	else throw std::invalid_argument("Invalid input specification.");
	
	//print
	std::cout<<"format = "<<format<<"\n";
	std::cout<<"struc  = "<<fstruc<<"\n";
	std::cout<<"sort   = "<<sort<<"\n";
	
	//check parameters
	if(format==FILE_FORMAT::UNKNOWN) throw std::invalid_argument("Invalid format.");
	if(sort==Sort::UNKNOWN) throw std::invalid_argument("Invalid sort.");
	
	//read
	std::cout<<"reading structure\n";
	read_struc(fstruc.c_str(),format,atomT,struc);
	
	//sort
	std::cout<<"sorting atoms\n";
	switch(sort){
		case Sort::MOLECULE: sort_mol(struc); break;
		default: throw std::invalid_argument("Invalid sort."); break;
	}
	
	//write 
	std::cout<<"writing structure\n";
	write_struc(fstruc.c_str(),format,atomT,struc);
}