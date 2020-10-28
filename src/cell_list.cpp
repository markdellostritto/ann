// c++ libraries
#include <iostream>
// ann - print
#include "print.hpp"
// ann - math
#include "math_func.hpp"
// ann - structure
#include "structure.hpp"
// ann - cell list
#include "cell_list.hpp"

//==== operators ====

std::ostream& operator<<(std::ostream& out, const CellList& c){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("CELL LIST",str)<<"\n";
	out<<"DIM  = "<<c.dim_[0]<<" "<<c.dim_[1]<<" "<<c.dim_[2]<<"\n";
	out<<"FLEN = "<<c.flen_[0]<<" "<<c.flen_[1]<<" "<<c.flen_[2]<<"\n";
	out<<print::title("CELL LIST",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

const std::vector<int>& CellList::atoms(int i, int j, int k)const{
	return atoms_[
		index(
			math::func::mod(i,dim_[0]),
			math::func::mod(j,dim_[1]),
			math::func::mod(k,dim_[2])
		)
	];
}

const std::vector<int>& CellList::atoms(const Eigen::Vector3i& i)const{
	return atoms_[
		index(
			math::func::mod(i[0],dim_[0]),
			math::func::mod(i[1],dim_[1]),
			math::func::mod(i[2],dim_[2])
		)
	];
}
	
void CellList::defaults(){
	dim_[0]=0; dim_[1]=0; dim_[2]=0;
	flen_[0]=0.0; flen_[1]=0.0; flen_[2]=0.0;
	natoms_=0;
	cell_.clear();
	atoms_.clear();
}

int CellList::index(int i, int j, int k)const{
	return i*dim_[1]*dim_[2]+j*dim_[2]+k;
}

void CellList::compute(double rc, const Structure& struc){
	//local variables
	Eigen::Vector3d p;
	//compute the dimension (ceil: at least 1)
	dim_[0]=std::ceil(struc.R().row(0).lpNorm<Eigen::Infinity>()/rc);//max vec in x-dir / rc
	dim_[1]=std::ceil(struc.R().row(1).lpNorm<Eigen::Infinity>()/rc);//max vec in y-dir / rc
	dim_[2]=std::ceil(struc.R().row(2).lpNorm<Eigen::Infinity>()/rc);//max vec in z-dir / rc
	flen_[0]=1.0/(1.0*dim_[0]);
	flen_[1]=1.0/(1.0*dim_[1]);
	flen_[2]=1.0/(1.0*dim_[2]);
	//resize
	natoms_=struc.nAtoms();
	atoms_.clear();
	atoms_.resize(dim_[0]*dim_[1]*dim_[2]);
	cell_.resize(struc.nAtoms());
	//iterate over all atoms
	for(int n=0; n<struc.nAtoms(); ++n){
		Eigen::Vector3d t=struc.posn(n);
		//convert to frac coordinates
		p.noalias()=struc.RInv()*struc.posn(n);
		//bin the position
		cell_[n][0]=math::func::mod((int)(p[0]/flen_[0]),dim_[0]);
		cell_[n][1]=math::func::mod((int)(p[1]/flen_[1]),dim_[1]);
		cell_[n][2]=math::func::mod((int)(p[2]/flen_[2]),dim_[2]);
		//add to the cell list
		atoms_[index(cell_[n][0],cell_[n][1],cell_[n][2])].push_back(n);
	}
}