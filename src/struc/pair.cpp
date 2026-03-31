//c++
#include <iostream>
#include <stdexcept>
// str
#include "str/string.hpp"
#include "str/token.hpp"
// struc
#include "struc/structure.hpp"
#include "struc/pair.hpp"

//=== operators ====

std::ostream& operator<<(std::ostream& out, const Pair& pair){
	return out<<"pair rc "<<pair.rcut_<<" stride "<<pair.stride_;
}

//==== member functions ====

void Pair::clear(){
	neigh_.clear();
}

void Pair::read(const Token& token){
	Token token_=token;
	int stride=0;
	double rcut=0;
	//pair rc 6.0 stride 10
	while(!token_.end()){
		const std::string tag=string::to_upper(token_.next());
		if(tag=="STRIDE"){
			stride=std::atoi(token_.next().c_str());
		} else if(tag=="RC"){
			rcut=std::atof(token_.next().c_str());
		} 
	}
	if(stride<=0) throw std::invalid_argument("Pair::read(const Token&): invalid stride.");
	if(rcut<=0) throw std::invalid_argument("Pair::read(const Token&): invalid rcut.");
	stride_=stride;
	rcut_=rcut;
	rcut2_=rcut*rcut;
}

void Pair::build(const Structure& struc, double rcut){
	if(PAIR_PRINT_FUNC>0) std::cout<<"Pair::build(const Structure&,double):\n";
	Eigen::Vector3d r;
	const int natoms=struc.nAtoms();
	rcut_=rcut;
	rcut2_=rcut_*rcut_;
	neigh_.resize(natoms);
	for(int i=0; i<natoms; ++i){
		neigh_[i].clear();
		for(int j=0; j<natoms; ++j){
			const double dr2=struc.dist2(struc.posn(i),struc.posn(j),r);
			if(1e-6<dr2 && dr2<rcut2_){
				neigh_[i].push_back(j);
			}
		}
	}
}

/*void Pair::build_cl(const Structure& struc, double rcut){
	//find the number of cells
	Eigen::Vector3i ncells;
	ncells[0]=floor(struc.R().col(0).norm()/rcut);
	ncells[1]=floor(struc.R().col(1).norm()/rcut);
	ncells[2]=floor(struc.R().col(2).norm()/rcut);
	//compute the small lattice vector matrix
	Eigen::Matrix3d Rs;
	Rs.col(0)=R.col(0)/ncells[0];
	Rs.col(1)=R.col(1)/ncells[1];
	Rs.col(2)=R.col(2)/ncells[2];
	Eigen::Matrix3d Rsi=Rs.inverse();
	//build the cell list
	Tensor<3,std::vector<int> > cell_list;
	for(int n=0; n<natoms; ++n){
		//compute the fractional position (smaller cell)
		Eigen::Vector3d rfrac=Rsi*struc.posn(n);
		//compute the integer index
		Eigen::Vector3i index=Eigen::Vector3i::Constant(-1);
		index[0]=floor(rfrac[0]);
		index[1]=floor(rfrac[1]);
		index[2]=floor(rfrac[2]);
		//add atom index to the appropriate cell list
		cell_list(index).push_back(n);
	}
	//make neighbor cells
	std::vector<Eigen::Vector3i> neighcells(3*3*3-1);
	
	//loop over all cell lists
	for(int i=0; i<ncells[0]; ++i){
		for(int j=0; j<ncells[1]; ++j){
			for(int k=0; k<ncells[2]; ++k){
				//loop over all atoms in the cell
				const int nAtoms=cell_list(i,j,k).size();
				for(int n=0; n<nAtoms; ++n){
					//loop over all atoms
				}
			}
		}
	}
}*/