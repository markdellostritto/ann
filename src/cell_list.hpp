#pragma once
#ifndef CELL_LIST_HPP
#define CELL_LIST_HPP

//c++ libraries
#include <iosfwd>
// eigen libraries
#include <Eigen/Dense>
// ann - structure
#include "structure_fwd.hpp"

class CellList{
private:
	int natoms_; //number of atoms
	std::vector<Eigen::Vector3i> cell_;//cell of atom - natoms
	int dim_[3];//dimension of the cell list
	double flen_[3];//fractional length of the cell lists
	std::vector<std::vector<int> > atoms_;//the atoms in each cell - dim^3
public:
	//==== constructors/destructors ====
	CellList(){defaults();}
	CellList(double rc, const Structure& struc){compute(rc,struc);}
	~CellList(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const CellList& cellList);
	
	//==== access ====
	int dim(int i)const{return dim_[i];}
	double flen(int i)const{return flen_[i];}
	const Eigen::Vector3i& cell(int i)const{return cell_[i];}
	const std::vector<int>& atoms(int i, int j, int k)const;
	const std::vector<int>& atoms(const Eigen::Vector3i& i)const;
	
	//==== member functions ====
	void defaults();
	void clear(){defaults();}
	int index(int i, int j, int k)const;
	void compute(double rc, const Structure& struc);
};

#endif