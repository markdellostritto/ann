#ifndef MOLDATA_HPP
#define MOLDATA_HPP

// c++ 
#include <iostream>
#include <vector>
#include <string>
// eigen
#include <Eigen/Dense>

class Atom{
public:
	int index;
	int type;
	int mol;
	double charge;
	Eigen::Vector3d posn;
	Eigen::Vector3d vel;
public:
	Atom(){clear();}
	~Atom(){}
	
	void clear();
};

class MolData{
public:
	//count
	int nAtoms;
	int nBonds;
	int nAngles;
	int nDihedrals;
	int nImpropers;
	//types
	int nTypes;
	int nBondTypes;
	int nAngleTypes;
	int nDihedralTypes;
	int nImproperTypes;
	std::vector<double> mass;
	std::vector<std::vector<double> > pair;
	//coeffs
	std::vector<std::vector<double> > cbond;
	std::vector<std::vector<double> > cangle;
	std::vector<std::vector<double> > cdihedral;
	std::vector<std::vector<double> > cimproper;
	//links
	std::vector<std::array<int,1+2> > bonds;
	std::vector<std::array<int,1+3> > angles;
	std::vector<std::array<int,1+4> > dihedrals;
	std::vector<std::array<int,1+4> > impropers;
	//atoms
	std::vector<Atom> atoms;
	//cell
	std::array<double,2> xlim;
	std::array<double,2> ylim;
	std::array<double,2> zlim;
public:
	//==== constructors/destructors ====
	MolData(){clear();}
	~MolData(){}
	
	//==== operators ====
	MolData& operator+=(const MolData& moldata);
	
	//==== member functions ====
	void clear();
	
	//==== static functions ====
	static MolData& read(const char* file, MolData& moldata);
	static const MolData& write(const char* file, const MolData& moldata);
	
};




#endif 