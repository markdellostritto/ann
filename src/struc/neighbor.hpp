#pragma once
#ifndef NEIGHBOR_HPP
#define NEIGHBOR_HPP

//c++ libraries
#include <iosfwd>
#include <string>
// struc
#include "struc/structure.hpp"

#ifndef NEIGH_PRINT_FUNC
#define NEIGH_PRINT_FUNC 0
#endif

#ifndef NEIGH_PRINT_STATUS
#define NEIGH_PRINT_STATUS 0
#endif

#ifndef NEIGH_PRINT_DATA
#define NEIGH_PRINT_DATA 0
#endif


//**********************************************************************************************
//Neighbor
//**********************************************************************************************

class Neighbor{
private:
	Eigen::Vector3d r_;//distance vector pointing from neighbor to central atom
	double dr_;//norm of r_;
	int type_;
	int index_;
	bool min_;
public:
	Neighbor(){clear();}
	~Neighbor(){};
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Neighbor& obj);
	
	Eigen::Vector3d& r(){return r_;}
	const Eigen::Vector3d& r()const{return r_;}
	double& dr(){return dr_;}
	const double& dr()const{return dr_;}
	int& type(){return type_;}
	const int& type()const{return type_;}
	int& index(){return index_;}
	const int& index()const{return index_;}
	bool& min(){return min_;}
	const bool& min()const{return min_;}
	
	void clear();
};

//**********************************************************************************************
// NeigborList
//**********************************************************************************************

class NeighborList{
private:
	int period_;
	double rc_;
	std::vector<std::vector<Neighbor> > neigh_;
public:
	//==== constructors/destructors ====
	NeighborList(){defaults();}
	NeighborList(const Structure& struc, double rc){defaults();build(struc,rc);}
	~NeighborList(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NeighborList& obj);
	
	//==== access ====
	int& period(){return period_;}
	const int& period()const{return period_;}
	double& rc(){return rc_;}
	const double& rc()const{return rc_;}
	int size(int i)const{return neigh_[i].size();}
	Neighbor& neigh(int i, int j){return neigh_[i][j];}
	const Neighbor& neigh(int i, int j)const{return neigh_[i][j];}
	
	//==== member functions ====
	void read(const char* str);
	void build(const Structure& struc, double rc);
	void build(const Structure& struc, double rc, int ii);
	void build(const Structure& struc, double rc, const std::vector<int>& subset);
	void build(const Structure& struc){build(struc,rc_);}
	void build(const Structure& struc, int ii){build(struc,rc_,ii);}
	void build(const Structure& struc, const std::vector<int>& subset){build(struc,rc_,subset);}
	void clear();
	void defaults(){clear();}
	
	static std::vector<Eigen::Vector3d>& ilist(const Structure& struc, double rc, std::vector<Eigen::Vector3d>& Rlist);
	
	static bool sortcmp(const Eigen::Vector3d& i, const Eigen::Vector3d& j);
};

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Neighbor& obj);
	template <> int nbytes(const NeighborList& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Neighbor& obj, char* arr);
	template <> int pack(const NeighborList& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Neighbor& obj, const char* arr);
	template <> int unpack(NeighborList& obj, const char* arr);
	
}

#endif