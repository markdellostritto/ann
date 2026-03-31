#pragma once
#ifndef VERLET_HPP
#define VERLET_HPP

// c++
#include <ostream>
#include <vector>
// eigen
#include <Eigen/Dense>
// str
#include "str/token.hpp"
#include "str/string.hpp"
// struc
#include "struc/structure_fwd.hpp"

#ifndef VERLET_PRINT_FUNC
#define VERLET_PRINT_FUNC 0
#endif

namespace verlet{

//**************************************************************************
// Neighbor
//**************************************************************************

class Neighbor{
private:
	int index_;
	Eigen::Vector3d cell_;
public:
	//==== constructors/destructors ====
	Neighbor():index_(-1),cell_(Eigen::Vector3d::Zero()){}
	Neighbor(int i):index_(i),cell_(Eigen::Vector3d::Zero()){}
	~Neighbor(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Neighbor& n);
	
	//==== access ====
	int& index(){return index_;}
	const int& index()const{return index_;}
	Eigen::Vector3d& cell(){return cell_;}
	const Eigen::Vector3d& cell()const{return cell_;}
};

class List{
private:
	int stride_;
	double rc_;
	std::vector<std::vector<Neighbor> > neigh_;
public:
	//==== constructors/destructors ====
	List():rc_(0){}
	~List(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const List& list);
	
	//==== access ====
	int& stride(){return stride_;}
	const int& stride()const{return stride_;}
	double& rc(){return rc_;}
	const double& rc()const{return rc_;}
	const int size(int i)const{return neigh_[i].size();}
	std::vector<Neighbor>& neigh(int i){return neigh_[i];}
	const std::vector<Neighbor>& neigh(int i)const{return neigh_[i];}
	Neighbor& neigh(int i, int j){return neigh_[i][j];}
	const Neighbor& neigh(int i, int j)const{return neigh_[i][j];}
	const std::vector<std::vector<Neighbor> >& neigh()const{return neigh_;}
	
	//==== member functions ====
	void clear(){neigh_.clear();}
	void read(const char* str);
	void read(Token& token);
	void resize(int natoms);
	void build(const Structure& struc);
	void build(const Structure& struc, int i);
	void build(const Structure& struc, double rc, int i);
	
	//=== static functions ====
	static bool sortcmp(const Eigen::Vector3d& i, const Eigen::Vector3d& j);
};

}

#endif