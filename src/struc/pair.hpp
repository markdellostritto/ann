#pragma once
#ifndef PAIR_HPP
#define PAIR_HPP

//c++
#include <iosfwd>
//str
#include "str/token.hpp"
//struc
#include "struc/structure_fwd.hpp"

#ifndef PAIR_PRINT_FUNC
#define PAIR_PRINT_FUNC 0
#endif

#ifndef PAIR_PRINT_STATUS
#define PAIR_PRINT_STATUS 0
#endif

#ifndef PAIR_PRINT_DATA
#define PAIR_PRINT_DATA 0
#endif

class Pair{
private:
	int stride_;
	double rcut_;
	double rcut2_;
	std::vector<std::vector<int> > neigh_;
public:
	//==== constructors/destructors ====
	Pair(){}
	~Pair(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Pair& pair);
	
	//==== access ====
	int& stride(){return stride_;}
	const int& stride()const{return stride_;}
	double& rcut(){return rcut_;}
	const double& rcut()const{return rcut_;}
	int size(int i)const{return neigh_[i].size();}
	const std::vector<int>& neigh(int i)const{return neigh_[i];}
	const int& neigh(int i, int j)const{return neigh_[i][j];}
	
	//==== member functions ====
	void clear();
	void read(const Token& token);
	void build(const Structure& struc, double rcut);
	void build(const Structure& struc){build(struc,rcut_);}
};

#endif