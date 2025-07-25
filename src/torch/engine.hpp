#pragma once
#ifndef ENGINE_HPP
#define ENGINE_HPP

// struc
#include "struc/verlet.hpp"
// torch
#include "torch/pot.hpp"

#ifndef ENGINE_PRINT_FUNC
#define ENGINE_PRINT_FUNC 0
#endif

//****************************************************************************
// Engine
//****************************************************************************

class Engine{
private:
	//types
	int ntypes_;
	//potentials
	double rcmax_;//max cutoff
	verlet::List vlist_;//verlet list
	std::vector<std::shared_ptr<ptnl::Pot> > pots_;//potentials
public:
	//==== constructors/destructors ====
	Engine():ntypes_(-1){}
	~Engine(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Engine& engine);
	
	//==== access ====
	//types
	const int ntypes()const{return ntypes_;}
	verlet::List& vlist(){return vlist_;}
	const verlet::List& vlist()const{return vlist_;}
	std::vector<std::shared_ptr<ptnl::Pot> >& pots(){return pots_;}
	const std::vector<std::shared_ptr<ptnl::Pot> >& pots()const{return pots_;}
	std::shared_ptr<ptnl::Pot>& pot(int i){return pots_[i];}
	const std::shared_ptr<ptnl::Pot>& pot(int i)const{return pots_[i];}
	
	//==== member functions ====
	//reading/writing
	void read(Token& token);
	//setup/initialization
	void clear();
	void resize(int ntypes);
	void init();
	//energy/forces
	double energy(const Structure& struc);
	double energy(const Structure& struc, int j);
	double compute(Structure& struc);
	
	//==== static functions ====
	static Structure& rand_step(Structure& struc, const Engine& engine);
	static double ke(const Structure& struc);
};

//**********************************************
// serialization
//**********************************************
	
namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Engine& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Engine& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Engine& obj, const char* arr);
	
}

#endif