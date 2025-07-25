// c
#include <cstdio>
// c++
#include <iostream>
#include <stdexcept>
// str
#include "str/string.hpp"
#include "str/print.hpp"
#include "str/token.hpp"
// math
#include "math/const.hpp"
// torch
#include "torch/engine.hpp"
#include "torch/phonon.hpp"
#include "torch/pot.hpp"
#include "torch/pot_factory.hpp"
#include "torch/pot_list.hpp"

//****************************************************************************
// Engine
//****************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Engine& engine){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("ENGINE",str)<<"\n";
	out<<"NTYPES = "<<engine.ntypes_<<"\n";
	out<<"RCMAX  = "<<engine.rcmax_<<"\n";
	out<<"VLIST  = "<<engine.vlist_<<"\n";
	for(int i=0; i<engine.pots_.size(); ++i){
		out<<"POT = ";
		switch(engine.pots_[i]->name()){
			case ptnl::Pot::Name::COUL_CUT:{out<<static_cast<const ptnl::PotCoulCut&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::COUL_LONG:{out<<static_cast<const ptnl::PotCoulLong&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::COUL_WOLF:{out<<static_cast<const ptnl::PotCoulWolf&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::COUL_DSF:{out<<static_cast<const ptnl::PotCoulDSF&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::GAUSS_CUT:{out<<static_cast<const ptnl::PotGaussCut&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::GAUSS_DSF:{out<<static_cast<const ptnl::PotGaussDSF&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::GAUSS_LONG:{out<<static_cast<const ptnl::PotGaussLong&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::LJ_CUT:{out<<static_cast<const ptnl::PotLJCut&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::LJ_LONG:{out<<static_cast<const ptnl::PotLJLong&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::LDAMP_CUT:{out<<static_cast<const ptnl::PotLDampCut&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::LDAMP_LONG:{out<<static_cast<const ptnl::PotLDampLong&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::QEQ_GL:{out<<static_cast<const ptnl::PotQEQGL&>(*engine.pots_[i])<<"\n";} break;
			case ptnl::Pot::Name::NNPE:{out<<static_cast<const ptnl::PotNNPE&>(*engine.pots_[i])<<"\n";} break;
			default: break;
		}
	}
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

//** setup/initialization **

void Engine::clear(){
	if(ENGINE_PRINT_FUNC>0) std::cout<<"Engine::clear():\n";
	ntypes_=-1;
	rcmax_=0;
	vlist_.clear();
	pots_.clear();
}

void Engine::resize(int ntypes){
	if(ENGINE_PRINT_FUNC>0) std::cout<<"Engine::resize(int):\n";
	if(ntypes<=0) throw std::invalid_argument("Invalid number of types.");
	ntypes_=ntypes;
	for(int i=0; i<pots_.size(); ++i) pots_[i]->resize(ntypes_);
}

void Engine::init(){
	if(ENGINE_PRINT_FUNC>0) std::cout<<"Engine::init():\n";
	rcmax_=0;
	for(int i=0; i<pots_.size(); ++i){
		if(pots_[i]->rc()>rcmax_) rcmax_=pots_[i]->rc();
		pots_[i]->init();
	}
	if(rcmax_==0) throw std::invalid_argument("Engine::init(): Invalid max cutoff.\n");
	vlist_.rc()=rcmax_;
}

//** energy/forces **

double Engine::energy(const Structure& struc){
	if(ENGINE_PRINT_FUNC>0) std::cout<<"Engine::energy(const Structure&):\n";
	double energy=0;
	for(int i=0; i<pots_.size(); ++i){
		energy+=pots_[i]->energy(struc,vlist_);
	}
	return energy;
}

double Engine::energy(const Structure& struc, int j){
	if(ENGINE_PRINT_FUNC>0) std::cout<<"Engine::energy(const Structure&):\n";
	double energy=0;
	for(int i=0; i<pots_.size(); ++i){
		energy+=pots_[i]->energy(struc,vlist_,j);
	}
	return energy;
}

double Engine::compute(Structure& struc){
	if(ENGINE_PRINT_FUNC>0) std::cout<<"Engine::compute(const Structure&):\n";
	//reset energy/forces
	double energy=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.force(i).setZero();
	}
	//compute energy/forces
	for(int i=0; i<pots_.size(); ++i){
		energy+=pots_[i]->compute(struc,vlist_);
	}
	return energy;
}

void Engine::read(Token& token){
	if(ENGINE_PRINT_FUNC>0) std::cout<<"Engine::read(Token&):\n";
	token.next();//stride
	const int stride=std::atoi(token.next().c_str());
	if(stride<=0) throw std::invalid_argument("Engine::read(Token&): invalid neighbor stride.");
	vlist_.stride()=stride;
}

Structure& Engine::rand_step(Structure& struc, const Engine& engine){
	/*switch(engine.sim().barostat()){
		case Barostat::NONE:{
			for(int i=0; i<struc.nAtoms(); ++i){
				struc.posn(i).noalias()+=Eigen::Vector3d::Random()*engine.met().dr()/math::constant::Rad3;
			}
		}break;
		case Barostat::ISO:{
			Eigen::Matrix3d R=struc.R();
			R*=(1.0+2.0*(((double)std::rand())/RAND_MAX-0.5)*engine.met().dv());
			static_cast<Cell&>(struc).init(R);
			for(int i=0; i<struc.nAtoms(); ++i){
				struc.posn(i).noalias()+=Eigen::Vector3d::Random()*engine.met().dr()/math::constant::Rad3;
			}
		}break;
		case Barostat::ANISO:{
			Eigen::Matrix3d R=struc.R();
			R.col(0)*=(1.0+2.0*(((double)std::rand())/RAND_MAX-0.5)*engine.met().dv()/3.0);
			R.col(1)*=(1.0+2.0*(((double)std::rand())/RAND_MAX-0.5)*engine.met().dv()/3.0);
			R.col(2)*=(1.0+2.0*(((double)std::rand())/RAND_MAX-0.5)*engine.met().dv()/3.0);
			static_cast<Cell&>(struc).init(R);
			for(int i=0; i<struc.nAtoms(); ++i){
				struc.posn(i).noalias()+=Eigen::Vector3d::Random()*engine.met().dr()/math::constant::Rad3;
			}
		}break;
		default: break;
	}*/
	return struc;
}

double Engine::ke(const Structure& struc){
	double energy=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		energy+=struc.vel(i).squaredNorm()*struc.mass(i);
	}
	return 0.5*energy;
}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Engine& obj){
		if(ENGINE_PRINT_FUNC>0) std::cout<<"nbytes(const Engine&):\n";
		int size=0;
		size+=sizeof(int);//ntypes_
		size+=nbytes(obj.vlist());
		size+=sizeof(int);//npots
		for(int i=0; i<obj.pots().size(); ++i){
			size+=sizeof(int);//name_
			switch(obj.pot(i)->name()){
				case ptnl::Pot::Name::LJ_CUT:{
					size+=nbytes(static_cast<const ptnl::PotLJCut&>(*obj.pot(i)));
				}break;
				default: break;
			}
		}
		return size;
	}
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Engine& obj, char* arr){
		if(ENGINE_PRINT_FUNC>0) std::cout<<"pack(const Engine&,char*):\n";
		int pos=0;
		const int ntypes=obj.ntypes();
		std::memcpy(arr+pos,&ntypes,sizeof(int)); pos+=sizeof(int);//ntypes_
		pos+=pack(obj.vlist(),arr+pos);
		int npots=obj.pots().size();
		std::memcpy(arr+pos,&npots,sizeof(int)); pos+=sizeof(int);//job_
		for(int i=0; i<obj.pots().size(); ++i){
			ptnl::Pot::Name name=obj.pot(i)->name();
			std::memcpy(arr+pos,&name,sizeof(int)); pos+=sizeof(int);//name_
			switch(obj.pot(i)->name()){
				case ptnl::Pot::Name::LJ_CUT:{
					pos+=pack(static_cast<const ptnl::PotLJCut&>(*obj.pot(i)),arr+pos);
				}break;
				default: break;
			}
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Engine& obj, const char* arr){
		if(ENGINE_PRINT_FUNC>0) std::cout<<"unpack(Engine&,const char*):\n";
		int pos=0;
		int ntypes=0;
		std::memcpy(&ntypes,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		obj.resize(ntypes);
		pos+=unpack(obj.vlist(),arr+pos);
		int npots=0;
		std::memcpy(&npots,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		obj.pots().resize(npots);
		for(int i=0; i<obj.pots().size(); ++i){
			ptnl::Pot::Name name;
			std::memcpy(&name,arr+pos,sizeof(int)); pos+=sizeof(int);//name_
			switch(name){
				case ptnl::Pot::Name::LJ_CUT:{
					obj.pot(i).reset(new ptnl::PotLJCut());
					pos+=unpack(static_cast<ptnl::PotLJCut&>(*obj.pot(i)),arr+pos);
				}break;
				default: break;
			}
		}
		return pos;
	}
	
}
