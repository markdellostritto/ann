// c++
#include <stdexcept>
// str
#include "str/string.hpp"
// torch
#include "torch/pot_factory.hpp"
#include "torch/pot_pauli.hpp"
#include "torch/pot_lj_cut.hpp"
#include "torch/pot_lj_long.hpp"
#include "torch/pot_lj_sm.hpp"
#include "torch/pot_ldamp_cut.hpp"
#include "torch/pot_ldamp_dsf.hpp"
#include "torch/pot_ldamp_long.hpp"
#include "torch/pot_coul_cut.hpp"
#include "torch/pot_coul_wolf.hpp"
#include "torch/pot_coul_dsf.hpp"
#include "torch/pot_coul_long.hpp"
#include "torch/pot_gauss_cut.hpp"
#include "torch/pot_gauss_dsf.hpp"
#include "torch/pot_gauss_long.hpp"
#include "torch/pot_qeq_gl.hpp"
#include "torch/pot_nnpe.hpp"

//factory

namespace ptnl{
	
	std::shared_ptr<Pot>& make(std::shared_ptr<Pot>& pot, const Pot::Name& name){
		if(POT_FAC_FUNC>0) std::cout<<"make(std::shared_ptr<Pot>&,const Pot::Name&)\n";
		switch(name){
			case Pot::Name::PAULI:{
				pot.reset(new PotPauli());
			}break;
			case Pot::Name::LJ_CUT:{
				pot.reset(new PotLJCut());
			}break;
			case Pot::Name::LJ_LONG:{
				pot.reset(new PotLJLong());
			}break;
			case Pot::Name::LJ_SM:{
				pot.reset(new PotLJSm());
			}break;
			case Pot::Name::LDAMP_CUT:{
				pot.reset(new PotLDampCut());
			}break;
			case Pot::Name::LDAMP_DSF:{
				pot.reset(new PotLDampDSF());
			}break;
			case Pot::Name::LDAMP_LONG:{
				pot.reset(new PotLDampLong());
			}break;
			case Pot::Name::COUL_CUT:{
				pot.reset(new PotCoulCut());
			}break;
			case Pot::Name::COUL_WOLF:{
				pot.reset(new PotCoulWolf());
			}break;
			case Pot::Name::COUL_DSF:{
				pot.reset(new PotCoulDSF());
			}break;
			case Pot::Name::COUL_LONG:{
				pot.reset(new PotCoulLong());
			}break;
			case Pot::Name::GAUSS_CUT:{
				pot.reset(new PotGaussCut());
			}break;
			case Pot::Name::GAUSS_DSF:{
				pot.reset(new PotGaussDSF());
			}break;
			case Pot::Name::GAUSS_LONG:{
				pot.reset(new PotGaussLong());
			}break;
			case Pot::Name::QEQ_GL:{
				pot.reset(new PotQEQGL());
			}break;
			case Pot::Name::NNPE:{
				pot.reset(new PotNNPE());
			}break;
			default:{
				throw std::invalid_argument("ptnl::make(std::shared_ptr<Pot>&,potnl::Pot::Name&): Invalid potential.");
			}break;
		}
		return pot;
	}
	
	std::shared_ptr<Pot>& read(std::shared_ptr<Pot>& pot, Token& token){
		if(POT_FAC_FUNC>0) std::cout<<"make(std::shared_ptr<Pot>&,Token&)\n";
		const Pot::Name name=Pot::Name::read(string::to_upper(token.next()).c_str());
		switch(name){
			case Pot::Name::PAULI:{
				pot.reset(new PotPauli());
				static_cast<PotPauli&>(*pot).read(token);
			}break;
			case Pot::Name::LJ_CUT:{
				pot.reset(new PotLJCut());
				static_cast<PotLJCut&>(*pot).read(token);
			}break;
			case Pot::Name::LJ_LONG:{
				pot.reset(new PotLJLong());
				static_cast<PotLJLong&>(*pot).read(token);
			}break;
			case Pot::Name::LJ_SM:{
				pot.reset(new PotLJSm());
				static_cast<PotLJSm&>(*pot).read(token);
			}break;
			case Pot::Name::LDAMP_CUT:{
				pot.reset(new PotLDampCut());
				static_cast<PotLDampCut&>(*pot).read(token);
			}break;
			case Pot::Name::LDAMP_DSF:{
				pot.reset(new PotLDampDSF());
				static_cast<PotLDampDSF&>(*pot).read(token);
			}break;
			case Pot::Name::LDAMP_LONG:{
				pot.reset(new PotLDampLong());
				static_cast<PotLDampLong&>(*pot).read(token);
			}break;
			case Pot::Name::COUL_CUT:{
				pot.reset(new PotCoulCut());
				static_cast<PotCoulCut&>(*pot).read(token);
			}break;
			case Pot::Name::COUL_WOLF:{
				pot.reset(new PotCoulWolf());
				static_cast<PotCoulWolf&>(*pot).read(token);
			}break;
			case Pot::Name::COUL_DSF:{
				pot.reset(new PotCoulDSF());
				static_cast<PotCoulDSF&>(*pot).read(token);
			}break;
			case Pot::Name::COUL_LONG:{
				pot.reset(new PotCoulLong());
				static_cast<PotCoulLong&>(*pot).read(token);
			}break;
			case Pot::Name::GAUSS_CUT:{
				pot.reset(new PotGaussCut());
				static_cast<PotGaussCut&>(*pot).read(token);
			}break;
			case Pot::Name::GAUSS_DSF:{
				pot.reset(new PotGaussDSF());
				static_cast<PotGaussDSF&>(*pot).read(token);
			}break;
			case Pot::Name::GAUSS_LONG:{
				pot.reset(new PotGaussLong());
				static_cast<PotGaussLong&>(*pot).read(token);
			}break;
			case Pot::Name::QEQ_GL:{
				pot.reset(new PotQEQGL());
				static_cast<PotQEQGL&>(*pot).read(token);
			}break;
			case Pot::Name::NNPE:{
				pot.reset(new PotNNPE());
				static_cast<PotNNPE&>(*pot).read(token);
			}break;
			default:{
				throw std::invalid_argument("ptnl::read(std::shared_ptr<Pot>&,Token&): Invalid potential.");
			}break;
		}
		return pot;
	}
	
	void coeff(std::vector<std::shared_ptr<Pot> > pots, Token& token){
		Pot::Name name=Pot::Name::read(string::to_upper(token.next()).c_str());
		for(int i=0; i<pots.size(); ++i){
			if(pots[i]->name()==name){
				pots[i]->coeff(token);
				break;
			}
		}
	}
	
	std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Pot>& pot){
		switch(pot->name()){
			case Pot::Name::PAULI: out<<static_cast<const PotPauli&>(*pot); break;
			case Pot::Name::LJ_CUT: out<<static_cast<const PotLJCut&>(*pot); break;
			case Pot::Name::LJ_LONG: out<<static_cast<const PotLJLong&>(*pot); break;
			case Pot::Name::LJ_SM: out<<static_cast<const PotLJSm&>(*pot); break;
			case Pot::Name::LDAMP_CUT: out<<static_cast<const PotLDampCut&>(*pot); break;
			case Pot::Name::LDAMP_DSF: out<<static_cast<const PotLDampDSF&>(*pot); break;
			case Pot::Name::LDAMP_LONG: out<<static_cast<const PotLDampLong&>(*pot); break;
			case Pot::Name::COUL_CUT: out<<static_cast<const PotCoulCut&>(*pot); break;
			case Pot::Name::COUL_WOLF: out<<static_cast<const PotCoulWolf&>(*pot); break;
			case Pot::Name::COUL_DSF: out<<static_cast<const PotCoulDSF&>(*pot); break;
			case Pot::Name::COUL_LONG: out<<static_cast<const PotCoulLong&>(*pot); break;
			case Pot::Name::GAUSS_CUT: out<<static_cast<const PotGaussCut&>(*pot); break;
			case Pot::Name::GAUSS_DSF: out<<static_cast<const PotGaussDSF&>(*pot); break;
			case Pot::Name::GAUSS_LONG: out<<static_cast<const PotGaussLong&>(*pot); break;
			case Pot::Name::QEQ_GL: out<<static_cast<const PotQEQGL&>(*pot); break;
			case Pot::Name::NNPE: out<<static_cast<const PotNNPE&>(*pot); break;
			default: break;
		}
		return out;
	}
}

//serialization

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const std::shared_ptr<ptnl::Pot>& obj){
		if(POT_FAC_FUNC>0) std::cout<<"nbytes(const std::shared_ptr<ptnl::Pot>&):\n";
		int size=0;
		size+=sizeof(int);
		if(obj!=nullptr){
			size+=sizeof(ptnl::Pot::Name);
			switch(obj->name()){
				case ptnl::Pot::Name::PAULI: size+=nbytes(static_cast<const ptnl::PotPauli&>(*obj)); break;
				case ptnl::Pot::Name::LJ_CUT: size+=nbytes(static_cast<const ptnl::PotLJCut&>(*obj)); break;
				case ptnl::Pot::Name::LJ_LONG: size+=nbytes(static_cast<const ptnl::PotLJLong&>(*obj)); break;
				case ptnl::Pot::Name::LJ_SM: size+=nbytes(static_cast<const ptnl::PotLJSm&>(*obj)); break;
				case ptnl::Pot::Name::LDAMP_CUT: size+=nbytes(static_cast<const ptnl::PotLDampCut&>(*obj)); break;
				case ptnl::Pot::Name::LDAMP_DSF: size+=nbytes(static_cast<const ptnl::PotLDampDSF&>(*obj)); break;
				case ptnl::Pot::Name::LDAMP_LONG: size+=nbytes(static_cast<const ptnl::PotLDampLong&>(*obj)); break;
				case ptnl::Pot::Name::COUL_CUT: size+=nbytes(static_cast<const ptnl::PotCoulCut&>(*obj)); break;
				case ptnl::Pot::Name::COUL_WOLF: size+=nbytes(static_cast<const ptnl::PotCoulWolf&>(*obj)); break;
				case ptnl::Pot::Name::COUL_DSF: size+=nbytes(static_cast<const ptnl::PotCoulDSF&>(*obj)); break;
				case ptnl::Pot::Name::COUL_LONG: size+=nbytes(static_cast<const ptnl::PotCoulLong&>(*obj)); break;
				case ptnl::Pot::Name::GAUSS_CUT: size+=nbytes(static_cast<const ptnl::PotGaussCut&>(*obj)); break;
				case ptnl::Pot::Name::GAUSS_DSF: size+=nbytes(static_cast<const ptnl::PotGaussDSF&>(*obj)); break;
				case ptnl::Pot::Name::GAUSS_LONG: size+=nbytes(static_cast<const ptnl::PotGaussLong&>(*obj)); break;
				case ptnl::Pot::Name::QEQ_GL: size+=nbytes(static_cast<const ptnl::PotQEQGL&>(*obj)); break;
				case ptnl::Pot::Name::NNPE: size+=nbytes(static_cast<const ptnl::PotNNPE&>(*obj)); break;
				default: throw std::invalid_argument("serialize::nbytes(const std::shared_ptr<ptnl::Pot>&): Invalid potential name."); break;
			}
		}
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const std::shared_ptr<ptnl::Pot>& obj, char* arr){
		if(POT_FAC_FUNC>0) std::cout<<"pack(const std::shared_ptr<ptnl::Pot>&,char*)\n";
		int pos=0;
		const int null=(obj==nullptr);
		std::memcpy(arr+pos,&null,sizeof(int)); pos+=sizeof(int);
		if(!null){
			std::memcpy(arr+pos,&obj->name(),sizeof(ptnl::Pot::Name)); pos+=sizeof(ptnl::Pot::Name);
			switch(obj->name()){
				case ptnl::Pot::Name::PAULI: pos+=pack(static_cast<const ptnl::PotPauli&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LJ_CUT: pos+=pack(static_cast<const ptnl::PotLJCut&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LJ_LONG: pos+=pack(static_cast<const ptnl::PotLJLong&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LJ_SM: pos+=pack(static_cast<const ptnl::PotLJSm&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LDAMP_CUT: pos+=pack(static_cast<const ptnl::PotLDampCut&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LDAMP_DSF: pos+=pack(static_cast<const ptnl::PotLDampDSF&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LDAMP_LONG: pos+=pack(static_cast<const ptnl::PotLDampLong&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::COUL_CUT: pos+=pack(static_cast<const ptnl::PotCoulCut&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::COUL_WOLF: pos+=pack(static_cast<const ptnl::PotCoulWolf&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::COUL_DSF: pos+=pack(static_cast<const ptnl::PotCoulDSF&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::COUL_LONG: pos+=pack(static_cast<const ptnl::PotCoulLong&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::GAUSS_CUT: pos+=pack(static_cast<const ptnl::PotGaussCut&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::GAUSS_DSF: pos+=pack(static_cast<const ptnl::PotGaussDSF&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::GAUSS_LONG: pos+=pack(static_cast<const ptnl::PotGaussLong&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::QEQ_GL: pos+=pack(static_cast<const ptnl::PotQEQGL&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::NNPE: pos+=pack(static_cast<const ptnl::PotNNPE&>(*obj),arr+pos); break;
				default: throw std::invalid_argument("serialize::pack(const std::shared_ptr<ptnl::Pot>&, char*): Invalid potential name."); break;
			}
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(std::shared_ptr<ptnl::Pot>& obj, const char* arr){
		if(POT_FAC_FUNC>0) std::cout<<"unpack(std::shared_ptr<ptnl::Pot>&,const char*)\n";
		int pos=0;
		int null=1;
		std::memcpy(&null,arr+pos,sizeof(int)); pos+=sizeof(int);
		if(!null){
			ptnl::Pot::Name name=ptnl::Pot::Name::UNKNOWN;
			std::memcpy(&name,arr+pos,sizeof(ptnl::Pot::Name)); pos+=sizeof(ptnl::Pot::Name);
			ptnl::make(obj,name);
			switch(obj->name()){
				case ptnl::Pot::Name::PAULI: pos+=unpack(static_cast<ptnl::PotPauli&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LJ_CUT: pos+=unpack(static_cast<ptnl::PotLJCut&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LJ_LONG: pos+=unpack(static_cast<ptnl::PotLJLong&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LJ_SM: pos+=unpack(static_cast<ptnl::PotLJSm&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LDAMP_CUT: pos+=unpack(static_cast<ptnl::PotLDampCut&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LDAMP_DSF: pos+=unpack(static_cast<ptnl::PotLDampDSF&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::LDAMP_LONG: pos+=unpack(static_cast<ptnl::PotLDampLong&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::COUL_CUT: pos+=unpack(static_cast<ptnl::PotCoulCut&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::COUL_WOLF: pos+=unpack(static_cast<ptnl::PotCoulWolf&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::COUL_DSF: pos+=unpack(static_cast<ptnl::PotCoulDSF&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::COUL_LONG: pos+=unpack(static_cast<ptnl::PotCoulLong&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::GAUSS_CUT: pos+=unpack(static_cast<ptnl::PotGaussCut&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::GAUSS_DSF: pos+=unpack(static_cast<ptnl::PotGaussDSF&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::GAUSS_LONG: pos+=unpack(static_cast<ptnl::PotGaussLong&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::QEQ_GL: pos+=unpack(static_cast<ptnl::PotQEQGL&>(*obj),arr+pos); break;
				case ptnl::Pot::Name::NNPE: pos+=unpack(static_cast<ptnl::PotNNPE&>(*obj),arr+pos); break;
				default: throw std::invalid_argument("serialize::unpack(std::shared_ptr<ptnl::Pot>&,const char*): Invalid potential name."); break;
			}
		} else obj.reset();
		return pos;
	}
	
}