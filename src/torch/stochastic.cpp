// c++
#include <iostream>
#include <stdexcept>
// math
#include "math/const.hpp"
// struc
#include "struc/structure.hpp"
// torch
#include "torch/stochastic.hpp"

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Stochastic& s){
	return out<<"dr "<<s.dr_<<" dv "<<s.dv_<<" met "<<s.met_;
}
	
//==== member functions ====

void Stochastic::read(Token& token){
	//read T
	token.next();//T
	const double T=std::atof(token.next().c_str());
	if(T<=0) throw std::invalid_argument("Stochastic::read(Token&): invalid dr.");
	met_.T()=T;
	//read dr
	token.next();//dr
	dr_=std::atof(token.next().c_str());
	if(dr_<=0) throw std::invalid_argument("Stochastic::read(Token&): invalid dr.");
	//read dv
	token.next();//dv
	dv_=std::atof(token.next().c_str());
	if(dv_<=0) throw std::invalid_argument("Stochastic::read(Token&): invalid dv.");
}

Structure& Stochastic::step(Structure& struc, Engine& engine){
	for(int i=0; i<struc.nAtoms(); ++i){
		engine.vlist().build(struc,i);
		const double energy_old=engine.energy(struc,i);
		const Eigen::Vector3d dp=Eigen::Vector3d::Random()*dr_/math::constant::Rad3;
		struc.posn(i).noalias()+=dp;
		engine.vlist().build(struc,i);
		const double energy_new=engine.energy(struc,i);
		const bool accept=met_.step(energy_new-energy_old);
		if(!accept){
			struc.posn(i).noalias()-=dp;
		}
	}
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