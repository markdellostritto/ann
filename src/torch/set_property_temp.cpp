//c++
#include <iostream>
//chem
#include "chem/units.hpp"
//torch
#include "torch/set_property_temp.hpp"

namespace property{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Temp& temp){
	return out<<static_cast<const Base&>(temp)<<" "<<temp.temp_;
}

//==== member functions ====
void Temp::read(Token& token){
	//property temp ${group} ${temp}
	group_.label()=token.next();
	temp_=std::atof(token.next().c_str());
}

void Temp::set(Structure& struc){
	//randomize velocities
	for(int i=0; i<group_.size(); ++i){
		struc.force(group_.atom(i)).setZero();
		struc.vel(group_.atom(i))=Eigen::Vector3d::Random();
	}
	//compute KE/T
	struc.ke()=0;
	for(int i=0; i<group_.size(); ++i){
		const int ii=group_.atom(i);
		struc.ke()+=struc.mass(ii)*struc.vel(ii).squaredNorm();
	}
	struc.ke()*=0.5;
	struc.temp()=struc.ke()*(2.0/3.0)/(group_.size()*units::Consts::kb());
	//rescale velocities
	const double fac=sqrt(temp_/(struc.temp()+1e-6));
	for(int i=0; i<group_.size(); ++i){
		struc.vel(group_.atom(i))*=fac;
	}
	//compute KE/T
	struc.ke()=0;
	for(int i=0; i<group_.size(); ++i){
		const int ii=group_.atom(i);
		struc.ke()+=struc.mass(ii)*struc.vel(ii).squaredNorm();
	}
	struc.ke()*=0.5;
	struc.temp()=struc.ke()*(2.0/3.0)/(group_.size()*units::Consts::kb());
}

}