//chem
#include "chem/units.hpp"
//torch
#include "torch/compute_state_temp.hpp"

namespace compute{

namespace state{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Temp& temp){
	return static_cast<const Base&>(temp);
}

//==== member functions ====

double Temp::compute(const Struc& struc){
	double struc.ke()=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.ke()+=struc.mass(i)*struc.vel(i).squaredNorm();
	}
	struc.ke()*=0.5;
	struc.T()=2.0/3.0*struc.ke()/(struc.nAtoms()*units::consts::kb());
	return struc.T();
}

}

}