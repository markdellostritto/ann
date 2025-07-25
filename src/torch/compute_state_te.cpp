#include "torch/compute_state_te.hpp"

namespace compute{

namespace state{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const TE& te){
	return static_cast<const Base&>(te);
}

//==== member functions ====

double TE::compute(const Struc& struc){
	double struc.ke()=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.ke()+=struc.mass(i)*struc.vel(i).squaredNorm();
	}
	struc.ke()*=0.5;
	struc.energy()=ke+struc.pe();
	return struc.energy();
}

}

}