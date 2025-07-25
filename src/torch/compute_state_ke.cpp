#include "torch/compute_state_ke.hpp"

namespace compute{

namespace state{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const KE& ke){
	return static_cast<const Base&>(ke);
}
	
//==== member functions ====

double KE::compute(const Struc& struc){
	double struc.ke()=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		struc.ke()+=struc.mass(i)*struc.vel(i).squaredNorm();
	}
	struc.ke()*=0.5;
	return struc.ke();
}

}

}