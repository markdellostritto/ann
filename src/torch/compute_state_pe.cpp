#include "torch/compute_state_pe.hpp"

namespace compute{

namespace state{

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PE& pe){
	return static_cast<const Base&>(pe);
}

//==== member functions ====

double PE::compute(const Struc& struc){
	return struc.pe();
}

}

}