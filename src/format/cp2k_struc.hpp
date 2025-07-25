#pragma once
#ifndef CP2K_STRUC_HPP
#define CP2K_STRUC_HPP

// eigen libraries
#include <Eigen/Dense>
// format
#include "format/cp2k.hpp"
// structure
#include "struc/structure_fwd.hpp"

#ifndef __cplusplus
	#error A C++ compiler is required
#endif

namespace CP2K{

//*****************************************************
//reading
//*****************************************************

void read(const char* file, const AtomType& atomT, Structure& struc);

}

#endif