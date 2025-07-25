#pragma once
#ifndef CUBE_HPP
#define CUBE_HPP

// eigen libraries
#include <Eigen/Dense>
// local libraries - structure
#include "struc/structure.hpp"
#include "struc/grid.hpp"

#ifndef CUBE_PRINT_FUNC
#define CUBE_PRINT_FUNC 0
#endif

#ifndef CUBE_PRINT_STATUS
#define CUBE_PRINT_STATUS 0
#endif

#ifndef CUBE_PRINT_DATA
#define CUBE_PRINT_DATA 0
#endif

namespace CUBE{
	
void read(const char* file, const AtomType& atomT, Structure& struc, Grid& grid);
void write(const char* file, const AtomType& atomT, const Structure& struc, const Grid& grid);

}

#endif
