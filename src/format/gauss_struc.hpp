#pragma once
#ifndef GAUSS_STRUC_HPP
#define GAUSS_STRUC_HPP

// eigen libraries
#include <Eigen/Dense>
// structure
#include "struc/structure.hpp"

#ifndef GAUSS_PRINT_FUNC
#define GAUSS_PRINT_FUNC 0
#endif

#ifndef GAUSS_PRINT_STATUS
#define GAUSS_PRINT_STATUS 0
#endif

#ifndef GAUSS_PRINT_DATA
#define GAUSS_PRINT_DATA 0
#endif

namespace GAUSSIAN{
	
//*****************************************************
//FORMAT struct
//*****************************************************

struct Format{
	std::string gauss;//gauss
	static Format& read(const std::vector<std::string>& strlist, Format& format);
};

//*****************************************************
//reading
//*****************************************************

void read(const char* file, const AtomType& atomT, Structure& struc);

//*****************************************************
//writing
//*****************************************************

void write(const char* file, const AtomType& atomT, const Structure& struc);

}

#endif