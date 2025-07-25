#pragma once
#ifndef RAW_STRUC_HPP
#define RAW_STRUC_HPP

// eigen libraries
#include <Eigen/Dense>
// structure
#include "struc/structure_fwd.hpp"

#ifndef RAW_PRINT_FUNC
#define RAW_PRINT_FUNC 0
#endif

#ifndef RAW_PRINT_STATUS
#define RAW_PRINT_STATUS 0
#endif

#ifndef RAW_PRINT_DATA
#define RAW_PRINT_DATA 0
#endif

namespace RAW{
	
//*****************************************************
//FORMAT struct
//*****************************************************

struct Format{
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
