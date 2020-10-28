#pragma once
#ifndef XYZ_HPP
#define XYZ_HPP

// eigen libraries
#include <Eigen/Dense>
// ann - structure
#include "structure_fwd.hpp"

#ifndef XYZ_PRINT_FUNC
#define XYZ_PRINT_FUNC 0
#endif

#ifndef XYZ_PRINT_STATUS
#define XYZ_PRINT_STATUS 0
#endif

#ifndef XYZ_PRINT_DATA
#define XYZ_PRINT_DATA 0
#endif

#ifndef __cplusplus
	#error A C++ compiler is required
#endif

namespace XYZ{
	
//*****************************************************
//FORMAT struct
//*****************************************************

struct Format{
	std::string xyz;//xyz
	static Format& read(const std::vector<std::string>& strlist, Format& format);
};

//unwrapping

void unwrap(Structure& struc);

//*****************************************************
//reading
//*****************************************************

void read(const char* file, const AtomType& atomT, Structure& struc);
//void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim);

//*****************************************************
//writing
//*****************************************************

void write(const char* file, const AtomType& atomT, const Structure& struc);
//void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim);

}

#endif