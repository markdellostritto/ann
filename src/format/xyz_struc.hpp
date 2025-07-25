#pragma once
#ifndef XYZ_STRUC_HPP
#define XYZ_STRUC_HPP

// eigen libraries
#include <Eigen/Dense>
// format
#include "format/xyz.hpp"
// structure
#include "struc/structure_fwd.hpp"

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

//*****************************************************
//writing
//*****************************************************

void write(const char* file, const AtomType& atomT, const Structure& struc);

}

#endif
