#pragma once
#ifndef QE_HPP
#define QE_HPP

// c++ libraries
#include <vector>
#include <string>
// ann - structure
#include "struc/structure_fwd.hpp"

#ifndef QE_PRINT_FUNC
#define QE_PRINT_FUNC 0
#endif

#ifndef QE_PRINT_STATUS
#define QE_PRINT_STATUS 0
#endif

namespace QE{

//static variables
static const char* NAMESPACE_GLOBAL="QE";

//*****************************************************
//FORMAT struct
//*****************************************************

struct Format{
	std::string fileIn;//input
	std::string filePos;//position
	std::string fileFor;//force
	std::string fileCel;//cell
	std::string fileEvp;//energy/volume/pressure
	std::string fileOut;//std output file
	static Format& read(const std::vector<std::string>& strlist, Format& format);
};

//*****************************************************
//OUT format
//*****************************************************

namespace OUT{
	
static const char* NAMESPACE_LOCAL="OUT";
void read(const char* file, const AtomType& atomT, Structure& struc);

}

}

#endif