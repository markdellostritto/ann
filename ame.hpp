#pragma once
#ifndef AME_HPP
#define AME_HPP

// c++ libraries
#include <vector>
#include <string>
// ann - structure
#include "structure_fwd.hpp"

#ifndef DEBUG_AME_PRINT_FUNC
#define DEBUG_AME_PRINT_FUNC 0
#endif

#ifndef DEBUG_AME_PRINT_STATUS
#define DEBUG_AME_PRINT_STATUS 0
#endif

#ifndef DEBUG_AME_PRINT_DATA
#define DEBUG_AME_PRINT_DATA 0
#endif

namespace AME{

struct Format{
	std::string ame;
	static Format& read(const std::vector<std::string>& strlist, Format& format);
};

static const char* NAMESPACE="AME";

void read(const char* file, const AtomType& atomT, Structure& struc);
void write(const char* file, const AtomType& atomT, const Structure& struc);
void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim);
void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim);

}

#endif