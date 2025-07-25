#pragma once
#ifndef VASP_STRUC_HPP
#define VASP_STRUC_HPP

// c++ libraries
#include <vector>
#include <string>
// format
#include "format/vasp.hpp"
// structure
#include "format/format.hpp"
#include "struc/structure_fwd.hpp"

namespace VASP{

//*****************************************************
//FORMAT struct
//*****************************************************

struct Format{
	std::string xdatcar;//xdatcar
	std::string poscar;//poscar
	std::string xml;//poscar
	std::string outcar;//outcar
	std::string procar;//procar
	std::string eigenval;//eigenval
	std::string energy;//energy
	static Format& read(const std::vector<std::string>& strlist, Format& format);
};

//*****************************************************
//POSCAR
//*****************************************************

namespace POSCAR{

static const char* NAMESPACE_LOCAL="POSCAR";
void read(const char* file, const AtomType& atomT, Structure& struc);
void write(const char* file, const AtomType& atomT, const Structure& struc);

}

//*****************************************************
//XML
//*****************************************************

namespace XML{

static const char* NAMESPACE_LOCAL="XML";
void read(const char* file, int t, const AtomType& atomT, Structure& struc);

}

//Simulation& read(const Format& format, const Interval& interval, const AtomType& atomT, Simulation& sim);
//const Simulation& write(const Format& format, const Interval& interval, const AtomType& atomT, const Simulation& sim);

}


#endif
