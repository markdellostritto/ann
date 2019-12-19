#pragma once
#ifndef VASP_HPP
#define VASP_HPP

// c++ libraries
#include <vector>
#include <string>
// ann - structure
#include "structure_fwd.hpp"

#ifndef DEBUG_VASP
#define DEBUG_VASP 0
#endif

#ifndef __cplusplus
	#error A C++ compiler is required
#endif

namespace VASP{

//static variables
static const int HEADER_SIZE=7;//number of lines in the header before the atomic positions
static const char* NAMESPACE_GLOBAL="VASP";


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
//reading
//*****************************************************

void read_cell(FILE* reader, Cell& cell);
void read_atoms(FILE* reader, std::vector<std::string>& names, std::vector<unsigned int>& natoms);
bool read_coord(FILE* reader);

//*****************************************************
//writing
//*****************************************************

void write_name(FILE* writer, const char* name);
void write_cell(FILE* writer, const Cell& cell);
void write_atoms(FILE* write, const std::vector<std::string>& names, const std::vector<unsigned int>& natoms);

//*****************************************************
//POSCAR
//*****************************************************

namespace POSCAR{

static const char* NAMESPACE_LOCAL="POSCAR";
void read(const char* file, const AtomType& atomT, Structure& struc);
void write(const char* file, const AtomType& atomT, const Structure& struc);

}

//*****************************************************
//XDATCAR
//*****************************************************

namespace XDATCAR{

static const char* NAMESPACE_LOCAL="XDATCAR";
void read(const char* file, const Interval interval, const AtomType& atomT, Simulation& sim);
void write(const char* file, const Interval interval, const AtomType& atomT, const Simulation& sim);

}

//*****************************************************
//XML
//*****************************************************

namespace XML{

static const char* NAMESPACE_LOCAL="XML";
void read(const char* file, unsigned int t, const AtomType& atomT, Structure& struc);

}

Simulation& read(const Format& format, const Interval& interval, const AtomType& atomT, Simulation& sim);
const Simulation& write(const Format& format, const Interval& interval, const AtomType& atomT, const Simulation& sim);

}


#endif
