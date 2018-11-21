#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "structure.hpp"
#include "string.hpp"
#include "cell.hpp"
#include "units.hpp"

#ifndef DEBUG_QE
#define DEBUG_QE 0
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
	std::string fileCel;//cell
	std::string fileEvp;//energy/volume/pressure
	std::string fileOut;//std output file
	static Format& read(const std::vector<std::string>& strlist, Format& format);
};

//*****************************************************
//IN format
//*****************************************************

namespace IN{

//static variables
static const char* NAMESPACE_LOCAL="IN";
	
Cell& read_cell(FILE* reader, Cell& cell);
void read_atoms(FILE* reader, std::vector<std::string>& atomNames, std::vector<unsigned int>& atomNumbers);
double read_timestep(FILE* reader);

}

//*****************************************************
//CEL format
//*****************************************************

namespace CEL{

//static variables
static const char* NAMESPACE_LOCAL="CEL";

void read_cell(FILE* reader, Simulation& sim);
	
}

//*****************************************************
//POS format
//*****************************************************

namespace POS{

//static variables
static const char* NAMESPACE_LOCAL="POS";
unsigned int load_timesteps(FILE* reader);
void load_posns(FILE* reader, Simulation& sim);

}

//*****************************************************
//EVP format
//*****************************************************

namespace EVP{
	
static const char* NAMESPACE_LOCAL="EVP";
void load_energy(FILE* reader, Simulation& sim);

}

//*****************************************************
//OUT format
//*****************************************************

namespace OUT{
	
static const char* NAMESPACE_LOCAL="OUT";
void read(const char* file, const AtomType& atomT, Structure& struc);

}

Simulation& read(const Format& format, const Interval& interval, const AtomType& atomT, Simulation& sim);

}