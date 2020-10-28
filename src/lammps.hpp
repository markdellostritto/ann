#ifndef LAMMPS_HPP
#define LAMMPS_HPP

//c++ libraries
#include <vector>
#include <map>
//simulation
#include "structure.hpp"

#ifndef __cplusplus
	#error A C++ compiler is required.
#endif

#ifndef LAMMPS_PRINT_FUNC
#define LAMMPS_PRINT_FUNC 0
#endif

#ifndef LAMMPS_PRINT_STATUS
#define LAMMPS_PRINT_STATUS 0
#endif

#ifndef LAMMPS_PRINT_DATA
#define LAMMPS_PRINT_DATA 0
#endif

namespace LAMMPS{

//static variables
static const char* NAMESPACE_GLOBAL="LAMMPS";

//formats
struct STYLE_ATOM{
	enum type{
		FULL,
		BOND,
		ATOMIC,
		CHARGE,
		UNKNOWN
	};
	static STYLE_ATOM::type read(const char* str);
};
std::ostream& operator<<(std::ostream& out, STYLE_ATOM::type& t);
struct FORMAT_ATOM{
	int atom,mol,specie;
	int x,y,z,q;
	FORMAT_ATOM():atom(-1),mol(-1),specie(-1),x(-1),y(-1),z(-1),q(-1){};
};
struct DATA_ATOM{
	int specie,index;
	double q;
	Eigen::Vector3d posn;
};

//*****************************************************
//FORMAT struct
//*****************************************************

struct Format{
	std::string in;//input file
	std::string data;//data file
	std::string dump;//dump file
	std::vector<std::string> name;
	std::vector<double> mass;
	std::vector<int> natoms;
	int N;
	int nSpecies;
	STYLE_ATOM::type styleAtom;
	std::vector<std::map<int,int> > indexmap;
	std::string units;
	static Format& read(const std::vector<std::string>& strlist, Format& format);
};

namespace DUMP{

//static variables
static const char* NAMESPACE_LOCAL="DUMP";

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim, Format& format);
void read_posn(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim);
void write(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim);

}

namespace DATA{

//static variables
static const char* NAMESPACE_LOCAL="DATA";

void read(const char* file, Format& format, Simulation& sim);

}

namespace IN{

//static variables
static const char* NAMESPACE_LOCAL="IN";

void read_style(const char* file, STYLE_ATOM::type& styleAtom);

}

void read(Format& format, const Interval& interval, const AtomType& atomT, Simulation& sim);

}

#endif
