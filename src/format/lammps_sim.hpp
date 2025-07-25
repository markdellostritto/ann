#pragma once
#ifndef LAMMPS_SIM_HPP
#define LAMMPS_SIM_HPP

// c++ libraries
#include <vector>
#include <map>
//format
#include "format/lammps.hpp"
// structure
#include "struc/sim.hpp"

namespace LAMMPS{

namespace DUMP{

void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim);
void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim);

}

}

#endif
