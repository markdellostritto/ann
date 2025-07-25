#ifndef FILE_SIM_HPP
#define FILE_SIM_HPP

// structure
#include "struc/sim.hpp"
#include "struc/interval.hpp"
// file i/o
#include "format/format.hpp"

Simulation& read_sim(const char* file, FILE_FORMAT::type format, const Interval& interval, const AtomType& atomT, Simulation& sim);
const Simulation& write_sim(const char* file, FILE_FORMAT::type format, const Interval& interval, const AtomType& atomT, const Simulation& sim);

#endif