#pragma once
#ifndef VASP_SIM_HPP
#define VASP_SIM_HPP

// c++ libraries
#include <vector>
#include <string>
// format
#include "format/vasp.hpp"
// structure
#include "format/format.hpp"
#include "struc/sim.hpp"

namespace VASP{

namespace XDATCAR{

static const char* NAMESPACE_LOCAL="XDATCAR";
void read(const char* file, const Interval& interval, const AtomType& atomT, Simulation& sim);
void write(const char* file, const Interval& interval, const AtomType& atomT, const Simulation& sim);

}

}

#endif