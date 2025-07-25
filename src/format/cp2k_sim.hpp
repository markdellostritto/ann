#pragma once
#ifndef CP2K_SIM_HPP
#define CP2K_SIM_HPP

// eigen libraries
#include <Eigen/Dense>
// format
#include "format/cp2k.hpp"
// structure
#include "struc/sim.hpp"

#ifndef __cplusplus
	#error A C++ compiler is required
#endif

namespace CP2K{

//*****************************************************
//FORMAT struct
//*****************************************************

struct Format{
	std::string fxyz;//fxyz
	std::string xyz;//xyz
	std::string input;//input
	static Format& read(const std::vector<std::string>& strlist, Format& format);
};

//*****************************************************
//reading
//*****************************************************

Simulation& read(const Format& format, const Interval& interval, const AtomType& atomT, Simulation& sim);

//*****************************************************
//writing
//*****************************************************

//const Simulation& write(const Format& format, const Interval& interval, const AtomType& atomT, const Simulation& sim);


}

#endif