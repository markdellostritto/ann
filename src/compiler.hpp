#pragma once
#ifndef COMPILER_HPP
#define COMPILER_HPP

// c++ libraries
#include <string>

namespace compiler{

	std::string standard();
	std::string name();
	std::string version();
	std::string date();
	std::string time();
	std::string arch();
	std::string instr();
	std::string omp();
	std::string os();
	
}

#endif