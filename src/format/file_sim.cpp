// c++ libraries
#include <iostream>
#include <exception>
// file i/o
#include "format/file_sim.hpp"
#include "format/vasp_sim.hpp"
#include "format/lammps_sim.hpp"
#include "format/xyz_sim.hpp"

Simulation& read_sim(const char* file, FILE_FORMAT::type format, const Interval& interval, const AtomType& atomT, Simulation& sim){
	switch(format){
		case FILE_FORMAT::XDATCAR:
			VASP::XDATCAR::read(file,interval,atomT,sim);
		break;
		case FILE_FORMAT::LAMMPS:
			LAMMPS::DUMP::read(file,interval,atomT,sim);
		break;
		case FILE_FORMAT::XYZ:
			XYZ::read(file,interval,atomT,sim);
		break;
		default:
			throw std::invalid_argument("ERROR in read_sim(const char*,FILE_Format::type,const AtomType&,Simulation&): invalid file format.");
		break;
	}
	return sim;
}

const Simulation& write_sim(const char* file, FILE_FORMAT::type format, const Interval& interval, const AtomType& atomT, const Simulation& sim){
	switch(format){
		case FILE_FORMAT::XDATCAR:
			VASP::XDATCAR::write(file,interval,atomT,sim);
		break;
		case FILE_FORMAT::LAMMPS:
			LAMMPS::DUMP::write(file,interval,atomT,sim);
		break;
		case FILE_FORMAT::XYZ:
			XYZ::write(file,interval,atomT,sim);
		break;
		default:
			throw std::invalid_argument("ERROR in write_sim(const char*,FILE_Format::type,const AtomType&,const Simulation&): invalid file format.");
		break;
	}
	return sim;
}