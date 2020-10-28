#pragma once
#ifndef FORMAT_HPP
#define FORMAT_HPP

#include <iosfwd>

//**********************************************************************************************
//FILE_FORMAT struct
//**********************************************************************************************

struct FILE_FORMAT{
	enum type{
		UNKNOWN,//Unknown format
		XDATCAR,//VASP xdatcar file
		POSCAR,//VASP poscar file
		OUTCAR,//VASP outcar file
		VASP_XML,//VASP XML file
		XYZ,//XYZ file
		LAMMPS,//LAMMPS input,data,dump files
		QE,//quantum espresso output files
		AME,//ame format
		ANN,//ame format
		BINARY
	};
	static FILE_FORMAT::type read(const char* str);
	static const char* name(const FILE_FORMAT::type& format);
};

std::ostream& operator<<(std::ostream& out, FILE_FORMAT::type& format);

#endif