#include <iostream>
#include <cstring>
#include "format/format.hpp"

//**********************************************************************************************
//FILE_FORMAT struct
//**********************************************************************************************

FILE_FORMAT::type FILE_FORMAT::read(const char* str){
	if(std::strcmp(str,"XDATCAR")==0) return FILE_FORMAT::XDATCAR;
	else if(std::strcmp(str,"POSCAR")==0) return FILE_FORMAT::POSCAR;
	else if(std::strcmp(str,"OUTCAR")==0) return FILE_FORMAT::OUTCAR;
	else if(std::strcmp(str,"VASP_XML")==0) return FILE_FORMAT::VASP_XML;
	else if(std::strcmp(str,"XYZ")==0) return FILE_FORMAT::XYZ;
	else if(std::strcmp(str,"CP2K")==0) return FILE_FORMAT::CP2K;
	else if(std::strcmp(str,"LAMMPS")==0) return FILE_FORMAT::LAMMPS;
	else if(std::strcmp(str,"QE")==0) return FILE_FORMAT::QE;
	else if(std::strcmp(str,"GAUSSIAN")==0) return FILE_FORMAT::GAUSSIAN;
	else if(std::strcmp(str,"RAW")==0) return FILE_FORMAT::RAW;
	else if(std::strcmp(str,"BINARY")==0) return FILE_FORMAT::BINARY;
	else return FILE_FORMAT::UNKNOWN;
}

static const char* name(const FILE_FORMAT::type& format){
	switch(format){
		case FILE_FORMAT::XDATCAR: return "XDATCAR";
		case FILE_FORMAT::POSCAR: return "POSCAR";
		case FILE_FORMAT::OUTCAR: return "OUTCAR";
		case FILE_FORMAT::VASP_XML: return "VASP_XML";
		case FILE_FORMAT::XYZ: return "XYZ";
		case FILE_FORMAT::CP2K: return "CP2K";
		case FILE_FORMAT::LAMMPS: return "LAMMPS";
		case FILE_FORMAT::QE: return "QE";
		case FILE_FORMAT::GAUSSIAN: return "GAUSSIAN";
		case FILE_FORMAT::RAW: return "RAW";
		case FILE_FORMAT::BINARY: return "BINARY";
		default: return "UNKNOWN";
	}
}

std::ostream& operator<<(std::ostream& out, FILE_FORMAT::type& format){
	switch(format){
		case FILE_FORMAT::XDATCAR: out<<"XDATCAR"; break;
		case FILE_FORMAT::POSCAR: out<<"POSCAR"; break;
		case FILE_FORMAT::OUTCAR: out<<"OUTCAR"; break;
		case FILE_FORMAT::VASP_XML: out<<"VASP_XML"; break;
		case FILE_FORMAT::XYZ: out<<"XYZ"; break;
		case FILE_FORMAT::CP2K: out<<"CP2K"; break;
		case FILE_FORMAT::LAMMPS: out<<"LAMMPS"; break;
		case FILE_FORMAT::QE: out<<"QE"; break;
		case FILE_FORMAT::GAUSSIAN: out<<"GAUSSIAN"; break;
		case FILE_FORMAT::RAW: out<<"RAW"; break;
		case FILE_FORMAT::BINARY: out<<"BINARY"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}
