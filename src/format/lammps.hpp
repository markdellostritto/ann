#pragma once
#ifndef LAMMPS_HPP
#define LAMMPS_HPP

//eigen
#include <Eigen/Dense>

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

//*****************************************************
//STYLE_ATOM struct
//*****************************************************

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

//*****************************************************
//FORMAT_ATOM struct
//*****************************************************

struct FORMAT_ATOM{
	int index,mol,type;
	int x,y,z,xu,yu,zu,q,m,fx,fy,fz,vx,vy,vz;
	FORMAT_ATOM():
		index(-1),mol(-1),type(-1),x(-1),y(-1),z(-1),xu(-1),yu(-1),zu(-1),
		q(-1),m(-1),fx(-1),fy(-1),fz(-1),vx(-1),vy(-1),vz(-1){};
};

//*****************************************************
//DATA_ATOM struct
//*****************************************************

std::ostream& operator<<(std::ostream& out, FORMAT_ATOM& f);
struct DATA_ATOM{
	int type,index;
	double q,m;
	Eigen::Vector3d posn;
	Eigen::Vector3d vel;
	Eigen::Vector3d force;
};

}

#endif
