#pragma once
#ifndef AME_HPP
#define AME_HPP

// eigen libraries
#include <Eigen/Dense>
// ann - structure
#include "struc/structure_fwd.hpp"

#ifndef AME_PRINT_FUNC
#define AME_PRINT_FUNC 0
#endif

#ifndef AME_PRINT_STATUS
#define AME_PRINT_STATUS 0
#endif

#ifndef AME_PRINT_DATA
#define AME_PRINT_DATA 0
#endif

#ifndef __cplusplus
	#error A C++ compiler is required
#endif

namespace AME{
	
//*****************************************************
//FORMAT struct
//*****************************************************

struct FORMAT_ATOM{
	int name,rx,ry,rz,fx,fy,fz,vx,vy,vz,q,m,c6;
	FORMAT_ATOM():name(-1),rx(-1),ry(-1),rz(-1),fx(-1),fy(-1),fz(-1),vx(-1),vy(-1),vz(-1),q(-1),m(-1),c6(-1){};
};

//*****************************************************
//reading
//*****************************************************

void read(const char* file, const AtomType& atomT, Structure& struc);

//*****************************************************
//writing
//*****************************************************

void write(const char* file, const AtomType& atomT, const Structure& struc);

}

#endif