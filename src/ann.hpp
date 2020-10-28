#ifndef ANN_HPP
#define ANN_HPP

// ann - structure
#include "structure_fwd.hpp"

#ifndef ANN_PRINT_FUNC
#define ANN_PRINT_FUNC 0
#endif

#ifndef ANN_PRINT_STATUS
#define ANN_PRINT_STATUS 0
#endif

namespace ANN{
	
struct Format{
//==== members ====
	//basic properties
	int name;
	//serial properties
	int mass;
	int charge;
	//vector properties
	int x,y,z;
	int vx,vy,vz;
	int fx,fy,fz;
	//nnp
	int symm;
//==== constructors/destructors ====
	Format():name(-1),mass(-1),charge(-1),x(-1),y(-1),z(-1),vx(-1),vy(-1),vz(-1),fx(-1),fy(-1),fz(-1){}
	~Format(){}
};

void read(const char* file, const AtomType& atomT, Structure& struc);
void write(const char* file, const AtomType& atomT, const Structure& struc);

}

#endif