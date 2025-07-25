#pragma once
#ifndef VASP_HPP
#define VASP_HPP

#ifndef VASP_PRINT_FUNC
#define VASP_PRINT_FUNC 0
#endif

#ifndef VASP_PRINT_STATUS
#define VASP_PRINT_STATUS 0
#endif

#ifndef VASP_PRINT_DATA
#define VASP_PRINT_DATA 0
#endif

#ifndef __cplusplus
	#error A C++ compiler is required
#endif

namespace VASP{

//static variables
static const int HEADER_SIZE=7;//number of lines in the header before the atomic positions
static const char* NAMESPACE_GLOBAL="VASP";

}


#endif
