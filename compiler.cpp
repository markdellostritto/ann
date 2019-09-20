#include "compiler.hpp"

namespace compiler{

std::string standard(){
	std::string str;
	#ifdef __cplusplus
	if (__cplusplus == 201703L) str="C++17";
	else if (__cplusplus == 201402L) str="C++14";
	else if (__cplusplus == 201103L) str="C++11";
	else if (__cplusplus == 199711L) str="C++98";
	else str="pre-standard C++\n";
	#endif
	return str;
}

std::string version(){
	std::string str;
	//clang
	#if defined __clang__
		str="CLANG/LLVM";
	#elif (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
		str="GCC/G++";
	#elif defined __ICC || defined __INTEL_COMPILER
		str="ICC/ICPC";
	#elif defined __HP_cc || defined __HP_aCC
		str="Hewlett-Packard C/aC++";
	#elif defined __IBMC__ || defined __IBMCPP__
		str="IBM XL C/C++";
	#elif defined _MSC_VER
		str="Microsoft Visual Studio";
	#elif defined __PGI
		str="PGCC/PGCPP";
	#elif defined __SUNPRO_C || defined __SUNPRO_CC
		str="Oracle Solaris Studio";
	#endif
	return str;
}

std::string instruction(){
	std::string str;
	#if defined __AVX__
		str="AVX";
	#elif defined __AVX2__
		str="AVX2";
	#endif
	return str;
}

}