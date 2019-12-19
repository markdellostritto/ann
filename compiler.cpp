// ann - compiler
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

std::string name(){
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

std::string version(){
	std::string str(__VERSION__);
	return str;
}

std::string date(){
	std::string str(__DATE__);
	return str;
}

std::string time(){
	std::string str(__TIME__);
	return str;
}

std::string arch(){
	std::string str;
	#ifdef defined __i386__
		str="i386"
	#elif defined __i486__
		str="i486"
	#elif defined __i586__
		str="i586"
	#elif defined __i686__
		str="i686"
	#elif defined __arm__
		str="arm";
	#elif defined __aarch64__
		str="arm64";
	#elif (defined(__amd64__) || defined(__x86_64__))
		str="amd64";
	#endif
	return str;
}

std::string instr(){
	std::string str;
	#if defined __AVX__
		str="AVX";
	#elif defined __AVX2__
		str="AVX2";
	#endif
	return str;
}

std::string omp(){
	std::string str="FALSE";
	#ifdef _OPENMP
		str="TRUE";
	#endif
	return str;
}

std::string os(){
	std::string str;
	#if (defined(__WIN32) || defined(__WIN64))
		str="windows";
	#elif defined __CYGWIN__
		str="cygwin";
	#elif (defined(__APPLE__) || defined(__MACH__))
		str="mac";
	#elif defined __linux__
		str="linux";
	#elif defined __unix__
		str="unix";
	#endif
	return str;
}

}