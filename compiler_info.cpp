#include <iostream>
#include "compiler.hpp"

int main(int argc, char* argv[]){

	std::cout<<"standard    = "<<compiler::standard()<<"\n";
	std::cout<<"version     = "<<compiler::version()<<"\n";
	std::cout<<"instruction = "<<compiler::instruction()<<"\n";
	#ifdef FP_FAST_FMA
	std::cout<<"fast_math   = true\n";
	#else 
	std::cout<<"fast_math   = false\n";
	#endif
	return 0;
}