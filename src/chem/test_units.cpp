//c++
#include <iostream>
//str
#include "str/print.hpp"
//chem
#include "chem/units.hpp"

int main(int argc, char* argv[]){
	
	char* str=new char[print::len_buf];
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("FUNDAMENTAL CONSTANTS",str)<<"\n";
	std::cout<<"ALPHA    = "<<units::ALPHA<<"\n";
	std::cout<<"p/e mass = "<<units::MPoME<<"\n";
	std::cout<<print::buf(str)<<"\n";
	
	std::cout<<print::buf(str)<<"\n";
	std::cout<<print::title("CONVERSION CONSTANTS",str)<<"\n";
	std::cout<<"**** Distance ****\n";
	std::cout<<"1.0 Bohr     = "<<units::Bohr2Ang<<" Angstrom\n";
	std::cout<<"1.0 Angstrom = "<<units::Ang2Bohr<<" Bohr\n";
	std::cout<<"**** Energy ****\n";
	std::cout<<"1.0 Hartree      = "<<units::Eh2Ev<<" Electronvolt\n";
	std::cout<<"1.0 Electronvolt = "<<units::Ev2Eh<<" Hartree\n";
	std::cout<<print::buf(str)<<"\n";
	
	delete[] str;
	
	return 0;
}