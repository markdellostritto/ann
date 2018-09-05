# include directories
DIR1   = /usr/local/include/eigen-eigen-67e894c6cd8f/
INC    = $(DIR1)
# compiler 
CXXFLAGS = -fopenmp -std=gnu++11 -w -O3 $(foreach d, $(INC), -I$d)
CXX     = g++ 

# objects for final executable
objects = nn_pot.o nn.o \
		basis_radial.o symm_radial.o symm_radial_g1.o symm_radial_g2.o \
		basis_angular.o symm_angular.o symm_angular_g3.o symm_angular_g4.o \
		structure.o cell.o property.o \
		math_function.o math_special.o \
		units.o eigen.o parallel.o ptable.o string.o \
		optimize.o serialize.o statistics.o cutoff.o \

nn_pot_train_omp: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_train_omp.exe nn_pot_train_omp.cpp $(objects)
serialize.o: serialize.cpp
	$(CXX) $(CXXFLAGS) -c serialize.cpp
parallel.o: parallel.cpp
	$(CXX) $(CXXFLAGS) -c parallel.cpp
units.o: units.cpp
	$(CXX) $(CXXFLAGS) -c units.cpp
ptable.o: ptable.cpp
	$(CXX) $(CXXFLAGS) -c ptable.cpp
string.o: string.cpp
	$(CXX) $(CXXFLAGS) -c string.cpp
statistics.o: statistics.cpp
	$(CXX) $(CXXFLAGS) -c statistics.cpp
math_function.o: math_function.cpp
	$(CXX) $(CXXFLAGS) -c math_function.cpp
math_special.o: math_special.cpp math_const.hpp
	$(CXX) $(CXXFLAGS) -c math_special.cpp
eigen.o: eigen.cpp string.hpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c eigen.cpp
cutoff.o: cutoff.cpp math_const.hpp
	$(CXX) $(CXXFLAGS) -c cutoff.cpp
optimize.o: optimize.cpp math_const.hpp math_function.hpp math_cmp.hpp
	$(CXX) $(CXXFLAGS) -c optimize.cpp
symm_radial.o: symm_radial.cpp cutoff.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial.cpp
symm_radial_g1.o: symm_radial_g1.cpp symm_radial.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial_g1.cpp
symm_radial_g2.o: symm_radial_g2.cpp symm_radial.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial_g2.cpp
symm_angular.o: symm_angular.cpp cutoff.hpp
	$(CXX) $(CXXFLAGS) -c symm_angular.cpp
symm_angular_g3.o: symm_angular_g3.cpp symm_angular.hpp
	$(CXX) $(CXXFLAGS) -c symm_angular_g3.cpp
symm_angular_g4.o: symm_angular_g4.cpp symm_angular.hpp
	$(CXX) $(CXXFLAGS) -c symm_angular_g4.cpp
basis_radial.o: basis_radial.cpp cutoff.hpp symm_radial_g1.hpp symm_radial_g2.hpp
	$(CXX) $(CXXFLAGS) -c basis_radial.cpp
basis_angular.o: basis_angular.cpp cutoff.hpp symm_angular_g3.hpp symm_angular_g4.hpp
	$(CXX) $(CXXFLAGS) -c basis_angular.cpp
property.o: property.cpp serialize.hpp eigen.hpp
	$(CXX) $(CXXFLAGS) -c property.cpp
cell.o: cell.cpp math_const.hpp math_function.hpp eigen.hpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c cell.cpp
structure.o: structure.cpp cell.hpp string.hpp property.hpp ptable.hpp
	$(CXX) $(CXXFLAGS) -c structure.cpp
nn.o: nn.cpp math_const.hpp math_special.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c nn.cpp 
nn_pot.o: nn_pot.cpp atom.hpp property.hpp cell.hpp structure.hpp nn.hpp ptable.hpp parallel.hpp optimize.hpp map.hpp basis_radial.hpp basis_angular.hpp
	$(CXX) $(CXXFLAGS) -c nn_pot.cpp 

clean: 
	rm $(objects)

test: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_test.exe nn_pot_test.cpp $(objects)
	
