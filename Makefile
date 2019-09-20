# include directories
EIGEN = /usr/local/include/eigen-eigen-67e894c6cd8f/ # eigen library
INC   = $(EIGEN)

# openmp
omp: CXXFLAGS = -fopenmp -std=gnu++11 -w -O3 $(foreach d, $(INC), -I$d)
omp: CXX = g++
# mpi
mpi: CXXFLAGS = -std=gnu++11 -w -O3 -mavx $(foreach d, $(INC), -I$d)
mpi: CXX = mpic++
# intel openmp
iomp: CXXFLAGS = -qopenmp -std=c++11 -w -O3 $(foreach d, $(INC), -I$d)
iomp: CXX = icpc
# intel mpi
impi: CXXFLAGS = -std=c++11 -w -O3 -mavx $(foreach d, $(INC), -I$d)
impi: CXX = mpiicpc
# test
test: CXXFLAGS = -fopenmp -std=gnu++11 -w -O3 $(foreach d, $(INC), -I$d)
test: CXX = g++
# convert
convert: CXXFLAGS = -std=gnu++11 -w -O3 $(foreach d, $(INC), -I$d)
convert: CXX = g++

# objects for final executable
objects = nn_pot.o nn.o \
		basis_radial.o symm_radial.o symm_radial_g1.o symm_radial_g2.o symm_radial_t1.o \
		basis_angular.o symm_angular.o symm_angular_g3.o symm_angular_g4.o \
		structure.o cell.o qe.o vasp.o ame.o atom.o \
		math_func.o math_special.o \
		ewald3D.o \
		units.o eigen.o parallel.o ptable.o string.o compiler.o \
		optimize.o serialize.o cutoff.o map.o accumulator.o \

nn_pot_train_omp: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_train_omp.exe nn_pot_train_omp.cpp $(objects)
accumulator.o: accumulator.cpp
	$(CXX) $(CXXFLAGS) -c accumulator.cpp
serialize.o: serialize.cpp
	$(CXX) $(CXXFLAGS) -c serialize.cpp
map.o: map.cpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c map.cpp
parallel.o: parallel.cpp
	$(CXX) $(CXXFLAGS) -c parallel.cpp
units.o: units.cpp
	$(CXX) $(CXXFLAGS) -c units.cpp
ptable.o: ptable.cpp
	$(CXX) $(CXXFLAGS) -c ptable.cpp
string.o: string.cpp
	$(CXX) $(CXXFLAGS) -c string.cpp
compiler.o: compiler.cpp
	$(CXX) $(CXXFLAGS) -c compiler.cpp
math_func.o: math_func.cpp
	$(CXX) $(CXXFLAGS) -c math_func.cpp
math_special.o: math_special.cpp math_const.hpp
	$(CXX) $(CXXFLAGS) -c math_special.cpp
eigen.o: eigen.cpp string.hpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c eigen.cpp
cutoff.o: cutoff.cpp math_const.hpp
	$(CXX) $(CXXFLAGS) -c cutoff.cpp
optimize.o: optimize.cpp math_const.hpp math_special.hpp math_cmp.hpp
	$(CXX) $(CXXFLAGS) -c optimize.cpp
symm_radial.o: symm_radial.cpp cutoff.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial.cpp
symm_radial_g1.o: symm_radial_g1.cpp symm_radial.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial_g1.cpp
symm_radial_g2.o: symm_radial_g2.cpp symm_radial.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial_g2.cpp
symm_radial_t1.o: symm_radial_t1.cpp symm_radial.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial_t1.cpp
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
cell.o: cell.cpp math_const.hpp math_special.hpp eigen.hpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c cell.cpp
structure.o: structure.cpp cell.hpp string.hpp ptable.hpp
	$(CXX) $(CXXFLAGS) -c structure.cpp
atom.o: atom.cpp
	$(CXX) $(CXXFLAGS) -c atom.cpp
ewald3D.o: ewald3D.cpp structure.hpp cell.hpp math_const.hpp
	$(CXX) $(CXXFLAGS) -c ewald3D.cpp
vasp.o: vasp.cpp structure.hpp cell.hpp string.hpp 
	$(CXX) $(CXXFLAGS) -c vasp.cpp
lammps.o: lammps.cpp structure.hpp cell.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c lammps.cpp
qe.o: qe.cpp structure.hpp cell.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c qe.cpp
ame.o: ame.cpp structure.hpp cell.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c ame.cpp
nn.o: nn.cpp math_const.hpp math_special.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c nn.cpp 
nn_pot.o: nn_pot.cpp cell.hpp structure.hpp nn.hpp ptable.hpp parallel.hpp optimize.hpp map.hpp basis_radial.hpp basis_angular.hpp
	$(CXX) $(CXXFLAGS) -c nn_pot.cpp 

# targets

clean: 
	rm $(objects)

omp: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_train_omp.exe nn_pot_train_omp.cpp $(objects)

mpi: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_train_mpi.exe nn_pot_train_mpi.cpp $(objects)

iomp: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_train_omp.exe nn_pot_train_omp.cpp $(objects)

impi: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_train_mpi.exe nn_pot_train_mpi.cpp $(objects)

test: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_test.exe nn_pot_test.cpp $(objects)

convert: convert.cpp structure.o eigen.o serialize.o units.o ptable.o vasp.o qe.o ame.o string.o cell.o
	$(CXX) $(CXXFLAGS) -o convert.exe convert.cpp structure.o cell.o string.o eigen.o serialize.o units.o ptable.o vasp.o qe.o ame.o
