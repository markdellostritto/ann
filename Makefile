# include directories
EIGEN = /usr/local/include/eigen-eigen-67e894c6cd8f/ # eigen library
INC   = $(EIGEN)

# mpi
mpi: CXXFLAGS = -std=gnu++11 -w -O3 -mavx $(foreach d, $(INC), -I$d)
mpi: CXX = mpic++
# intel mpi
impi: CXXFLAGS = -std=c++11 -w -O3 -mavx $(foreach d, $(INC), -I$d)
impi: CXX = mpiicpc
# test
test: CXXFLAGS = -std=c++11 -w -O3 $(foreach d, $(INC), -I$d)
test: CXX = mpic++
# fit
fit: CXXFLAGS = -std=gnu++11 -w -O3 $(foreach d, $(INC), -I$d)
fit: CXX = mpic++
# convert
convert: CXXFLAGS = -std=gnu++11 -w -O3 $(foreach d, $(INC), -I$d)
convert: CXX = g++
# intgrad
intgrad: CXXFLAGS = -std=gnu++11 -w -O3 $(foreach d, $(INC), -I$d)
intgrad: CXX = g++
# compute
compute: CXXFLAGS = -std=gnu++11 -w -O3 $(foreach d, $(INC), -I$d)
compute: CXX = g++

# objects for final executable
neural = nn.o nn_pot.o cutoff.o batch.o
train = nn_train.o 
basisr = basis_radial.o symm_radial.o symm_radial_g1.o symm_radial_g2.o symm_radial_t1.o
basisa = basis_angular.o symm_angular.o symm_angular_g3.o symm_angular_g4.o
struct = structure.o cell.o cell_list.o atom.o format.o qe.o vasp.o xyz.o ann.o cp2k.o
math = math_func.o math_special.o accumulator.o optimize.o eigen.o random.o
chem = units.o ptable.o ewald3D.o
utility = string.o compiler.o print.o serialize.o map.o time.o
thread = parallel.o
objects = $(neural) $(basisr) $(basisa) $(struct) $(math) $(chem) $(utility) $(thread)

# neural 
nn.o: nn.cpp math_const.hpp math_special.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c nn.cpp
nn_pot.o: nn_pot.cpp cell.hpp structure.hpp nn.hpp ptable.hpp parallel.hpp optimize.hpp map.hpp basis_radial.hpp basis_angular.hpp
	$(CXX) $(CXXFLAGS) -c nn_pot.cpp
cutoff.o: cutoff.cpp math_const.hpp
	$(CXX) $(CXXFLAGS) -c cutoff.cpp
# train
nn_train.o: nn_train.cpp
	$(CXX) $(CXXFLAGS) -c nn_train.cpp
batch.o: batch.cpp
	$(CXX) $(CXXFLAGS) -c batch.cpp
# basis - radial
basis_radial.o: basis_radial.cpp cutoff.hpp symm_radial_g1.hpp symm_radial_g2.hpp symm_radial_t1.hpp
	$(CXX) $(CXXFLAGS) -c basis_radial.cpp
symm_radial.o: symm_radial.cpp cutoff.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial.cpp
symm_radial_g1.o: symm_radial_g1.cpp symm_radial.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial_g1.cpp
symm_radial_g2.o: symm_radial_g2.cpp symm_radial.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial_g2.cpp
symm_radial_t1.o: symm_radial_t1.cpp symm_radial.hpp
	$(CXX) $(CXXFLAGS) -c symm_radial_t1.cpp
# basis - angular
basis_angular.o: basis_angular.cpp cutoff.hpp symm_angular_g3.hpp symm_angular_g4.hpp
	$(CXX) $(CXXFLAGS) -c basis_angular.cpp
symm_angular.o: symm_angular.cpp cutoff.hpp
	$(CXX) $(CXXFLAGS) -c symm_angular.cpp
symm_angular_g3.o: symm_angular_g3.cpp symm_angular.hpp
	$(CXX) $(CXXFLAGS) -c symm_angular_g3.cpp
symm_angular_g4.o: symm_angular_g4.cpp symm_angular.hpp
	$(CXX) $(CXXFLAGS) -c symm_angular_g4.cpp
# struct
structure.o: structure.cpp cell.hpp string.hpp ptable.hpp
	$(CXX) $(CXXFLAGS) -c structure.cpp
cell.o: cell.cpp math_const.hpp math_special.hpp eigen.hpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c cell.cpp
cell_list.o: cell_list.cpp
	$(CXX) $(CXXFLAGS) -c cell_list.cpp
atom.o: atom.cpp
	$(CXX) $(CXXFLAGS) -c atom.cpp
format.o: format.cpp
	$(CXX) $(CXXFLAGS) -c format.cpp
vasp.o: vasp.cpp structure.hpp cell.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c vasp.cpp
lammps.o: lammps.cpp structure.hpp cell.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c lammps.cpp
qe.o: qe.cpp structure.hpp cell.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c qe.cpp
xyz.o: xyz.cpp structure.hpp cell.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c xyz.cpp
cp2k.o: cp2k.cpp structure.hpp cell.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c cp2k.cpp
ann.o: ann.cpp structure.hpp cell.hpp string.hpp
	$(CXX) $(CXXFLAGS) -c ann.cpp
# math
math_func.o: math_func.cpp
	$(CXX) $(CXXFLAGS) -c math_func.cpp
math_special.o: math_special.cpp math_const.hpp
	$(CXX) $(CXXFLAGS) -c math_special.cpp
accumulator.o: accumulator.cpp
	$(CXX) $(CXXFLAGS) -c accumulator.cpp
optimize.o: optimize.cpp math_const.hpp math_special.hpp math_cmp.hpp
	$(CXX) $(CXXFLAGS) -c optimize.cpp
eigen.o: eigen.cpp string.hpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c eigen.cpp
random.o: random.cpp
	$(CXX) $(CXXFLAGS) -c random.cpp
# chem
units.o: units.cpp
	$(CXX) $(CXXFLAGS) -c units.cpp
ptable.o: ptable.cpp
	$(CXX) $(CXXFLAGS) -c ptable.cpp
ewald3D.o: ewald3D.cpp structure.hpp cell.hpp math_const.hpp
	$(CXX) $(CXXFLAGS) -c ewald3D.cpp
# utility
serialize.o: serialize.cpp
	$(CXX) $(CXXFLAGS) -c serialize.cpp
map.o: map.cpp serialize.hpp
	$(CXX) $(CXXFLAGS) -c map.cpp
parallel.o: parallel.cpp
	$(CXX) $(CXXFLAGS) -c parallel.cpp
string.o: string.cpp
	$(CXX) $(CXXFLAGS) -c string.cpp
compiler.o: compiler.cpp
	$(CXX) $(CXXFLAGS) -c compiler.cpp
print.o: print.cpp
	$(CXX) $(CXXFLAGS) -c print.cpp
input.o: input.cpp
	$(CXX) $(CXXFLAGS) -c input.cpp
time.o: time.cpp
	$(CXX) $(CXXFLAGS) -c time.cpp

# targets

clean: 
	rm *.o
mpi: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_train_mpi.exe nn_pot_train_mpi.cpp $(objects)
impi: $(objects)
	$(CXX) $(CXXFLAGS) -o nn_pot_train_mpi.exe nn_pot_train_mpi.cpp $(objects)
compute: nn_pot_compute.cpp $(neural) $(basisr) $(basisa) $(struct) $(math) $(chem) $(utility)
	$(CXX) $(CXXFLAGS) -o nn_pot_compute.exe nn_pot_compute.cpp $(neural) $(basisr) $(basisa) $(struct) $(math) $(chem) $(utility)
test: $(objects)
	$(CXX) $(CXXFLAGS) -o test_unit.exe test_unit.cpp $(objects)
	$(CXX) $(CXXFLAGS) -o test_mem.exe test_mem.cpp $(objects)
	$(CXX) $(CXXFLAGS) -o test_format.exe test_format.cpp $(objects)
fit: nn_fit.cpp nn_train.o nn.o optimize.o string.o print.o eigen.o math_special.o input.o compiler.o serialize.o random.o parallel.o batch.o time.o
	$(CXX) $(CXXFLAGS) -o nn_fit.exe nn_fit.cpp nn_train.o nn.o optimize.o string.o print.o eigen.o math_special.o input.o compiler.o serialize.o random.o parallel.o batch.o time.o
convert: convert.cpp $(struct) $(chem) $(utility) eigen.o
	$(CXX) $(CXXFLAGS) -o convert.exe convert.cpp $(struct) $(chem) $(utility) eigen.o
intgrad: intgrad.cpp $(struct) $(chem) $(utility) $(neural) $(basisr) $(basisa) $(math) 
	$(CXX) $(CXXFLAGS) -o intgrad.exe intgrad.cpp $(struct) $(chem) $(utility) $(neural) $(basisr) $(basisa) $(math) 
