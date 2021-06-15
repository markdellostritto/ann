##################################################################################
# EXTERNAL LIBRARY PATHS
##################################################################################
EIGEN = /usr/local/include/eigen-3.4-rc1/ # eigen library
INC_LIST = $(EIGEN)
INC = $(foreach d, $(INC_LIST), -I$d)

##################################################################################
# GENERAL SETTINGS
##################################################################################
COMP=gnu

# Include global settings.
include Makefile.$(COMP)

##################################################################################
# OBJECT FILES
##################################################################################
neural = nn.o nn_pot.o cutoff.o batch.o
train = nn_train.o 
basisr = basis_radial.o symm_radial.o symm_radial_g1.o symm_radial_g2.o symm_radial_t1.o
basisa = basis_angular.o symm_angular.o symm_angular_g3.o symm_angular_g4.o
struct = structure.o cell.o cell_list.o atom.o
format = format.o qe.o vasp.o xyz.o ann.o cp2k.o
math = math_func.o math_special.o accumulator.o optimize.o eigen.o random.o
chem = units.o ptable.o ewald3D.o
io = string.o print.o 
utility = compiler.o serialize.o map.o time.o
thread = parallel.o
objects = $(neural) $(basisr) $(basisa) $(struct) $(format) $(math) $(chem) $(io) $(utility) $(thread)

##################################################################################
# OBJECT FILE MAKE RULES
##################################################################################

# neural 
nn.o: nn.cpp math_const.hpp math_special.hpp string.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nn.cpp
nn_pot.o: nn_pot.cpp cell.hpp structure.hpp nn.hpp ptable.hpp parallel.hpp optimize.hpp map.hpp basis_radial.hpp basis_angular.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nn_pot.cpp
cutoff.o: cutoff.cpp math_const.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c cutoff.cpp
# train
nn_train.o: nn_train.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c nn_train.cpp
nn_train_ekf.o: nn_train_ekf.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c nn_train_ekf.cpp
batch.o: batch.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c batch.cpp
ekf.o: ekf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ekf.cpp
# basis - radial
basis_radial.o: basis_radial.cpp cutoff.hpp symm_radial_g1.hpp symm_radial_g2.hpp symm_radial_t1.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c basis_radial.cpp
symm_radial.o: symm_radial.cpp cutoff.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c symm_radial.cpp
symm_radial_g1.o: symm_radial_g1.cpp symm_radial.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c symm_radial_g1.cpp
symm_radial_g2.o: symm_radial_g2.cpp symm_radial.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c symm_radial_g2.cpp
symm_radial_t1.o: symm_radial_t1.cpp symm_radial.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c symm_radial_t1.cpp
# basis - angular
basis_angular.o: basis_angular.cpp cutoff.hpp symm_angular_g3.hpp symm_angular_g4.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c basis_angular.cpp
symm_angular.o: symm_angular.cpp cutoff.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c symm_angular.cpp
symm_angular_g3.o: symm_angular_g3.cpp symm_angular.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c symm_angular_g3.cpp
symm_angular_g4.o: symm_angular_g4.cpp symm_angular.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c symm_angular_g4.cpp
# struct
structure.o: structure.cpp cell.hpp string.hpp ptable.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c structure.cpp
sim.o: sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c sim.cpp
cell.o: cell.cpp math_const.hpp math_special.hpp eigen.hpp serialize.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c cell.cpp
cell_list.o: cell_list.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c cell_list.cpp
atom.o: atom.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c atom.cpp
# format
format.o: format.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format.cpp
vasp.o: vasp.cpp structure.hpp cell.hpp string.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c vasp.cpp
lammps.o: lammps.cpp structure.hpp cell.hpp string.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c lammps.cpp
qe.o: qe.cpp structure.hpp cell.hpp string.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c qe.cpp
xyz.o: xyz.cpp structure.hpp cell.hpp string.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c xyz.cpp
cp2k.o: cp2k.cpp structure.hpp cell.hpp string.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c cp2k.cpp
ann.o: ann.cpp structure.hpp cell.hpp string.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ann.cpp
config.o: config.cpp structure.hpp cell.hpp string.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c config.cpp
# math
math_func.o: math_func.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math_func.cpp
math_special.o: math_special.cpp math_const.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math_special.cpp
accumulator.o: accumulator.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c accumulator.cpp
optimize.o: optimize.cpp math_const.hpp math_special.hpp math_cmp.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c optimize.cpp
eigen.o: eigen.cpp string.hpp serialize.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c eigen.cpp
random.o: random.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c random.cpp
# chem
units.o: units.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c units.cpp
ptable.o: ptable.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ptable.cpp
ewald3D.o: ewald3D.cpp structure.hpp cell.hpp math_const.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ewald3D.cpp
# utility
serialize.o: serialize.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c serialize.cpp
map.o: map.cpp serialize.hpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c map.cpp
parallel.o: parallel.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c parallel.cpp
string.o: string.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c string.cpp
compiler.o: compiler.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c compiler.cpp
print.o: print.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c print.cpp
input.o: input.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c input.cpp
time.o: time.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c time.cpp

##################################################################################
# TARGETS
##################################################################################

clean: 
	rm *.o
test: $(objects) sim.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_unit.exe test_unit.cpp $(objects) sim.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_mem.exe test_mem.cpp $(objects)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_format.exe test_format.cpp $(objects)
train: $(objects) sim.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nn_pot_train_mpi.exe nn_pot_train_mpi.cpp $(objects) sim.o
train_ekf: $(objects) ekf.o sim.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nn_pot_train_ekf.exe nn_pot_train_ekf.cpp $(objects) ekf.o sim.o
fit: nn_fit.cpp nn_train.o nn.o optimize.o string.o print.o eigen.o math_special.o input.o compiler.o serialize.o random.o parallel.o batch.o time.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nn_fit.exe nn_fit.cpp nn_train.o nn.o optimize.o string.o print.o eigen.o math_special.o input.o compiler.o serialize.o random.o parallel.o batch.o time.o
fit_ekf: nn_fit_ekf.cpp nn_train_ekf.o nn.o optimize.o string.o print.o eigen.o math_special.o input.o compiler.o ekf.o serialize.o random.o parallel.o batch.o time.o 
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nn_fit_ekf.exe nn_fit_ekf.cpp nn_train_ekf.o nn.o optimize.o string.o print.o eigen.o math_special.o input.o compiler.o ekf.o serialize.o random.o parallel.o batch.o time.o
compute: nn_pot_compute.cpp $(neural) $(basisr) $(basisa) $(struct) $(math) $(chem) $(utility) $(thread)
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nn_pot_compute.exe nn_pot_compute.cpp $(neural) $(basisr) $(basisa) $(struct) $(math) $(chem) $(utility) $(thread)
compute_sim: nn_pot_compute_sim.cpp $(neural) $(basisr) $(basisa) $(struct) $(math) $(chem) $(utility) $(thread) $(io) sim.o vasp.o format.o xyz.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nn_pot_compute_sim.exe nn_pot_compute_sim.cpp $(neural) $(basisr) $(basisa) $(struct) $(math) $(chem) $(utility) $(thread) $(io) sim.o vasp.o format.o xyz.o
intgrad: intgrad.cpp $(struct) $(chem) $(utility) $(neural) $(basisr) $(basisa) $(math) 
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o intgrad.exe intgrad.cpp $(struct) $(chem) $(utility) $(neural) $(basisr) $(basisa) $(math) 
convert: convert.cpp $(struct) $(chem) $(utility) eigen.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o convert.exe convert.cpp $(struct) $(chem) $(utility) eigen.o
qe2config: qe2config.cpp config.o qe.o eigen.o $(chem) $(utility) structure.o cell.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o qe2config.exe qe2config.cpp config.o qe.o eigen.o $(chem) $(utility) structure.o cell.o
qe2raw: qe2raw.cpp qe.o eigen.o $(chem) $(utility) structure.o cell.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o qe2raw.exe qe2raw.cpp qe.o eigen.o $(chem) $(utility) structure.o cell.o

