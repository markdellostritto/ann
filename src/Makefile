##################################################################################
# EXTERNAL LIBRARY PATHS
##################################################################################
EIGEN = /usr/local/include/eigen-3.4-rc1/ # eigen library
SRC  := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INC_LIST = $(EIGEN) $(SRC) 
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
neural = batch.o data.o nn.o nn_train.o
nnp = nnh.o nnp.o cutoff.o type.o
basis = basis.o basis_radial.o basis_angular.o 
struc = structure.o cell.o neighbor.o state.o atom_type.o
format = format.o lammps.o
format_struc = file_struc.o qe_struc.o vasp_struc.o xyz_struc.o cp2k_struc.o raw_struc.o gauss_struc.o
format_sim = file_sim.o vasp_sim.o lammps_sim.o xyz_sim.o cp2k_sim.o interval.o
math = special.o eigen.o reduce.o corr.o poly.o 
rand = rng.o
opt = loss.o stop.o algo.o objective.o decay.o iter.o
chem = units.o ptable.o 
str = string.o print.o token.o
util = compiler.o time.o
mem = serialize.o map.o
thread = comm.o dist.o
entropy = entropy_rho.o rho_atomic.o
signal = fft.o window.o
pot = pot.o pot_factory.o pot_coul_cut.o pot_coul_wolf.o pot_coul_dsf.o pot_coul_long.o pot_gauss_cut.o pot_gauss_dsf.o pot_gauss_long.o pot_pauli.o pot_lj_cut.o pot_lj_long.o pot_lj_sm.o pot_ldamp_cut.o pot_ldamp_long.o pot_ldamp_dsf.o pot_spin_ex.o pot_qeq_gl.o pot_nnpe.o 
kspace = kspace.o kspace_coul.o kspace_london.o
analyze = pair.o group.o compute.o shell.o hbond.o rdf.o distmat.o avg_posn.o
property = set_property.o set_property_type.o set_property_mass.o set_property_charge.o set_property_velocity.o set_property_temp.o set_property_factory.o
objects = $(neural) $(basis) $(struct) $(format) $(math) $(chem) $(str) $(util) $(thread)

##################################################################################
# OBJECT FILE MAKE RULES
##################################################################################

# machine learning
nn.o: ml/nn.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ml/nn.cpp
batch.o: ml/batch.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ml/batch.cpp
data.o: ml/data.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ml/data.cpp
nn_train.o: ml/nn_train.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c ml/nn_train.cpp
pca.o: ml/pca.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c ml/pca.cpp
# nnp
nnh.o: nnp/nnh.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/nnh.cpp
nnp.o: nnp/nnp.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/nnp.cpp
type.o: nnp/type.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/type.cpp
cutoff.o: nnp/cutoff.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/cutoff.cpp
# nnp - basis
basis.o: nnp/basis.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/basis.cpp
basis_radial.o: nnp/basis_radial.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/basis_radial.cpp
basis_angular.o: nnp/basis_angular.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/basis_angular.cpp
basis_dihedral.o: nnp/basis_dihedral.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c nnp/basis_dihedral.cpp
# struc
structure.o: struc/structure.cpp 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/structure.cpp
interval.o: struc/interval.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/interval.cpp
sim.o: struc/sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/sim.cpp
cell.o: struc/cell.cpp 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/cell.cpp
neighbor.o: struc/neighbor.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/neighbor.cpp
verlet.o: struc/verlet.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/verlet.cpp
pair.o: struc/pair.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/pair.cpp
atom_type.o: struc/atom_type.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/atom_type.cpp
state.o: struc/state.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/state.cpp
grid.o: struc/grid.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c struc/grid.cpp
# format
file_struc.o: format/file_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/file_struc.cpp
file_sim.o: format/file_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/file_sim.cpp
format.o: format/format.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/format.cpp
vasp_struc.o: format/vasp_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/vasp_struc.cpp
vasp_sim.o: format/vasp_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/vasp_sim.cpp
lammps.o: format/lammps.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/lammps.cpp
lammps_sim.o: format/lammps_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/lammps_sim.cpp
xyz_sim.o: format/xyz_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/xyz_sim.cpp
cp2k_sim.o: format/cp2k_sim.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/cp2k_sim.cpp
qe_struc.o: format/qe_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/qe_struc.cpp
xyz_struc.o: format/xyz_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/xyz_struc.cpp
raw_struc.o: format/raw_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/raw_struc.cpp
cp2k_struc.o: format/cp2k_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/cp2k_struc.cpp
gauss_struc.o: format/gauss_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/gauss_struc.cpp
cube_struc.o: format/cube_struc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c format/cube_struc.cpp
# math
special.o: math/special.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/special.cpp
euclid.o: math/euclid.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/euclid.cpp
interp.o: math/interp.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/interp.cpp
poly.o: math/poly.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/poly.cpp
rbf.o: math/rbf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/rbf.cpp
accumulator.o: math/accumulator.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/accumulator.cpp
corr.o: math/corr.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/corr.cpp
reduce.o: math/reduce.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/reduce.cpp
eigen.o: math/eigen.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/eigen.cpp
hist.o: math/hist.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/hist.cpp
graph.o: math/graph.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c math/graph.cpp
# rand
mc.o: rand/mc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c rand/mc.cpp
rng.o: rand/rng.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c rand/rng.cpp
# optimization
loss.o: opt/loss.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/loss.cpp
stop.o: opt/stop.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/stop.cpp
algo.o: opt/algo.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/algo.cpp
decay.o: opt/decay.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/decay.cpp
objective.o: opt/objective.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/objective.cpp
iter.o: opt/iter.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/iter.cpp
optimize.o: opt/optimize.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c opt/optimize.cpp
# chem
units.o: chem/units.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/units.cpp
ptable.o: chem/ptable.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/ptable.cpp
molkit.o: chem/molkit.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/molkit.cpp
molkit2.o: chem/molkit2.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/molkit2.cpp
alias.o: chem/alias.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/alias.cpp
moldata.o: chem/moldata.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c chem/moldata.cpp
# qchem
dmc.o: qchem/dmc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c qchem/dmc.cpp
sdmc.o: qchem/sdmc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c qchem/sdmc.cpp
pdmc.o: qchem/pdmc.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c qchem/pdmc.cpp
qdo.o: qchem/qdo.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c qchem/qdo.cpp
# memory
serialize.o: mem/serialize.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c mem/serialize.cpp
map.o: mem/map.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c mem/map.cpp
# thread
dist.o: thread/dist.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c thread/dist.cpp
comm.o: thread/comm.cpp
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -c thread/comm.cpp
# string
string.o: str/string.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/string.cpp
print.o: str/print.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/print.cpp
token.o: str/token.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/token.cpp
parse.o: str/parse.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c str/parse.cpp
# utility
compiler.o: util/compiler.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c util/compiler.cpp
time.o: util/time.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c util/time.cpp
# entropy
rho_atomic.o: entropy/rho_atomic.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c entropy/rho_atomic.cpp
# signal
fft.o: signal/fft.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c signal/fft.cpp -lfftw3
window.o: signal/window.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c signal/window.cpp
# analysis
group.o: analysis/group.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c analysis/group.cpp
# torch
job.o: torch/job.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/job.cpp
drude.o: torch/drude.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/drude.cpp
qeq.o: torch/qeq.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/qeq.cpp
qtpie.o: torch/qtpie.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/qtpie.cpp
set_property.o: torch/set_property.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property.cpp
set_property_type.o: torch/set_property_type.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_type.cpp
set_property_mass.o: torch/set_property_mass.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_mass.cpp
set_property_charge.o: torch/set_property_charge.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_charge.cpp
set_property_velocity.o: torch/set_property_velocity.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_velocity.cpp
set_property_temp.o: torch/set_property_temp.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_temp.cpp
set_property_factory.o: torch/set_property_factory.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/set_property_factory.cpp
pot.o: torch/pot.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot.cpp
pot_factory.o: torch/pot_factory.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_factory.cpp
pot_coul_cut.o: torch/pot_coul_cut.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_coul_cut.cpp
pot_coul_wolf.o: torch/pot_coul_wolf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_coul_wolf.cpp
pot_coul_dsf.o: torch/pot_coul_dsf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_coul_dsf.cpp
pot_coul_long.o: torch/pot_coul_long.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_coul_long.cpp
pot_gauss_cut.o: torch/pot_gauss_cut.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_gauss_cut.cpp
pot_gauss_long.o: torch/pot_gauss_long.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_gauss_long.cpp
pot_gauss_dsf.o: torch/pot_gauss_dsf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_gauss_dsf.cpp
pot_qeq_gl.o: torch/pot_qeq_gl.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_qeq_gl.cpp
pot_pauli.o: torch/pot_pauli.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_pauli.cpp 
pot_pslater.o: torch/pot_pslater.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_pslater.cpp
pot_lj_cut.o: torch/pot_lj_cut.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_lj_cut.cpp 
pot_lj_long.o: torch/pot_lj_long.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_lj_long.cpp
pot_lj_sm.o: torch/pot_lj_sm.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_lj_sm.cpp
pot_ldamp_cut.o: torch/pot_ldamp_cut.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_ldamp_cut.cpp
pot_ldamp_long.o: torch/pot_ldamp_long.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_ldamp_long.cpp
pot_ldamp_dsf.o: torch/pot_ldamp_dsf.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_ldamp_dsf.cpp
pot_spin_ex.o: torch/pot_spin_ex.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_spin_ex.cpp
pot_nnpe.o: torch/pot_nnpe.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/pot_nnpe.cpp
kspace.o: torch/kspace.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/kspace.cpp
kspace_coul.o: torch/kspace_coul.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/kspace_coul.cpp
kspace_london.o: torch/kspace_london.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/kspace_london.cpp
engine.o: torch/engine.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/engine.cpp
monte_carlo.o: torch/monte_carlo.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/monte_carlo.cpp
integrator.o: torch/integrator.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/integrator.cpp
dump.o: torch/dump.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/dump.cpp
stochastic.o: torch/stochastic.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/stochastic.cpp
thole.o: torch/thole.cpp
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -c torch/thole.cpp

##################################################################################
# TARGETS
##################################################################################

clean: 
	rm *.o

#==== UNIT TESTS ====

#** optimization**
test_opt_algo: $(opt) $(str) serialize.o eigen.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_opt_algo.exe opt/test_opt_algo.cpp $(opt) $(str) serialize.o eigen.o
test_opt_loss: $(opt) $(str) serialize.o eigen.o reduce.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_opt_loss.exe opt/test_opt_loss.cpp $(opt) $(str) serialize.o eigen.o reduce.o
#** mem **
test_tensor: 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_tensor.exe mem/test_tensor.cpp
test_tensortri:
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_tensortri.exe mem/test_tensortri.cpp
#** math **
test_math_special: time.o special.o poly.o reduce.o print.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_math_special.exe math/test_math_special.cpp time.o special.o poly.o reduce.o print.o
test_math_eigen: print.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_math_eigen.exe math/test_math_eigen.cpp print.o
test_math_euclid: euclid.o print.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_math_euclid.exe math/test_math_euclid.cpp euclid.o print.o
test_math_interp: time.o interp.o print.o rbf.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_math_interp.exe math/test_math_interp.cpp time.o interp.o print.o rbf.o
test_math_poly: time.o poly.o special.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_math_poly.exe math/test_math_poly.cpp time.o poly.o special.o
test_fft_unit: fft.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_fft_unit.exe signal/test_fft_unit.cpp fft.o -lfftw3
test_smooth: fft.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_smooth.exe signal/test_smooth.cpp fft.o -lfftw3
test_math_reduce: reduce.o print.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_math_reduce.exe math/test_math_reduce.cpp reduce.o print.o
test_density_gauss: print.o rng.o special.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_density_gauss.exe math/test_density_gauss.cpp print.o rng.o special.o
test_ylm: $(struc) $(math) $(mem) $(str) grid.o cube_struc.o $(chem)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_ylm.exe math/test_ylm.cpp $(struc) $(math) $(mem) $(str) grid.o cube_struc.o $(chem)
#** rand **
test_rand_rng: rng.o hist.o special.o $(str)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_rand_rng.exe rand/test_rand_rng.cpp rng.o hist.o special.o $(str)
test_rand_mc: rng.o reduce.o mc.o special.o $(str)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_rand_mc.exe rand/test_rand_mc.cpp rng.o reduce.o mc.o special.o $(str)
#** machine learning **
test_nn_unit: nn.o string.o serialize.o eigen.o print.o special.o rng.o token.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nn_unit.exe ml/test_nn_unit.cpp nn.o string.o serialize.o eigen.o print.o special.o rng.o token.o
test_nn_neuron: nn.o string.o serialize.o eigen.o print.o special.o rng.o token.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nn_neuron.exe ml/test_nn_neuron.cpp nn.o string.o serialize.o eigen.o print.o special.o rng.o token.o
test_nn_grad: nn.o string.o serialize.o eigen.o print.o special.o rng.o token.o reduce.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nn_grad.exe ml/test_nn_grad.cpp nn.o string.o serialize.o eigen.o print.o special.o rng.o token.o reduce.o
test_nn_cost: nn.o string.o serialize.o eigen.o print.o special.o rng.o token.o reduce.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nn_cost.exe ml/test_nn_cost.cpp nn.o string.o serialize.o eigen.o print.o special.o rng.o token.o reduce.o
test_pca_gauss: pca.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pca_gauss.exe ml/test_pca_gauss.cpp pca.o
#** torch **
test_kspace_coul: $(struc) $(str) $(math) $(mem) $(kspace) $(chem) pot_coul_long.o pot.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_kspace_coul.exe torch/test_kspace_coul.cpp $(struc) $(str) $(math) $(mem) $(kspace) $(chem) pot_coul_long.o pot.o
test_kspace_london: $(struc) $(str) $(chem) $(kspace) $(pot) $(thread) $(basis) $(nnp) $(math) nn.o $(mem)
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o test_kspace_london.exe torch/test_kspace_london.cpp $(struc) $(str) $(chem) $(kspace) $(pot) $(thread) $(basis) $(nnp) $(math) nn.o $(mem)
test_pot_pauli: $(struc) $(str) eigen.o serialize.o units.o pot_pauli.o pot.o special.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_pauli.exe torch/test_pot_pauli.cpp $(struc) $(str) eigen.o serialize.o units.o pot_pauli.o pot.o special.o
test_pot_pslater: $(struc) $(str) eigen.o serialize.o units.o pot_pslater.o pot.o special.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_pslater.exe torch/test_pot_pslater.cpp $(struc) $(str) eigen.o serialize.o units.o pot_pslater.o pot.o special.o
test_pot_lj_cut: $(struc) $(str) eigen.o serialize.o units.o pot_lj_cut.o pot.o special.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_lj_cut.exe torch/test_pot_lj_cut.cpp $(struc) $(str) eigen.o serialize.o units.o pot_lj_cut.o pot.o special.o
test_pot_lj_sm: $(struc) $(str) eigen.o serialize.o units.o pot_lj_sm.o pot.o special.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_lj_sm.exe torch/test_pot_lj_sm.cpp $(struc) $(str) eigen.o serialize.o units.o pot_lj_sm.o pot.o special.o
test_pot_ldamp_dsf: $(struc) $(str) eigen.o serialize.o units.o pot_ldamp_dsf.o pot.o special.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_ldamp_dsf.exe torch/test_pot_ldamp_dsf.cpp $(struc) $(str) eigen.o serialize.o units.o pot_ldamp_dsf.o pot.o special.o
test_pot_coul_cut: $(struc) $(str) eigen.o serialize.o units.o pot_coul_cut.o pot.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_coul_cut.exe torch/test_pot_coul_cut.cpp $(struc) $(str) eigen.o serialize.o units.o pot_coul_cut.o pot.o
test_pot_coul_dsf: $(struc) $(str) eigen.o serialize.o units.o pot_coul_dsf.o pot.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_coul_dsf.exe torch/test_pot_coul_dsf.cpp $(struc) $(str) eigen.o serialize.o units.o pot_coul_dsf.o pot.o
test_pot_coul_long: $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_coul.o pot_coul_long.o pot_coul_cut.o pot.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_coul_long.exe torch/test_pot_coul_long.cpp $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_coul.o pot_coul_long.o pot_coul_cut.o pot.o
test_pot_gauss_cut: $(struc) $(str) eigen.o serialize.o units.o pot_gauss_cut.o pot.o ptable.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_gauss_cut.exe torch/test_pot_gauss_cut.cpp $(struc) $(str) eigen.o serialize.o units.o pot_gauss_cut.o pot.o ptable.o
test_pot_gauss_long: $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_coul.o pot_gauss_long.o pot_coul_long.o pot.o ptable.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_gauss_long.exe torch/test_pot_gauss_long.cpp $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_coul.o pot_gauss_long.o pot_coul_long.o pot.o ptable.o
test_thole: thole.o $(struc) $(str) eigen.o serialize.o units.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_thole.exe torch/test_thole.cpp thole.o $(struc) $(str) eigen.o serialize.o units.o
test_pot_gauss_dsf: $(struc) $(str) eigen.o serialize.o units.o pot_gauss_dsf.o pot_coul_dsf.o pot.o ptable.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_gauss_dsf.exe torch/test_pot_gauss_dsf.cpp $(struc) $(str) eigen.o serialize.o units.o pot_gauss_dsf.o pot_coul_dsf.o pot.o ptable.o
test_pot_ldamp_long: $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_london.o pot_ldamp_long.o pot.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pot_ldamp_long.exe torch/test_pot_ldamp_long.cpp $(struc) $(str) eigen.o serialize.o units.o kspace.o kspace_london.o pot_ldamp_long.o pot.o
test_torch_force: $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) $(property) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o job.o group.o verlet.o rng.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_torch_force.exe torch/test_torch_force.cpp $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) $(property) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o job.o group.o verlet.o rng.o
#** nnp **
test_nnp_symm: $(basis) $(mem) $(struc) $(str) nnh.o nnp.o nn.o eigen.o special.o rng.o cutoff.o type.o poly.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nnp_symm.exe nnp/test_nnp_symm.cpp $(basis) $(mem) $(struc) $(str) nnh.o nnp.o nn.o eigen.o special.o rng.o cutoff.o type.o poly.o
test_nnp_force: $(basis) $(mem) $(struc) $(str) nnh.o nnp.o nn.o eigen.o special.o rng.o cutoff.o type.o poly.o reduce.o $(format_struc) $(chem)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nnp_force.exe nnp/test_nnp_force.cpp $(basis) $(mem) $(struc) $(str) nnh.o nnp.o nn.o eigen.o special.o rng.o cutoff.o type.o poly.o reduce.o $(format_struc) $(chem)
test_struc_unit: $(struc) $(str) eigen.o serialize.o units.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_struc_unit.exe struc/test_struc_unit.cpp $(struc) $(str) eigen.o serialize.o units.o
test_neigh_unit: $(struc) $(str) eigen.o serialize.o units.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_neigh_unit.exe struc/test_neigh_unit.cpp $(struc) $(str) eigen.o serialize.o units.o
test_nnp_cutoff: $(mem) $(str) $(math) basis.o cutoff.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nnp_cutoff.exe nnp/test_nnp_cutoff.cpp $(mem) $(str) $(math) basis.o cutoff.o
test_basis_radial: $(mem) $(str) $(math) basis.o basis_radial.o cutoff.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_basis_radial.exe nnp/test_basis_radial.cpp $(mem) $(str) $(math) basis.o basis_radial.o cutoff.o
test_basis_angular: $(mem) $(str) $(math) basis.o basis_angular.o cutoff.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_basis_angular.exe nnp/test_basis_angular.cpp $(mem) $(str) $(math) basis.o basis_angular.o cutoff.o
test_omp_out:
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -fopenmp -o test_omp_out.exe omp/test_omp_out.cpp
test_omp_for: time.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -fopenmp -o test_omp_for.exe omp/test_omp_for.cpp time.o
test_omp_eigen: 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -fopenmp -o test_omp_eigen.exe omp/test_omp_eigen.cpp 
test_perturb: special.o rng.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_perturb.exe nnp/test_perturb.cpp rng.o special.o
# chem
test_units: units.o print.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_units.exe chem/test_units.cpp units.o print.o
# qchem
test_sdmc_d1e1: units.o sdmc.o dmc.o rng.o print.o hist.o $(math)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_sdmc_d1e1.exe qchem/test_sdmc_d1e1.cpp units.o sdmc.o dmc.o rng.o print.o hist.o $(math)
test_sdmc_d1e2: units.o sdmc.o dmc.o rng.o print.o hist.o $(math)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_sdmc_d1e2.exe qchem/test_sdmc_d1e2.cpp units.o sdmc.o dmc.o rng.o print.o hist.o $(math)
test_sdmc_d3e1: units.o sdmc.o dmc.o rng.o print.o hist.o $(math)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_sdmc_d3e1.exe qchem/test_sdmc_d3e1.cpp units.o sdmc.o dmc.o rng.o print.o hist.o $(math)
test_pdmc_d1e1: units.o pdmc.o rng.o print.o hist.o $(math)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pdmc_d1e1.exe qchem/test_pdmc_d1e1.cpp units.o pdmc.o rng.o print.o hist.o $(math)
test_pdmc_d3e1: units.o pdmc.o rng.o print.o hist.o $(math)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_pdmc_d3e1.exe qchem/test_pdmc_d3e1.cpp units.o pdmc.o rng.o print.o hist.o $(math)
sqdflow: units.o dmc.o sdmc.o rng.o qdo.o $(str) $(math)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o sqdflow.exe qchem/sqdflow.cpp units.o dmc.o sdmc.o rng.o qdo.o $(str) $(math)
# string
test_token: $(str)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_token.exe str/test_token.cpp $(str)
# nlopt
test_nlopt: 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o test_nlopt.exe opt/test_nlopt.cpp -lnlopt

#==== NN OPTIMIZATION ====

nn_fit: $(neural) $(str) $(math) $(util) $(thread) $(opt) serialize.o rng.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nn_fit.exe ml/nn_fit.cpp $(neural) $(str) $(math) $(util) $(thread) $(opt) serialize.o rng.o

#==== NNP OPTIMIZATION ====

symm: $(nnp) $(str) $(util) $(mem) $(basis) $(struct) $(format_struc) $(struc) $(chem) nn.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o symm.exe nnp/symm.cpp $(nnp) $(str) $(util) $(mem) $(basis) $(struct) $(format_struc) $(struc) $(chem) $(math) nn.o format.o
nnpte: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o rng.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpte.exe nnp/nnpte.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o pot_pauli.o format.o alias.o rng.o
nnpte2: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpte2.exe nnp/nnpte2.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o pot_pauli.o format.o alias.o rng.o
nnptk2: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o rng.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptk2.exe nnp/nnptk2.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o pot_pauli.o format.o alias.o rng.o
nnpta: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o thole.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpta.exe nnp/nnpta.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o pot_pauli.o format.o alias.o thole.o
nnptefr: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) $(rand) batch.o nn.o format.o alias.o pot_ldamp_dsf.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptefr.exe nnp/nnptefr.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) $(rand) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o pot_pauli.o format.o alias.o pot_ldamp_dsf.o
nnptefr2: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) $(rand) batch.o nn.o format.o alias.o pca.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptefr2.exe nnp/nnptefr2.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) $(rand) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o pot_pauli.o format.o alias.o pca.o pot_ldamp_dsf.o
nnptefr3: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) $(rand) batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptefr3.exe nnp/nnptefr3.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) $(rand) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o pot_pauli.o format.o alias.o
nnptef: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptef.exe nnp/nnptef.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o
nnptefa: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o 
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptefa.exe nnp/nnptefa.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o pot_pauli.o
nnpqa: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpqa.exe nnp/nnpqa.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o alias.o
nnptk: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) batch.o nn.o format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptk.exe nnp/nnptk.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) batch.o nn.o pot.o pot_ldamp_long.o pot_gauss_long.o pot_ldamp_cut.o format.o
nnpteq: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) $(pot) qeq.o batch.o nn.o format.o alias.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpteq.exe nnp/nnpteq.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) $(pot) qeq.o batch.o nn.o format.o alias.o
nnptex: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) $(opt) $(pot) qtpie.o batch.o nn.o format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptex.exe nnp/nnptex.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(kspace) $(opt) $(pot) qtpie.o batch.o nn.o format.o
nnptes: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o 
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptes.exe nnp/nnptes.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o
nnptev: $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnptev.exe nnp/nnptev.cpp $(nnp) $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) optimize.o batch.o nn.o
nnpce: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) format.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpce.exe nnp/nnpce.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) format.o
nnpces: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_sim) $(chem) $(struc) format.o sim.o rng.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpces.exe nnp/nnpces.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_sim) $(chem) $(struc) format.o sim.o rng.o
nnpceq: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) format.o qeq.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o nnpceq.exe nnp/nnpceq.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(thread) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) format.o qeq.o
nnpceqs: $(nnp) nn.o $(str) $(math) $(util) $(mem) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) format.o qeq.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -fopenmp -o nnpceqs.exe nnp/nnpceqs.cpp $(nnp) nn.o $(str) $(math) $(util) $(mem) $(basis) $(struct) $(format_struc) $(chem) $(struc) $(pot) $(kspace) format.o qeq.o

#==== DRUDE ====

drudem: drude.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(kspace) $(math) format.o type.o verlet.o dump.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o drudem.exe torch/drudem.cpp drude.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(kspace) $(math) format.o type.o verlet.o dump.o

#==== CGEM ====
fit_cgem: $(struc) $(format_struc) $(str) $(chem) $(math) $(util) $(mem) format.o type.o cube_struc.o grid.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o fit_cgem.exe cgem/fit_cgem.cpp $(struc) $(format_struc) $(str) $(chem) $(math) $(util) $(mem) format.o type.o cube_struc.o grid.o

#==== CHARGE EQULIBRATION ====

qeqm: qeq.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(kspace) $(nnp) $(basis) $(math) nn.o format.o 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o qeqm.exe torch/qeqm.cpp qeq.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(kspace) $(nnp) $(basis) $(math) nn.o format.o 
qtpiem: qtpie.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(kspace) eigen.o format.o reduce.o type.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o qtpiem.exe torch/qtpiem.cpp qtpie.o $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(kspace) eigen.o format.o reduce.o type.o

#==== TORCH ====

torch: $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) $(property) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o job.o group.o verlet.o rng.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o torch.exe torch/torch.cpp $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) $(property) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o job.o group.o verlet.o rng.o
torch_train: $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) $(property) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o verlet.o group.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o torch_train.exe torch/torch_train.cpp $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) $(pot) $(math) $(kspace) $(basis) $(nnp) $(property) batch.o data.o nn.o monte_carlo.o integrator.o dump.o stochastic.o format.o engine.o qeq.o verlet.o group.o
phonon: $(struc) format.o vasp_struc.o vasp_sim.o lammps_sim.o $(str) $(mem) $(util) units.o ptable.o special.o eigen.o $(thread) sim.o
	$(CXX_THREAD) $(CXX_FLAGS) $(INC) -o phonon.exe torch/phonon.cpp $(struc) format.o vasp_struc.o vasp_sim.o lammps_sim.o $(str) $(mem) $(util) units.o ptable.o special.o eigen.o $(thread) sim.o -lfftw3

#==== ANALYSIS ====

atomsort: $(struc) $(format_struc) $(str) $(chem) $(mem) format.o parse.o eigen.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o atomsort.exe analysis/atomsort.cpp $(struc) $(format_struc) $(str) $(chem) $(mem) format.o parse.o eigen.o
vibrlx: $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) fft.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o vibrlx.exe analysis/vibrlx.cpp $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) fft.o -lfftw3
smod: $(struc) format.o $(format_struc) $(str) $(mem) $(chem) eigen.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o smod.exe analysis/smod.cpp $(struc) format.o $(format_struc) $(str) $(mem) $(chem) eigen.o
simmod: $(struc) format.o $(format_sim) $(str) $(mem) $(chem) eigen.o sim.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o simmod.exe analysis/simmod.cpp $(struc) format.o $(format_sim) $(str) $(mem) $(chem) eigen.o sim.o
cff: $(struc) format.o $(format_struc) $(str) $(mem) $(chem) $(math) molkit2.o graph.o alias.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o cff.exe chem/cff.cpp $(struc) format.o $(format_struc) $(str) $(mem) $(chem) $(math) molkit2.o graph.o alias.o
molgraph: $(struc) format.o $(format_struc) $(str) $(mem) $(chem) $(math) molkit2.o graph.o 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o molgraph.exe chem/molgraph.cpp $(struc) format.o $(format_struc) $(str) $(mem) $(chem) $(math) molkit2.o graph.o 
vdos: $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) $(signal) group.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o vdos.exe analysis/vdos.cpp $(struc) sim.o format.o $(format_sim) $(str) $(mem) $(chem) $(math) $(signal) group.o -lfftw3
vdos2: $(str) $(mem) $(chem) $(math) $(signal) group.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o vdos2.exe analysis/vdos2.cpp $(str) $(mem) $(chem) $(math) $(signal) group.o -lfftw3
adf: $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o $(signal)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o adf.exe analysis/adf.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o $(signal) -lfftw3
rdf: $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o $(signal)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o rdf.exe analysis/rdf.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o $(signal) -lfftw3
angle_atom: $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim) pair.o $(signal)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o angle_atom.exe analysis/angle_atom.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim) pair.o $(signal) -lfftw3
bond_atom: $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim) pair.o $(signal)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o bond_atom.exe analysis/bond_atom.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim) pair.o $(signal) -lfftw3
distmat: $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim) 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o distmat.exe analysis/distmat.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim)
bond_max: $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim) pair.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o bond_max.exe analysis/bond_max.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim) pair.o  
angle_max: $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim) pair.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o angle_max.exe analysis/angle_max.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o sim.o format.o $(format_sim) pair.o
cluster: $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o 
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o cluster.exe analysis/cluster.cpp $(str) $(mem) $(chem) $(math) $(struc) group.o hist.o sim.o format.o $(format_sim) pair.o
dos_avg: $(str) parse.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o dos_avg.exe analysis/dos_avg.cpp $(str) parse.o

#==== TOOLS ====

convert_struc: $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) eigen.o parse.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o convert_struc.exe format/convert_struc.cpp $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) eigen.o parse.o format.o
convert_sim: $(struc) $(format_sim) $(str) $(mem) $(util) $(chem) eigen.o parse.o sim.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o convert_sim.exe format/convert_sim.cpp $(format_sim) $(struc) $(str) $(mem) $(util) $(chem) eigen.o parse.o sim.o format.o
print_struc: $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) eigen.o parse.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o print_struc.exe format/print_struc.cpp $(struc) $(format_struc) $(str) $(mem) $(util) $(chem) eigen.o parse.o format.o
make_basis: $(str) $(basis) $(math) cutoff.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o make_basis.exe nnp/make_basis.cpp $(str) $(basis) $(math) cutoff.o
make_eigen: $(str) $(struc) $(format_struc) $(chem) $(mem) eigen.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o make_eigen.exe nnp/make_eigen.cpp $(str) $(struc) $(format_struc) $(chem) $(mem) eigen.o format.o
make_eigen2: $(str) $(struc) $(format_struc) $(chem) $(mem) eigen.o format.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o make_eigen2.exe nnp/make_eigen2.cpp $(str) $(struc) $(format_struc) $(chem) $(mem) eigen.o format.o
cat_data: $(str) $(chem) moldata.o parse.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o cat_data.exe chem/cat_data.cpp $(str) $(chem) moldata.o parse.o
move_data: $(str) $(chem) moldata.o parse.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o move_data.exe chem/move_data.cpp $(str) $(chem) moldata.o parse.o
xyz_data: $(str) $(chem) moldata.o parse.o
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o xyz_data.exe chem/xyz_data.cpp $(str) $(chem) moldata.o parse.o

#==== QE ====
avg_dos: $(string)
	$(CXX_SERIAL) $(CXX_FLAGS) $(INC) -o avg_dos.exe qe/avg_dos.cpp $(str)
