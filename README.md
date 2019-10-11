# ANN - Atomic Nueral Network

Author: Mark J. DelloStritto

A code to train neural network potentials using periodic atomic structures and ab-initio references energies.  
Also included is a an extension to LAMMPS (https://lammps.sandia.gov/) which implements neural network potentials 
for molecular dynamics simulations.

## CODE ORGANIZATION

**SYSTEM**
* compiler.hpp - utilities for printing compiler information

**MEMORY**
* serialize.hpp - serialization of complex objects
* map.hpp       - mapping two different objects to each other

**MULTITHREADING**
* parallel.hpp - utilities for thread distribution
* mpi_util.hpp - utilities for mpi communication

**MATH**
* math_const.hpp   - mathematical constants
* math_special.hpp - special functions
* math_func.hpp    - function evalulation
* math_cmp.hpp     - comparison functions
* accumulator.hpp  - statistical accumulator
* eigen.hpp        - Eigen utilities
* lmat.hpp         - lower triangular matrix

**OPTIMIZATION**
* optimize.hpp - optimization

**STRING**
* string.hpp - string utilities

**CHEMISTRY**
* units.hpp  - physical units
* ptable.hpp - physical constants
* atom.hpp   - atomic species

**NEURAL NETWORK**
* nn.hpp - neural network implementation

**NEURAL NETWORK POTENTIAL**
* cutoff.hpp          - distance cutoff
* symm_radial.hpp     - radial symmetry function header
* symm_radial_g1.hpp  - "G1" symmetry function [1]
* symm_radial_g2.hpp  - "G2" symmetry function [1]
* symm_radial_t1.hpp  - "T1" symmetry function (tanh)
* symm_angular.hpp    - angular symmetry function header
* symm_angular_g3.hpp - "G3" symmetry function [1]
* symm_angular_g4.hpp - "G4" symmetry function [1]
* basis_radial.hpp    - radial basis (table of radial symmetry functions)
* basis_angular.hpp   - angular basis (table of angular symmetry functions)
* nn_pot.hpp          - neural network potential

**TRAINING**
* nn_pot_train_omp.hpp - nn pot - training - OpenMP (deprecated)
* nn_pot_train_mpi.hpp - nn pot - training - MPI
* nn_train.hpp - neural network training (testing)

**STRUCTURE**
* cell.hpp      - unit cell
* structure.hpp - atomic trajectories/properties

**FILE**
* vasp.hpp - read/write structures - VASP (https://www.vasp.at/)
* qe.hpp   - read/write structures - QE (https://www.quantum-espresso.org/)
* ame.hpp  - read/write structures - AME

![Code Layout](/code_layout.png)

## INSTALLATION

This code requires the Eigen Matrix Library (http://eigen.tuxfamily.org)

The Eigen library is a header library, and thus does not need to be compiled. 
The location of the Eigen library must be specified in the Makefile.

**Makefile options:**
* make omp     - make training binary - parallelized - OMP - gnu (deprecated)
* make mpi     - make training binary - parallelized - MPI - gnu
* make impi    - make training binary - parallelized - MPI - intel
* make test    - make testing binary
* make convert - make conversion binary
* make clean   - removes all object files
	
## TRAINING - DATA

### data - files:

The training, validation, and testing data are read from data files, each of which
contains a list of files providing structures and associated energies. The structures
listed are read from each data file and combined to form the total data set in each
category. Each data file can be individually classified as training, validation, or testing.
Although one can use a single data file for each category, it is recommended to split 
the data into several categories based on different constructions, symmetries, chemistries, etc.
The use of separate data files facilitates the organization and division of data into
different categories for training and analysis.

### data - training:

The data used for training the neural network potential. This data is the ONLY data 
which determines the gradient for optimization. The training data should comprise 
the bulk of the reference data.  

### data - validation:

The data used for determining the end of the optimization of the neural network training. 
Although a stopping criterion is implemented, one should monitor the validation error 
closely and end optimization when the change the in validation error is negligible or 
when the validation error is less than a threshold value. The validation data should be a 
small (~10-20%) fraction of the training data and should reflect a target of the 
neural network potential.

### data - test:

Optional data for testing.  When provided in training mode, energies 
and forces are computed for this data at the end of optimization. 
When provided in test mode, no optimization is performed, and energies 
and forces are computed for this data.

## NEURAL NETWORK POTENTIAL - SPECIFICATION

The neural network potential is stored separately for each element. The potential constists of three parts:

* vacuum energy of atom (sets cohesive energy)
* radial and angular basis (set of symmetry functions)
* atomic neural network

The basis can be the same for all atoms, as the neural networks for each element are 
completely independent.  Since the energy is basis + network, the basis can be the 
same for all elements.  The basis can of course vary per element, reflecting different 
local environments for each element.

If no basis is provided, an automatically generated basis will be used. 
It is not recommended to use the automatically generated parameters of the atomic basis. 
Instead, one should specify a set of symmetry functions which best reflect the atomic forces.
Essentially, one should ensure that there are a large number of symmetry functions with large gradients
at the interatomic distances and angles where the gradient of the energy is greatest.
As the interatomic forces computed as sums of the gradients of the symmetry functions,
the symmetry functions should also resemble the expected interatomic forces.

## NEURAL NETWORK POTENTIAL - TRAINING

The data can be organized in any number of data files. These data files are simply 
lists of file names which will be loaded into memory. Each data file can contain any number of structure files. 
The file format specifies the format of the data files.  The same format must be used for all files.

The units define the physical constants used to interpret the input variables and data.
The unit systems follow the same conventions as the LAMMPS unit systems. 
Currently, only AU and METAL units are implemented.

If no files are provided for reading the potential or restart, an automatically generated 
basis set and randomly initialized neural network is created. If a neural network potential 
file is provided, the potential will be read from file. If only the basis for a given set of
atoms is provided, the basis will be read for those atoms and the neural network will be
randomly initialized.  If a restart file is provided, 
the potential and optimization parameters will be read from the restart file. (Note: 
the restart file does not store structures, and so any set of structures can be used when 
restarting, though changing the structures may affect the results.) 

The cutoff radius sets the distance at which the potential goes to zero. The batch size 
specifies the size of the random collection of items in the training data used for training. 
Pre-conditioning shifts and normalizes the inputs of the neural network using the average 
and standard deviation of the inputs.  These values are then incorporated into the neural network, 
implemented as "scaling" layers modifying the input and output layers.  It is generally _not_ 
recommended to pre-scale the nodes, as they are already normalized by half the volume of the 
cutoff sphere (sphere formed by cutoff radius).

## OPTIMIZATION

There are three basic types of optimization:

* stochastic descent (rprop)
* gradient descent (sdg,sdm,nag,adagrad,adadelta,adam,nadam)
* conjugate-gradient (bfgs)

All gradient descent algorithms require a descent parameter (gamma).  Most also require 
a memory parameter (eta) which adds different forms of momentum to the descent by 
combining the current gradient with past gradients.  All gradient descent algorithms also
offer a decay parameter which decreases the descent parameter (gamma) over time.  The decay
parameter should be an integer which sets the time constant of the exponential decay of
the step.  At step n, the optimization step will thus be reduced by a factor of exp(-n/T), where
T is the number specified in the parameter file.  When using a gradient descent method,
it is recommended to use a relatively large but non-zero decay parameter, typically on the order
of the number of steps taking before stopping (max_iter).

The RPROP algorithm uses the sign of the gradient only, with the step sized determined by 
the sign of the change in the objective function. The RPROP algorithms is recommended, 
as it is extremently stable, exhibiting a monotonic decrease in the objective function 
for essentially any parameter values. However, the RPROP algorithm can severely overfit
the training data, and generally requires a large regularization parameter (lambda).

The ADAM algorithm is also effective, though it requires a good choice of gamma, and is 
the best gradient descent method available. It is recommended that one use a small gamma (around 1e-3) 
and a small batch size (~25% of the total number of training structures), as the smaller 
batch size results in a stochastic gradient direction which tends to benifit 
training performance. Though ADAM is less prone to overfitting, it is still recommended that 
one use a non-zero regularization parameter.  For more difficult systems, one might consider using 
the NADAM algorithm, with is ADAM with a Nesterov-style momentum term which results in a more stable 
gradient in the optimization step.

All other algorithms work to some degree, but are not recommended.

An important aspect of optimization is the regularization parameter (lambda). This parameter is a 
Lagrange multiplier which enforces small weights. When lambda is non-zero, a term is added to the 
error which is proportional to the average squared magnitude of the weights, and a corresponding 
term is added to the gradients. This regularization term thus prevents the weights of the newtork 
from becoming too large. Generally speaking, the order of magnitude of lambda will set the maximum order 
of magnitude of the weights of the network, independent of the size of the network itself.  For example, 
if lambda is of the order 1e-3 the largest weights of the network will typically be on the order of 1e2 and 
will not exceed 1e3. The regularization parameter (lambda) is very important for optimization as it 
is an excellent tool for preventing overfitting. If lambda is zero, the network may yield very good 
training and validation errors, but will generally yield an unphysical potential. A non-zero lambda, typically 
on the order of 1e-3, will yield slightly worse training and validation errors, but will yield a
more accurate and more general potential.

When restarting, all optimization parameters (lambda, gamma, decay, etc.) are overwritten by the 
values provided in the parameter file. There are times when this is not desirable, i.e. when restarting 
the optimization when using a decay schedule for the optimization step (gamma). In this case, if one removes
or comments out a given parameter, then the optimization parameters are taken from the restart file.

## WRITING

Every "nsave" iterations the neural network potential is written to file, and a restart
file is written.  The optimization count and all optimization data is retained in the
restart file, allowing for consistent restarting from a completed run.  When the max
number of iterations is reached, the final potential and restart files are written,
allowing for rapid restart without moving or copying files. Note that the max iterations 
applies only to the current run, not to the total optimization count.

The error for the training and validation data is written to file very "nprint" timesteps.
The error is accumulated in the same file when restarting, appending the new iteration 
count and error to the end of the file.

Currently, the basis set, energies, and forces are optionally written to file.  
The basis funtions are written as functions of distance/angle.  
The reference and nn energies for the training and validation sets are written.  
The reference and nn forces for the training and validation sets are written.  

## MOLECULAR DYNAMICS

An add-on package for LAMMPS is included so that the neural network potential may be used
for molecular dynamics simulations.  The potential reads in the same neural network potential
files output by the training code.
Installation of the package is documented in the README files in USER-ANN

## FILE FORMATS

A number of different file formats can be read to obtain atomic structures, forces, and energies.

**FORMATS**
* VASP xml - XML file produced by VASP which contains all relevant information, including positions, forces, and the energy.
* QE output - Quantum Espresso does not aggregate all relevant information in a single file, however, all relevant information 
is included in the standard output. Thus, if one pipes the output into a file, this can be read for training.
* AME file - special format designed for this code

A conversion utility is included.  The arguments are as follows:

* -format-in "format-in" = input format
* -format-out "format-out" = output format
* -frac = fractional coordinate ouput
* -cart = Cartesian coordinate output
* -offset x:y:z = uniform position offset
* -interval beg : end : stride = interval to load trajectory
* -sep = seperate the output into separate files for each timestep
* -poscar "file" = VASP poscar file 
* -xdatcar "file" = VASP xdatcar file
* -xml "file" = xml file
* -qeout "file" = qe output 
* -pos "file" = qe pos file
* -evp "file" = qe evp file
* -cel "file" = qe cel file
* -in "file" = qe input file
* -ame "file" = ame file
* -out "file" = output file

## PARAMETER FILE REFERENCE

### GENERAL PARAMETERS

* __seed__: (integer) The seed for the random number generator. If negative, the seed is taken from the current time.
* __mode__: (string) The execution mode, either "train" or "test".
* __format__: (string) The format of the structure files (must all be the same format). Possible values:
	* __VASP_XML__ VASP XML file
	* __QE__ Quantum Espresso output 
* __units__: (string) The unit system (follows LAMMPS conventions). Possible values:
	* __au__ atomic units
	* __metal__ metal units
* __charge__: (boolean) Whether the atoms have a constant charge. If "true," then ewald energies are computed for each
structure and removed, allowing one to train on the remainder.

### PARAMETERS - DATA

* __data_train__ (string) Data file containing files for training.
* __data_val__ (string) Data file containing files for validation.
* __data_test__ (string) Data file containing files for testing.

### PARAMETERS - ATOMS

Each __atom__ entry (atom name mass (charge)) lists the properties of a given atomic species in the nueral network potential. This consists of a space-separated line with the atom name, mass, energy, and optionally the charge. The atom name defines the species and must match a corresponding species in at least one of the structure files. The atom mass is unimportant for training, but is important for maintaining consistency between training and molecular dynamics simulations. The energy is used to "post-scale" the output: the atom energy is added to the output for each atom to obtain the total energy of the system. The atomic energy thus sets the cohesive energy of the system, and should be computed as the limit of the energy of a pure crystal of the species as the lattice constant goes to infinity. The charge needs to be specified only if __charge__ is set to true and is used to compute the ewald energies of the structures.

### READ/WRITE

* __read_pot__ - (atomname filename) Read the nueral network potential for "atomname" from "file". The name of the atom must match a species specified with an __atom__ command.
* __read_basis_radial__ - (atomname filename) Read the radial basis for "atomname" from "file".
* __read_basis_angular__ - (atomname filename) Read the radial basis for "atomname" from "file".
* __read_restart__ - (filename) Read the restart data from "filename".

### OPTIMIZATION

* __algo__ - (string) The optimization algorithm. Possible values:
	* __sgd__ - steepest gradient descent
	* __sdm__ - steepest descent with momentum
	* __nag__ - Nesterov accelerated gradient
	* __adagrad__ - adagrad
	* __adadelta__ - adadelta
	* __rmsprop__ - rmsprop
	* __adam__ - adam
	* __bfgs__ - bfgs
	* __rprop__ - rprop
* __opt_val__ - (string) Optimization value, determining how to stop optimization. Possible values:
	* __ftol_abs__ - Stop when the absolute value of the objective function is below the tolerance.
	* __ftol_rel__ - Stop when the change in the value of the objective function is below the tolerance.
	* __xtol_rel__ - Stop when the change in the parameters is below the tolerance.
* __tol__ - (float) Optimization tolerance, determining the stopping point, must be positive.
* __max_iter__ - (int) The maximum number of iterations in a given optimization run, must be greater than zero.
* __n_print__ - (int) Print the error to standard output every "n_print" steps.
* __n_write__ - (int) Write the restart file every "n_write" steps.
* __gamma__ - (float) Optimization step size (gradient descent), must be greater than zero.
* __eta__ - (float) Memory term for gradient descent methods which use momentum, must be between zero and one.
* __decay__ - (int) Exponential decay rate for the optimization step __gamma__.

### NN POTENTIAL

* __r_cut__ - (float) The cutoff radius of the potential in distance units, must be greater than zero.
* __lambda__ - (float) Regularization parameter, must be greater than zero.
* __n_hidden__ - (int array) The number of nodes in each hidden layer, with the leftmost integer adjacent to the input and the rightmost integer adjacent to the output, with each input accepting inputs from the left and sending outputs to the right.
* __transfer__ - (string) The name of the transfer function. Possible values:
	* __tanh__ - hyperbolic tanh function
	* __sigmoid__ - sigmoid function
	* __softplus__ - softplus function
	* __relu__ - rectified linear function
	* __lin__ - linear function
* __n_batch__ - (int) Number of structures in the batch, must be less than the total number of structures owned by a process, superceded by __p_batch__.
* __p_batch__ - (float) Percentage of structures in the batch, must be between zero and one.
* __pre_cond__ - (bool) Whether to precondition the inputs by shifting to zero using the average and scaling the magnitude by the inverse of the standard deviation.
* __calc_force__ - (bool) Whether to compute the forces on the atoms at the end of optimization. An expensive calculation that is often unnecessary, though useful for testing purposes.

### OUTPUT

* __write_basis__ - Whether to write the basis as a function of distance/angle to file.
* __write_energy__ - Whether to write the training/validation/testing energies to file.
* __write_force__ - Whether to write the force on each atom to file.

## REFERENCES

[1] Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050. https://doi.org/10.1002/qua.24890.
