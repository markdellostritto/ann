# ANN - Atomic Nueral Network

Author: Mark J. DelloStritto

A code to train neural network potentials using periodic atomic structures and ab-initio references energies.  
Also included is a an extension to LAMMPS (https://lammps.sandia.gov/) which implements neural network potentials 
for molecular dynamics simulations.

## CODE ORGANIZATION

**SYSTEM**
* compiler.hpp - utitilties for printing compiler information

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
* atom.hpp   - aggregate of data for different types of atoms

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
* basis_angular.hpp   - angular basis (table of radial symmetry functions)
* nn_pot.hpp          - neural network potential

**TRAINING**
* nn_pot_train_omp.hpp - nn pot - training - OpenMP
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
In the makefile, the location of the Eigen library must be specified in the indicated line.

**Makefile options:**
* make omp     - makes training binary - parallelized - OMP - gnu (deprecated)
* make mpi     - makes training binary - parallelized - MPI - gnu
* make impi    - makes training binary - parallelized - MPI - intel
* make test    - makes testing binary
* make convert - makes conversion binary
* make clean   - removes all object files
	
## TRAINING - DATA

### data - files:

The code reads in each data file provided. Each data file in turn should contain a list of 
files containing structures to be read for training, validation, and testing. Each data 
file can be classified as training, validation, or testing. The file names should either 
be accessible from the working directory, or they should contain the full path name (e.g. 
starting at $HOME)

### data - training:

The data used for training the neural network potential. This data is the ONLY data 
which determines the gradient for optimization. The training data should comprise 
the bulk of the reference data.  

### data - validation:

The data used for determining the end of the optimization of the neural network training. 
No stopping criterion is implemented, rather one should monitor the validation error 
closely and end the optimization when the change in validation error is negligible or 
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
same for all elements.  The basis can of course vary per element, reflecting different local environments for each element.

If no basis is provided, an automatically generated basis will be used. 
It is not recommended to use the automatically generated parameters of the atomic basis. 
Instead, one should specify a set of symmetry functions which best reflect the atomic forces.
Essentially, one should ensure that there are a large number of symmetry functions with large gradients
at the interatomic distances and angles where the gradient of the energy is greatest.
As the interatomic forces are constrained to be sums of the gradients of the symmetry functions,
the symmetry functions should also resemble the expected interatomic forces in these regions.

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
randomlly initialized.  If a restart file is provided, 
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
for essentially any parameter values. That being said, the RPROP algorithm can severely overfit
the training data, and generally requires a large regulation parameter (lambda).

The ADAM algorithm is also effective, though it requires a good choice of gamma, and is 
the best gradient descent method available. It is recommended that one use a small gamma (around 1e-3) 
and a small batch size (~25% of the total number of training structures), as the smaller 
batch sizes introduces a randomness to the gradient direction which tends to benifit 
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
the optimization with a using a decay schedule for the optimization step (gamma). In this case, if one removes
or comments out a given parameter, then the optimization parameters are taken from the restart file.

## WRITING

Every "nsave" iterations the neural network potential is written to a file, and a restart
file is written.  Each have the current optimization count appended to the end.  The count is stored in
the restart file and thereby consistently maintained when restarting.  When the max
number of iterations is reached, the final potential and restart files are written,
allowing for rapid restart without moving or copying files.

The error for the training and validation data is written to file very nprint timesteps.
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

A conversion utility is included.  The arguments are as follows:

* -format-in "format-in" = input format
* -format-out "format-out" = output format
* -frac = fractional coordinate ouput
* -cart = Cartesian coordinate output
* -offset x:y:z = uniform position offset
* -interval beg : end : stride = interval to load trajectory
* -sep = seperate the output into separate files for each timestep
* -poscar "file" = poscar file
* -xdatcar "file" = xdatcar file
* -xml "file" = xml file
* -qeout "file" = qe output 
* -pos "file" = qe pos file
* -evp "file" = qe evp file
* -cel "file" = qe cel file
* -in "file" = qe input file
* -ame "file" = ame file
* -out "file" = output file

## REFERENCES

[1] Behler, J. Constructing High-Dimensional Neural Network Potentials: A Tutorial Review. Int. J. Quantum Chem. 2015, 115 (16), 1032–1050. https://doi.org/10.1002/qua.24890.
