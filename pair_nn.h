/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nn,PairNN)

#else

#ifndef LMP_PAIR_NN_H
#define LMP_PAIR_NN_H

#include "pair.h"
//c++ libraries
#include <vector>
// Eigen
#include <Eigen/Dense>
// ann library - basis functions - sum of symmetry functions
#include "cutoff.h"
#include "basis_radial.h"
#include "basis_angular.h"
// ann library - neural network
#include "nn.h"
// ann library - lower triangular matrix
#include "lmat.h"
// ann library - serialization
#include "serialize.h"
// ann library - chemical info
#include "ptable.h"

#ifndef PAIR_NN_PRINT_FUNC
#define PAIR_NN_PRINT_FUNC 0
#endif

#ifndef PAIR_NN_PRINT_STATUS
#define PAIR_NN_PRINT_STATUS 0
#endif

#ifndef PAIR_NN_PRINT_DATA
#define PAIR_NN_PRINT_DATA 0
#endif

namespace LAMMPS_NS{

class PairNN: public Pair{
protected:
	//==== global cutoff ====
	double rc_;
	//==== basis for valence species (X-specie) (symmetry functions) ====
	std::vector<std::vector<BasisR> > basisR_;//radial basis functions (nspecies x nspecies)
	std::vector<LMat<BasisA> > basisA_;//angular basis functions (nspecies (nspecies x (nspecies+1)/2))
	std::vector<unsigned int> nInput_;//number of radial + angular symmetry functions (nspecies)
	std::vector<unsigned int> nInputR_;//number of radial symmetry functions (nspecies)
	std::vector<unsigned int> nInputA_;//number of angular symmetry functions (nspecies)
	std::vector<std::vector<unsigned int> > offsetR_;//offset for the given radial basis
	std::vector<LMat<unsigned int> > offsetA_;//offset for the given radial basis
	//==== element nn's ====
	std::vector<NN::Network> nn_;//neural networks for each specie (nspecies)
	std::vector<double> energyAtom_;//energy of isolated atom
	//==== allocate data ====
	virtual void allocate();
public:
	//constructors/destructors
	PairNN(class LAMMPS *);
	virtual ~PairNN();
	//compute
	virtual void compute(int, int);
	//initialization
	void init_style();
	double init_one(int, int);
	void settings(int, char **);
	void coeff(int, char **);
	//reading/writing - restart
	void write_restart(FILE *);
	void read_restart(FILE *);
	void write_restart_settings(FILE *);
	void read_restart_settings(FILE *);
	//reading/writing - data
	void write_data(FILE *);
	void write_data_all(FILE *);
	//reading/writing - local
	void read_pot(int type, const char* file);
	//force evaluation 
	double single(int, int, int, int, double, double, double, double &);
	//parameter access
	void *extract(const char *, int &);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
