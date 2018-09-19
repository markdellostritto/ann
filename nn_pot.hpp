#ifndef ANN_NN_POT_HPP
#define ANN_NN_POT_HPP

// c libraries
#include <cstdlib>
#include <cmath>
// c++ libraries
#include <iostream>
// lower triangular matrix
#include "lmat.hpp"
// simulation libraries
#include "atom.hpp"
#include "property.hpp"
#include "cell.hpp"
#include "structure.hpp"
// neural networks
#include "nn.hpp"
// chemistry
#include "ptable.hpp"
// parallel
#include "parallel.hpp"
// optimization
#include "optimize.hpp"
// map
#include "map.hpp"
// math
#include "math_const.hpp"
// basis functions
#include "cutoff.hpp"
#include "basis_radial.hpp"
#include "basis_angular.hpp"

#ifndef NN_POT_PRINT_FUNC
#define NN_POT_PRINT_FUNC 0
#endif

#ifndef NN_POT_PRINT_STATUS
#define NN_POT_PRINT_STATUS 0
#endif

#ifndef NN_POT_PRINT_DATA
#define NN_POT_PRINT_DATA 0
#endif

//************************************************************
// TERMINOLOGY:
// symmetry function - measure of local symmetry around a given atom
// basis - group of symmetry functions associated with a given interaction
// Each species has its own basis for each pair interaction and each
// triple interaction (i.e. H has (H)-H, (H)-O, (H)-O-O, (H)-H-H, (H)-H-O/(H)-O-H)
// Since the atomic neural networks are different, the basis for pair/triple
// interactions on different species of center atoms can be the same, reducing
// the number of necessary parameters.  Note however, that they can be different.
//************************************************************


//************************************************************
// TYPEDEFS
//************************************************************

typedef Atom<Name,AN,Species,Index,Position,Symm,Force> AtomT;
typedef std::vector<Eigen::VectorXd,Eigen::aligned_allocator<Eigen::VectorXd> > VecList;

//************************************************************
// FORWARD DECLARATIONS
//************************************************************

class NNPotOpt;

//************************************************************
// NEURAL NETWORK POTENTIAL
//************************************************************

class NNPot{
private:
	//basis for pair/triple interactions
	double rc_;//global cutoff
	std::vector<std::vector<BasisR> > basisR_;//radial basis functions (nspecies x nspecies)
	std::vector<LMat<BasisA> > basisA_;//angular basis functions (nspecies (nspecies x (nspecies+1)/2))
	
	//element nn's
	Map<std::string,unsigned int> speciesMap_;//map atom names to nn indices
	std::vector<NN::Network> nn_;//neural networks for each specie (nspecies)
	
	//network configuration
	std::vector<unsigned int> nInput_;//number of radial + angular symmetry functions (nspecies)
	std::vector<unsigned int> nInputR_;//number of radial symmetry functions (nspecies)
	std::vector<unsigned int> nInputA_;//number of angular symmetry functions (nspecies)
	std::vector<std::vector<unsigned int> > offsetR_;//offset for the given radial basis
	std::vector<LMat<unsigned int> > offsetA_;//offset for the given radial basis
	static const unsigned int nOutput_=1;//only output = energy
	
	//pre-/post-conditioning
	std::vector<double> energyAtom_;//energy of isolated atom
	
	//input/output
	std::string header_;
	
	//utility
	Eigen::Vector3d rIJ_,rIK_,rJK_;
	Eigen::Vector3d rIJt_,rIKt_,rJKt_;
	VecList R_;
public:
	//initialization data - aggregates data necessary for starting training from scratch
	struct Init{
		//basis functions
		PhiRN::type phiRN;//radial basis name
		PhiAN::type phiAN;//angular basis name
		unsigned int nR,nA;//number of radial and angular symmetry functions
		//cutoff 
		double rm;//min radius
		double rc;//cutoff radius
		CutoffN::type tcut;//the type of cutoff function
		//network configuration
		std::vector<unsigned int> nh;//number of hidden nodes
		//regularization
		double lambda;
		//transfer function
		NN::TransferN::type tfType;//transfer function type
		//operators
		friend std::ostream& operator<<(std::ostream& out, const Init& init);
		//constructors/destructors
		Init(){defaults();};
		~Init(){};
		//member functions
		void defaults();
	};
public:
	//constructors/destructors
	NNPot(){defaults();};
	~NNPot(){};
	
	//friend declarations
	friend class NNPotOpt;
	
	//access
	//species
		const unsigned int nSpecies()const{return speciesMap_.size();};
		const std::string& speciesName(unsigned int i)const{return speciesMap_.key(i);};
		unsigned int speciesIndex(const std::string& name)const{return speciesMap_[name];};
		Map<std::string,unsigned int>& speciesMap(){return speciesMap_;};
		const Map<std::string,unsigned int>& speciesMap()const{return speciesMap_;};
		double& energyAtom(unsigned int i){return energyAtom_[i];};
		const double& energyAtom(unsigned int i)const{return energyAtom_[i];};
	//global cutoff
		double& rc(){return rc_;};
		const double& rc()const{return rc_;};
	//basis
		std::vector<std::vector<BasisR> >& basisR(){return basisR_;};
		const std::vector<std::vector<BasisR> >& basisR()const{return basisR_;};
		BasisR& basisR(unsigned int i, unsigned int j){return basisR_[i][j];};
		const BasisR& basisR(unsigned int i, unsigned int j)const{return basisR_[i][j];};
		std::vector<LMat<BasisA> >& basisA(){return basisA_;};
		const std::vector<LMat<BasisA> >& basisA()const{return basisA_;};
		BasisA& basisA(unsigned int n, unsigned int i, unsigned int j){return basisA_[n](i,j);};
		const BasisA& basisA(unsigned int n, unsigned int i, unsigned int j)const{return basisA_[n](i,j);};
		std::vector<NN::Network>& nn(){return nn_;};
		const std::vector<NN::Network>& nn()const{return nn_;};
		NN::Network& nn(unsigned int i){return nn_[i];};
		const NN::Network& nn(unsigned int i)const{return nn_[i];};
	//hidden nodes
		const unsigned int& nInput(unsigned int i)const{return nInput_[i];};
	//input/output
		std::string& header(){return header_;};
		const std::string& header()const{return header_;};
		
	//operators
	friend std::ostream& operator<<(std::ostream& out, const NNPot& nnpot);
	
	//member functions
	//auxilliary
		void defaults();//set defaults
		void clear(){defaults();};//clear the potential
		void init(const Init& init_);//initialize the basis functions and element networks
	//resizing
		void resize(const Structure<AtomT>& struc);//assign vector of all species in the simulations
		void resize(const std::vector<Structure<AtomT> >& simv);//assign vector of all species in the simulations
		void resize(const std::vector<std::string>& speciesNames);//set the number of species and species names to the total number of species in the simulations
	//nn-struc
		void initSymm(Structure<AtomT>& struc);//assign vector of all species in the simulations
		void initSymm(std::vector<Structure<AtomT> >& simv);//assign vector of all species in the simulations
		void inputs_symm(Structure<AtomT>& struc);//calculate inputs - symmetry functions
		void forces(Structure<AtomT>& struc);//calculate forces
		double energy(Structure<AtomT>& struc);//sum over atomic energyies and return total energy
		void forces_radial(Structure<AtomT>& struc);
		void forces_angular(Structure<AtomT>& struc);
	//static functions
		void write()const;
		void read();
		void write(unsigned int index, const std::string& filename)const;
		void read(unsigned int index, const std::string& filename);
};

#endif