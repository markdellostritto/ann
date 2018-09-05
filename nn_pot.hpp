#ifndef NN_POT_HPP
#define NN_POT_HPP

// c libraries
#include <cstdlib>
#include <cmath>
// c++ libraries
#include <iostream>
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
// basis functions
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
// TYPEDEFS
//************************************************************

typedef Atom<Name,AN,Species,Index,Position,Symm,Force> AtomT;
typedef std::vector<Eigen::VectorXd,Eigen::aligned_allocator<Eigen::VectorXd> > VecList;

//************************************************************
// FORWARD DECLARATIONS
//************************************************************

class NNPotOpt;

//************************************************************
// Lower Triangular Matrix
//************************************************************

template <class T>
class LMat{
private:
	unsigned int n_;
	std::vector<std::vector<T> > mat_;
public:
	//constructors/destructors
	LMat():n_(0){};
	LMat(unsigned int n){resize(n);};
	~LMat(){};
	
	//access
	unsigned int& n(){return n_;};
	const unsigned int& n()const{return n_;};
	T& operator()(unsigned int i, unsigned int j);
	const T& operator()(unsigned int i, unsigned int j)const;
	
	//member functions
	void clear();
	unsigned int size()const;
	void resize(unsigned int n);
	unsigned int index(unsigned int i, unsigned int j);
};

template <class T>
void LMat<T>::clear(){
	n_=0;
	mat_.clear();
}

template <class T>
unsigned int LMat<T>::size()const{
	return (n_*(n_+1))/2;
}

template <class T>
void LMat<T>::resize(unsigned int n){
	n_=n; mat_.resize(n_);
	for(unsigned int i=0; i<n_; ++i) mat_[i].resize(i+1);
}

template <class T>
unsigned LMat<T>::index(unsigned int i, unsigned int j){
	unsigned int ii=i,jj=j;
	if(i<j){ii=j;jj=i;}
	unsigned int index=0;
	for(unsigned int n=0; n<ii; ++n) index+=mat_[n].size();
	return index+jj;
}

template <class T>
T& LMat<T>::operator()(unsigned int i, unsigned int j){
	return (i>=j)?mat_[i][j]:mat_[j][i];
}

template <class T>
const T& LMat<T>::operator()(unsigned int i, unsigned int j)const{
	return (i>=j)?mat_[i][j]:mat_[j][i];
}

//************************************************************
// Neural Network Potential
//************************************************************

class NNPot{
private:
	//basis for valence species (X-specie) (symmetry functions)
	PhiRN::type phiRN_;//radial basis name
	PhiAN::type phiAN_;//angular basis name
	unsigned int nR_,nA_;//number of radial and angular symmetry functions
	std::vector<BasisR> basisR_;//radial basis functions (number of valence species)
	LMat<BasisA> basisA_;//angular basis functions (number of combinations of valence species)
	
	//cutoff 
	double rm_;//min radius
	double rc_;//cutoff radius
	CutoffN::type tcut_;//the type of cutoff function
	
	//element nn's
	unsigned int nParams_;//number of nn parameters (number of weights and biases)
	Map<std::string,unsigned int> speciesMap_;//map atom names to nn indices
	std::vector<NN::Network> nn_;//neural networks for each specie (number of species)
	
	//network configuration
	unsigned int nInput_;//number of radial + angular symmetry functions
	std::vector<unsigned int> nh_;//number of hidden nodes
	static const unsigned int nOutput_=1;//only output = energy
	
	//pre-/post-conditioning
	bool preCond_;//pre-condition 
	bool postCond_;//post-condition
	VecList preBias_,preScale_;//one for each specie (nn)
	double postBias_,postScale_;//energy is only output
	
	//transfer function
	NN::TransferN::type tfType_;//transfer function type
	
	//regularization
	double lambda_;
	
	//input/output
	unsigned int nPrint_;
	unsigned int nSave_;
	static const char* header;
	
	//utility vectrors
	Eigen::Vector3d rIJ_,rIK_,rJK_;
	Eigen::Vector3d rIJt_,rIKt_,rJKt_;
	VecList lvShifts_;
	VecList dOutdIn_;
public:
	//constructors/destructors
	NNPot(){defaults();};
	~NNPot(){};
	
	//friend declarations
	friend class NNPotOpt;
	
	//access
	//number of functions
		unsigned int& nR(){return nR_;};
		const unsigned int& nR()const{return nR_;};
		unsigned int& nA(){return nA_;};
		const unsigned int& nA()const{return nA_;};
	//cutoff
		double& rm(){return rm_;};
		const double& rm()const{return rm_;};
		double& rc(){return rc_;};
		const double& rc()const{return rc_;};
		CutoffN::type& tcut(){return tcut_;};
		const CutoffN::type& tcut()const{return tcut_;};
	//species
		const unsigned int nSpecies()const{return speciesMap_.size();};
		const std::string& speciesName(unsigned int i)const{return speciesMap_.key(i);};
		unsigned int speciesIndex(const std::string& name)const{return speciesMap_[name];};
	//basis
		PhiRN::type& phiRN(){return phiRN_;};
		const PhiRN::type& phiRN()const{return phiRN_;};
		PhiAN::type& phiAN(){return phiAN_;};
		const PhiAN::type& phiAN()const{return phiAN_;};
		std::vector<BasisR>& basisR(){return basisR_;};
		const std::vector<BasisR>& basisR()const{return basisR_;};
		BasisR& basisR(unsigned int i){return basisR_[i];};
		const BasisR& basisR(unsigned int i)const{return basisR_[i];};
		LMat<BasisA>& basisA(){return basisA_;};
		const LMat<BasisA>& basisA()const{return basisA_;};
		std::vector<NN::Network>& nn(){return nn_;};
		const std::vector<NN::Network>& nn()const{return nn_;};
		NN::Network& nn(unsigned int i){return nn_[i];};
		const NN::Network& nn(unsigned int i)const{return nn_[i];};
	//hidden nodes
		const unsigned int& nParams()const{return nParams_;};
		const unsigned int& nInput()const{return nInput_;};
		std::vector<unsigned int>& nh(){return nh_;};
		const std::vector<unsigned int>& nh()const{return nh_;};
	//pre-/post-conditioning
		bool& preCond(){return preCond_;};
		const bool& preCond()const{return preCond_;};
		bool& postCond(){return postCond_;};
		const bool& postCond()const{return postCond_;};
		const VecList& preScale()const{return preScale_;};
		const VecList& preBias()const{return preBias_;};
		const double& postScale()const{return postScale_;};
		const double& postBias()const{return postBias_;};
	//transfer function
		NN::TransferN::type& tfType(){return tfType_;};
		const NN::TransferN::type& tfType()const{return tfType_;};
	//regularization
		double& lambda(){return lambda_;};
		const double& lambda()const{return lambda_;};
	//input/output
		unsigned int& nPrint(){return nPrint_;};
		const unsigned int& nPrint()const{return nPrint_;};
		unsigned int& nSave(){return nSave_;};
		const unsigned int& nSave()const{return nSave_;};
	
	//operators
	friend std::ostream& operator<<(std::ostream& out, const NNPot& nnpot);
	
	//member functions
	//auxilliary
		void defaults();//set defaults
		void clear(){defaults();};//clear the potential
		void init();//initialize the basis functions and element networks
	//nn-struc
		void initSpecies(const Structure<AtomT>& struc);//assign vector of all species in the simulations
		void initSymm(Structure<AtomT>& struc);//assign vector of all species in the simulations
		void initSpecies(const std::vector<Structure<AtomT> >& simv);//assign vector of all species in the simulations
		void initSymm(std::vector<Structure<AtomT> >& simv);//assign vector of all species in the simulations
		void setSpecies(const std::vector<std::string>& speciesNames);//set the number of species and species names to the total number of species in the simulations
		void inputs_symm(Structure<AtomT>& struc);//calculate inputs - symmetry functions
		void forces(Structure<AtomT>& struc);//calculate forces
		double energy(Structure<AtomT>& struc);//sum over atomic energyies and return total energy
		
	//static functions
		static void write(const char* file, const NNPot& nnpot);
		static void read(const char* file, NNPot& nnpot);
};

#endif