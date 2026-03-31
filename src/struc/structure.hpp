#pragma once
#ifndef STRUCTURE_HPP
#define STRUCTURE_HPP

//no bounds checking in Eigen
#define EIGEN_NO_DEBUG

// c++ libraries
#include <iosfwd>
// Eigen
#include <Eigen/Dense>
// ann - structure
#include "struc/cell.hpp"
#include "struc/state.hpp"
#include "struc/atom_type.hpp"
// ann - serialize
#include "mem/serialize.hpp"

#ifndef STRUC_PRINT_FUNC
#define STRUC_PRINT_FUNC 0
#endif

#ifndef STRUC_PRINT_STATUS
#define STRUC_PRINT_STATUS 0
#endif

#ifndef STRUC_PRINT_DATA
#define STRUC_PRINT_DATA 0
#endif

typedef Eigen::Matrix<int,3,1> Vec3i;
typedef Eigen::Matrix<double,3,1> Vec3d;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecXd;

//**********************************************************************************************
//AtomData
//**********************************************************************************************

class AtomData{
protected:
	//atom type
	AtomType atomType_;
	//number of atoms
	int nAtoms_;
	//basic properties
	std::vector<std::string> name_;//name
	std::vector<int>	an_;//atomic_number
	std::vector<int>	type_;//type
	std::vector<int>	index_;//index
	//serial properties
	std::vector<double>	mass_;//mass
	std::vector<double>	charge_;//charge
	std::vector<double> radius_;//radius
	std::vector<double>	entropy_;//entropy
	std::vector<double>	chi_;//chi
	std::vector<double>	eta_;//eta
	std::vector<double>	c6_;//c6 - london dispersion coefficient
	std::vector<double>	js_;//js - spin interaction coefficient
	std::vector<double> alpha_;//atomic polarizability
	std::vector<double> weight_;//weight
	std::vector<double> drudeQ_;//drude - charge
	std::vector<double> drudeM_;//drude - mass
	std::vector<double> drudeW_;//drude - frequency
	std::vector<double> drudeN_;//drude - normalization
	//vector properties
	std::vector<Vec3i>	image_;//image
	std::vector<Vec3d>	posn_;//position
	std::vector<Vec3d>	vel_;//velocity
	std::vector<Vec3d>	force_;//force
	std::vector<Vec3d>	spin_;//spin
	std::vector<Vec3d>	drudeR_;//drude - position
	//nnp
	std::vector<VecXd>	symm_;//symmetry function
public:
	//==== constructors/destructors ====
	AtomData():nAtoms_(0){}
	~AtomData(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AtomData& ad);
	
	//==== access - global ====
	const AtomType& atomType()const{return atomType_;}
	const int& nAtoms()const{return nAtoms_;}
	
	//==== access - vectors ====
	//basic properties
	std::vector<std::string>& name(){return name_;}
	const std::vector<std::string>& name()const{return name_;}
	std::vector<int>& an(){return an_;}
	const std::vector<int>& an()const{return an_;}
	std::vector<int>& type(){return type_;}
	const std::vector<int>& type()const{return type_;}
	std::vector<int>& index(){return index_;}
	const std::vector<int>& index()const{return index_;}
	//serial properties
	std::vector<double>& mass(){return mass_;}
	const std::vector<double>& mass()const{return mass_;}
	std::vector<double>& charge(){return charge_;}
	const std::vector<double>& charge()const{return charge_;}
	std::vector<double>& radius(){return radius_;}
	const std::vector<double>& radius()const{return radius_;}
	std::vector<double>& entropy(){return entropy_;}
	const std::vector<double>& entropy()const{return entropy_;}
	std::vector<double>& chi(){return chi_;}
	const std::vector<double>& chi()const{return chi_;}
	std::vector<double>& eta(){return eta_;}
	const std::vector<double>& eta()const{return eta_;}
	std::vector<double>& c6(){return c6_;}
	const std::vector<double>& c6()const{return c6_;}
	std::vector<double>& js(){return js_;}
	const std::vector<double>& js()const{return js_;}
	std::vector<double>& alpha(){return alpha_;}
	const std::vector<double>& alpha()const{return alpha_;}
	std::vector<double>& weight(){return weight_;}
	const std::vector<double>& weight()const{return weight_;}
	std::vector<double>& drudeQ(){return drudeQ_;}
	const std::vector<double>& drudeQ()const{return drudeQ_;}
	std::vector<double>& drudeM(){return drudeM_;}
	const std::vector<double>& drudeM()const{return drudeN_;}
	std::vector<double>& drudeW(){return drudeW_;}
	const std::vector<double>& drudeW()const{return drudeW_;}
	std::vector<double>& drudeN(){return drudeN_;}
	const std::vector<double>& drudeN()const{return drudeN_;}
	//vector properties
	std::vector<Vec3i>& image(){return image_;}
	const std::vector<Vec3i>& image()const{return image_;}
	std::vector<Vec3d>& posn(){return posn_;}
	const std::vector<Vec3d>& posn()const{return posn_;}
	std::vector<Vec3d>& vel(){return vel_;}
	const std::vector<Vec3d>& vel()const{return vel_;}
	std::vector<Vec3d>& force(){return force_;}
	const std::vector<Vec3d>& force()const{return force_;}
	std::vector<Vec3d>& spin(){return spin_;}
	const std::vector<Vec3d>& spin()const{return spin_;}
	std::vector<Vec3d>& drudeR(){return drudeR_;}
	const std::vector<Vec3d>& drudeR()const{return drudeR_;}
	//nnp
	std::vector<VecXd>& symm(){return symm_;}
	const std::vector<VecXd>& symm()const{return symm_;}
	
	//==== access - atoms ====
	//basic properties
	std::string& name(int i){return name_[i];}
	const std::string& name(int i)const{return name_[i];}
	int& an(int i){return an_[i];}
	const int& an(int i)const{return an_[i];}
	int& type(int i){return type_[i];}
	const int& type(int i)const{return type_[i];}
	int& index(int i){return index_[i];}
	const int& index(int i)const{return index_[i];}
	//serial properties
	double& mass(int i){return mass_[i];}
	const double& mass(int i)const{return mass_[i];}
	double& charge(int i){return charge_[i];}
	const double& charge(int i)const{return charge_[i];}
	double& radius(int i){return radius_[i];}
	const double& radius(int i)const{return radius_[i];}
	double& entropy(int i){return entropy_[i];}
	const double& entropy(int i)const{return entropy_[i];}
	double& chi(int i){return chi_[i];}
	const double& chi(int i)const{return chi_[i];}
	double& eta(int i){return eta_[i];}
	const double& eta(int i)const{return eta_[i];}
	double& c6(int i){return c6_[i];}
	const double& c6(int i)const{return c6_[i];}
	double& js(int i){return js_[i];}
	const double& js(int i)const{return js_[i];}
	double& alpha(int i){return alpha_[i];}
	const double& alpha(int i)const{return alpha_[i];}
	double& weight(int i){return weight_[i];}
	const double& weight(int i)const{return weight_[i];}
	double& drudeQ(int i){return drudeQ_[i];}
	const double& drudeQ(int i)const{return drudeQ_[i];}
	double& drudeM(int i){return drudeM_[i];}
	const double& drudeM(int i)const{return drudeM_[i];}
	double& drudeW(int i){return drudeW_[i];}
	const double& drudeW(int i)const{return drudeW_[i];}
	double& drudeN(int i){return drudeN_[i];}
	const double& drudeN(int i)const{return drudeN_[i];}
	//vector properties
	Vec3i& image(int i){return image_[i];}
	const Vec3i& image(int i)const{return image_[i];}
	Vec3d& posn(int i){return posn_[i];}
	const Vec3d& posn(int i)const{return posn_[i];}
	Vec3d& vel(int i){return vel_[i];}
	const Vec3d& vel(int i)const{return vel_[i];}
	Vec3d& force(int i){return force_[i];}
	const Vec3d& force(int i)const{return force_[i];}
	Vec3d& spin(int i){return spin_[i];}
	const Vec3d& spin(int i)const{return spin_[i];}
	Vec3d& drudeR(int i){return drudeR_[i];}
	const Vec3d& drudeR(int i)const{return drudeR_[i];}
	//nnp
	VecXd& symm(int i){return symm_[i];}
	const VecXd& symm(int i)const{return symm_[i];}
	
	//==== member functions ====
	void clear();
	void resize(int nAtoms, const AtomType& atomT);
};

//**********************************************************************************************
//Structure
//**********************************************************************************************

class Structure: public Cell, public State, public AtomData{
public:
	//==== constructors/destructors ====
	Structure(){}
	Structure(int nAtoms, const AtomType& atomT){resize(nAtoms,atomT);}
	~Structure(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Structure& sim);
	
	//==== member functions ====
	void clear();
	
	//==== static functions ====
	static void write_binary(const Structure& struc, const char* file);
	static void read_binary(Structure& struc, const char* file);
	static Structure& super(const Structure& struc, Structure& superc, const Eigen::Vector3i nlat);
	static Structure& cnnn(Structure& struc, double rc);
};

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const AtomData& obj);
	template <> int nbytes(const Structure& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const AtomData& obj, char* arr);
	template <> int pack(const Structure& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(AtomData& obj, const char* arr);
	template <> int unpack(Structure& obj, const char* arr);
	
}

#endif
