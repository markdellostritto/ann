// math
#include "math/const.hpp"
#include "math/special.hpp"
// chem
#include "chem/units.hpp"
// pot
#include "torch/pot_pauli.hpp"
// thread
#include "thread/dist.hpp"

namespace ptnl{

using math::constant::PI;
using math::special::coscut;
using math::special::sincut;

// constants

const double PotPauli::CC=4.0;

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotPauli& pot){
	return out<<static_cast<const Pot&>(pot);
}
	
//==== member functions ====

void PotPauli::read(Token& token){
	static_cast<Pot&>(*this).read(token);
}

void PotPauli::coeff(Token& token){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::coeff(Token&):\n";
	//coeff pauli type epsilon sigma
	const int type=std::atoi(token.next().c_str())-1;
	const double z=std::atof(token.next().c_str());
	const double r=std::atof(token.next().c_str());
	
	if(type>=ntypes_) throw std::invalid_argument("Invalid type.");
	
	z_[type]=z;
	r_[type]=r;
	f_[type]=1;
}

void PotPauli::resize(int ntypes){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotPauli::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::VectorXi::Zero(ntypes_);
		r_=Eigen::VectorXd::Zero(ntypes_);
		z_=Eigen::VectorXd::Zero(ntypes_);
		zz_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		rr_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotPauli::init(){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::init():\n";
	for(int j=0; j<ntypes_; ++j){
		for(int i=0; i<ntypes_; ++i){
			zz_(i,j)=sqrt(z_[i]*z_[j]);
			rr_(i,j)=0.5*(r_[i]+r_[j]);
		}
	}
}

double PotPauli::energy(const Structure& struc, const NeighborList& nlist){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::energy(const Structure&,const NeighborList&):\n";
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			if(nlist.neigh(i,j).dr()<rc_){
				const double dr=nlist.neigh(i,j).dr();
				energy_+=zz_(ti,tj)*exp(-CC*(dr-rr_(ti,tj))/rr_(ti,tj));
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotPauli::energy(const Structure& struc, const NeighborList& nlist, int i){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::energy(const Structure&,const NeighborList&,int):\n";
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<nlist.size(i); ++j){
		const int tj=struc.type(nlist.neigh(i,j).index());
		if(nlist.neigh(i,j).dr()<rc_){
			const double dr=nlist.neigh(i,j).dr();
			energy_+=zz_(ti,tj)*exp(-CC*(dr-rr_(ti,tj))/rr_(ti,tj));
		}
	}
	return energy_;
}

double PotPauli::compute(Structure& struc, const NeighborList& nlist){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::compute(const Structure&,const NeighborList&):\n";
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			if(nlist.neigh(i,j).dr()<rc_){
				const Eigen::Vector3d& r=nlist.neigh(i,j).r();
				const double dr=nlist.neigh(i,j).dr();
				const double fexp=exp(-CC*(dr-rr_(ti,tj))/rr_(ti,tj));
				energy_+=zz_(ti,tj)*fexp;
				const Eigen::Vector3d fij=zz_(ti,tj)*fexp*CC/(rr_(ti,tj)*dr)*r;
				struc.force(i).noalias()+=fij;
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotPauli::compute(Structure& struc, const NeighborList& nlist, int i){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::compute(const Structure&,const NeighborList&,int):\n";
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<nlist.size(i); ++j){
		const int tj=struc.type(nlist.neigh(i,j).index());
		if(nlist.neigh(i,j).dr()<rc_){
			const Eigen::Vector3d& r=nlist.neigh(i,j).r();
			const double dr=nlist.neigh(i,j).dr();
			const double fexp=exp(-CC*(dr-rr_(ti,tj))/rr_(ti,tj));
			energy_+=zz_(ti,tj)*fexp;
			const Eigen::Vector3d fij=zz_(ti,tj)*fexp*CC/(rr_(ti,tj)*dr)*r;
			struc.force(i).noalias()+=fij;
		}
	}
	return energy_;
}

double PotPauli::energy(const Structure& struc, const verlet::List& vlist){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::energy(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				energy_+=zz_(ti,tj)*exp(-CC*(sqrt(dr2)-rr_(ti,tj))/rr_(ti,tj));
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotPauli::energy(const Structure& struc, const verlet::List& vlist, int i){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::energy(const Structure&,const verlet::List&,int):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<vlist.size(i); ++j){
		const int jj=vlist.neigh(i,j).index();
		const int tj=struc.type(jj);
		const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
		if(dr2<rc2_){
			energy_+=zz_(ti,tj)*exp(-CC*(sqrt(dr2)-rr_(ti,tj))/rr_(ti,tj));
		}
	}
	return energy_;
}

double PotPauli::compute(Structure& struc, const verlet::List& vlist){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::compute(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double fexp=exp(-CC*(dr-rr_(ti,tj))/rr_(ti,tj));
				energy_+=zz_(ti,tj)*fexp;
				const Eigen::Vector3d fij=zz_(ti,tj)*fexp*CC/(rr_(ti,tj)*dr)*drv;
				struc.force(i).noalias()+=fij;
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotPauli::compute(Structure& struc, const verlet::List& vlist, int i){
	if(POT_PAULI_PRINT_FUNC>0) std::cout<<"ptnl::PotPauli::compute(const Structure&,const verlet::List&,int):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<vlist.size(i); ++j){
		const int jj=vlist.neigh(i,j).index();
		const int tj=struc.type(jj);
		struc.diff(struc.posn(i),struc.posn(jj),drv);
		drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
		const double dr2=drv.squaredNorm();
		if(dr2<rc2_){
			const double dr=sqrt(dr2);
			const double fexp=exp(-CC*(dr-rr_(ti,tj))/rr_(ti,tj));
			energy_+=zz_(ti,tj)*fexp;
			const Eigen::Vector3d fij=zz_(ti,tj)*fexp*CC/(rr_(ti,tj)*dr)*drv;
			struc.force(i).noalias()+=fij;
		}
	}
	return energy_;
}

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotPauli& obj){
		if(POT_PAULI_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotPauli&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(int);//name_
		size+=sizeof(int);//ntypes_
		const int nt=obj.ntypes();
		size+=nt*sizeof(int);//f_
		size+=nt*sizeof(double);//r_
		size+=nt*sizeof(double);//z_
		return size;
	}
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotPauli& obj, char* arr){
		if(POT_PAULI_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotPauli&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.name(),sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		const int nt=obj.ntypes();
		if(nt>0){
			std::memcpy(arr+pos,obj.f().data(),nt*sizeof(int)); pos+=nt*sizeof(int);//f_
			std::memcpy(arr+pos,obj.r().data(),nt*sizeof(double)); pos+=nt*sizeof(double);//s_
			std::memcpy(arr+pos,obj.z().data(),nt*sizeof(double)); pos+=nt*sizeof(double);//e_
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotPauli& obj, const char* arr){
		if(POT_PAULI_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotPauli&,const char*):\n";
		int pos=0,nt=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		ptnl::Pot::Name name;
		std::memcpy(&name,arr+pos,sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		if(name!=ptnl::Pot::Name::PAULI) throw std::invalid_argument("serialize::unpack(PotPauli&,const char*): Invalid name.");
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.f().data(),arr+pos,nt*sizeof(int)); pos+=nt*sizeof(int);//f_
			std::memcpy(obj.r().data(),arr+pos,nt*sizeof(double)); pos+=nt*sizeof(double);//r_
			std::memcpy(obj.z().data(),arr+pos,nt*sizeof(double)); pos+=nt*sizeof(double);//z_
		}
		obj.init();
		return pos;
	}
	
}