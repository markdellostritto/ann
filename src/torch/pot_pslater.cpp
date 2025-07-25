// math
#include "math/const.hpp"
#include "math/special.hpp"
// chem
#include "chem/units.hpp"
// pot
#include "torch/pot_pslater.hpp"
// thread
#include "thread/dist.hpp"

namespace ptnl{

using math::constant::PI;
using math::special::coscut;
using math::special::sincut;

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotPSlater& pot){
	return out<<static_cast<const Pot&>(pot);
}
	
//==== member functions ====

void PotPSlater::read(Token& token){
	static_cast<Pot&>(*this).read(token);
}

void PotPSlater::coeff(Token& token){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::coeff(Token&):\n";
	//coeff pauli type epsilon sigma
	const int type=std::atoi(token.next().c_str())-1;
	const double z=std::atof(token.next().c_str());
	const double r=std::atof(token.next().c_str());
	
	if(type>=ntypes_) throw std::invalid_argument("Invalid type.");
	
	z_[type]=z;
	r_[type]=r;
	f_[type]=1;
}

void PotPSlater::resize(int ntypes){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotPSlater::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::VectorXi::Zero(ntypes_);
		r_=Eigen::VectorXd::Zero(ntypes_);
		z_=Eigen::VectorXd::Zero(ntypes_);
		a_=Eigen::VectorXd::Zero(ntypes_);
		x_=Eigen::MatrixXd::Zero(ntypes_,ntypes);
		c6_=Eigen::MatrixXd::Zero(ntypes_,ntypes);
		z2_=Eigen::MatrixXd::Zero(ntypes_,ntypes);
	}
}

void PotPSlater::init(){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::init():\n";
	for(int i=0; i<ntypes_; ++i){
		a_[i]=1.0/r_[i];
		const double ri3=r_[i]*r_[i]*r_[i];
		for(int j=0; j<ntypes_; ++j){
			const double rj3=r_[j]*r_[j]*r_[j];
			x_(i,j)=0.25*(1.0/(r_[i]*r_[i])-1.0/(r_[j]*r_[j]));
			const double x3=x_(i,j)*x_(i,j)*x_(i,j);
			const double x6=x3*x3;
			c6_(i,j)=1.0/(4.0*x6*ri3*rj3);
			z2_(i,j)=z_[i]*z_[j];
		}
	}
}

double PotPSlater::energy(const Structure& struc, const NeighborList& nlist){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::energy(const Structure&,const NeighborList&):\n";
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			if(nlist.neigh(i,j).dr()<rc_){
				const double dr=nlist.neigh(i,j).dr();
				const double f=(
					a_[ti]*(dr*x_(ti,tj)-2.0*a_[tj])*exp(-0.5*a_[tj]*dr)+
					a_[tj]*(dr*x_(ti,tj)+2.0*a_[ti])*exp(-0.5*a_[ti]*dr)
				);
				energy_+=z2_(ti,tj)*c6_(ti,tj)/(dr*dr)*f*f;
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotPSlater::energy(const Structure& struc, const NeighborList& nlist, int i){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::energy(const Structure&,const NeighborList&,int):\n";
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<nlist.size(i); ++j){
		const int tj=struc.type(nlist.neigh(i,j).index());
		if(nlist.neigh(i,j).dr()<rc_){
			const double dr=nlist.neigh(i,j).dr();
			const double f=(
				a_[ti]*(dr*x_(ti,tj)-2.0*a_[tj])*exp(-0.5*a_[tj]*dr)+
				a_[tj]*(dr*x_(ti,tj)+2.0*a_[ti])*exp(-0.5*a_[ti]*dr)
			);
			energy_+=z2_(ti,tj)*c6_(ti,tj)/(dr*dr)*f*f;
		}
	}
	return energy_;
}

double PotPSlater::compute(Structure& struc, const NeighborList& nlist){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::compute(const Structure&,const NeighborList&):\n";
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			if(nlist.neigh(i,j).dr()<rc_){
				const Eigen::Vector3d& r=nlist.neigh(i,j).r();
				const double dr=nlist.neigh(i,j).dr();
				const double f=(
					a_[ti]*(dr*x_(ti,tj)-2.0*a_[tj])*exp(-0.5*a_[tj]*dr)+
					a_[tj]*(dr*x_(ti,tj)+2.0*a_[ti])*exp(-0.5*a_[ti]*dr)
				);
				const double fp=(
					(a_[ti]*x_(ti,tj)-0.5*a_[ti]*a_[tj]*dr*x_(ti,tj)+a_[ti]*a_[tj]*a_[tj])*exp(-0.5*a_[tj]*dr)+
					(a_[tj]*x_(ti,tj)-0.5*a_[ti]*a_[tj]*dr*x_(ti,tj)-a_[ti]*a_[ti]*a_[tj])*exp(-0.5*a_[ti]*dr)
				);
				const double dri=1.0/dr;
				energy_+=z2_(ti,tj)*c6_(ti,tj)/(dr*dr)*f*f;
				const Eigen::Vector3d fij=
					2.0*z2_(ti,tj)*c6_(ti,tj)/(dr*dr*dr*dr)*f*f*r
					-z2_(ti,tj)*c6_(ti,tj)/(dr*dr*dr)*f*fp*r;
				struc.force(i).noalias()+=fij;
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotPSlater::compute(Structure& struc, const NeighborList& nlist, int i){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::compute(const Structure&,const NeighborList&,int):\n";
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<nlist.size(i); ++j){
		const int tj=struc.type(nlist.neigh(i,j).index());
		if(nlist.neigh(i,j).dr()<rc_){
			const Eigen::Vector3d& r=nlist.neigh(i,j).r();
			
		}
	}
	return energy_;
}

double PotPSlater::energy(const Structure& struc, const verlet::List& vlist){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::energy(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotPSlater::energy(const Structure& struc, const verlet::List& vlist, int i){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::energy(const Structure&,const verlet::List&,int):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	const int ti=struc.type(i);
	for(int j=0; j<vlist.size(i); ++j){
		const int jj=vlist.neigh(i,j).index();
		const int tj=struc.type(jj);
		const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
		if(dr2<rc2_){
			
		}
	}
	return energy_;
}

double PotPSlater::compute(Structure& struc, const verlet::List& vlist){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::compute(const Structure&,const verlet::List&):\n";
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
				
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotPSlater::compute(Structure& struc, const verlet::List& vlist, int i){
	if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"ptnl::PotPSlater::compute(const Structure&,const verlet::List&,int):\n";
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
	
	template <> int nbytes(const ptnl::PotPSlater& obj){
		if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotPSlater&):\n";
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
	
	template <> int pack(const ptnl::PotPSlater& obj, char* arr){
		if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotPSlater&,char*):\n";
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
	
	template <> int unpack(ptnl::PotPSlater& obj, const char* arr){
		if(POT_PSLATER_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotPSlater&,const char*):\n";
		int pos=0,nt=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		ptnl::Pot::Name name;
		std::memcpy(&name,arr+pos,sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		if(name!=ptnl::Pot::Name::PAULI) throw std::invalid_argument("serialize::unpack(PotPSlater&,const char*): Invalid name.");
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.f().data(),arr+pos,nt*sizeof(int)); pos+=nt*nt*sizeof(int);//f_
			std::memcpy(obj.r().data(),arr+pos,nt*sizeof(double)); pos+=nt*nt*sizeof(double);//r_
			std::memcpy(obj.z().data(),arr+pos,nt*sizeof(double)); pos+=nt*nt*sizeof(double);//z_
		}
		obj.init();
		return pos;
	}
	
}