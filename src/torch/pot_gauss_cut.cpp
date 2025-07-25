// math
#include "math/const.hpp"
// chem
#include "chem/units.hpp"
// pot
#include "torch/pot_gauss_cut.hpp"

//==== operator ====

namespace ptnl{

#define PGC_SELF_ENERGY

//==== using statements ====

using math::constant::Rad2;
using math::constant::RadPI;

//==== operators ====

std::ostream& operator<<(std::ostream& out, const PotGaussCut& pot){
	return out<<static_cast<const Pot&>(pot)<<" eps "<<pot.eps_;
}
	
//==== member functions ====

void PotGaussCut::read(Token& token){
	static_cast<Pot&>(*this).read(token);
	if(!token.end()){
		eps_=std::atof(token.next().c_str());
		if(eps_<=0.0) throw std::invalid_argument("PotGaussCut::read(Token&): Invalid epsilon.");
	}
}

void PotGaussCut::coeff(Token& token){
	if(PGC_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussCut::coeff(Token& token):\n";
	//coeff gauss_long type radius
	const int t=std::atof(token.next().c_str())-1;
	const double rad=std::atof(token.next().c_str());
	
	if(t>=ntypes_) throw std::invalid_argument("ptnl::PotGaussCut::coeff(Token& token): Invalid type.");
	
	int tmin=t,tmax=t;
	if(t<0){tmin=0;tmax=ntypes_-1;}
	for(int i=tmin; i<=tmax; ++i){
		f_[i]=1;
		radius_[i]=rad;
	}
}

void PotGaussCut::resize(int ntypes){
	if(PGC_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussCut::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotGaussCut::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::VectorXi::Zero(ntypes_);
		radius_=Eigen::VectorXd::Zero(ntypes_);
		rij_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotGaussCut::init(){
	if(PGC_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussCut::init():\n";
	for(int i=0; i<ntypes_; ++i){
		if(f_[i]==0) throw std::invalid_argument("ptnl::PotGaussCut::init(): No radius set.");
		const double ri=radius_[i];
		if(ri<=0) throw std::invalid_argument("ptnl::PotGaussLong::init(): Invalid radius.");
		rij_(i,i)=2.0*ri;
		for(int j=i+1; j<ntypes_; ++j){
			const double rj=radius_(j,j);
			rij_(i,j)=Rad2*sqrt(ri*ri+rj*rj);
			rij_(j,i)=rij_(i,j);
		}
	}
}

double PotGaussCut::energy(const Structure& struc, const NeighborList& nlist){
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		const int ti=struc.type(i);
		#ifdef PGC_SELF_ENERGY
		energy_+=qi*qi*2.0/(rij_(ti,ti)*RadPI);
		#endif
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_) energy_+=qi*qj*erf(dr/rij_(ti,tj))/dr;
		}
	}
	return 0.5*ke*energy_;
}

double PotGaussCut::compute(Structure& struc, const NeighborList& nlist){
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		const int ti=struc.type(i);
		#ifdef PGC_SELF_ENERGY
		energy_+=qi*qi*2.0/(rij_(ti,ti)*RadPI);
		#endif
		for(int j=0; j<nlist.size(i); ++j){
			const Eigen::Vector3d& r=nlist.neigh(i,j).r();
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			const double dr=nlist.neigh(i,j).dr();
			const double irij=1.0/rij_(ti,tj);
			const double erff=erf(dr*irij);
			const double expf=exp(-dr*dr*irij*irij);
			energy_+=qi*qj*erff/dr;
			struc.force(i).noalias()-=ke*qi*qj*(2.0*irij*dr/RadPI*expf-erff)/(dr*dr*dr)*r;
		}
	}
	return 0.5*ke*energy_;
}

Eigen::MatrixXd& PotGaussCut::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		#ifdef PGC_SELF_ENERGY
		J(i,i)+=2.0/(rij_(ti,ti)*RadPI);
		#endif
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_) J(i,jj)+=erf(dr/rij_(ti,tj))/dr;
		}
	}
	//return matrix
	J*=ke;
	return J;
}

double PotGaussCut::energy(const Structure& struc, const verlet::List& vlist){
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		const int ti=struc.type(i);
		#ifdef PGC_SELF_ENERGY
		energy_+=qi*qi*2.0/(rij_(ti,ti)*RadPI);
		#endif
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				energy_+=qi*qj*erf(dr/rij_(ti,tj))/dr;
			}
		}
	}
	return 0.5*ke*energy_;
}

double PotGaussCut::compute(Structure& struc, const verlet::List& vlist){
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		const int ti=struc.type(i);
		#ifdef PGC_SELF_ENERGY
		energy_+=qi*qi*2.0/(rij_(ti,ti)*RadPI);
		#endif
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double irij=1.0/rij_(ti,tj);
				const double erff=erf(dr*irij);
				const double expf=exp(-dr*dr*irij*irij);
				energy_+=qi*qj*erff/dr;
				struc.force(i).noalias()-=ke*qi*qj*(2.0*irij*dr/RadPI*expf-erff)/(dr*dr*dr)*drv;
			}
		}
	}
	return 0.5*ke*energy_;
}

Eigen::MatrixXd& PotGaussCut::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		#ifdef PGC_SELF_ENERGY
		J(i,i)+=2.0/(rij_(ti,ti)*RadPI);
		#endif
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				J(i,jj)+=erf(dr/rij_(ti,tj))/dr;
			}
		}
	}
	//return matrix
	J*=ke;
	return J;
}

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotGaussCut& obj){
		if(PGC_PRINT_FUNC>0) std::cout<<"nbytes(const PotGaussCut&):\n";
		int size=0;
		const int nt=obj.ntypes();
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(int);//ntypes_
		size+=sizeof(double);//eps_
		size+=sizeof(int)*nt;//f_
		size+=sizeof(double)*nt;//radius
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotGaussCut& obj, char* arr){
		if(PGC_PRINT_FUNC>0) std::cout<<"pack(const PotGaussCut&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);//eps_
		const int nt=obj.ntypes();
		if(nt>0){
			std::memcpy(arr+pos,obj.f().data(),sizeof(int)*nt); pos+=sizeof(int);//radius
			std::memcpy(arr+pos,obj.radius().data(),sizeof(double)*nt); pos+=sizeof(double);//radius
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotGaussCut& obj, const char* arr){
		if(PGC_PRINT_FUNC>0) std::cout<<"unpack(PotGaussCut&,const char*):\n";
		int pos=0,nt=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);//eps_
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.f().data(),arr+pos,sizeof(int)*nt); pos+=sizeof(int);//radius
			std::memcpy(obj.radius().data(),arr+pos,sizeof(double)*nt); pos+=sizeof(double);//radius
		}
		obj.init();
		return pos;
	}
	
}