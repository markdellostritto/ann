// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"
// pot
#include "torch/pot_gauss_dsf.hpp"

namespace ptnl{

//#define PGDSF_SELF_ENERGY

//==== using statements ====

using math::constant::RadPI;
using math::constant::Rad2;

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotGaussDSF& pot){
	return out<<static_cast<const Pot&>(pot)<<" eps "<<pot.eps_;
}
	
//==== member functions ====

void PotGaussDSF::read(Token& token){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::read(Token&)\n";
	static_cast<Pot&>(*this).read(token);
	//pot coul_wolf 6.0 0.1
	alpha_=std::atof(token.next().c_str());
	if(alpha_<=0) throw std::invalid_argument("ptnl::PotGaussDSF::read(Token&): invalid alpha.");
	if(!token.end()){
		eps_=std::atof(token.next().c_str());
		if(eps_<=0.0) throw std::invalid_argument("ptnl::PotGaussDSF::read(Token&): Invalid epsilon.");
	}
}

void PotGaussDSF::coeff(Token& token){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::coeff(Token& token)\n";
	//coeff gauss_long type radius
	const int t=std::atof(token.next().c_str())-1;
	const double rad=std::atof(token.next().c_str());
	
	if(t>=ntypes_) throw std::invalid_argument("ptnl::PotGaussDSF::coeff(Token& token): Invalid type.");
	
	int tmin=t,tmax=t;
	if(t<0){tmin=0;tmax=ntypes_-1;}
	for(int i=tmin; i<=tmax; ++i){
		f_[i]=1;
		radius_[i]=rad;
	}
}

void PotGaussDSF::resize(int ntypes){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotGaussDSF::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::VectorXi::Zero(ntypes_);
		radius_=Eigen::VectorXd::Zero(ntypes_);
		rij_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		erfij_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		expij_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotGaussDSF::init(){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::init():\n";
	for(int i=0; i<ntypes_; ++i){
		if(f_[i]==0) throw std::invalid_argument("ptnl::PotGaussDSF::init(): No radius set.");
		const double ri=radius_[i];
		if(ri<=0) throw std::invalid_argument("ptnl::PotGaussDSF::init(): Invalid radius.");
		rij_(i,i)=2.0*ri;
		erfij_(i,i)=erf(rc_/rij_(i,i));
		expij_(i,i)=exp(-rc_*rc_/(rij_(i,i)*rij_(i,i)));
		for(int j=i+1; j<ntypes_; ++j){
			const double rj=radius_(j,j);
			rij_(i,j)=Rad2*sqrt(ri*ri+rj*rj);
			rij_(j,i)=rij_(i,j);
			erfij_(i,j)=erf(rc_/rij_(i,j));
			erfij_(j,i)=erfij_(i,j);
			expij_(i,j)=exp(-rc_*rc_/(rij_(i,j)*rij_(i,j)));
			expij_(j,i)=expij_(i,j);
		}
	}
}

double PotGaussDSF::energy(const Structure& struc, const NeighborList& nlist){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::energy(const Structure&,const NeighborList&)\n";
	const double ke=units::Consts::ke()*eps_;
	double energy=0;
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	double q2s=0.0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		const int ti=struc.type(i);
		q2s+=qi*qi;
		#ifdef PGDSF_SELF_ENERGY
		energy+=qi*qi*(
			2.0/(RadPI*rij_(ti,ti))-erfij_(ti,ti)*verfc
			-(vc*erfij_(ti,ti)-2.0*expij_(ti,ti)*verfc/(math::constant::RadPI*rij_(ti,ti)))*rc_
		);
		#endif
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			const double dr=nlist.neigh(i,j).dr();
			energy+=qi*qj*(
				erf(dr/rij_(ti,tj))*erfc(alpha_*dr)/dr-erfij_(ti,tj)*verfc
				+(vc*erfij_(ti,tj)-2.0*expij_(ti,tj)*verfc/(math::constant::RadPI*rij_(ti,tj)))*(dr-rc_)
			);
		}
	}
	energy-=q2s*(verfc+2.0*alpha_/RadPI-rc_*vc);
	return 0.5*ke*energy;
}

double PotGaussDSF::compute(Structure& struc, const NeighborList& nlist){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::compute(Structure&,const NeighborList&)\n";
	throw std::invalid_argument("ptnl::PotGaussDSF::compute(Structure&,const NeighborList&): NOT YET IMPLEMENTED\n");
	const double ke=units::Consts::ke()*eps_;
	double energy=0;
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const Eigen::Vector3d& r=nlist.neigh(i,j).r();
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const double ferfc=erfc(alpha_*dr);
				const double fexp=exp(-alpha_*alpha_*dr*dr);
				energy+=qi*qj*(
					erf(dr/rij_(ti,tj))*erfc(alpha_*dr)/dr-erfij_(ti,tj)*verfc
					+(vc*erfij_(ti,tj)-2.0*expij_(ti,tj)*verfc/(math::constant::RadPI*rij_(ti,tj)))*(dr-rc_)
				);
				energy+=qi*qj*(ferfc/dr-verfc+vc*(dr-rc_));
				struc.force(i).noalias()-=ke*qi*qj*(ferfc+2.0*alpha_*dr/RadPI*fexp)/(dr*dr*dr)*r-vc/dr*r;
			}
		}
	}
	return 0.5*ke*energy;
}

Eigen::MatrixXd& PotGaussDSF::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::J(const Structure&,const NeighborList&,Eigen::MatrixXd&):\n";
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	const double ec=(verfc+2.0*alpha_/RadPI-rc_*vc);
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		#ifdef PGDSF_SELF_ENERGY
		J(i,i)+=(
			2.0/(RadPI*rij_(ti,ti))-erfij_(ti,ti)*verfc
			-(vc*erfij_(ti,ti)-2.0*expij_(ti,ti)*verfc/(math::constant::RadPI*rij_(ti,ti)))*rc_
		);
		#endif
		J(i,i)-=ec;
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_) J(i,jj)+=(
				erf(dr/rij_(ti,tj))*erfc(alpha_*dr)/dr-erfij_(ti,tj)*verfc
				+(vc*erfij_(ti,tj)-2.0*expij_(ti,tj)*verfc/(math::constant::RadPI*rij_(ti,tj)))*(dr-rc_)
			);
		}
	}
	//return matrix
	J*=ke;
	return J;
}

double PotGaussDSF::energy(const Structure& struc, const verlet::List& vlist){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::energy(const Structure&,const verlet::List&)\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	double energy=0;
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	double q2s=0.0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		const int ti=struc.type(i);
		q2s+=qi*qi;
		#ifdef PGDSF_SELF_ENERGY
		energy+=qi*qi*(
			2.0/(RadPI*rij_(ti,ti))-erfij_(ti,ti)*verfc
			-(vc*erfij_(ti,ti)-2.0*expij_(ti,ti)*verfc/(math::constant::RadPI*rij_(ti,ti)))*rc_
		);
		#endif
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				energy+=qi*qj*(
					erf(dr/rij_(ti,tj))*erfc(alpha_*dr)/dr-erfij_(ti,tj)*verfc
					+(vc*erfij_(ti,tj)-2.0*expij_(ti,tj)*verfc/(math::constant::RadPI*rij_(ti,tj)))*(dr-rc_)
				);
			}
		}
	}
	energy-=q2s*(verfc+2.0*alpha_/RadPI-rc_*vc);
	return 0.5*ke*energy;
}

double PotGaussDSF::compute(Structure& struc, const verlet::List& vlist){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::compute(Structure&,const verlet::List&)\n";
	throw std::invalid_argument("ptnl::PotGaussDSF::compute(Structure&,const verlet::List&): NOT YET IMPLEMENTED\n");
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	double energy=0;
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double qj=struc.charge(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double ferfc=erfc(alpha_*dr);
				const double fexp=exp(-alpha_*alpha_*dr*dr);
				energy+=qi*qj*(
					erf(dr/rij_(ti,tj))*erfc(alpha_*dr)/dr-erfij_(ti,tj)*verfc
					+(vc*erfij_(ti,tj)-2.0*expij_(ti,tj)*verfc/(math::constant::RadPI*rij_(ti,tj)))*(dr-rc_)
				);
				energy+=qi*qj*(ferfc/dr-verfc+vc*(dr-rc_));
				struc.force(i).noalias()-=ke*qi*qj*(ferfc+2.0*alpha_*dr/RadPI*fexp)/(dr*dr*dr)*drv-vc/dr*drv;
			}
		}
	}
	return 0.5*ke*energy;
}

Eigen::MatrixXd& PotGaussDSF::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	if(PGDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotGaussDSF::J(const Structure&,const verlet::List&,Eigen::MatrixXd&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	const double ec=(verfc+2.0*alpha_/RadPI-rc_*vc);
	for(int i=0; i<nAtoms; ++i){
		const int ti=struc.type(i);
		#ifdef PGDSF_SELF_ENERGY
		J(i,i)+=(
			2.0/(RadPI*rij_(ti,ti))-erfij_(ti,ti)*verfc
			-(vc*erfij_(ti,ti)-2.0*expij_(ti,ti)*verfc/(math::constant::RadPI*rij_(ti,ti)))*rc_
		);
		#endif
		J(i,i)-=ec;
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				J(i,jj)+=(
					erf(dr/rij_(ti,tj))*erfc(alpha_*dr)/dr-erfij_(ti,tj)*verfc
					+(vc*erfij_(ti,tj)-2.0*expij_(ti,tj)*verfc/(math::constant::RadPI*rij_(ti,tj)))*(dr-rc_)
				);
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
	
	template <> int nbytes(const ptnl::PotGaussDSF& obj){
		if(PGDSF_PRINT_FUNC>0) std::cout<<"nbytes(const PotGaussDSF&):\n";
		int size=0;
		const int nt=obj.ntypes();
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(int);//ntypes_
		size+=sizeof(double);//eps_
		size+=sizeof(double);//alpha_
		size+=sizeof(int)*nt;//f_
		size+=sizeof(double)*nt;//radius
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotGaussDSF& obj, char* arr){
		if(PGDSF_PRINT_FUNC>0) std::cout<<"pack(const PotGaussDSF&,char*):\n";
		int pos=0;
		const int nt=obj.ntypes();
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&nt,sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(arr+pos,&obj.alpha(),sizeof(double)); pos+=sizeof(double);//alpha_
		if(nt>0){
			std::memcpy(arr+pos,obj.f().data(),sizeof(int)*nt); pos+=sizeof(int)*nt;//radius
			std::memcpy(arr+pos,obj.radius().data(),sizeof(double)*nt); pos+=sizeof(double)*nt;//radius
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotGaussDSF& obj, const char* arr){
		if(PGDSF_PRINT_FUNC>0) std::cout<<"unpack(PotGaussDSF&,const char*):\n";
		int pos=0,nt=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(&obj.alpha(),arr+pos,sizeof(double)); pos+=sizeof(double);//alpha_
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.f().data(),arr+pos,sizeof(int)*nt); pos+=sizeof(int)*nt;//f
			std::memcpy(obj.radius().data(),arr+pos,sizeof(double)*nt); pos+=sizeof(double)*nt;//radius
		}
		obj.init();
		return pos;
	}
	
}