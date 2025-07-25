// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"
// pot
#include "torch/pot_coul_dsf.hpp"

namespace ptnl{

//==== using statements ====

using math::constant::RadPI;

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotCoulDSF& pot){
	return out<<static_cast<const Pot&>(pot)<<" eps "<<pot.eps_;
}
	
//==== member functions ====

void PotCoulDSF::read(Token& token){
	if(PCDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulDSF::read(Token&)\n";
	static_cast<Pot&>(*this).read(token);
	//pot coul_wolf 6.0 0.1
	alpha_=std::atof(token.next().c_str());
	if(alpha_<=0) throw std::invalid_argument("ptnl::PotCoulDSF::read(Token&): invalid alpha.");
	if(!token.end()){
		eps_=std::atof(token.next().c_str());
		if(eps_<=0.0) throw std::invalid_argument("ptnl::PotCoulLong::read(Token&): Invalid epsilon.");
	}
}

double PotCoulDSF::energy(const Structure& struc, const NeighborList& nlist){
	if(PCDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulDSF::energy(const Structure&,const NeighborList&)\n";
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	double q2s=0.0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		q2s+=qi*qi;
		for(int j=0; j<nlist.size(i); ++j){
			const double qj=struc.charge(nlist.neigh(i,j).index());
			const double dr=nlist.neigh(i,j).dr();
			energy_+=qi*qj*(erfc(alpha_*dr)/dr-verfc+vc*(dr-rc_));
		}
	}
	energy_-=q2s*(2.0*alpha_/RadPI+verfc-rc_*vc);
	return 0.5*ke*energy_;
}

double PotCoulDSF::compute(Structure& struc, const NeighborList& nlist){
	if(PCDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulDSF::compute(Structure&,const NeighborList&)\n";
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	double q2s=0.0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		q2s+=qi*qi;
		for(int j=0; j<nlist.size(i); ++j){
			const Eigen::Vector3d& r=nlist.neigh(i,j).r();
			const double qj=struc.charge(nlist.neigh(i,j).index());
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const double ferfc=erfc(alpha_*dr);
				const double fexp=exp(-alpha_*alpha_*dr*dr);
				energy_+=qi*qj*(ferfc/dr-verfc+vc*(dr-rc_));
				struc.force(i).noalias()-=ke*qi*qj*(ferfc+2.0*alpha_*dr/RadPI*fexp)/(dr*dr*dr)*r-vc/dr*r;
			}
		}
	}
	energy_-=q2s*(verfc+2.0*alpha_/RadPI-rc_*vc);
	return 0.5*ke*energy_;
}

Eigen::MatrixXd& PotCoulDSF::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	if(PCDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulDSF::J(const Structure&,const NeighborList&,Eigen::MatrixXd&):\n";
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	const double ec=(verfc+2.0*alpha_/RadPI-rc_*vc);
	for(int i=0; i<nAtoms; ++i){
		J(i,i)+=ec;
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const double& dr=nlist.neigh(i,j).dr();
			if(dr<rc_) J(i,jj)+=(erf(alpha_*dr)/dr-verfc+vc*(dr-rc_));
		}
	}
	//return matrix
	J*=ke;
	return J;
}

double PotCoulDSF::energy(const Structure& struc, const verlet::List& vlist){
	if(PCDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulDSF::energy(const Structure&,const verlet::List&)\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	double q2s=0.0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		q2s+=qi*qi;
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double qj=struc.charge(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				energy_+=qi*qj*(erfc(alpha_*dr)/dr-verfc+vc*(dr-rc_));
			}
		}
	}
	energy_-=q2s*(2.0*alpha_/RadPI+verfc-rc_*vc);
	return 0.5*ke*energy_;
}

double PotCoulDSF::compute(Structure& struc, const verlet::List& vlist){
	if(PCDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulDSF::compute(Structure&,const verlet::List&)\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	double q2s=0.0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		q2s+=qi*qi;
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double qj=struc.charge(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double ferfc=erfc(alpha_*dr);
				const double fexp=exp(-alpha_*alpha_*dr*dr);
				energy_+=qi*qj*(ferfc/dr-verfc+vc*(dr-rc_));
				struc.force(i).noalias()-=ke*qi*qj*(ferfc+2.0*alpha_*dr/RadPI*fexp)/(dr*dr*dr)*drv-vc/dr*drv;
			}
		}
	}
	energy_-=q2s*(verfc+2.0*alpha_/RadPI-rc_*vc);
	return 0.5*ke*energy_;
}

Eigen::MatrixXd& PotCoulDSF::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	if(PCDSF_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulDSF::J(const Structure&,const verlet::List&,Eigen::MatrixXd&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	const double verfc=erfc(rc_*alpha_)/rc_;
	const double vc=(verfc+2.0*alpha_*exp(-rc_*rc_*alpha_*alpha_)/(math::constant::RadPI))/rc_;
	const double ec=(verfc+2.0*alpha_/RadPI-rc_*vc);
	for(int i=0; i<nAtoms; ++i){
		J(i,i)+=ec;
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				J(i,jj)+=(erf(alpha_*dr)/dr-verfc+vc*(dr-rc_));
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
	
	template <> int nbytes(const ptnl::PotCoulDSF& obj){
		if(PCDSF_PRINT_FUNC>0) std::cout<<"nbytes(const PotCoulDSF&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(double);//eps_
		size+=sizeof(double);//alpha_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotCoulDSF& obj, char* arr){
		if(PCDSF_PRINT_FUNC>0) std::cout<<"pack(const PotCoulDSF&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(arr+pos,&obj.alpha(),sizeof(double)); pos+=sizeof(double);//alpha_
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotCoulDSF& obj, const char* arr){
		if(PCDSF_PRINT_FUNC>0) std::cout<<"unpack(PotCoulDSF&,const char*):\n";
		int pos=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(&obj.alpha(),arr+pos,sizeof(double)); pos+=sizeof(double);//alpha_
		return pos;
	}
	
}