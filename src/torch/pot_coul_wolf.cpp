// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"
// pot
#include "torch/pot_coul_wolf.hpp"

namespace ptnl{

//==== using statements ====

using math::constant::RadPI;

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotCoulWolf& pot){
	return out<<static_cast<const Pot&>(pot)<<" alpha "<<pot.alpha_<<" eps "<<pot.eps_;
}
	
//==== member functions ====

void PotCoulWolf::read(Token& token){
	if(PCW_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulWolf::energy(const Structure&,const NeighborList&)\n";
	static_cast<Pot&>(*this).read(token);
	//pot coul_wolf 6.0 0.1
	alpha_=std::atof(token.next().c_str());
	if(alpha_<=0) throw std::invalid_argument("ptnl::PotCoulWolf::read(Token&): invalid alpha.");
	if(!token.end()){
		eps_=std::atof(token.next().c_str());
		if(eps_<=0.0) throw std::invalid_argument("ptnl::PotCoulWolf::read(Token&): Invalid epsilon.");
	}
}

double PotCoulWolf::energy(const Structure& struc, const NeighborList& nlist){
	if(PCW_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulWolf::energy(const Structure&,const NeighborList&)\n";
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	double q2s_=0;
	const double vc=erfc(alpha_*rc_)/rc_;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		q2s_+=qi*qi;
		for(int j=0; j<nlist.size(i); ++j){
			const double qj=struc.charge(nlist.neigh(i,j).index());
			const double dr=nlist.neigh(i,j).dr();
			energy_+=qi*qj*(erfc(alpha_*dr)/dr-vc);
		}
	}
	energy_*=0.5;
	energy_-=(0.5*vc+alpha_/RadPI)*q2s_;
	return ke*energy_;
}

double PotCoulWolf::compute(Structure& struc, const NeighborList& nlist){
	if(PCW_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulWolf::compute(Structure&,const NeighborList&)\n";
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	double q2s_=0;
	const double vc=erfc(alpha_*rc_)/rc_;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		q2s_+=qi*qi;
		for(int j=0; j<nlist.size(i); ++j){
			const Eigen::Vector3d& r=nlist.neigh(i,j).r();
			const double qj=struc.charge(nlist.neigh(i,j).index());
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const double ferfc=erfc(alpha_*dr);
				const double fexp=exp(-alpha_*alpha_*dr*dr);
				energy_+=qi*qj*(ferfc/dr-vc);
				struc.force(i).noalias()+=ke*qi*qj*(2.0*alpha_*dr/RadPI*fexp+ferfc)/(dr*dr*dr)*r;
			}
		}
	}
	energy_*=0.5;
	energy_-=(0.5*vc+alpha_/RadPI)*q2s_;
	return ke*energy_;
}

Eigen::MatrixXd& PotCoulWolf::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	if(PCW_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulWolf::J(const Structure&,const NeighborList&,Eigen::MatrixXd&):\n";
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	//interaction terms
	double q2s_=0;
	const double vc=erfc(alpha_*rc_)/rc_;
	for(int i=0; i<nAtoms; ++i){
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const double& dr=nlist.neigh(i,j).dr();
			if(dr<rc_) J(i,jj)+=(erfc(alpha_*dr)/dr-vc);
		}
	}
	//constant terms
	for(int i=0; i<nAtoms; ++i){
		J(i,i)+=(0.5*vc+alpha_/RadPI);
	}
	//return matrix
	J*=ke;
	return J;
}

double PotCoulWolf::energy(const Structure& struc, const verlet::List& vlist){
	if(PCW_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulWolf::energy(const Structure&,const verlet::List&)\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	double q2s_=0;
	const double vc=erfc(alpha_*rc_)/rc_;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		q2s_+=qi*qi;
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double qj=struc.charge(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				energy_+=qi*qj*(erfc(alpha_*dr)/dr-vc);
			}
		}
	}
	energy_*=0.5;
	energy_-=(0.5*vc+alpha_/RadPI)*q2s_;
	return ke*energy_;
}

double PotCoulWolf::compute(Structure& struc, const verlet::List& vlist){
	if(PCW_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulWolf::compute(Structure&,const verlet::List&)\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	double q2s_=0;
	const double vc=erfc(alpha_*rc_)/rc_;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double qi=struc.charge(i);
		q2s_+=qi*qi;
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double qj=struc.charge(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double ferfc=erfc(alpha_*dr);
				const double fexp=exp(-alpha_*alpha_*dr*dr);
				energy_+=qi*qj*(ferfc/dr-vc);
				struc.force(i).noalias()+=ke*qi*qj*(2.0*alpha_*dr/RadPI*fexp+ferfc)/(dr*dr*dr)*drv;
			}
		}
	}
	energy_*=0.5;
	energy_-=(0.5*vc+alpha_/RadPI)*q2s_;
	return ke*energy_;
}

Eigen::MatrixXd& PotCoulWolf::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	if(PCW_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulWolf::J(const Structure&,const verlet::List&,Eigen::MatrixXd&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	//interaction terms
	double q2s_=0;
	const double vc=erfc(alpha_*rc_)/rc_;
	for(int i=0; i<nAtoms; ++i){
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				J(i,jj)+=(erfc(alpha_*dr)/dr-vc);
			}
		}
	}
	//constant terms
	for(int i=0; i<nAtoms; ++i){
		J(i,i)+=(0.5*vc+alpha_/RadPI);
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
	
	template <> int nbytes(const ptnl::PotCoulWolf& obj){
		if(PCW_PRINT_FUNC>0) std::cout<<"nbytes(const PotCoulWolf&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(double);//eps_
		size+=sizeof(double);//alpha_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotCoulWolf& obj, char* arr){
		if(PCW_PRINT_FUNC>0) std::cout<<"pack(const PotCoulWolf&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(arr+pos,&obj.alpha(),sizeof(double)); pos+=sizeof(double);//alpha_
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotCoulWolf& obj, const char* arr){
		if(PCW_PRINT_FUNC>0) std::cout<<"unpack(PotCoulWolf&,const char*):\n";
		int pos=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(&obj.alpha(),arr+pos,sizeof(double)); pos+=sizeof(double);//alpha_
		return pos;
	}
	
}