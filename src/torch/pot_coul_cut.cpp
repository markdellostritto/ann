// c++
#include <stdexcept>
// chem
#include "chem/units.hpp"
// pot
#include "torch/pot_coul_cut.hpp"

//==== operator ====

namespace ptnl{

std::ostream& operator<<(std::ostream& out, const PotCoulCut& pot){
	return out<<static_cast<const Pot&>(pot)<<" eps "<<pot.eps_;
}
	
//==== member functions ====

void PotCoulCut::read(Token& token){
	if(PCC_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulCut::read(Token&):\n";
	static_cast<Pot&>(*this).read(token);
	if(!token.end()){
		eps_=std::atof(token.next().c_str());
		if(eps_<=0.0) throw std::invalid_argument("PotCoulCut::read(Token&): Invalid epsilon.");
	}
}

double PotCoulCut::energy(const Structure& struc, const NeighborList& nlist){
	if(PCC_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulCut::energy(const Structure&,NeighborList&):\n";
	double energy_=0;
	const double ke=units::Consts::ke()*eps_;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		for(int j=0; j<nlist.size(i); ++j){
			const double& qj=struc.charge(nlist.neigh(i,j).index());
			const double& dr=nlist.neigh(i,j).dr();
			energy_+=qi*qj/dr;
		}
	}
	return 0.5*ke*energy_;
}

double PotCoulCut::compute(Structure& struc, const NeighborList& nlist){
	if(PCC_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulCut::compute(Structure&,NeighborList&):\n";
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		for(int j=0; j<nlist.size(i); ++j){
			const Eigen::Vector3d& r=nlist.neigh(i,j).r();
			const double& qj=struc.charge(nlist.neigh(i,j).index());
			const double& dri=1.0/nlist.neigh(i,j).dr();
			const double e=ke*qi*qj*dri;
			energy_+=e;
			struc.force(i).noalias()+=e*dri*dri*r;
		}
	}
	return 0.5*energy_;
}

Eigen::MatrixXd& PotCoulCut::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	if(PCC_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulCut::J(const Structure&,const NeighborList&,Eigen::MatrixXd&):\n";
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	//r-space
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<nlist.size(i); ++j){
			const double dr=nlist.neigh(i,j).dr();
			const int jj=nlist.neigh(i,j).index();
			if(dr<rc_) J(i,jj)+=1.0/dr;
		}
	}
	//return matrix
	J*=ke;
	return J;
}

double PotCoulCut::energy(const Structure& struc, const verlet::List& vlist){
	if(PCC_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulCut::energy(const Structure&,verlet::List&):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	const double ke=units::Consts::ke()*eps_;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double& qj=struc.charge(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				energy_+=qi*qj/sqrt(dr2);
			}
		}
	}
	return 0.5*ke*energy_;
}

double PotCoulCut::compute(Structure& struc, const verlet::List& vlist){
	if(PCC_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulCut::compute(Structure&,verlet::List&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double& qj=struc.charge(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dri=1.0/sqrt(dr2);
				const double e=ke*qi*qj*dri;
				energy_+=e;
				struc.force(i).noalias()+=e*dri*dri*drv;
			}
		}
	}
	return 0.5*energy_;
}

Eigen::MatrixXd& PotCoulCut::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	if(PCC_PRINT_FUNC>0) std::cout<<"ptnl::PotCoulCut::J(const Structure&,const verlet::List&,Eigen::MatrixXd&):\n";
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	const int nAtoms=struc.nAtoms();
	J=Eigen::MatrixXd::Zero(nAtoms,nAtoms);
	//r-space
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_) J(i,jj)+=1.0/sqrt(dr2);
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
	
	template <> int nbytes(const ptnl::PotCoulCut& obj){
		if(PCC_PRINT_FUNC>0) std::cout<<"nbytes(const PotCoulCut&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(double);//eps_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotCoulCut& obj, char* arr){
		if(PCC_PRINT_FUNC>0) std::cout<<"pack(const PotCoulCut&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);//eps_
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotCoulCut& obj, const char* arr){
		if(PCC_PRINT_FUNC>0) std::cout<<"unpack(PotCoulCut&,const char*):\n";
		int pos=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);//eps_
		return pos;
	}
	
}