// chem
#include "chem/units.hpp"
// math
#include "math/const.hpp"
// pot
#include "torch/pot_coul_long.hpp"

namespace ptnl{

//==== using statements ====

using math::constant::RadPI;

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotCoulLong& pot){
	return out<<static_cast<const Pot&>(pot)<<" "<<pot.prec_;
}
	
//==== member functions ====

void PotCoulLong::read(Token& token){
	static_cast<Pot&>(*this).read(token);
	//pot coul_long 6.0 1e-12
	prec_=std::atof(token.next().c_str());
	if(prec_<=0) throw std::invalid_argument("ptnl::PotCoulLong::read(Token&): invalid precision.");
	if(!token.end()){
		eps_=std::atof(token.next().c_str());
		if(eps_<=0.0) throw std::invalid_argument("ptnl::PotCoulLong::read(Token&): Invalid epsilon.");
		coul_.eps()=eps_;
	}
}

void PotCoulLong::init(){
	coul_.prec()=prec_;
	coul_.rc()=rc_;
}

double PotCoulLong::energy(const Structure& struc, const NeighborList& nlist){
	const double ke=units::Consts::ke()*eps_;
	coul_.init(struc);
	const double a=coul_.alpha();
	if(PCL_PRINT_DATA>0) std::cout<<"a = "<<a<<"\n";
	// r-space
	double energyR=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		for(int j=0; j<nlist.size(i); ++j){
			const double& qj=struc.charge(nlist.neigh(i,j).index());
			const double& dr=nlist.neigh(i,j).dr();
			if(dr<rc_) energyR+=qi*qj*erfc(a*dr)/dr;
		}
	}
	energyR*=0.5*ke;
	// k-space
	const double energyK=coul_.energy(struc);
	if(PCL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR<<"\n";
		std::cout<<"energyK = "<<energyK<<"\n";
	}
	//total
	return energyR+energyK;
}

double PotCoulLong::compute(Structure& struc, const NeighborList& nlist){
	const double ke=units::Consts::ke()*eps_;
	// k-space
	coul_.init(struc);
	const double energyK=coul_.compute(struc,nlist);
	const double a=coul_.alpha();
	if(PCL_PRINT_DATA>0) std::cout<<"a = "<<a<<"\n";
	// r-space
	double energyR=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		for(int j=0; j<nlist.size(i); ++j){
			const Eigen::Vector3d& rij=nlist.neigh(i,j).r();
			const double& qj=struc.charge(nlist.neigh(i,j).index());
			const double& dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const double erfcf=erfc(a*dr);
				const double expf=exp(-a*a*dr*dr);
				energyR+=qi*qj*erfcf/dr;
				struc.force(i).noalias()+=ke*qi*qj*(erfcf+2.0*a*dr/RadPI*expf)/(dr*dr*dr)*rij;
			}
		}
	}
	energyR*=0.5*ke;
	if(PCL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR<<"\n";
		std::cout<<"energyK = "<<energyK<<"\n";
	}
	//total
	return energyR+energyK;
}

Eigen::MatrixXd& PotCoulLong::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	const double ke=units::Consts::ke()*eps_;
	coul_.init(struc);
	const double a=coul_.alpha();
	if(PCL_PRINT_DATA>0) std::cout<<"a = "<<a<<"\n";
	const int nAtoms=struc.nAtoms();
	//k-space
	coul_.J(struc,J);
	Eigen::MatrixXd Jr=Eigen::MatrixXd::Zero(struc.nAtoms(),struc.nAtoms());
	//r-space
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<nlist.size(i); ++j){
			const double& dr=nlist.neigh(i,j).dr();
			const int jj=nlist.neigh(i,j).index();
			if(dr<rc_) Jr(i,jj)+=erfc(a*dr)/dr;
		}
	}
	//return matrix
	J.noalias()+=Jr*ke;
	return J;
}

double PotCoulLong::energy(const Structure& struc, const verlet::List& vlist){
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	coul_.init(struc);
	const double a=coul_.alpha();
	if(PCL_PRINT_DATA>0) std::cout<<"a = "<<a<<"\n";
	// r-space
	double energyR=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double& qj=struc.charge(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				energyR+=qi*qj*erfc(a*dr)/dr;
			}
		}
	}
	energyR*=0.5*ke;
	// k-space
	const double energyK=coul_.energy(struc);
	if(PCL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR<<"\n";
		std::cout<<"energyK = "<<energyK<<"\n";
	}
	//total
	return energyR+energyK;
}

double PotCoulLong::compute(Structure& struc, const verlet::List& vlist){
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	// k-space
	coul_.init(struc);
	const double energyK=coul_.compute(struc,vlist);
	const double a=coul_.alpha();
	if(PCL_PRINT_DATA>0) std::cout<<"a = "<<a<<"\n";
	// r-space
	double energyR=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const double& qi=struc.charge(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double& qj=struc.charge(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				const double erfcf=erfc(a*dr);
				const double expf=exp(-a*a*dr2);
				energyR+=qi*qj*erfcf/dr;
				struc.force(i).noalias()+=ke*qi*qj*(erfcf+2.0*a*dr/RadPI*expf)/(dr*dr2)*drv;
			}
		}
	}
	energyR*=0.5*ke;
	if(PCL_PRINT_DATA>0){
		std::cout<<"energyR = "<<energyR<<"\n";
		std::cout<<"energyK = "<<energyK<<"\n";
	}
	//total
	return energyR+energyK;
}

Eigen::MatrixXd& PotCoulLong::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	Eigen::Vector3d drv;
	const double ke=units::Consts::ke()*eps_;
	coul_.init(struc);
	const double a=coul_.alpha();
	if(PCL_PRINT_DATA>0) std::cout<<"a = "<<a<<"\n";
	const int nAtoms=struc.nAtoms();
	//k-space
	coul_.J(struc,J);
	Eigen::MatrixXd Jr=Eigen::MatrixXd::Zero(struc.nAtoms(),struc.nAtoms());
	//r-space
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr=sqrt(dr2);
				Jr(i,jj)+=erfc(a*dr)/dr;
			}
		}
	}
	//return matrix
	J.noalias()+=Jr*ke;
	return J;
}

double PotCoulLong::cQ(Structure& struc){
	coul_.init(struc);
	return units::Consts::ke()*0.5*math::constant::PI/(coul_.alpha()*struc.vol());
}

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotCoulLong& obj){
		if(PCL_PRINT_FUNC>0) std::cout<<"nbytes(const PotCoulLong&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(double);//eps_
		size+=sizeof(double);//prec_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotCoulLong& obj, char* arr){
		if(PCL_PRINT_FUNC>0) std::cout<<"pack(const PotCoulLong&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.eps(),sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(arr+pos,&obj.prec(),sizeof(double)); pos+=sizeof(double);//prec_
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotCoulLong& obj, const char* arr){
		if(PCL_PRINT_FUNC>0) std::cout<<"unpack(PotCoulLong&,const char*):\n";
		int pos=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&obj.eps(),arr+pos,sizeof(double)); pos+=sizeof(double);//eps_
		std::memcpy(&obj.prec(),arr+pos,sizeof(double)); pos+=sizeof(double);//prec_
		obj.init();
		return pos;
	}
	
}