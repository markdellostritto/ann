// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// pot
#include "torch/pot_spin_ex.hpp"
// thread
#include "thread/dist.hpp"

namespace ptnl{

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotSpinEx& pot){
	return out<<static_cast<const Pot&>(pot);
}
	
//==== member functions ====

void PotSpinEx::read(Token& token){
	static_cast<Pot&>(*this).read(token);
}

void PotSpinEx::coeff(Token& token){
	//coeff london_cut rcut
	static_cast<Pot&>(*this).read(token);
	//coeff lj_cut type1 type2 radius alpha
	const int t1=std::atoi(token.next().c_str())-1;
	const int t2=std::atoi(token.next().c_str())-1;
	const double radius=std::atof(token.next().c_str());
	const double alpha=std::atof(token.next().c_str());
	
	if(t1>=ntypes_) throw std::invalid_argument("Invalid type.");
	if(t2>=ntypes_) throw std::invalid_argument("Invalid type.");
	
	int t1min=t1,t1max=t1;
	int t2min=t2,t2max=t2;
	if(t1<0){t1min=0;t1max=ntypes_-1;}
	if(t2<0){t2min=0;t2max=ntypes_-1;}
	for(int i=t1min; i<=t1max; ++i){
		for(int j=t2min; j<=t2max; ++j){
			rad_(i,j)=radius; rad_(j,i)=radius;
			alpha_(i,j)=alpha; alpha_(j,i)=alpha;
			f_(i,j)=1; f_(j,i)=1;
		}
	}
}

void PotSpinEx::resize(int ntypes){
	if(POT_SPINEX_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotSpinEx::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotSpinEx::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::MatrixXi::Zero(ntypes_,ntypes_);
		rad_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		alpha_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotSpinEx::init(){
	if(POT_SPINEX_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotSpinEx::init():\n";
	for(int i=0; i<ntypes_; ++i){
		for(int j=i+1; j<ntypes_; ++j){
			if(f_(i,j)==0){
				rad_(i,j)=0.5*(rad_(i,i)+rad_(j,j));
				rad_(j,i)=rad_(i,j);
				alpha_(i,j)=sqrt(alpha_(i,i)*alpha_(j,j));
				alpha_(j,i)=alpha_(i,j);
				f_(i,j)=1; f_(j,i)=1;
			}			
		}
	}
}

double PotSpinEx::energy(const Structure& struc, const NeighborList& nlist){
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			if(nlist.neigh(i,j).dr()<rc_){
				const double rij=rad_(i,j);
				const double dr=nlist.neigh(i,j).dr();
				const double aij=alpha_(i,j);
				const double r2=(dr*dr)/(rij*rij);
				energy_-=4.0*aij*r2*exp(-r2)*struc.spin(i).dot(struc.spin(jj));
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

Eigen::MatrixXd& PotSpinEx::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	J=Eigen::MatrixXd::Zero(struc.nAtoms(),struc.nAtoms());
	for(int i=0; i<struc.nAtoms(); ++i){
		const double ri=ptable::radius_covalent(struc.an(i));
		for(int j=0; j<nlist.size(i); ++j){
			const int jj=nlist.neigh(i,j).index();
			const double rj=ptable::radius_covalent(struc.an(jj));
			if(nlist.neigh(i,j).dr()<rc_){
				const double rij=rad_(i,j);
				const double dr=nlist.neigh(i,j).dr();
				const double aij=alpha_(i,j);
				const double r2=(dr*dr)/(rij*rij);
				J(i,jj)-=4.0*aij*r2*exp(-r2)*struc.spin(i).dot(struc.spin(jj));
			}
		}
	}
	//return matrix
	return J;
}

double PotSpinEx::energy(const Structure& struc, const verlet::List& vlist){
	Eigen::Vector3d drv;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double rij=rad_(i,j);
				const double aij=alpha_(i,j);
				const double r2=dr2/(rij*rij);
				energy_-=4.0*aij*r2*exp(-r2)*struc.spin(i).dot(struc.spin(jj));
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

Eigen::MatrixXd& PotSpinEx::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	Eigen::Vector3d drv;
	J=Eigen::MatrixXd::Zero(struc.nAtoms(),struc.nAtoms());
	for(int i=0; i<struc.nAtoms(); ++i){
		const double ri=ptable::radius_covalent(struc.an(i));
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double rij=rad_(i,j);
				const double aij=alpha_(i,j);
				const double r2=dr2/(rij*rij);
				J(i,jj)-=4.0*aij*r2*exp(-r2)*struc.spin(i).dot(struc.spin(jj));
			}
		}
	}
	//return matrix
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
	
	template <> int nbytes(const ptnl::PotSpinEx& obj){
		if(POT_SPINEX_CUT_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotSpinEx&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(int);//name_
		size+=sizeof(int);//ntypes_
		return size;
	}
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotSpinEx& obj, char* arr){
		if(POT_SPINEX_CUT_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotSpinEx&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.name(),sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotSpinEx& obj, const char* arr){
		if(POT_SPINEX_CUT_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotSpinEx&,const char*):\n";
		int pos=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		ptnl::Pot::Name name;
		std::memcpy(&name,arr+pos,sizeof(int)); pos+=sizeof(int);//name_
		if(name!=ptnl::Pot::Name::LJ_CUT) throw std::invalid_argument("serialize::unpack(ptnl::PotSpinEx&,const char*): Invalid name.");
		int nt=0;
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		return pos;
	}
	
}
