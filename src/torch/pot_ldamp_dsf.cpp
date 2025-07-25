// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// pot
#include "torch/pot_ldamp_dsf.hpp"
// thread
#include "thread/dist.hpp"

namespace ptnl{

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotLDampDSF& pot){
	return out<<static_cast<const Pot&>(pot)<<" "<<pot.a_;
}

double operator-(const PotLDampDSF& pot1, const PotLDampDSF& pot2){
	return (static_cast<const Pot&>(pot1)-static_cast<const Pot&>(pot2))
		+(pot1.rvdw()-pot2.rvdw()).norm()
		+(pot1.c6()-pot2.c6()).norm();
}

//==== member functions ====

void PotLDampDSF::read(Token& token){
	static_cast<Pot&>(*this).read(token);
	//pot ldamp_dsf 6.0 a
	a_=std::atof(token.next().c_str());
	if(a_<=0) throw std::invalid_argument("ptnl::PotLDampDSF::read(Token&): Invalid damping parameter.");
}

void PotLDampDSF::coeff(Token& token){
	if(POT_LDAMP_DSF_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampDSF::coeff(Token& token)\n";
	//coeff ldamp_cut type1 type2 c6 rvdw
	const int t1=std::atoi(token.next().c_str())-1;
	const int t2=std::atoi(token.next().c_str())-1;
	const double c6=std::atof(token.next().c_str());
	const double rvdw=std::atof(token.next().c_str());
	
	if(t1>=ntypes_) throw std::invalid_argument("ptnl::PotLDampDSF::coeff(Token&): Invalid type.");
	if(t2>=ntypes_) throw std::invalid_argument("ptnl::PotLDampDSF::coeff(Token&): Invalid type.");
	
	int t1min=t1,t1max=t1;
	int t2min=t2,t2max=t2;
	if(t1<0){t1min=0;t1max=ntypes_-1;}
	if(t2<0){t2min=0;t2max=ntypes_-1;}
	for(int i=t1min; i<=t1max; ++i){
		for(int j=t2min; j<=t2max; ++j){
			const double rvdw3=rvdw*rvdw*rvdw;
			const double rvdw6=rvdw3*rvdw3;
			c6_(i,j)=c6; c6_(j,i)=c6;
			rvdw_(i,j)=rvdw; rvdw_(j,i)=rvdw;
			rvdw6_(i,j)=rvdw6;
			rvdw6_(j,i)=rvdw6_(i,j);
			f_(i,j)=1; f_(j,i)=1;
		}
	}
}

void PotLDampDSF::resize(int ntypes){
	if(POT_LDAMP_DSF_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampDSF::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotLDampDSF::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::MatrixXi::Zero(ntypes_,ntypes_);
		c6_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		rvdw_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		rvdw6_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		potfRc_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		potgRc_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotLDampDSF::init(){
	a2_=a_*a_;
	rc2_=rc_*rc_;
	rc6_=rc2_*rc2_*rc2_;
	for(int i=0; i<ntypes_; ++i){
		for(int j=i+1; j<ntypes_; ++j){
			if(f_(i,j)==0){
				c6_(i,j)=sqrt(c6_(i,i)*c6_(j,j));
				c6_(j,i)=c6_(i,j);
				rvdw_(i,j)=0.5*(rvdw_(i,i)+rvdw_(j,j));
				rvdw_(j,i)=rvdw_(i,j);
				f_(i,j)=1; f_(j,i)=1;
			}
		}
	}
	for(int i=0; i<ntypes_; ++i){
		for(int j=i; j<ntypes_; ++j){
			const double rvdw=rvdw_(i,j);
			const double rvdw3=rvdw*rvdw*rvdw;
			const double rvdw6=rvdw3*rvdw3;
			rvdw6_(i,j)=rvdw6;
			rvdw6_(j,i)=rvdw6_(i,j);
		}
	}
	const double bc2=rc2_*a2_;
	for(int i=0; i<ntypes_; ++i){
		for(int j=i; j<ntypes_; ++j){
			potfRc_(i,j)=(
				-1.0/(rc6_+rvdw6_(i,j))
				-(-1.0/rc6_)
				-exp(-bc2)*(1.0+bc2*(1.0+0.5*bc2))/rc6_
			);
			potfRc_(j,i)=potfRc_(i,j);
		}
	}
	for(int i=0; i<ntypes_; ++i){
		for(int j=i; j<ntypes_; ++j){
			potgRc_(i,j)=-1.0/rc_*(
				-6.0*rc6_/((rc6_+rvdw6_(i,j))*(rc6_+rvdw6_(i,j)))
				-(-6.0/rc6_)
				-exp(-bc2)*(6.0+bc2*(6.0+bc2*(3.0+bc2)))/rc6_
			);
			potgRc_(j,i)=potgRc_(i,j);
		}
	}
}

double PotLDampDSF::energy(const Structure& struc, const NeighborList& nlist){
	if(POT_LDAMP_DSF_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampDSF::resize(const Structure&,const NeighborList&):\n";
	//compute r-space energy
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const int tj=struc.type(nlist.neigh(i,j).index());
				const double dr2=dr*dr;
				const double dr6=dr2*dr2*dr2;
				const double b2=dr2*a2_;
				const double potfr_=(
					-1.0/(dr6+rvdw6_(ti,tj))
					-(-1.0/dr6)
					-exp(-b2)*(1.0+b2*(1.0+0.5*b2))/dr6
				);
				energy_+=c6_(ti,tj)*(potfr_-potfRc_(ti,tj)-potgRc_(ti,tj)*(dr-rc_));
			}
		}
	}
	energy_*=0.5;
	//return total energy
	return energy_;
}

double PotLDampDSF::compute(Structure& struc, const NeighborList& nlist){
	if(POT_LDAMP_DSF_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampDSF::compute(Structure&,const NeighborList&):\n";
	//compute r-space energy
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const Eigen::Vector3d& r=nlist.neigh(i,j).r();
				const double dr2=dr*dr;
				const double dr6=dr2*dr2*dr2;
				const double b2=dr2*a2_;
				const double expf=exp(-b2);
				const double den=1.0/(dr6+rvdw6_(ti,tj));
				const double potfr_=(
					-1.0*den
					-(-1.0/dr6)
					-expf*(1.0+b2*(1.0+0.5*b2))/dr6
				);
				const double potgr_=-1.0/dr*(
					-6.0*dr6*den*den
					-(-6.0/dr6)
					-expf*(6.0+b2*(6.0+b2*(3.0+b2)))/dr6
				);
				energy_+=c6_(ti,tj)*(potfr_-potfRc_(ti,tj)-potgRc_(ti,tj)*(dr-rc_));
				struc.force(i).noalias()+=-1.0*r*c6_(ti,tj)/dr*(potgr_-potgRc_(ti,tj));
			}
		}
	}
	energy_*=0.5;
	//return total energy
	return energy_;
}

double PotLDampDSF::energy(const Structure& struc, const verlet::List& vlist){
	if(POT_LDAMP_DSF_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampDSF::resize(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	//compute r-space energy
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr6=dr2*dr2*dr2;
				const double b2=dr2*a2_;
				const double potfr_=(
					-1.0/(dr6+rvdw6_(ti,tj))
					-(-1.0/dr6)
					-exp(-b2)*(1.0+b2*(1.0+0.5*b2))/dr6
				);
				energy_+=c6_(ti,tj)*(potfr_-potfRc_(ti,tj)-potgRc_(ti,tj)*(sqrt(dr2)-rc_));
			}
		}
	}
	energy_*=0.5;
	//return total energy
	return energy_;
}

double PotLDampDSF::compute(Structure& struc, const verlet::List& vlist){
	if(POT_LDAMP_DSF_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampDSF::compute(Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	//compute r-space energy
	const double a2=a_*a_;
	const double a4=a2*a2;
	const double a6=a4*a2;
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
				const double dr6=dr2*dr2*dr2;
				const double b2=dr2*a2_;
				const double expf=exp(-b2);
				const double den=1.0/(dr6+rvdw6_(ti,tj));
				const double potfr_=(
					-1.0*den
					-(-1.0/dr6)
					-expf*(1.0+b2*(1.0+0.5*b2))/dr6
				);
				const double potgr_=-1.0/dr*(
					-6.0*dr6*den*den
					-(-6.0/dr6)
					-expf*(6.0+b2*(6.0+b2*(3.0+b2)))/dr6
				);
				energy_+=c6_(ti,tj)*(potfr_-potfRc_(ti,tj)-potgRc_(ti,tj)*(dr-rc_));
				struc.force(i).noalias()+=drv*c6_(ti,tj)/dr*(potgr_-potgRc_(ti,tj));
			}
		}
	}
	energy_*=0.5;
	//return total energy
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
	
	template <> int nbytes(const ptnl::PotLDampDSF& obj){
		if(POT_LDAMP_DSF_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotLDampDSF&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(ptnl::Pot::Name);//name_
		size+=sizeof(int);//ntypes_
		size+=sizeof(double);//a_
		const int nt=obj.ntypes();
		size+=nt*nt*sizeof(double);//rvdw_
		size+=nt*nt*sizeof(double);//c6_
		size+=nt*nt*sizeof(double);//potfRc_
		size+=nt*nt*sizeof(double);//potgRc_
		return size;
	}
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLDampDSF& obj, char* arr){
		if(POT_LDAMP_DSF_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotLDampDSF&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.name(),sizeof(ptnl::Pot::Name)); pos+=sizeof(ptnl::Pot::Name);//name_
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(arr+pos,&obj.a(),sizeof(double)); pos+=sizeof(double);//a_
		const int nt=obj.ntypes();
		if(nt>0){
			std::memcpy(arr+pos,obj.rvdw().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//rvdw_
			std::memcpy(arr+pos,obj.c6().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//c6_
			std::memcpy(arr+pos,obj.potfRc().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//rvdw_
			std::memcpy(arr+pos,obj.potgRc().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//c6_
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotLDampDSF& obj, const char* arr){
		if(POT_LDAMP_DSF_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotLDampDSF&,const char*):\n";
		int pos=0,nt=0;
		ptnl::Pot::Name name;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&name,arr+pos,sizeof(ptnl::Pot::Name)); pos+=sizeof(ptnl::Pot::Name);//name_
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//nt_
		std::memcpy(&obj.a(),arr+pos,sizeof(double)); pos+=sizeof(double);//a_
		if(name!=ptnl::Pot::Name::LDAMP_DSF) throw std::invalid_argument("serialize::unpack(ptnl::PotLDampDSF&,const char*): Invalid name.");
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.rvdw().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//rvdw_
			std::memcpy(obj.c6().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//c6_
			std::memcpy(obj.potfRc().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//rvdw_
			std::memcpy(obj.potgRc().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//c6_
		}
		obj.init();
		return pos;
	}
	
}