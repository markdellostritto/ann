// math
// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// pot
#include "torch/pot_ldamp_cut.hpp"
// thread
#include "thread/dist.hpp"

namespace ptnl{

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotLDampCut& pot){
	return out<<static_cast<const Pot&>(pot)<<" LDAMP_A "<<LDAMP_A;
}
	
//==== member functions ====

void PotLDampCut::read(Token& token){
	static_cast<Pot&>(*this).read(token);
}

void PotLDampCut::coeff(Token& token){
	if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampCut::coeff(Token&):\n";
	//coeff ldamp_cut type1 type2 c6 rvdw
	const int t1=std::atof(token.next().c_str())-1;
	const int t2=std::atof(token.next().c_str())-1;
	const double c6=std::atof(token.next().c_str());
	const double rvdw=std::atof(token.next().c_str());
	
	if(t1>=ntypes_) throw std::invalid_argument("ptnl::PotLDampCut::coeff(Token&): Invalid type.");
	if(t2>=ntypes_) throw std::invalid_argument("ptnl::PotLDampCut::coeff(Token&): Invalid type.");
	
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
			rvdw6_(i,j)=rvdw6; rvdw6_(j,i)=rvdw6;
			f_(i,j)=1; f_(j,i)=1;
		}
	}
}

void PotLDampCut::resize(int ntypes){
	if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampCut::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::MatrixXi::Zero(ntypes_,ntypes_);
		c6_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		rvdw_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		rvdw6_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotLDampCut::init(){
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
}

double PotLDampCut::energy(const Structure& struc, const NeighborList& nlist){
	if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampCut::energy(const Structure&,const NeighborList&):\n";
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const int tj=struc.type(nlist.neigh(i,j).index());
				const double dr2=dr*dr;
				const double dr6=dr2*dr2*dr2;
				#if LDAMP_A == 3
					//a=3,b=2
					const double rij=rvdw_(ti,tj);
					const double den=1.0/(dr*dr*dr+rij*rij*rij);
					energy_-=c6_(ti,tj)*den*den;
				#elif LDAMP_A == 6
					//a=6,b=1
					energy_-=c6_(ti,tj)/(dr6+rvdw6_(ti,tj));
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double rij6=rvdw6_(ti,tj);
					energy_-=c6_(ti,tj)/sqrt(dr6*dr6+rij6*rij6);
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotLDampCut::compute(Structure& struc, const NeighborList& nlist){
	if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampCut::compute(const Structure&,const NeighborList&):\n";
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const Eigen::Vector3d& r=nlist.neigh(i,j).r();
				#if LDAMP_A == 3
					const double rij=rvdw_(ti,tj);
					const double den=1.0/(dr*dr*dr+rij*rij*rij);
					energy_-=c6_(ti,tj)*den*den;
					struc.force(i).noalias()-=6.0*c6_(ti,tj)*dr*den*den*den*r;
				#elif LDAMP_A == 6
					//a=6,b=1
					const double dr2=dr*dr;
					const double dr6=dr2*dr2*dr2;
					const double den=1.0/(dr6+rvdw6_(ti,tj));
					energy_-=c6_(ti,tj)*den;
					struc.force(i).noalias()-=6.0*c6_(ti,tj)*dr2*dr2*den*den*r;
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double dr2=dr*dr;
					const double dr6=dr2*dr2*dr2;
					const double rij6=rvdw6_(ti,tj);
					const double den=1.0/sqrt(dr6*dr6+rij6*rij6);
					energy_-=c6_(ti,tj)*den;
					struc.force(i).noalias()-=6.0*c6_(ti,tj)*dr2*dr2*dr6*den*den*den*r;
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

Eigen::MatrixXd& PotLDampCut::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampCut::coeff(const Structure&,const NeighborList&,Eigen::MatrixXd&):\n";
	J=Eigen::MatrixXd::Zero(struc.nAtoms(),struc.nAtoms());
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			const int jj=nlist.neigh(i,j).index();
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				#if LDAMP_A == 3
					const double rij=rvdw_(ti,tj);
					const double den=1.0/(dr*dr*dr+rij*rij*rij);
					J(i,jj)-=den*den;
				#elif LDAMP_A == 6
					const double dr2=dr*dr;
					const double dr6=dr2*dr2*dr2;
					J(i,jj)-=1.0/(dr6+rvdw6_(ti,tj));
				#elif LDAMP_A == 12
					const double dr2=dr*dr;
					const double dr6=dr2*dr2*dr2;
					const double rij6=rvdw6_(ti,tj);
					J(i,jj)=1.0/sqrt(dr6*dr6+rij6*rij6);
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	return J;
}

double PotLDampCut::energy(const Structure& struc, const verlet::List& vlist){
	if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampCut::energy(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	double energy_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr6=dr2*dr2*dr2;
				#if LDAMP_A == 3
					//a=3,b=2
					const double dr=sqrt(dr2);
					const double rij=rvdw_(ti,tj);
					const double den=1.0/(dr*dr*dr+rij*rij*rij);
					energy_-=c6_(ti,tj)*den*den;
				#elif LDAMP_A == 6
					//a=6,b=1
					energy_-=c6_(ti,tj)/(dr6+rvdw6_(ti,tj));
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double rij6=rvdw6_(ti,tj);
					energy_-=c6_(ti,tj)/sqrt(dr6*dr6+rij6*rij6);
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

double PotLDampCut::compute(Structure& struc, const verlet::List& vlist){
	if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampCut::compute(const Structure&,const verlet::List&):\n";
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
				#if LDAMP_A == 3
					const double dr=sqrt(dr2);
					const double rij=rvdw_(ti,tj);
					const double den=1.0/(dr*dr*dr+rij*rij*rij);
					energy_-=c6_(ti,tj)*den*den;
					struc.force(i).noalias()-=6.0*c6_(ti,tj)*dr*den*den*den*drv;
				#elif LDAMP_A == 6
					//a=6,b=1
					const double dr6=dr2*dr2*dr2;
					const double den=1.0/(dr6+rvdw6_(ti,tj));
					energy_-=c6_(ti,tj)*den;
					struc.force(i).noalias()-=6.0*c6_(ti,tj)*dr2*dr2*den*den*drv;
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double dr6=dr2*dr2*dr2;
					const double rij6=rvdw6_(ti,tj);
					const double den=1.0/sqrt(dr6*dr6+rij6*rij6);
					energy_-=c6_(ti,tj)*den;
					struc.force(i).noalias()-=6.0*c6_(ti,tj)*dr2*dr2*dr6*den*den*den*drv;
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	energy_*=0.5;
	return energy_;
}

Eigen::MatrixXd& PotLDampCut::J(const Structure& struc, const verlet::List& vlist, Eigen::MatrixXd& J){
	if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampCut::coeff(const Structure&,const verlet::List&,Eigen::MatrixXd&):\n";
	Eigen::Vector3d drv;
	J=Eigen::MatrixXd::Zero(struc.nAtoms(),struc.nAtoms());
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int tj=struc.type(vlist.neigh(i,j).index());
			const int jj=vlist.neigh(i,j).index();
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				#if LDAMP_A == 3
					J(i,jj)=0;
				#elif LDAMP_A == 6
					const double dr6=dr2*dr2*dr2;
					J(i,jj)-=1.0/(dr6+rvdw6_(ti,tj));
				#elif LDAMP_A == 12
					J(i,jj)=0;
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
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
	
	template <> int nbytes(const ptnl::PotLDampCut& obj){
		if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotLDampCut&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(int);//name_
		size+=sizeof(int);//ntypes_
		const int nt=obj.ntypes();
		if(nt>0){
			size+=nt*nt*sizeof(double);//rvdw_
			size+=nt*nt*sizeof(double);//c6_
		}
		return size;
	}
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLDampCut& obj, char* arr){
		if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotLDampCut&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.name(),sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		const int nt=obj.ntypes();
		if(nt>0){
			std::memcpy(arr+pos,obj.rvdw().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//rvdw_
			std::memcpy(arr+pos,obj.c6().data(),nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//c6_
		}
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(ptnl::PotLDampCut& obj, const char* arr){
		if(POT_LDAMP_CUT_PRINT_FUNC>0) std::cout<<"unpack(PotLDampCut&,const char*):\n";
		int pos=0,nt=0;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		ptnl::Pot::Name name;
		std::memcpy(&name,arr+pos,sizeof(int)); pos+=sizeof(int);//name_
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//ntypes_
		if(name!=ptnl::Pot::Name::LDAMP_CUT) throw std::invalid_argument("serialize::unpack(PotLDampCut&,const char*): Invalid name.");
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.rvdw().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//rvdw_
			std::memcpy(obj.c6().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//c6_
		}
		obj.init();
		return pos;
	}
	
}