// math
// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// pot
#include "torch/pot_ldamp_long.hpp"
// thread
#include "thread/dist.hpp"

namespace ptnl{

//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotLDampLong& pot){
	return out<<static_cast<const Pot&>(pot)<<" "<<pot.prec_<<" LDAMP_A "<<LDAMP_A;
}

double operator-(const PotLDampLong& pot1, const PotLDampLong& pot2){
	return (static_cast<const Pot&>(pot1)-static_cast<const Pot&>(pot2))
		+std::fabs(pot1.prec()+pot2.prec())
		+(pot1.rvdw()-pot2.rvdw()).norm()
		+(pot1.c6()-pot2.c6()).norm()
		+(pot1.ksl()-pot2.ksl());
}

//==== member functions ====

void PotLDampLong::read(Token& token){
	static_cast<Pot&>(*this).read(token);
	//pot ldamp_long 6.0 1e-12
	prec_=std::atof(token.next().c_str());
	if(prec_<=0) throw std::invalid_argument("ptnl::PotLDampLong::read(Token&): Invalid precision.");
}

void PotLDampLong::coeff(Token& token){
	if(POT_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampLong::coeff(Token& token)\n";
	//coeff ldamp_cut type1 type2 c6 rvdw
	const int t1=std::atoi(token.next().c_str())-1;
	const int t2=std::atoi(token.next().c_str())-1;
	const double c6=std::atof(token.next().c_str());
	const double rvdw=std::atof(token.next().c_str());
	
	if(t1>=ntypes_) throw std::invalid_argument("ptnl::PotLDampLong::coeff(Token&): Invalid type.");
	if(t2>=ntypes_) throw std::invalid_argument("ptnl::PotLDampLong::coeff(Token&): Invalid type.");
	
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

void PotLDampLong::resize(int ntypes){
	if(POT_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampLong::resize(int):\n";
	if(ntypes<0) throw std::invalid_argument("ptnl::PotLDampLong::resize(int): Invalid number of types.");
	ntypes_=ntypes;
	if(ntypes_>0){
		f_=Eigen::MatrixXi::Zero(ntypes_,ntypes_);
		c6_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		rvdw_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
		rvdw6_=Eigen::MatrixXd::Zero(ntypes_,ntypes_);
	}
}

void PotLDampLong::init(){
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
	ksl_.prec()=prec_;
	ksl_.rc()=rc_;
}

double PotLDampLong::energy(const Structure& struc, const NeighborList& nlist){
	if(POT_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampLong::resize(const Structure&,const NeighborList&):\n";
	//compute k-space energy
	ksl_.init(struc,c6_);
	const double energyK_=ksl_.energy(struc);
	//compute r-space energy
	const double a=ksl_.alpha();
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const int tj=struc.type(nlist.neigh(i,j).index());
				const double dr3=dr*dr*dr;
				const double dr6=dr3*dr3;
				const double b=dr*a;
				const double b2=b*b;
				const double c=exp(-b2)*(1.0+b2*(1.0+0.5*b2));
				#if LDAMP_A == 3
					//a=3,b=2
					const double rij=rvdw_(ti,tj);
					const double den=1.0/(dr3+rij*rij*rij);
					energyR_+=c6_(ti,tj)*((1.0-c)/dr6-den*den);
				#elif LDAMP_A == 6
					//a=6,b=1
					energyR_+=c6_(ti,tj)*((1.0-c)/dr6-1.0/(dr6+rvdw6_(ti,tj)));
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double rij6=rvdw6_(ti,tj);
					energyR_+=c6_(ti,tj)*((1.0-c)/dr6-1.0/sqrt(dr6*dr6+rij6*rij6));
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	energyR_*=0.5;
	//return total energy
	return energyR_+energyK_;
}

double PotLDampLong::compute(Structure& struc, const NeighborList& nlist){
	if(POT_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampLong::compute(Structure&,const NeighborList&):\n";
	//compute k-space energy
	ksl_.init(struc,c6_);
	const double energyK_=ksl_.compute(struc);
	//compute r-space energy
	const double a=ksl_.alpha();
	const double a2=a*a;
	const double a4=a2*a2;
	const double a6=a4*a2;
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const int tj=struc.type(nlist.neigh(i,j).index());
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const Eigen::Vector3d& r=nlist.neigh(i,j).r();
				const double dr2=dr*dr;
				const double dr6=dr2*dr2*dr2;
				const double b=dr*a;
				const double b2=b*b;
				const double expf=exp(-b2);
				const double cc=(1.0-exp(-b2)*(1.0+b2*(1.0+0.5*b2)));
				#if LDAMP_A == 3
					//a=3,b=2
					const double rij=rvdw_(ti,tj);
					const double den=1.0/(dr*dr*dr+rij*rij*rij);
					energyR_+=c6_(ti,tj)*(cc/dr6-den*den);
					struc.force(i).noalias()+=r*c6_(ti,tj)*(
						(6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*dr2)-6.0*dr*den*den*den
					);
				#elif LDAMP_A == 6
					//a=6,b=1
					const double den=1.0/(dr6+rvdw6_(ti,tj));
					energyR_+=c6_(ti,tj)*(cc/dr6-den);
					struc.force(i).noalias()+=r*c6_(ti,tj)*(
						(6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*dr2)-6.0*dr2*dr2*den*den
					);
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double rij6=rvdw6_(ti,tj);
					const double den=1.0/sqrt(dr6*dr6+rij6*rij6);
					energyR_+=c6_(ti,tj)*(cc/dr6-den);
					struc.force(i).noalias()+=r*c6_(ti,tj)*(
						(6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*dr2)-6.0*dr2*dr2*dr6*den*den*den
					);
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	energyR_*=0.5;
	//return total energy
	return energyR_+energyK_;
}

/*
Eigen::MatrixXd& PotLDampLong::J(const Structure& struc, const NeighborList& nlist, Eigen::MatrixXd& J){
	const double ke=units::consts::ke()*eps_;
	coul_.init(struc);
	const double a=coul_.alpha();
	if(PCL_PRINT_DATA>0) std::cout<<"a = "<<a<<"\n";
	const int nAtoms=struc.nAtoms();
	//k-space
	coul_.J(struc,J);
	Eigen::MatrixXd Jr=Eigen::MatrixXd::Zero(struc.nAtoms(),struc.nAtoms());
	//r-space
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<nlist.size(i); ++j){
			const double dr=nlist.neigh(i,j).dr();
			if(dr<rc_){
				const int tj=struc.type(nlist.neigh(i,j).index());
				const double dr2=dr*dr;
				const double dr6=dr2*dr2*dr2;
				const double rij=rvdw_(ti,tj);
				const double rij2=rij*rij;
				const double rij6=rij2*rij2*rij2;
				const double b=dr*a;
				const double b2=b*b;
				#if LDAMP_A == 3
					//a=3,b=2
					const double den=1.0/(dr*dr*dr+rij2*rij);
					energyR_+=(1.0-exp(-b2)*(1.0+b2*(1.0+0.5*b2)))/dr6-den*den;
				#elif LDAMP_A == 6
					//a=6,b=1
					energyR_+=(1.0-exp(-b2)*(1.0+b2*(1.0+0.5*b2)))/dr6-1.0/(dr6+rij6);
				#elif LDAMP_A == 12
					//a=12,b=1/2
					energyR_+=(1.0-exp(-b2)*(1.0+b2*(1.0+0.5*b2)))/dr6-1.0/sqrt(dr6*dr6+rij6*rij6);
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	//return matrix
	J.noalias()+=Jr*ke;
	return J;
}
*/

double PotLDampLong::energy(const Structure& struc, const verlet::List& vlist){
	if(POT_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampLong::resize(const Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	//compute k-space energy
	ksl_.init(struc,c6_);
	const double energyK_=ksl_.energy(struc);
	//compute r-space energy
	const double a=ksl_.alpha();
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			const double dr2=(struc.diff(struc.posn(i),struc.posn(jj),drv)-struc.R()*vlist.neigh(i,j).cell()).squaredNorm();
			if(dr2<rc2_){
				const double dr6=dr2*dr2*dr2;
				const double b2=dr2*a*a;
				const double c=exp(-b2)*(1.0+b2*(1.0+0.5*b2));
				#if LDAMP_A == 3
					//a=3,b=2
					const double dr3=sqrt(dr)*dr2;
					const double rij=rvdw_(ti,tj);
					const double den=1.0/(dr3+rij*rij*rij);
					energyR_+=c6_(ti,tj)*((1.0-c)/dr6-den*den);
				#elif LDAMP_A == 6
					//a=6,b=1
					energyR_+=c6_(ti,tj)*((1.0-c)/dr6-1.0/(dr6+rvdw6_(ti,tj)));
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double rij6=rvdw6_(ti,tj);
					energyR_+=c6_(ti,tj)*((1.0-c)/dr6-1.0/sqrt(dr6*dr6+rij6*rij6));
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	energyR_*=0.5;
	//return total energy
	return energyR_+energyK_;
}

double PotLDampLong::compute(Structure& struc, const verlet::List& vlist){
	if(POT_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"ptnl::PotLDampLong::compute(Structure&,const verlet::List&):\n";
	Eigen::Vector3d drv;
	//compute k-space energy
	ksl_.init(struc,c6_);
	const double energyK_=ksl_.compute(struc);
	//compute r-space energy
	const double a=ksl_.alpha();
	const double a2=a*a;
	const double a4=a2*a2;
	const double a6=a4*a2;
	double energyR_=0;
	for(int i=0; i<struc.nAtoms(); ++i){
		const int ti=struc.type(i);
		for(int j=0; j<vlist.size(i); ++j){
			const int jj=vlist.neigh(i,j).index();
			const int tj=struc.type(jj);
			struc.diff(struc.posn(i),struc.posn(jj),drv);
			drv.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dr2=drv.squaredNorm();
			if(dr2<rc2_){
				const double dr6=dr2*dr2*dr2;
				const double b2=dr2*a*a;
				const double expf=exp(-b2);
				const double cc=(1.0-exp(-b2)*(1.0+b2*(1.0+0.5*b2)));
				#if LDAMP_A == 3
					//a=3,b=2
					const double rij=rvdw_(ti,tj);
					const double dr3=sqrt(dr)*dr2;
					const double den=1.0/(dr3+rij*rij*rij);
					energyR_+=c6_(ti,tj)*(cc/dr6-den*den);
					struc.force(i).noalias()+=drv*c6_(ti,tj)*(
						(6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*dr2)-6.0*dr*den*den*den
					);
				#elif LDAMP_A == 6
					//a=6,b=1
					const double den=1.0/(dr6+rvdw6_(ti,tj));
					energyR_+=c6_(ti,tj)*(cc/dr6-den);
					struc.force(i).noalias()+=drv*c6_(ti,tj)*(
						(6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*dr2)-6.0*dr2*dr2*den*den
					);
				#elif LDAMP_A == 12
					//a=12,b=1/2
					const double rij6=rvdw6_(ti,tj);
					const double den=1.0/sqrt(dr6*dr6+rij6*rij6);
					energyR_+=c6_(ti,tj)*(cc/dr6-den);
					struc.force(i).noalias()+=drv*c6_(ti,tj)*(
						(6.0-expf*(6.0+b2*(6.0+b2*(3.0+b2))))/(dr6*dr2)-6.0*dr2*dr2*dr6*den*den*den
					);
				#else 
					#error Unsupported LDAMP exponent
				#endif
			}
		}
	}
	energyR_*=0.5;
	//return total energy
	return energyR_+energyK_;
}

} // namespace ptnl

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const ptnl::PotLDampLong& obj){
		if(POT_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"nbytes(const ptnl::PotLDampLong&):\n";
		int size=0;
		size+=nbytes(static_cast<const ptnl::Pot&>(obj));
		size+=sizeof(ptnl::Pot::Name);//name_
		size+=sizeof(int);//ntypes_
		size+=sizeof(double);//prec_
		const int nt=obj.ntypes();
		size+=nt*nt*sizeof(double);//rvdw_
		size+=nt*nt*sizeof(double);//c6_
		return size;
	}
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const ptnl::PotLDampLong& obj, char* arr){
		if(POT_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"pack(const ptnl::PotLDampLong&,char*):\n";
		int pos=0;
		pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
		std::memcpy(arr+pos,&obj.name(),sizeof(ptnl::Pot::Name)); pos+=sizeof(ptnl::Pot::Name);//name_
		std::memcpy(arr+pos,&obj.ntypes(),sizeof(int)); pos+=sizeof(int);//ntypes_
		std::memcpy(arr+pos,&obj.prec(),sizeof(double)); pos+=sizeof(double);//prec_
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
	
	template <> int unpack(ptnl::PotLDampLong& obj, const char* arr){
		if(POT_LDAMP_LONG_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotLDampLong&,const char*):\n";
		int pos=0,nt=0;
		ptnl::Pot::Name name;
		pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
		std::memcpy(&name,arr+pos,sizeof(ptnl::Pot::Name)); pos+=sizeof(ptnl::Pot::Name);//name_
		std::memcpy(&nt,arr+pos,sizeof(int)); pos+=sizeof(int);//nt_
		std::memcpy(&obj.prec(),arr+pos,sizeof(double)); pos+=sizeof(double);//prec_
		if(name!=ptnl::Pot::Name::LDAMP_LONG) throw std::invalid_argument("serialize::unpack(ptnl::PotLDampLong&,const char*): Invalid name.");
		obj.resize(nt);
		if(nt>0){
			std::memcpy(obj.rvdw().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//rvdw_
			std::memcpy(obj.c6().data(),arr+pos,nt*nt*sizeof(double)); pos+=nt*nt*sizeof(double);//c6_
		}
		obj.init();
		return pos;
	}
	
}