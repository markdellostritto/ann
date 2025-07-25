// torch
#include "torch/kspace.hpp"

namespace KSpace{

//==== constants ====

const std::complex<double> Base::I_=std::complex<double>(0,1);
	
//==== operators ====

std::ostream& operator<<(std::ostream& out, const Base& k){
	return out<<"prec "<<k.prec_<<" alpha "<<k.alpha_<<" nk "<<k.nk_.transpose();
}

double operator-(const Base& ks1, const Base& ks2){
	return std::fabs(ks1.rc()-ks2.rc())+std::fabs(ks1.prec()-ks2.prec());
}

}

//**********************************************
// serialization
//**********************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const KSpace::Base& obj){
		if(KSPACE_PRINT_FUNC>0) std::cout<<"nbytes(const KSpace::Base&):\n";
		int size=0;
		size+=sizeof(double);//rcut_
		size+=sizeof(double);//prec_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const KSpace::Base& obj, char* arr){
		if(KSPACE_PRINT_FUNC>0) std::cout<<"pack(const KSpace::Base&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.rcut(),sizeof(double)); pos+=sizeof(double);//rcut_
		std::memcpy(arr+pos,&obj.prec(),sizeof(double)); pos+=sizeof(double);//prec_
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(KSpace::Base& obj, const char* arr){
		if(KSPACE_PRINT_FUNC>0) std::cout<<"unpack(KSpace::Base&,const char*):\n";
		int pos=0;
		std::memcpy(&obj.rcut(),arr+pos,sizeof(double)); pos+=sizeof(double);//rcut_
		std::memcpy(&obj.prec(),arr+pos,sizeof(double)); pos+=sizeof(double);//prec_
		return pos;
	}
	
}