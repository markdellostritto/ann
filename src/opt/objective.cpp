// c++
#include <iostream>
#include <stdexcept>
// math
#include "math/eigen.hpp"
// str
#include "str/print.hpp"
// opt
#include "opt/objective.hpp"

namespace opt{

//***************************************************
//Objective class
//***************************************************

std::ostream& operator<<(std::ostream& out, const Objective& obj){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("OPT::OBJECTIVE",str)<<"\n";
	//status
	out<<"GAMMA   = "<<obj.gamma_<<"\n";
	out<<"VAL     = "<<obj.val_<<"\n";
	out<<"DV      = "<<obj.dv_<<"\n";
	out<<"DP      = "<<obj.dp_<<"\n";
	//parameters
	out<<"DIM     = "<<obj.dim_<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void Objective::defaults(){
	if(OPT_OBJ_PRINT_FUNC>0) std::cout<<"Objective::defaults()\n";
	//status
	gamma_=0;
	val_=0; valOld_=0;
	dv_=0; dp_=0;
	//parameters
	dim_=0;
}

void Objective::resize(int dim){
	if(OPT_OBJ_PRINT_FUNC>0) std::cout<<"Objective::resize(int)\n";
	if(dim<0) throw std::invalid_argument("Objective::resize(int): Invalid dimension");
	dim_=dim;
	if(dim_>0){
		p_=Eigen::VectorXd::Zero(dim_);
		pOld_=Eigen::VectorXd::Zero(dim_);
		g_=Eigen::VectorXd::Zero(dim_);
		gOld_=Eigen::VectorXd::Zero(dim_);
	}
}

}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const opt::Objective& obj){
	if(OPT_OBJ_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const opt::Objective&)\n";
	int size=0;
	//status
		size+=sizeof(double);//gamma_;
		size+=sizeof(double);//val_;
		size+=sizeof(double);//valOld_;
		size+=sizeof(double);//dv_;
		size+=sizeof(double);//dp_;
	//parameters
		const int dim=obj.dim();
		size+=sizeof(int);//dim_
		size+=sizeof(double)*dim;//p_
		size+=sizeof(double)*dim;//pOld_
		size+=sizeof(double)*dim;//g_
		size+=sizeof(double)*dim;//gOld_
	//return the size
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const opt::Objective& obj, char* arr){
	if(OPT_OBJ_PRINT_FUNC>0) std::cout<<"serialize::pack(const opt::Objective&,char*)\n";
	int pos=0;
	//status
		std::memcpy(arr+pos,&obj.gamma(),sizeof(double)); pos+=sizeof(double);//gamma_
		std::memcpy(arr+pos,&obj.val(),sizeof(double)); pos+=sizeof(double);//val_
		std::memcpy(arr+pos,&obj.valOld(),sizeof(double)); pos+=sizeof(double);//valOld_
		std::memcpy(arr+pos,&obj.dv(),sizeof(double)); pos+=sizeof(double);//dv_
		std::memcpy(arr+pos,&obj.dp(),sizeof(double)); pos+=sizeof(double);//dp_
	//parameters
		const int dim=obj.dim();
		std::memcpy(arr+pos,&dim,sizeof(int)); pos+=sizeof(int);//dim_
		if(dim>0){
			std::memcpy(arr+pos,obj.p().data(),sizeof(double)*dim); pos+=sizeof(double)*dim;
			std::memcpy(arr+pos,obj.pOld().data(),sizeof(double)*dim); pos+=sizeof(double)*dim;
			std::memcpy(arr+pos,obj.g().data(),sizeof(double)*dim); pos+=sizeof(double)*dim;
			std::memcpy(arr+pos,obj.gOld().data(),sizeof(double)*dim); pos+=sizeof(double)*dim;
		}
	//return bytes written
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(opt::Objective& obj, const char* arr){
	if(OPT_OBJ_PRINT_FUNC>0) std::cout<<"serialize::unpack(opt::Objective&,const char*)\n";
	int pos=0;
	//status
		std::memcpy(&obj.gamma(),arr+pos,sizeof(double)); pos+=sizeof(double);//gamma_
		std::memcpy(&obj.val(),arr+pos,sizeof(double)); pos+=sizeof(double);//val_
		std::memcpy(&obj.valOld(),arr+pos,sizeof(double)); pos+=sizeof(double);//valOld_
		std::memcpy(&obj.dv(),arr+pos,sizeof(double)); pos+=sizeof(double);//dv_
		std::memcpy(&obj.dp(),arr+pos,sizeof(double)); pos+=sizeof(double);//dp_
	//parameters
		int dim=0;
		std::memcpy(&dim,arr+pos,sizeof(int)); pos+=sizeof(int);//dim_
		obj.resize(dim);
		if(dim>0){
			std::memcpy(obj.p().data(),arr+pos,sizeof(double)*dim); pos+=sizeof(double)*dim;
			std::memcpy(obj.pOld().data(),arr+pos,sizeof(double)*dim); pos+=sizeof(double)*dim;
			std::memcpy(obj.g().data(),arr+pos,sizeof(double)*dim); pos+=sizeof(double)*dim;
			std::memcpy(obj.gOld().data(),arr+pos,sizeof(double)*dim); pos+=sizeof(double)*dim;
		}
	//return bytes read
	return pos;
}

}