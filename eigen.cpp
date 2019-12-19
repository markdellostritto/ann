//c++ libraries
#include <iostream>
//ann - string
#include "string.hpp"
// ann - eigen
#include "eigen.hpp"

namespace eigen{

LIN_SOLVER::type LIN_SOLVER::read(const char* str){
	if(std::strcmp(str,"LLT")==0) return LIN_SOLVER::LLT;
	else if(std::strcmp(str,"LDLT")==0) return LIN_SOLVER::LDLT;
	else if(std::strcmp(str,"PPLU")==0) return LIN_SOLVER::PPLU;
	else if(std::strcmp(str,"FPLU")==0) return LIN_SOLVER::FPLU;
	else if(std::strcmp(str,"HQR")==0) return LIN_SOLVER::HQR;
	else if(std::strcmp(str,"CPHQR")==0) return LIN_SOLVER::CPHQR;
	else return LIN_SOLVER::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, const LIN_SOLVER::type& t){
	switch(t){
		case LIN_SOLVER::LLT: out<<"LLT"; break;
		case LIN_SOLVER::LDLT: out<<"LDLT"; break;
		case LIN_SOLVER::PPLU: out<<"PPLU"; break;
		case LIN_SOLVER::FPLU: out<<"FPLU"; break;
		case LIN_SOLVER::HQR: out<<"HQR"; break;
		case LIN_SOLVER::CPHQR: out<<"CPHQR"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const Eigen::Vector3d& obj){return 3*sizeof(double);};
template <> unsigned int nbytes(const Eigen::VectorXd& obj){return obj.size()*sizeof(double)+sizeof(unsigned int);};
template <> unsigned int nbytes(const Eigen::Matrix3d& obj){return 9*sizeof(double);};
template <> unsigned int nbytes(const Eigen::MatrixXd& obj){return obj.size()*sizeof(double)+2*sizeof(unsigned int);};

//**********************************************
// packing
//**********************************************

template <> unsigned int pack(const Eigen::Vector3d& obj, char* arr){std::memcpy(arr,obj.data(),nbytes(obj));return 3*sizeof(double);};
template <> unsigned int pack(const Eigen::VectorXd& obj, char* arr){
	unsigned int pos=0;
	unsigned int size=obj.size();
	std::memcpy(arr+pos,&size,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(size>0){
		std::memcpy(arr+pos,obj.data(),size*sizeof(double)); pos+=size*sizeof(double);
	}
	return pos;
}
template <> unsigned int pack(const Eigen::Matrix3d& obj, char* arr){std::memcpy(arr,obj.data(),nbytes(obj));return 9*sizeof(double);};
template <> unsigned int pack(const Eigen::MatrixXd& obj, char* arr){
	unsigned int pos=0;
	const unsigned int nrows=obj.rows();
	const unsigned int ncols=obj.cols();
	std::memcpy(arr+pos,&nrows,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(arr+pos,&ncols,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(nrows>0 && ncols>0){
		std::memcpy(arr+pos,obj.data(),obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
	}
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> unsigned int unpack(Eigen::Vector3d& obj, const char* arr){std::memcpy(obj.data(),arr,nbytes(obj));return 3*sizeof(double);};
template <> unsigned int unpack(Eigen::VectorXd& obj, const char* arr){
	unsigned int pos=0,size=0;
	std::memcpy(&size,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(size>0){
		obj.resize(size);
		std::memcpy(obj.data(),arr+pos,size*sizeof(double)); pos+=size*sizeof(double);
	}
	return pos;
}
template <> unsigned int unpack(Eigen::Matrix3d& obj, const char* arr){std::memcpy(obj.data(),arr,nbytes(obj));return 9*sizeof(double);};
template <> unsigned int unpack(Eigen::MatrixXd& obj, const char* arr){
	unsigned int pos=0,nrows=0,ncols=0;
	std::memcpy(&nrows,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(&ncols,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(nrows>0 && ncols>0){
		obj.resize(nrows,ncols);
		std::memcpy(obj.data(),arr+pos,obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
	}
	return pos;
}

}
