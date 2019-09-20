#include "eigen.hpp"

namespace eigen{

LIN_SOLVER::type LIN_SOLVER::load(const char* str){
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
	
Eigen::Vector3d& load(const char* str, Eigen::Vector3d& vec){
	if(string::substrN(str,",")!=3) throw std::invalid_argument("Invalid Eigen::Vector3d format.");
	char* temp=new char[std::strlen(str)];
	std::strcpy(temp,str);
	vec[0]=std::atof(std::strtok(temp,","));
	vec[1]=std::atof(std::strtok(NULL,","));
	vec[2]=std::atof(std::strtok(NULL,","));
	delete[] temp;
	return vec;
}

const char* print(char* str, const Eigen::Vector3d& vec){
	str[0]='\0';
	sprintf(str,"%f,%f,%f",vec[0],vec[1],vec[2]);
	return str;
}

}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const Eigen::Vector3d& obj){return 3*sizeof(double);};
template <> unsigned int nbytes(const Eigen::VectorXd& obj){return obj.size()*sizeof(double)+sizeof(unsigned int);};
template <> unsigned int nbytes(const Eigen::Matrix3d& obj){return 9*sizeof(double);};
template <> unsigned int nbytes(const Eigen::MatrixXd& obj){return obj.rows()*obj.cols()*sizeof(double)+2*sizeof(unsigned int);};

//**********************************************
// packing
//**********************************************

template <> void pack(const Eigen::Vector3d& obj, char* arr){std::memcpy(arr,obj.data(),nbytes(obj));};
template <> void pack(const Eigen::VectorXd& obj, char* arr){
	unsigned int pos=0;
	unsigned int size=obj.size();
	std::memcpy(arr+pos,&size,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(size>0){
		std::memcpy(arr+pos,obj.data(),size*sizeof(double)); pos+=size*sizeof(double);
	}
}
template <> void pack(const Eigen::Matrix3d& obj, char* arr){std::memcpy(arr,obj.data(),nbytes(obj));};
template <> void pack(const Eigen::MatrixXd& obj, char* arr){
	unsigned int pos=0;
	unsigned int nrows=obj.rows();
	unsigned int ncols=obj.cols();
	std::memcpy(arr+pos,&nrows,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(arr+pos,&ncols,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(nrows>0 && ncols>0){
		std::memcpy(arr+pos,obj.data(),obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
	}
}

//**********************************************
// unpacking
//**********************************************

template <> void unpack(Eigen::Vector3d& obj, const char* arr){std::memcpy(obj.data(),arr,nbytes(obj));};
template <> void unpack(Eigen::VectorXd& obj, const char* arr){
	unsigned int pos=0;
	unsigned int size=0;
	std::memcpy(&size,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(size>0){
		obj.resize(size);
		std::memcpy(obj.data(),arr+pos,size*sizeof(double)); pos+=size*sizeof(double);
	}
}
template <> void unpack(Eigen::Matrix3d& obj, const char* arr){std::memcpy(obj.data(),arr,nbytes(obj));};
template <> void unpack(Eigen::MatrixXd& obj, const char* arr){
	unsigned int pos=0;
	unsigned int nrows=0;
	unsigned int ncols=0;
	std::memcpy(&nrows,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(&ncols,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(nrows>0 && ncols>0){
		obj.resize(nrows,ncols);
		std::memcpy(obj.data(),arr+pos,obj.size()*sizeof(double)); pos+=obj.size()*sizeof(double);
	}
}

}