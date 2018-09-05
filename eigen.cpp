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
	if(t==LIN_SOLVER::LLT) out<<"LLT";
	else if(t==LIN_SOLVER::LDLT) out<<"LDLT";
	else if(t==LIN_SOLVER::PPLU) out<<"PPLU";
	else if(t==LIN_SOLVER::PPLU) out<<"PPLU";
	else if(t==LIN_SOLVER::FPLU) out<<"FPLU";
	else if(t==LIN_SOLVER::HQR) out<<"HQR";
	else if(t==LIN_SOLVER::CPHQR) out<<"CPHQR";
	else out<<"UNKNOWN";
	return out;
}
	
Eigen::Vector3d& load(const char* str, Eigen::Vector3d& vec){
	if(string::substrN(str,",")!=3) throw std::invalid_argument("Invalid Eigen::Vector3d format.");
	char* temp=(char*)malloc(sizeof(char)*std::strlen(str));
	std::strcpy(temp,str);
	vec[0]=std::atof(std::strtok(temp,","));
	vec[1]=std::atof(std::strtok(NULL,","));
	vec[2]=std::atof(std::strtok(NULL,","));
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
template <> unsigned int nbytes(const Eigen::VectorXd& obj){return obj.size()*sizeof(double);};
template <> unsigned int nbytes(const Eigen::Matrix3d& obj){return 9*sizeof(double);};
template <> unsigned int nbytes(const Eigen::MatrixXd& obj){return obj.rows()*obj.cols()*sizeof(double);};

//**********************************************
// packing
//**********************************************

template <> void pack(const Eigen::Vector3d& obj, char* arr){std::memcpy(arr,obj.data(),nbytes(obj));};
template <> void pack(const Eigen::VectorXd& obj, char* arr){std::memcpy(arr,obj.data(),nbytes(obj));};
template <> void pack(const Eigen::Matrix3d& obj, char* arr){std::memcpy(arr,obj.data(),nbytes(obj));};
template <> void pack(const Eigen::MatrixXd& obj, char* arr){std::memcpy(arr,obj.data(),nbytes(obj));};

//**********************************************
// unpacking
//**********************************************

template <> void unpack(Eigen::Vector3d& obj, const char* arr){std::memcpy(obj.data(),arr,nbytes(obj));};
template <> void unpack(Eigen::VectorXd& obj, const char* arr){std::memcpy(obj.data(),arr,nbytes(obj));};
template <> void unpack(Eigen::Matrix3d& obj, const char* arr){std::memcpy(obj.data(),arr,nbytes(obj));};
template <> void unpack(Eigen::MatrixXd& obj, const char* arr){std::memcpy(obj.data(),arr,nbytes(obj));};

}