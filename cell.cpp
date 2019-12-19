// c++ libraries
#include <ostream>
// ann - math
#include "math_const.hpp"
#include "math_special.hpp"
// ann - eigen
#include "eigen.hpp"
// ann - cell
#include "cell.hpp"

//operators

Cell& Cell::operator=(const Cell& cell){
	init(cell.R(),1.0);
	return *this;
}

std::ostream& operator<<(std::ostream& out, const Cell& cell){
	out<<"scale  = "<<cell.scale_<<"\n";
	out<<"volume = "<<cell.vol_<<"\n";
	out<<"R      = \n"<<cell.R_<<"\n";
	out<<"K      = \n"<<cell.K_;
	return out;
}

bool operator==(const Cell& c1, const Cell& c2){
	if(std::fabs(c1.scale()-c2.scale())>num_const::ZERO) return false;
	else if(std::fabs(c1.vol()-c2.vol())>num_const::ZERO) return false;
	else if((c1.R()-c2.R()).norm()>num_const::ZERO) return false;
	else return true;
}

bool operator!=(const Cell& c1, const Cell& c2){
	return !(c1==c2);
}

//member functions

void Cell::defaults(){
	scale_=1;
	vol_=0;
	R_=Eigen::Matrix3d::Identity();
	RInv_=Eigen::Matrix3d::Identity();
	K_=Eigen::Matrix3d::Identity()*2*num_const::PI;
	KInv_=Eigen::Matrix3d::Identity()*2*num_const::PI;
}

void Cell::init(const Eigen::Matrix3d& R, double scale){
	scale_=scale;
	R_=R*scale_;
	RInv_.noalias()=R_.inverse();
	K_.col(0)=2.0*num_const::PI*R_.col(1).cross(R_.col(2))/(R_.col(0).dot(R_.col(1).cross(R_.col(2))));
	K_.col(1)=2.0*num_const::PI*R_.col(2).cross(R_.col(0))/(R_.col(1).dot(R_.col(2).cross(R_.col(0))));
	K_.col(2)=2.0*num_const::PI*R_.col(0).cross(R_.col(1))/(R_.col(2).dot(R_.col(0).cross(R_.col(1))));
	KInv_.noalias()=K_.inverse();
	vol_=std::fabs(R_.determinant());
}

void Cell::init(const Eigen::Vector3d& R1,const Eigen::Vector3d& R2,const Eigen::Vector3d& R3, double scale){
	scale_=scale;
	R_.col(0)=R1*scale_;
	R_.col(1)=R2*scale_;
	R_.col(2)=R3*scale_;
	RInv_.noalias()=R_.inverse();
	K_.col(0)=2.0*num_const::PI*R_.col(1).cross(R_.col(2))/(R_.col(0).dot(R_.col(1).cross(R_.col(2))));
	K_.col(1)=2.0*num_const::PI*R_.col(2).cross(R_.col(0))/(R_.col(1).dot(R_.col(2).cross(R_.col(0))));
	K_.col(2)=2.0*num_const::PI*R_.col(0).cross(R_.col(1))/(R_.col(2).dot(R_.col(0).cross(R_.col(1))));
	KInv_.noalias()=K_.inverse();
	vol_=std::fabs(R_.determinant());
}

//static functions - vectors

Eigen::Vector3d& Cell::sum(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& sum, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//find sum (in fractional coordinates)
	sum.noalias()=RInv*(v1+v2);
	//return to cell
	sum[0]=special::mod(sum[0],-0.5,0.5);
	sum[1]=special::mod(sum[1],-0.5,0.5);
	sum[2]=special::mod(sum[2],-0.5,0.5);
	//switch back to Cartesian coordinates
	sum=R*sum;
	return sum;
}

Eigen::Vector3d& Cell::diff(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& diff, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//find difference (in fractional coordinates)
	diff.noalias()=RInv*(v1-v2);
	//return to cell
	diff[0]=special::mod(diff[0],-0.5,0.5);
	diff[1]=special::mod(diff[1],-0.5,0.5);
	diff[2]=special::mod(diff[2],-0.5,0.5);
	//switch back to Cartesian coordinates
	diff=R*diff;
	return diff;
}

double Cell::dist(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& temp, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//find difference (in fractional coordinates)
	temp.noalias()=RInv*(v1-v2);
	//return to cell
	temp[0]=special::mod(temp[0],-0.5,0.5);
	temp[1]=special::mod(temp[1],-0.5,0.5);
	temp[2]=special::mod(temp[2],-0.5,0.5);
	//return the modulus (in Cartesian coordinates)
	return (R*temp).norm();
}

double Cell::dist(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//find difference (in fractional coordinates)
	Eigen::Vector3d temp=RInv*(v1-v2);
	//return to cell
	temp[0]=special::mod(temp[0],-0.5,0.5);
	temp[1]=special::mod(temp[1],-0.5,0.5);
	temp[2]=special::mod(temp[2],-0.5,0.5);
	//return the modulus (in Cartesian coordinates)
	return (R*temp).norm();
}

Eigen::Vector3d& Cell::returnToCell(const Eigen::Vector3d& v1, Eigen::Vector3d& v2, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//convert the vector to fractional coordinates
	v2=RInv*v1;
	//use the method of images to return the vector to the cell
	v2[0]=special::mod(v2[0],0.0,1.0);
	v2[1]=special::mod(v2[1],0.0,1.0);
	v2[2]=special::mod(v2[2],0.0,1.0);
	//return the vector in cartesian coordinates
	v2=R*v2;
	return v2;
}

Eigen::Vector3d& Cell::fracToCart(const Eigen::Vector3d& vFrac, Eigen::Vector3d& vCart, const Eigen::Matrix3d& R){
	vCart=R*vFrac;//not assuming vCart!=vFrac
	return vCart;
}

Eigen::Vector3d& Cell::cartToFrac(const Eigen::Vector3d& vCart, Eigen::Vector3d& vFrac, const Eigen::Matrix3d& RInv){
	vFrac=RInv*vCart;//not assuming vCart!=vFrac
	return vFrac;
}

//static functions - modification

Cell& Cell::super(const Eigen::Vector3i& f, Cell& cell){
	Eigen::Matrix3d R=cell.R();
	R.col(0)*=f[0];
	R.col(1)*=f[1];
	R.col(2)*=f[2];
	cell.init(R);
	return cell;
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************

	template <> unsigned int nbytes(const Cell& obj){
		return nbytes(obj.R());
	};
	
	//**********************************************
	// packing
	//**********************************************

	template <> unsigned int pack(const Cell& obj, char* arr){
		std::memcpy(arr,obj.R().data(),nbytes(obj));
		return nbytes(obj);
	};
	
	//**********************************************
	// unpacking
	//**********************************************

	template <> unsigned int unpack(Cell& obj, const char* arr){
		Eigen::Matrix3d lv;
		unpack(lv,arr);
		obj.init(lv);
		return nbytes(obj);
	}
	
}
