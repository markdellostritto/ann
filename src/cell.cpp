// c++ libraries
#include <ostream>
// ann - math
#include "math_const.hpp"
#include "math_func.hpp"
// ann - eigen
#include "eigen.hpp"
// ann - cell
#include "cell.hpp"

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Cell& cell){
	out<<"vol = "<<cell.vol_<<"\n";
	out<<"R   = \n"<<cell.R_<<"\n";
	out<<"K   = \n"<<cell.K_;
	return out;
}

bool operator==(const Cell& c1, const Cell& c2){
	if(std::fabs(c1.vol()-c2.vol())>math::constant::ZERO) return false;
	else if((c1.R()-c2.R()).norm()>math::constant::ZERO) return false;
	else return true;
}

bool operator!=(const Cell& c1, const Cell& c2){
	return !(c1==c2);
}

//==== member functions ====

void Cell::defaults(){
	vol_=0;
	R_=Eigen::Matrix3d::Zero();
	RInv_=Eigen::Matrix3d::Zero();
	K_=Eigen::Matrix3d::Zero();
	KInv_=Eigen::Matrix3d::Zero();
}

void Cell::init(const Eigen::Matrix3d& R){
	R_=R;
	RInv_.noalias()=R_.inverse();
	K_.col(0)=2.0*math::constant::PI*R_.col(1).cross(R_.col(2))/(R_.col(0).dot(R_.col(1).cross(R_.col(2))));
	K_.col(1)=2.0*math::constant::PI*R_.col(2).cross(R_.col(0))/(R_.col(1).dot(R_.col(2).cross(R_.col(0))));
	K_.col(2)=2.0*math::constant::PI*R_.col(0).cross(R_.col(1))/(R_.col(2).dot(R_.col(0).cross(R_.col(1))));
	KInv_.noalias()=K_.inverse();
	vol_=std::fabs(R_.determinant());
}

//==== static functions - vector operations ====

Eigen::Vector3d& Cell::sum(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& sum, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//find sum (in fractional coordinates)
	sum.noalias()=RInv*(v1+v2);
	//return to cell
	sum[0]=math::func::mod(sum[0],-0.5,0.5);
	sum[1]=math::func::mod(sum[1],-0.5,0.5);
	sum[2]=math::func::mod(sum[2],-0.5,0.5);
	//switch back to Cartesian coordinates
	sum=R*sum;
	return sum;
}

Eigen::Vector3d& Cell::diff(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& diff, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//find difference (in fractional coordinates)
	diff.noalias()=RInv*(v1-v2);
	//return to cell
	diff[0]=math::func::mod(diff[0],-0.5,0.5);
	diff[1]=math::func::mod(diff[1],-0.5,0.5);
	diff[2]=math::func::mod(diff[2],-0.5,0.5);
	//switch back to Cartesian coordinates
	diff=R*diff;
	return diff;
}

double Cell::dist(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& temp, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//find difference (in fractional coordinates)
	temp.noalias()=RInv*(v1-v2);
	//return to cell
	temp[0]=math::func::mod(temp[0],-0.5,0.5);
	temp[1]=math::func::mod(temp[1],-0.5,0.5);
	temp[2]=math::func::mod(temp[2],-0.5,0.5);
	//return the modulus (in Cartesian coordinates)
	return (R*temp).norm();
}

double Cell::dist(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//find difference (in fractional coordinates)
	Eigen::Vector3d temp=RInv*(v1-v2);
	//return to cell
	temp[0]=math::func::mod(temp[0],-0.5,0.5);
	temp[1]=math::func::mod(temp[1],-0.5,0.5);
	temp[2]=math::func::mod(temp[2],-0.5,0.5);
	//return the modulus (in Cartesian coordinates)
	return (R*temp).norm();
}

Eigen::Vector3d& Cell::returnToCell(const Eigen::Vector3d& v1, Eigen::Vector3d& v2, const Eigen::Matrix3d& R, const Eigen::Matrix3d& RInv){
	//convert the vector to fractional coordinates
	v2=RInv*v1;
	//use the method of images to return the vector to the cell
	v2[0]=math::func::mod(v2[0],0.0,1.0);
	v2[1]=math::func::mod(v2[1],0.0,1.0);
	v2[2]=math::func::mod(v2[2],0.0,1.0);
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

//==== vector operations ====

Eigen::Vector3d& Cell::sum(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& sum){
	//find sum (in fractional coordinates)
	sum.noalias()=RInv_*(v1+v2);
	//return to cell
	sum[0]=math::func::mod(sum[0],-0.5,0.5);
	sum[1]=math::func::mod(sum[1],-0.5,0.5);
	sum[2]=math::func::mod(sum[2],-0.5,0.5);
	//switch back to Cartesian coordinates
	sum=R_*sum;
	return sum;
}

Eigen::Vector3d& Cell::diff(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& diff){
	//find difference (in fractional coordinates)
	diff.noalias()=RInv_*(v1-v2);
	//return to cell
	diff[0]=math::func::mod(diff[0],-0.5,0.5);
	diff[1]=math::func::mod(diff[1],-0.5,0.5);
	diff[2]=math::func::mod(diff[2],-0.5,0.5);
	//switch back to Cartesian coordinates
	diff=R_*diff;
	return diff;
}

double Cell::dist(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2){
	//find difference (in fractional coordinates)
	Eigen::Vector3d temp=RInv_*(v1-v2);
	//return to cell
	temp[0]=math::func::mod(temp[0],-0.5,0.5);
	temp[1]=math::func::mod(temp[1],-0.5,0.5);
	temp[2]=math::func::mod(temp[2],-0.5,0.5);
	//return the modulus (in Cartesian coordinates)
	return (R_*temp).norm();
}

double Cell::dist(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& temp){
	//find difference (in fractional coordinates)
	temp.noalias()=RInv_*(v1-v2);
	//return to cell
	temp[0]=math::func::mod(temp[0],-0.5,0.5);
	temp[1]=math::func::mod(temp[1],-0.5,0.5);
	temp[2]=math::func::mod(temp[2],-0.5,0.5);
	//return the modulus (in Cartesian coordinates)
	return (R_*temp).norm();
}

//==== static functions ====

Cell& Cell::make_super(const Eigen::Vector3i& s, const Cell& cell1, Cell& cell2){
	if(&cell1==&cell2) throw std::runtime_error("Cell::make_super(const Eigen::Vector3i&,const Cell&,Cell&): identical references.\n");
	if(s[0]<=0 || s[1]<=0 || s[2]<=0) throw std::runtime_error("Cell::make_super(const Eigen::Vector3i&,const Cell&,Cell&): invalid supercell vector.\n");
	Eigen::Matrix3d R=cell1.R();
	R.col(0)*=s[0];
	R.col(1)*=s[1];
	R.col(2)*=s[2];
	cell2.init(R);
	return cell2;
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************

	template <> int nbytes(const Cell& obj){
		return nbytes(obj.R());
	};
	
	//**********************************************
	// packing
	//**********************************************

	template <> int pack(const Cell& obj, char* arr){
		std::memcpy(arr,obj.R().data(),nbytes(obj));
		return nbytes(obj);
	};
	
	//**********************************************
	// unpacking
	//**********************************************

	template <> int unpack(Cell& obj, const char* arr){
		Eigen::Matrix3d lv;
		unpack(lv,arr);
		obj.init(lv);
		return nbytes(obj);
	}
	
}
