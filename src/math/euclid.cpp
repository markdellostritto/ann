// math 
#include "math/euclid.hpp"

namespace math{

namespace euclid{

Eigen::Matrix3d rotate(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2){
	const Eigen::Vector3d u1=v1/v1.norm();
	const Eigen::Vector3d u2=v2/v2.norm();
	const Eigen::Vector3d ucu=u1.cross(u2);
	const double s=ucu.norm();
	const double c=u1.dot(u2);
	Eigen::Matrix3d vv;
	vv(0,0)=0.0; vv(0,1)=-ucu[2]; vv(0,2)=ucu[1];
	vv(1,0)=ucu[2]; vv(1,1)=0.0; vv(1,2)=-ucu[0];
	vv(2,0)=-ucu[1]; vv(2,1)=ucu[0]; vv(2,2)=0.0;
	Eigen::Matrix3d mat=Eigen::Matrix3d::Identity();
	mat.noalias()+=vv;
	mat.noalias()+=vv*vv*(1.0-c)/(s*s+1.0e-12);
	return mat;
}
	
}

}
