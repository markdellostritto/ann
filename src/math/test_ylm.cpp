// c
#include <cmath>
// c++
#include <iostream>
#include <complex>
// eigen
#include <Eigen/Dense>
// structure
#include "struc/structure.hpp"
#include "struc/grid.hpp"
// format
#include "format/cube_struc.hpp"
// math
#include "math/poly.hpp"
#include "math/special.hpp"

int main(int argc, char* argv[]){
	
	std::complex<double> li(0.0,1.0);
	
	//structure - lattice
	Structure struc;
	Eigen::Vector3d Rd; Rd<<5.0,5.0,5.0;
	Eigen::Vector3d ld; ld<<0.05,0.05,0.05;
	Eigen::Vector3d origin=0.5*Rd;
	Eigen::Matrix3d R=Rd.asDiagonal();
	struc.init(R);
	//structure - atom type
	AtomType atomT;
	atomT.an=true;
	atomT.mass=true;
	atomT.name=true;
	atomT.type=true;
	atomT.posn=true;
	//structure - atoms
	struc.resize(1,atomT);
	struc.an(0)=1;
	struc.mass(0)=1.0;
	struc.name(0)="H";
	struc.posn(0)=origin;
	
	//grid
	Grid grid;
	grid.resize(R,ld);
	Eigen::Vector3i nd=grid.n();
	
	const int l=1;
	const int m=1;
	const double mf=1.0*m;
	for(int i=0; i<nd[0]; ++i){
		for(int j=0; j<nd[1]; ++j){
			for(int k=0; k<nd[2]; ++k){
				Eigen::Vector3d r;
				r<<i*ld[0],j*ld[1],k*ld[2];
				r-=origin;
				const double cos=r[2]/r.norm();
				const double phi=std::acos(r[0]/sqrt(r[0]*r[0]+r[1]*r[1]))*math::special::sgn(r[1]);
				std::complex<double> wf=exp(-r.norm())*math::poly::plegendre(l,m,cos)*exp(li*mf*phi);
				grid.data(i,j,k)=wf.real();
			}
		}
	}
	
	CUBE::write("wf.cube",atomT,struc,grid,origin);
}