// c++
#include <iostream>
#include <stdexcept>
// struc
#include "struc/structure.hpp"
// torch
#include "dump.hpp"

//=== operators ====

std::ostream& operator<<(std::ostream& out, const Dump& dump){
	return out<<"file "<<dump.file_<<" nprint "<<dump.nprint_<<" nwrite "<<dump.nwrite_;
}

//==== member functions ====
	
void Dump::read(Token& token){
	//read nprint
	token.next();//NPRINT
	nprint_=std::atoi(token.next().c_str());
	if(nprint_<=0) throw std::invalid_argument("Dump::read(Token&): invalid nprint.");
	//read nwrite
	token.next();//NWRITE
	nwrite_=std::atoi(token.next().c_str());
	if(nwrite_<=0) throw std::invalid_argument("Dump::read(Token&): invalid nwrite.");
	//read file
	token.next();//file
	file_=token.next();
}

//==== static functions ====

void Dump::write(Structure& struc, FILE* writer){
	const Eigen::Matrix3d& R=struc.R();
	const Eigen::Vector3d A=R.col(0);
	const Eigen::Vector3d B=R.col(1);
	const Eigen::Vector3d C=R.col(2);
	const double a=A.norm();
	const double b=B.norm();
	const double c=C.norm();
	const double cosa=B.dot(C)/(b*c);
	const double cosb=A.dot(C)/(a*c);
	const double cosc=A.dot(B)/(a*b);
	const double lx=a;
	const double xy=b*cosc;
	const double xz=c*cosb;
	const double ly=std::sqrt(b*b-xy*xy);
	const double yz=(b*c*cosa-xy*xz)/ly;
	const double lz=std::sqrt(c*c-xz*xz-yz*yz);
	const double xlo=0.0;
	const double ylo=0.0;
	const double zlo=0.0;
	const double xhi=lx;
	const double yhi=ly;
	const double zhi=lz;
	const double xlob=xlo+std::min(0.0,std::min(xy,std::min(xz,xy+xz)));
	const double xhib=xhi+std::max(0.0,std::max(xy,std::max(xz,xy+xz)));
	const double ylob=ylo+std::min(0.0,yz);
	const double yhib=yhi+std::max(0.0,yz);
	const double zlob=zlo;
	const double zhib=zhi;
	
	fprintf(writer,"ITEM: TIMESTEP\n%i\n",struc.t());
	fprintf(writer,"ITEM: NUMBER OF ATOMS\n%i\n",struc.nAtoms());
	fprintf(writer,"ITEM: BOX BOUNDS pp pp pp\n");
	fprintf(writer,"%f %f %f\n",xlob,xhib,xy);
	fprintf(writer,"%f %f %f\n",ylob,yhib,xz);
	fprintf(writer,"%f %f %f\n",zlob,zhib,yz);
	if(struc.atomType().charge){
		fprintf(writer,"ITEM: ATOMS id type q x y z fx fy fz\n");
		for(int i=0; i<struc.nAtoms(); ++i){
			fprintf(writer,"%i %i %f %f %f %f %f %f\n",i+1,struc.type(i)+1,struc.charge(i),
				struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2],
				struc.force(i)[0],struc.force(i)[1],struc.force(i)[2]
			);
		}
	} else {
		fprintf(writer,"ITEM: ATOMS id type x y z fx fy fz\n");
		for(int i=0; i<struc.nAtoms(); ++i){
			fprintf(writer,"%i %i %f %f %f %f %f %f\n",i+1,struc.type(i)+1,
				struc.posn(i)[0],struc.posn(i)[1],struc.posn(i)[2],
				struc.force(i)[0],struc.force(i)[1],struc.force(i)[2]
			);
		}
	}
}
