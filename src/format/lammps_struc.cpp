// c libraries
#include <ctime>
// c++ libraries
#include <iostream>
// str
#include "str/string.hpp"
// chem
#include "chem/ptable.hpp"
#include "chem/units.hpp"
// structure
#include "format/lammps_sim.hpp"

namespace LAMMPS{

namespace DUMP{

void write(FILE* writer, const AtomType& atomT, Structure& struc){
	static const char* funcName="write<AtomT>(FILE*,const Interval&,const AtomType&,Structure&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
	//local variables
	bool error=false;
	
	try{
		const Eigen::Matrix3d& R=sim.frame(t).R();
		const double a=R.col(0).norm();
		const double b=R.col(1).norm();
		const double c=R.col(2).norm();
		const double cosa=R.col(1).dot(R.col(2))/(b*c);
		const double cosb=R.col(0).dot(R.col(2))/(a*c);
		const double cosc=R.col(0).dot(R.col(1))/(a*b);
		const double lx=a;
		const double xy=b*cosc;
		const double xz=c*cosb;
		const double ly=std::sqrt(b*b-xy*xy);
		const double yz=(b*c*cosa-xy*yz)/ly;
		const double lz=std::sqrt(c*c-xz*xz-yz*yz);
		
		fprintf(writer,"ITEM: TIMESTEP\n");
		fprintf(writer,"%i\n",t);
		fprintf(writer,"ITEM: NUMBER OF ATOMS\n");
		fprintf(writer,"%i\n",struc.nAtoms());
		fprintf(writer,"ITEM: BOX BOUNDS pp pp pp\n");
		fprintf(writer,"%f %f\n",0.0,lx,xy);
		fprintf(writer,"%f %f\n",0.0,ly,xz);
		fprintf(writer,"%f %f\n",0.0,lz,yz);
		fprintf(writer,"ITEM: ATOMS id type x y z fx fy fz\n");
		for(int n=0; n<sim.frame(t).nAtoms(); ++n){
			fprintf(writer,"%i %i %f %f %f\n",n+1,struc.type(n)+1,
				struc.posn(n)[0],struc.posn(n)[1],struc.posn(n)[2],
				struc.force(n)[0],struc.force(n)[1],struc.force(n)[2]
			);
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}

void write(const char* file, const Interval& interval, const AtomType& atomT, Structure& struc){
	static const char* funcName="write<AtomT>(const char*,const Interval&,const AtomType&,Structure&)";
	if(LAMMPS_PRINT_FUNC>0) std::cout<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
	//local variables
	FILE* writer=NULL;
	bool error=false;
	
	try{
		//open the file
		writer=fopen(file,"w");
		if(writer==NULL) throw std::runtime_error("Unable to open file.");
		
		write(writer,atomT,struc);
		
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in "<<NAMESPACE_GLOBAL<<"::"<<funcName<<":\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	if(error) throw std::runtime_error("I/O Exception Occurred.");
}
	
}

}
