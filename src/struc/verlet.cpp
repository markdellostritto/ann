// c++
#include <stdexcept>
#include <iostream>
// struc
#include "struc/verlet.hpp"
#include "struc/structure.hpp"
// math
#include "math/const.hpp"

namespace verlet{
	
//**************************************************************************
// Neighbor
//**************************************************************************

//==== operators ====
	
std::ostream& operator<<(std::ostream& out, const Neighbor& n){
	return out<<n.index_<<" "<<n.cell_.transpose();
}

//**************************************************************************
// List
//**************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const List& list){
	return out<<"rc "<<list.rc_<<" stride "<<list.stride_<<" size "<<list.neigh_.size();
}

//==== member functions ====

void List::read(const char* str){
	//vlist rc stride
	Token token(str,string::WS);
	read(token);
}

void List::read(Token& token){
	//vlist rc stride
	rc_=std::atof(token.next().c_str());
	stride_=std::atoi(token.next().c_str());
}

void List::resize(int natoms){
	if(natoms<0) throw std::invalid_argument("verlet::List::resize(int): Invalid list size.");
	neigh_.resize(natoms);
}

void List::build(const Structure& struc){
	if(VERLET_PRINT_FUNC>0) std::cout<<"List::build(const Structure&):\n";
	neigh_.resize(struc.nAtoms());
	const double rc2=rc_*rc_;
	if(struc.R().norm()>0){
		//store lattice vectors
		std::vector<Eigen::Vector3d> Rv(3);
		for(int i=0; i<3; ++i) Rv[i]=struc.R().col(i);
		//sort from largest to smallest
		std::sort(Rv.begin(),Rv.end(),sortcmp);
		//compute multiples
		int shell[3];
		shell[0]=floor(2.0*rc_/Rv[0].norm());
		shell[1]=floor(2.0*rc_/(Rv[0].cross(Rv[1]).norm()/Rv[0].norm()));
		shell[2]=floor(2.0*rc_/fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm())));
		const int Rsize=(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1);
		std::vector<Eigen::Vector3d> Rf(Rsize);
		std::vector<Eigen::Vector3d> Ri(Rsize);
		int count=0;
		for(int i=-shell[0]; i<=shell[0]; ++i){
			for(int j=-shell[1]; j<=shell[1]; ++j){
				for(int k=-shell[2]; k<=shell[2]; ++k){
					Rf[count]=i*Rv[0]+j*Rv[1]+k*Rv[2];
					Ri[count]<<i,j,k;
					count++;
				}
			}
		}
		//build the neighbor list
		Eigen::Vector3d tmp;
		for(int i=0; i<struc.nAtoms(); ++i){
			neigh_[i].clear();
			for(int j=0; j<struc.nAtoms(); ++j){
				const Eigen::Vector3d rIJ=struc.diff(struc.posn(i),struc.posn(j),tmp);
				for(int n=0; n<Rsize; ++n){
					const Eigen::Vector3d rIJt_=rIJ-Rf[n];
					const double dIJ2=rIJt_.squaredNorm();
					if(math::constant::ZERO<dIJ2 && dIJ2<rc2){
						neigh_[i].push_back(Neighbor());
						neigh_[i].back().index()=j;
						neigh_[i].back().cell()=Ri[n];
					}
				}
			}
		}
	} else {
		for(int i=0; i<struc.nAtoms(); ++i){
			neigh_[i].clear();
			for(int j=0; j<struc.nAtoms(); ++j){
				if(i==j) continue;
				const double dr2=(struc.posn(i)-struc.posn(j)).squaredNorm();
				if(dr2<rc2) neigh_[i].push_back(Neighbor(j));
			}
		}
	}
}

void List::build(const Structure& struc, int i){
	neigh_.resize(struc.nAtoms());
	const double rc2=rc_*rc_;
	if(struc.R().norm()>0){
		//store lattice vectors
		std::vector<Eigen::Vector3d> Rv(3);
		for(int i=0; i<3; ++i) Rv[i]=struc.R().col(i);
		//sort from largest to smallest
		std::sort(Rv.begin(),Rv.end(),sortcmp);
		//compute multiples
		int shell[3];
		shell[0]=floor(2.0*rc_/Rv[0].norm());
		shell[1]=floor(2.0*rc_/(Rv[0].cross(Rv[1]).norm()/Rv[0].norm()));
		shell[2]=floor(2.0*rc_/fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm())));
		const int Rsize=(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1);
		std::vector<Eigen::Vector3d> Rf(Rsize);
		std::vector<Eigen::Vector3d> Ri(Rsize);
		int count=0;
		for(int i=-shell[0]; i<=shell[0]; ++i){
			for(int j=-shell[1]; j<=shell[1]; ++j){
				for(int k=-shell[2]; k<=shell[2]; ++k){
					Rf[count]=i*Rv[0]+j*Rv[1]+k*Rv[2];
					Ri[count]<<i,j,k;
					count++;
				}
			}
		}
		//build the neighbor list
		Eigen::Vector3d tmp;
		for(int i=0; i<struc.nAtoms(); ++i){
			neigh_[i].clear();
			for(int j=0; j<struc.nAtoms(); ++j){
				const Eigen::Vector3d rIJ=struc.diff(struc.posn(i),struc.posn(j),tmp);
				for(int n=0; n<Rsize; ++n){
					const Eigen::Vector3d rIJt_=rIJ-Rf[n];
					const double dIJ2=rIJt_.squaredNorm();
					if(math::constant::ZERO<dIJ2 && dIJ2<rc2){
						neigh_[i].push_back(Neighbor());
						neigh_[i].back().index()=j;
						neigh_[i].back().cell()=Ri[n];
					}
				}
			}
		}
	} else {
		for(int i=0; i<struc.nAtoms(); ++i){
			neigh_[i].clear();
			for(int j=0; j<struc.nAtoms(); ++j){
				if(i==j) continue;
				const double dr2=(struc.posn(i)-struc.posn(j)).squaredNorm();
				if(dr2<rc2) neigh_[i].push_back(Neighbor(j));
			}
		}
	}
}

void List::build(const Structure& struc, double rc, int i){
	neigh_.resize(struc.nAtoms());
	const double rc2=rc_*rc_;
	if(struc.R().norm()>0){
		//store lattice vectors
		std::vector<Eigen::Vector3d> Rv(3);
		for(int i=0; i<3; ++i) Rv[i]=struc.R().col(i);
		//sort from largest to smallest
		std::sort(Rv.begin(),Rv.end(),sortcmp);
		//compute multiples
		int shell[3];
		shell[0]=floor(2.0*rc_/Rv[0].norm());
		shell[1]=floor(2.0*rc_/(Rv[0].cross(Rv[1]).norm()/Rv[0].norm()));
		shell[2]=floor(2.0*rc_/fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm())));
		const int Rsize=(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1);
		std::vector<Eigen::Vector3d> Rf(Rsize);
		std::vector<Eigen::Vector3d> Ri(Rsize);
		int count=0;
		for(int i=-shell[0]; i<=shell[0]; ++i){
			for(int j=-shell[1]; j<=shell[1]; ++j){
				for(int k=-shell[2]; k<=shell[2]; ++k){
					Rf[count]=i*Rv[0]+j*Rv[1]+k*Rv[2];
					Ri[count]<<i,j,k;
					count++;
				}
			}
		}
		//build the neighbor list
		Eigen::Vector3d tmp;
		for(int i=0; i<struc.nAtoms(); ++i){
			neigh_[i].clear();
			for(int j=0; j<struc.nAtoms(); ++j){
				const Eigen::Vector3d rIJ=struc.diff(struc.posn(i),struc.posn(j),tmp);
				for(int n=0; n<Rsize; ++n){
					const Eigen::Vector3d rIJt_=rIJ-Rf[n];
					const double dIJ2=rIJt_.squaredNorm();
					if(math::constant::ZERO<dIJ2 && dIJ2<rc2){
						neigh_[i].push_back(Neighbor());
						neigh_[i].back().index()=j;
						neigh_[i].back().cell()=Ri[n];
					}
				}
			}
		}
	} else {
		for(int i=0; i<struc.nAtoms(); ++i){
			neigh_[i].clear();
			for(int j=0; j<struc.nAtoms(); ++j){
				if(i==j) continue;
				const double dr2=(struc.posn(i)-struc.posn(j)).squaredNorm();
				if(dr2<rc2) neigh_[i].push_back(Neighbor(j));
			}
		}
	}
}

//==== static functions ====

bool List::sortcmp(const Eigen::Vector3d& i, const Eigen::Vector3d& j){
	return i.norm()>j.norm();
}

};