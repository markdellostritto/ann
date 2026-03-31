//c++ libraries
#include <iostream>
#include <algorithm>
// ann - str
#include "str/print.hpp"
// ann - structure
#include "struc/structure.hpp"
// ann - math
#include "math/const.hpp"
// ann - neighbor
#include "struc/neighbor.hpp"

//**********************************************************************************************
// Neighbor
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Neighbor& obj){
	out<<"r     = "<<obj.r_.transpose()<<"\n";
	out<<"dr    = "<<obj.dr_<<"\n";
	out<<"type  = "<<obj.type_<<"\n";
	out<<"index = "<<obj.index_<<"\n";
	out<<"min   = "<<obj.min_;
	return out;
}

//==== member functions ====

void Neighbor::clear(){
	r_.setZero();
	dr_=0;
	type_=-1;
	index_=-1;
	min_=false;
}

//**********************************************************************************************
// NeigborList
//**********************************************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const NeighborList& obj){
	return out<<"rc "<<obj.rc_;
}

//==== member functions ====

bool NeighborList::sortcmp(const Eigen::Vector3d& i, const Eigen::Vector3d& j){
	return i.norm()>j.norm();
}

void NeighborList::clear(){
	if(NEIGH_PRINT_FUNC>0) std::cout<<"NeighborList::clear():\n";
	rc_=0.0;
	neigh_.clear();
}

void NeighborList::build(const Structure& struc, double rc){
	if(NEIGH_PRINT_FUNC>0) std::cout<<"NeighborList::build(const Structure&,double):\n";
	rc_=rc;
	//resize the neighbor list
	neigh_.resize(struc.nAtoms());
	//local variables
	const double rc2=rc*rc;
	//lattice vector shifts
	if(struc.R().norm()>0){
		if(NEIGH_PRINT_STATUS>0) std::cout<<"storing lattice vectors\n";
		Eigen::Vector3d tmp;
		//store lattice vectors
		std::vector<Eigen::Vector3d> Rv(3);
		for(int i=0; i<3; ++i) Rv[i]=struc.R().col(i);
		//sort from largest to smallest
		std::sort(Rv.begin(),Rv.end(),sortcmp);
		if(NEIGH_PRINT_STATUS>0) std::cout<<"compting common multiples\n";
		//compute multiples
		std::vector<int> shell(3);
		shell[0]=floor(2.0*rc/Rv[0].norm());
		shell[1]=floor(2.0*rc/(Rv[0].cross(Rv[1]).norm()/Rv[0].norm()));
		shell[2]=floor(2.0*rc/fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm())));
		const int Rsize=(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1);
		if(NEIGH_PRINT_DATA>0) {
			std::cout<<"V  = "<<Rv[0].norm()<<" "<<(Rv[0].cross(Rv[1]).norm()/Rv[0].norm())<<" "<<fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm()))<<"\n";
			std::cout<<"shell = ("<<shell[0]<<","<<shell[1]<<","<<shell[2]<<") = "<<(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1)<<"\n";
			std::cout<<"Rsize = "<<Rsize<<"\n";
		}
		std::vector<Eigen::Vector3d> R_(Rsize);
		int count=0;
		for(int i=-shell[0]; i<=shell[0]; ++i){
			for(int j=-shell[1]; j<=shell[1]; ++j){
				for(int k=-shell[2]; k<=shell[2]; ++k){
					R_[count++].noalias()=i*Rv[0]+j*Rv[1]+k*Rv[2];
				}
			}
		}
		//loop over all atoms
		if(NEIGH_PRINT_STATUS>0) std::cout<<"computing neighbor list\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			//clear the neighbor list
			neigh_[i].clear();
			//loop over all atoms
			for(int j=0; j<struc.nAtoms(); ++j){
				//set up local neighbor list
				std::vector<Neighbor> neighj_;
				//compute the distance vector with p.b.c.
				const Eigen::Vector3d rIJ_=struc.diff(struc.posn(i),struc.posn(j),tmp);
				//loop over lattice vector shifts - atom j
				for(int n=0; n<Rsize; ++n){
					//shift the rIJ_ distance by a lattice vector shift
					const Eigen::Vector3d rIJt_=rIJ_-R_[n];
					const double dIJ2=rIJt_.squaredNorm();
					if(j!=i){
						if(dIJ2<=rc2){
							neighj_.push_back(Neighbor());
							neighj_.back().r()=rIJt_;
							neighj_.back().dr()=std::sqrt(dIJ2);
							neighj_.back().type()=struc.type(j);
							neighj_.back().index()=struc.index(j);
						}
					} else{
						if(math::constant::ZERO<dIJ2 && dIJ2<=rc2){
							neighj_.push_back(Neighbor());
							neighj_.back().r()=rIJt_;
							neighj_.back().dr()=std::sqrt(dIJ2);
							neighj_.back().type()=struc.type(j);
							neighj_.back().index()=struc.index(j);
						}
					}
				}
				//find and label the minimum image (if it exists)
				if(j!=i && neighj_.size()>0){
					int loc=0; double min=neighj_[0].dr();
					for(int n=1; n<neighj_.size(); ++n){
						if(neighj_[n].dr()<min){
							min=neighj_[n].dr();
							loc=n;
						}
					}
					neighj_[loc].min()=true;
					//std::cout<<"neigh index "<<neighj_[loc].index()<<"\n";
				}
				//append to full neighbor list
				for(int n=0; n<neighj_.size(); ++n){
					neigh_[i].push_back(neighj_[n]);
				}
			}
		}
	} else {
		//loop over all atoms
		if(NEIGH_PRINT_STATUS>0) std::cout<<"computing neighbor list\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			//clear the neighbor list
			neigh_[i].clear();
			//loop over all atoms
			for(int j=0; j<struc.nAtoms(); ++j){
				const Eigen::Vector3d rIJ_=struc.posn(i)-struc.posn(j);
				//loop over lattice vector shifts - atom j
				const double dIJ2=rIJ_.squaredNorm();
				if(math::constant::ZERO<dIJ2 && dIJ2<=rc2){
					neigh_[i].push_back(Neighbor());
					neigh_[i].back().r()=rIJ_;
					neigh_[i].back().dr()=std::sqrt(dIJ2);
					neigh_[i].back().type()=struc.type(j);
					neigh_[i].back().index()=j;
					//if(neigh_[i].back().index()>=0) std::cout<<"neigh index "<<neigh_[i].back().index()<<"\n";
				}
			}
		}
	}
}

void NeighborList::build(const Structure& struc, double rc, int ii){
	if(NEIGH_PRINT_FUNC>0) std::cout<<"NeighborList::build(const Structure&,double,int):\n";
	rc_=rc;
	//resize the neighbor list
	neigh_.resize(struc.nAtoms());
	//local variables
	const double rc2=rc*rc;
	//lattice vector shifts
	if(struc.R().norm()>0){
		if(NEIGH_PRINT_STATUS>0) std::cout<<"storing lattice vectors\n";
		Eigen::Vector3d tmp;
		//store lattice vectors
		std::vector<Eigen::Vector3d> Rv(3);
		for(int i=0; i<3; ++i) Rv[i]=struc.R().col(i);
		//sort from largest to smallest
		std::sort(Rv.begin(),Rv.end(),sortcmp);
		if(NEIGH_PRINT_STATUS>0) std::cout<<"compting common multiples\n";
		//compute multiples
		std::vector<int> shell(3);
		shell[0]=floor(2.0*rc/Rv[0].norm());
		shell[1]=floor(2.0*rc/(Rv[0].cross(Rv[1]).norm()/Rv[0].norm()));
		shell[2]=floor(2.0*rc/fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm())));
		const int Rsize=(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1);
		if(NEIGH_PRINT_DATA>0) {
			std::cout<<"V  = "<<Rv[0].norm()<<" "<<(Rv[0].cross(Rv[1]).norm()/Rv[0].norm())<<" "<<fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm()))<<"\n";
			std::cout<<"shell = ("<<shell[0]<<","<<shell[1]<<","<<shell[2]<<") = "<<(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1)<<"\n";
			std::cout<<"Rsize = "<<Rsize<<"\n";
		}
		std::vector<Eigen::Vector3d> R_(Rsize);
		int count=0;
		for(int i=-shell[0]; i<=shell[0]; ++i){
			for(int j=-shell[1]; j<=shell[1]; ++j){
				for(int k=-shell[2]; k<=shell[2]; ++k){
					R_[count++].noalias()=i*Rv[0]+j*Rv[1]+k*Rv[2];
				}
			}
		}
		//loop over all atoms
		if(NEIGH_PRINT_STATUS>0) std::cout<<"computing neighbor list\n";
		//clear the neighbor list
		neigh_[ii].clear();
		//loop over all atoms
		for(int j=0; j<struc.nAtoms(); ++j){
			//set up local neighbor list
			std::vector<Neighbor> neighj_;
			//compute the distance vector with p.b.c.
			const Eigen::Vector3d rIJ_=struc.diff(struc.posn(ii),struc.posn(j),tmp);
			//loop over lattice vector shifts - atom j
			for(int n=0; n<Rsize; ++n){
				//shift the rIJ_ distance by a lattice vector shift
				const Eigen::Vector3d rIJt_=rIJ_-R_[n];
				const double dIJ2=rIJt_.squaredNorm();
				if(j!=ii){
					if(dIJ2<=rc2){
						neighj_.push_back(Neighbor());
						neighj_.back().r()=rIJt_;
						neighj_.back().dr()=std::sqrt(dIJ2);
						neighj_.back().type()=struc.type(j);
						neighj_.back().index()=struc.index(j);
					}
				} else{
					if(math::constant::ZERO<dIJ2 && dIJ2<=rc2){
						neighj_.push_back(Neighbor());
						neighj_.back().r()=rIJt_;
						neighj_.back().dr()=std::sqrt(dIJ2);
						neighj_.back().type()=struc.type(j);
						neighj_.back().index()=struc.index(j);
					}
				}
			}
			//find and label the minimum image (if it exists)
			if(j!=ii && neighj_.size()>0){
				double min=struc.R().norm();
				int loc=-1;
				for(int n=0; n<neighj_.size(); ++n){
					if(neighj_[n].dr()<min){
						min=neighj_[n].dr();
						loc=n;
					}
				}
				if(loc>=0) neighj_[loc].min()=true;
				//std::cout<<"neigh index "<<neighj_[loc].index()<<"\n";
			}
			//append to full neighbor list
			for(int n=0; n<neighj_.size(); ++n){
				neigh_[ii].push_back(neighj_[n]);
			}
		}
	} else {
		//loop over all atoms
		if(NEIGH_PRINT_STATUS>0) std::cout<<"computing neighbor list\n";
		//clear the neighbor list
		neigh_[ii].clear();
		//loop over all atoms
		for(int j=0; j<struc.nAtoms(); ++j){
			const Eigen::Vector3d rIJ_=struc.posn(ii)-struc.posn(j);
			//loop over lattice vector shifts - atom j
			const double dIJ2=rIJ_.squaredNorm();
			if(math::constant::ZERO<dIJ2 && dIJ2<=rc2){
				neigh_[ii].push_back(Neighbor());
				neigh_[ii].back().r()=rIJ_;
				neigh_[ii].back().dr()=std::sqrt(dIJ2);
				neigh_[ii].back().type()=struc.type(j);
				neigh_[ii].back().index()=j;
				//if(neigh_[ii].back().index()>=0) std::cout<<"neigh index "<<neigh_[ii].back().index()<<"\n";
			}
		}
	}
}

void NeighborList::build(const Structure& struc, double rc, const std::vector<int>& subset){
	if(NEIGH_PRINT_FUNC>0) std::cout<<"NeighborList::build(const Structure&,double):\n";
	rc_=rc;
	//resize the neighbor list
	neigh_.resize(struc.nAtoms());
	//local variables
	const double rc2=rc*rc;
	//lattice vector shifts
	if(struc.R().norm()>0){
		if(NEIGH_PRINT_STATUS>0) std::cout<<"storing lattice vectors\n";
		Eigen::Vector3d tmp;
		//store lattice vectors
		std::vector<Eigen::Vector3d> Rv(3);
		for(int i=0; i<3; ++i) Rv[i]=struc.R().col(i);
		//sort from largest to smallest
		std::sort(Rv.begin(),Rv.end(),sortcmp);
		if(NEIGH_PRINT_STATUS>0) std::cout<<"compting common multiples\n";
		//compute multiples
		std::vector<int> shell(3);
		shell[0]=floor(2.0*rc/Rv[0].norm());
		shell[1]=floor(2.0*rc/(Rv[0].cross(Rv[1]).norm()/Rv[0].norm()));
		shell[2]=floor(2.0*rc/fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm())));
		const int Rsize=(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1);
		if(NEIGH_PRINT_DATA>0) {
			std::cout<<"V  = "<<Rv[0].norm()<<" "<<(Rv[0].cross(Rv[1]).norm()/Rv[0].norm())<<" "<<fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm()))<<"\n";
			std::cout<<"shell = ("<<shell[0]<<","<<shell[1]<<","<<shell[2]<<") = "<<(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1)<<"\n";
			std::cout<<"Rsize = "<<Rsize<<"\n";
		}
		std::vector<Eigen::Vector3d> R_(Rsize);
		int count=0;
		for(int i=-shell[0]; i<=shell[0]; ++i){
			for(int j=-shell[1]; j<=shell[1]; ++j){
				for(int k=-shell[2]; k<=shell[2]; ++k){
					R_[count++].noalias()=i*Rv[0]+j*Rv[1]+k*Rv[2];
				}
			}
		}
		//loop over all atoms
		if(NEIGH_PRINT_STATUS>0) std::cout<<"computing neighbor list\n";
		for(int ii=0; ii<subset.size(); ++ii){
			const int i=subset[ii];
			//clear the neighbor list
			neigh_[i].clear();
			//loop over all atoms
			for(int j=0; j<struc.nAtoms(); ++j){
				//set up local neighbor list
				std::vector<Neighbor> neighj_;
				//compute the distance vector with p.b.c.
				const Eigen::Vector3d rIJ_=struc.diff(struc.posn(i),struc.posn(j),tmp);
				//loop over lattice vector shifts - atom j
				for(int n=0; n<Rsize; ++n){
					//shift the rIJ_ distance by a lattice vector shift
					const Eigen::Vector3d rIJt_=rIJ_-R_[n];
					const double dIJ2=rIJt_.squaredNorm();
					if(j!=i){
						if(dIJ2<=rc2){
							neighj_.push_back(Neighbor());
							neighj_.back().r()=rIJt_;
							neighj_.back().dr()=std::sqrt(dIJ2);
							neighj_.back().type()=struc.type(j);
							neighj_.back().index()=struc.index(j);
						}
					} else{
						if(math::constant::ZERO<dIJ2 && dIJ2<=rc2){
							neighj_.push_back(Neighbor());
							neighj_.back().r()=rIJt_;
							neighj_.back().dr()=std::sqrt(dIJ2);
							neighj_.back().type()=struc.type(j);
							neighj_.back().index()=struc.index(j);
						}
					}
				}
				//find and label the minimum image (if it exists)
				if(j!=i && neighj_.size()>0){
					double min=struc.R().norm();
					int loc=-1;
					for(int n=0; n<neighj_.size(); ++n){
						if(neighj_[n].dr()<min){
							min=neighj_[n].dr();
							loc=n;
						}
					}
					if(loc>=0) neighj_[loc].min()=true;
					//std::cout<<"neigh index "<<neighj_[loc].index()<<"\n";
				}
				//append to full neighbor list
				for(int n=0; n<neighj_.size(); ++n){
					neigh_[i].push_back(neighj_[n]);
				}
			}
		}
	} else {
		//loop over all atoms
		if(NEIGH_PRINT_STATUS>0) std::cout<<"computing neighbor list\n";
		for(int ii=0; ii<subset.size(); ++ii){
			const int i=subset[ii];
			//clear the neighbor list
			neigh_[i].clear();
			//loop over all atoms
			for(int j=0; j<struc.nAtoms(); ++j){
				const Eigen::Vector3d rIJ_=struc.posn(i)-struc.posn(j);
				//loop over lattice vector shifts - atom j
				const double dIJ2=rIJ_.squaredNorm();
				if(math::constant::ZERO<dIJ2 && dIJ2<=rc2){
					neigh_[i].push_back(Neighbor());
					neigh_[i].back().r()=rIJ_;
					neigh_[i].back().dr()=std::sqrt(dIJ2);
					neigh_[i].back().type()=struc.type(j);
					neigh_[i].back().index()=j;
					//if(neigh_[i].back().index()>=0) std::cout<<"neigh index "<<neigh_[i].back().index()<<"\n";
				}
			}
		}
	}
}

std::vector<Eigen::Vector3d>& NeighborList::ilist(const Structure& struc, double rc, std::vector<Eigen::Vector3d>& Rlist){
	if(NEIGH_PRINT_FUNC>0) std::cout<<"NeighborList::ilist():\n";
	//store lattice vectors
	std::vector<Eigen::Vector3d> Rv(3);
	for(int i=0; i<3; ++i) Rv[i]=struc.R().col(i);
	//sort from largest to smallest
	std::sort(Rv.begin(),Rv.end(),sortcmp);
	//compute multiples
	std::vector<int> shell(3);
	shell[0]=floor(2.0*rc/Rv[0].norm());
	shell[1]=floor(2.0*rc/(Rv[0].cross(Rv[1]).norm()/Rv[0].norm()));
	shell[2]=floor(2.0*rc/fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm())));
	const int Rsize=(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1);
	if(STRUC_PRINT_DATA>0) {
		std::cout<<"V  = "<<Rv[0].norm()<<" "<<(Rv[0].cross(Rv[1]).norm()/Rv[0].norm())<<" "<<fabs(Rv[2].dot(Rv[0].cross(Rv[1]))/(Rv[0].norm()*Rv[1].norm()))<<"\n";
		std::cout<<"shell = ("<<shell[0]<<","<<shell[1]<<","<<shell[2]<<") = "<<(2*shell[0]+1)*(2*shell[1]+1)*(2*shell[2]+1)<<"\n";
	}
	Rlist.resize(Rsize);
	int count=0;
	for(int i=-shell[0]; i<=shell[0]; ++i){
		for(int j=-shell[1]; j<=shell[1]; ++j){
			for(int k=-shell[2]; k<=shell[2]; ++k){
				Rlist[count++].noalias()=i*Rv[0]+j*Rv[1]+k*Rv[2];
			}
		}
	}
	return Rlist;
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Neighbor& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const Neighbor&)\n";
		int size=0;
		size+=nbytes(obj.r());//r_
		size+=sizeof(double);//dr_
		size+=sizeof(int);//type_
		size+=sizeof(int);//index_
		size+=sizeof(bool);//min_
		return size;
	}
	template <> int nbytes(const NeighborList& obj){
		if(STRUC_PRINT_FUNC>0) std::cout<<"nbytes(const NeighborList&)\n";
		int size=0;
		size+=sizeof(double);//rcut_
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Neighbor& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const Neighbor&,char*):\n";
		int pos=0;
		pos+=pack(obj.r(),arr+pos);
		std::memcpy(arr+pos,&obj.dr(),sizeof(double)); pos+=sizeof(double);
		std::memcpy(arr+pos,&obj.type(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.index(),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,&obj.min(),sizeof(bool)); pos+=sizeof(bool);
		return pos;
	}
	template <> int pack(const NeighborList& obj, char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"pack(const NeighborList&,char*):\n";
		int pos=0;
		std::memcpy(arr+pos,&obj.rc(),sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Neighbor& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(Neighbor&,const char*):\n";
		int pos=0;
		pos+=unpack(obj.r(),arr+pos);
		std::memcpy(&obj.dr(),arr+pos,sizeof(double)); pos+=sizeof(double);
		std::memcpy(&obj.type(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.index(),arr+pos,sizeof(int)); pos+=sizeof(int);
		std::memcpy(&obj.min(),arr+pos,sizeof(bool)); pos+=sizeof(bool);
		return pos;
	}
	template <> int unpack(NeighborList& obj, const char* arr){
		if(STRUC_PRINT_FUNC>0) std::cout<<"unpack(NeighborList&,const char*):\n";
		int pos=0;
		std::memcpy(&obj.rc(),arr+pos,sizeof(double)); pos+=sizeof(double);
		return pos;
	}
	
}
