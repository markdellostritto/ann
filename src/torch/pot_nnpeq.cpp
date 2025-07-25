// c++
#include <iostream>
// chem
#include "chem/units.hpp"
// pot
#include "torch/pot_nnpeq.hpp"

namespace ptnl{
	
//==== operator ====

std::ostream& operator<<(std::ostream& out, const PotNNPEQ& pot){
	return out<<static_cast<const Pot&>(pot);
}
	
//==== member functions ====

void PotNNPEQ::read(Token& token){
	static_cast<Pot&>(*this).read(token);
}

void PotNNPEQ::coeff(Token& token){
	//coeff nn file name1 name2 name3 ...
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::coeff(Token&):\n";
	std::string pfile=token.next();
	std::vector<std::string> names;
	while(!token.end()) names.push_back(token.next());
	//read file
	read(pfile.c_str());
}

void PotNNPEQ::resize(int ntypes){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::resize(int):\n";
	if(ntypes<=0) throw std::invalid_argument("Invalid number of types.");
	ntypes_=ntypes;
	nnh_.resize(ntypes_);
	for(int i=0; i<ntypes_; ++i){
		nnh_[i].resize(ntypes_);
	}
}

/**
* execute all atomic networks and return energy
* @param nnp - the neural network potential
* @param struc - the structure for which we will compute the energy
* @return total energy
* it is assumed that the symmetry functions have been computed
*/
double PotNNPEQ::energy(const Structure& struc, const NeighborList& nlist){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::energy(Structure&,const NeighborList&):\n";
	double energy=0;
	//compute all atoms
	for(int i=0; i<struc.nAtoms(); ++i){
		//get the index of species i
		const int II=struc.type(i);
		
		//compute the symmetry functions
		symm_[II].setZero();
		for(int j=0; j<nlist.size(i); ++j){
			//get the index of species j
			//const int JJ=nlist.neigh(i,j).type();
			const int JJ=index(struc.name(nlist.neigh(i,j).index()));
			//get the distance from J to I
			const Eigen::Vector3d& rIJ=nlist.neigh(i,j).r();
			const double dIJ=nlist.neigh(i,j).dr();
			{
			//set the basis
			const int offsetR_=nnh_[II].offsetR(JJ);
			BasisR& basisRij_=nnh_[II].basisR(JJ);
			//compute the IJ contribution to all radial basis functions
			basisRij_.symm(dIJ);
			for(int nr=0; nr<basisRij_.size(); ++nr){
				symm_[II][offsetR_+nr]+=basisRij_.symm()[nr];
			}
			}
			//loop over all unique triplets
			for(int k=j+1; k<nlist.size(i); ++k){
				//find the index of the species of atom k
				//const int KK=nlist.neigh(i,k).type();
				const int KK=index(struc.name(nlist.neigh(i,k).index()));
				//get the distance from K to I
				const Eigen::Vector3d& rIK=nlist.neigh(i,k).r();
				const double dIK=nlist.neigh(i,k).dr();
				//get the distance from J to K
				const double dJK=(rIK-rIJ).norm();
				//compute the cosIJK angle and store the distances
				const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);
				const double d[3]={dIJ,dIK,dJK};
				//set the basis
				const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
				BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
				//compute the IJ,IK,JK contribution to all angular basis functions
				basisAijk_.symm(cosIJK,d);
				for(int na=0; na<basisAijk_.size(); ++na){
					symm_[II][offsetA_+na]+=basisAijk_.symm()[na];
				}
			}
		}
		
		//execute newtwork
		nnh_[II].nn().fp(symm_[II]);
		//compute the energy
		energy+=nnh_[II].nn().out()[0]+nnh_[II].type().energy().val();
		//compute the electronegativity
		//struc.chi(i)=nnh_[II].nn().out()[1]+nnh_[II].type().chi().val();
	}
	
	return energy;
}

double PotNNPEQ::energy(const Structure& struc, const NeighborList& nlist, int i){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"NNP::energy(Structure&,const NeighborList&, int):\n";
	//get the index of species i
	const int II=struc.type(i);
	
	//compute the symmetry functions
	symm_[II].setZero();
	for(int j=0; j<nlist.size(i); ++j){
		//get the index of species j
		//const int JJ=nlist.neigh(i,j).type();
		const int JJ=index(struc.name(nlist.neigh(i,j).index()));
		//get the distance from J to I
		const Eigen::Vector3d& rIJ=nlist.neigh(i,j).r();
		const double dIJ=nlist.neigh(i,j).dr();
		{
		//set the basis
		const int offsetR_=nnh_[II].offsetR(JJ);
		BasisR& basisRij_=nnh_[II].basisR(JJ);
		//compute the IJ contribution to all radial basis functions
		basisRij_.symm(dIJ);
		for(int nr=0; nr<basisRij_.size(); ++nr){
			symm_[II][offsetR_+nr]+=basisRij_.symm()[nr];
		}
		}
		//loop over all unique triplets
		for(int k=j+1; k<nlist.size(i); ++k){
			//find the index of the species of atom k
			//const int KK=nlist.neigh(i,k).type();
			const int KK=index(struc.name(nlist.neigh(i,k).index()));
			//get the distance from K to I
			const Eigen::Vector3d& rIK=nlist.neigh(i,k).r();
			const double dIK=nlist.neigh(i,k).dr();
			//get the distance from J to K
			const double dJK=(rIK-rIJ).norm();
			//compute the cosIJK angle and store the distances
			const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);
			const double d[3]={dIJ,dIK,dJK};
			//set the basis
			const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
			BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
			//compute the IJ,IK,JK contribution to all angular basis functions
			basisAijk_.symm(cosIJK,d);
			for(int na=0; na<basisAijk_.size(); ++na){
				symm_[II][offsetA_+na]+=basisAijk_.symm()[na];
			}
		}
	}
	
	//execute newtwork
	nnh_[II].nn().fp(symm_[II]);
	//compute the energy
	const double energy=nnh_[II].nn().out()[0]+nnh_[II].type().energy().val();
	//compute the electronegativity
	//struc.chi(i)=nnh_[II].nn().out()[1]+nnh_[II].type().chi().val();
	
	return energy;
}

/**
* Compute the forces on the atoms for a given structure
* @param nnp - the neural network potential
* @param struc - the structure which we will compute
* @param nlist - neighbor list for each atom (includes periodic images)
* The forces are computed by looping over all nearest-neighbor pairs
* and all unique nearest-neighbor triples.  Thus, the neighbor list must be set 
* for the structure, and the type must correspond to the index of the atomic species
* in the NNP.  In addition, the index for each neighbor must be set to -1 if it is
* a periodic image or within [0,natoms] in which case the Newton's third law force
* pair must be added to the neighbor atom.
*/
double PotNNPEQ::compute(Structure& struc, const NeighborList& nlist){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"NNP::force(Structure&,const NeighborList&):\n";
	double energy=0;
	//compute all atoms
	for(int i=0; i<struc.nAtoms(); ++i){
		//get the index of species i
		const int II=struc.type(i);
		
		//compute the symmetry functions
		symm_[II].setZero();
		for(int j=0; j<nlist.size(i); ++j){
			//get the index of species j
			//const int JJ=nlist.neigh(i,j).type();
			const int JJ=index(struc.name(nlist.neigh(i,j).index()));
			//get the distance from J to I
			const Eigen::Vector3d& rIJ=nlist.neigh(i,j).r();
			const double dIJ=nlist.neigh(i,j).dr();
			{
			//set the basis
			const int offsetR_=nnh_[II].offsetR(JJ);
			BasisR& basisRij_=nnh_[II].basisR(JJ);
			//compute the IJ contribution to all radial basis functions
			basisRij_.symm(dIJ);
			for(int nr=0; nr<basisRij_.size(); ++nr){
				symm_[II][offsetR_+nr]+=basisRij_.symm()[nr];
			}
			}
			//loop over all unique triplets
			for(int k=j+1; k<nlist.size(i); ++k){
				//find the index of the species of atom k
				//const int KK=nlist.neigh(i,k).type();
				const int KK=index(struc.name(nlist.neigh(i,k).index()));
				//get the distance from K to I
				const Eigen::Vector3d& rIK=nlist.neigh(i,k).r();
				const double dIK=nlist.neigh(i,k).dr();
				//get the distance from J to K
				const double dJK=(rIK-rIJ).norm();
				//compute the cosIJK angle and store the distances
				const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);
				const double d[3]={dIJ,dIK,dJK};
				//set the basis
				const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
				BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
				//compute the IJ,IK,JK contribution to all angular basis functions
				basisAijk_.symm(cosIJK,d);
				for(int na=0; na<basisAijk_.size(); ++na){
					symm_[II][offsetA_+na]+=basisAijk_.symm()[na];
				}
			}
		}
		
		//execute the network
		nnh_[II].nn().fpbp(symm_[II]);
		//compute the energy
		energy+=nnh_[II].nn().out()[0]+nnh_[II].type().energy().val();
		//compute the electronegativity
		struc.chi(i)=nnh_[II].nn().out()[1]+nnh_[II].type().chi().val();
		//compute the network gradient
		nnh_[II].dOdZ().grad(nnh_[II].nn());
		const Eigen::VectorXd& dEdG=nnh_[II].dOdZ().dodi().row(0);
		
		//compute the forces
		for(int j=0; j<nlist.size(i); ++j){
			//get the indices of the jth neighbor
			const int jj=nlist.neigh(i,j).index();
			//const int JJ=nlist.neigh(i,j).type();
			const int JJ=index(struc.name(jj));
			const bool jmin=nlist.neigh(i,j).min();
			//get the distance from J to I
			const Eigen::Vector3d& rIJ=nlist.neigh(i,j).r();
			const double dIJ=nlist.neigh(i,j).dr();
			if(dIJ<rc_){
				const double dIJi=1.0/dIJ;
				//compute the IJ contribution to the radial force
				{
				const int offsetR_=nnh_[II].offsetR(JJ);
				const double amp=nnh_[II].basisR(JJ).force(dIJ,dEdG.data()+offsetR_)*dIJi;
				struc.force(i).noalias()+=amp*rIJ;
				if(jmin) struc.force(jj).noalias()-=amp*rIJ;
				}
				//loop over all unique triplets
				for(int k=j+1; k<nlist.size(i); ++k){
					//find the index of the species of atom k
					const int kk=nlist.neigh(i,k).index();
					//const int KK=nlist.neigh(i,k).type();
					const int KK=index(struc.name(kk));
					const bool kmin=nlist.neigh(i,k).min();
					//get the distance from K to I
					const Eigen::Vector3d& rIK=nlist.neigh(i,k).r();
					const double dIK=nlist.neigh(i,k).dr();
					if(dIK<rc_){
						const double dIKi=1.0/dIK;
						//get the distance from J to K
						const Eigen::Vector3d rJK=(rIK-rIJ);
						const double dJK=rJK.norm();
						const double dJKi=1.0/dJK;
						//set the basis
						const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
						BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
						//compute the IJ,IK,JK contribution to the angular force
						double phi=0; double eta[3]={0,0,0};
						const double cosIJK=rIJ.dot(rIK)*dIJi*dIKi;
						const double d[3]={dIJ,dIK,dJK};
						nnh_[II].basisA(JJ,KK).force(phi,eta,cosIJK,d,dEdG.data()+offsetA_);
						struc.force(i).noalias()+=(phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJ*dIJi;
						struc.force(i).noalias()+=(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIK*dIKi;
						if(jmin){
							struc.force(jj).noalias()-=(-phi*cosIJK*dIJi+eta[0])*rIJ*dIJi+phi*dIJi*rIK*dIKi;
							struc.force(jj).noalias()-=eta[2]*rJK*dJKi;
						}
						if(kmin){
							struc.force(kk).noalias()-=(-phi*cosIJK*dIKi+eta[1])*rIK*dIKi+phi*dIKi*rIJ*dIJi;
							struc.force(kk).noalias()+=eta[2]*rJK*dJKi;
						}
					}
				}
			}
		}
	}
	
	struc.pe()+=energy;
	return energy;
}

/**
* execute all atomic networks and return energy
* @param nnp - the neural network potential
* @param struc - the structure for which we will compute the energy
* @return total energy
* it is assumed that the symmetry functions have been computed
*/
double PotNNPEQ::energy(const Structure& struc, const verlet::List& vlist){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::energy(Structure&,const verlet::List&):\n";
	double energy=0;
	//compute all atoms
	for(int i=0; i<struc.nAtoms(); ++i){
		//get the index of species i
		const int II=struc.type(i);
		
		//compute the symmetry functions
		symm_[II].setZero();
		for(int j=0; j<vlist.size(i); ++j){
			//get the index of species j
			const int jj=vlist.neigh(i,j).index();
			const int JJ=index(struc.name(jj));
			//get the distance from J to I
			Eigen::Vector3d rIJ;
			struc.diff(struc.posn(i),struc.posn(jj),rIJ);
			rIJ.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dIJ=rIJ.norm();
			{
			//set the basis
			const int offsetR_=nnh_[II].offsetR(JJ);
			BasisR& basisRij_=nnh_[II].basisR(JJ);
			//compute the IJ contribution to all radial basis functions
			basisRij_.symm(dIJ);
			for(int nr=0; nr<basisRij_.size(); ++nr){
				symm_[II][offsetR_+nr]+=basisRij_.symm()[nr];
			}
			}
			//loop over all unique triplets
			for(int k=j+1; k<vlist.size(i); ++k){
				//find the index of the species of atom k
				const int kk=vlist.neigh(i,k).index();
				const int KK=index(struc.name(k));
				//get the distance from K to I
				Eigen::Vector3d rIK;
				struc.diff(struc.posn(i),struc.posn(jj),rIK);
				rIK.noalias()-=struc.R()*vlist.neigh(i,j).cell();
				const double dIK=rIK.norm();
				//get the distance from J to K
				const double dJK=(rIK-rIJ).norm();
				//compute the cosIJK angle and store the distances
				const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);
				const double d[3]={dIJ,dIK,dJK};
				//set the basis
				const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
				BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
				//compute the IJ,IK,JK contribution to all angular basis functions
				basisAijk_.symm(cosIJK,d);
				for(int na=0; na<basisAijk_.size(); ++na){
					symm_[II][offsetA_+na]+=basisAijk_.symm()[na];
				}
			}
		}
		
		//execute newtwork
		nnh_[II].nn().fp(symm_[II]);
		//compute the energy
		energy+=nnh_[II].nn().out()[0]+nnh_[II].type().energy().val();
		//compute the electronegativity
		//struc.chi(i)=nnh_[II].nn().out()[1]+nnh_[II].type().chi().val();
	}
	
	return energy;
}

double PotNNPEQ::energy(const Structure& struc, const verlet::List& vlist, int i){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"NNP::energy(Structure&,const verlet::List&, int):\n";
	//get the index of species i
	const int II=struc.type(i);
	
	//compute the symmetry functions
	symm_[II].setZero();
	for(int j=0; j<vlist.size(i); ++j){
		//get the index of species j
		const int jj=vlist.neigh(i,j).index();
		const int JJ=index(struc.name(JJ));
		//get the distance from J to I
		Eigen::Vector3d rIJ;
		struc.diff(struc.posn(i),struc.posn(jj),rIJ);
		rIJ.noalias()-=struc.R()*vlist.neigh(i,j).cell();
		const double dIJ=rIJ.norm();
		{
		//set the basis
		const int offsetR_=nnh_[II].offsetR(JJ);
		BasisR& basisRij_=nnh_[II].basisR(JJ);
		//compute the IJ contribution to all radial basis functions
		basisRij_.symm(dIJ);
		for(int nr=0; nr<basisRij_.size(); ++nr){
			symm_[II][offsetR_+nr]+=basisRij_.symm()[nr];
		}
		}
		//loop over all unique triplets
		for(int k=j+1; k<vlist.size(i); ++k){
			//find the index of the species of atom k
			const int kk=vlist.neigh(i,k).index();
			const int KK=index(struc.name(k));
			//get the distance from K to I
			Eigen::Vector3d rIK;
			struc.diff(struc.posn(i),struc.posn(jj),rIK);
			rIK.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dIK=rIK.norm();
			//get the distance from J to K
			const double dJK=(rIK-rIJ).norm();
			//compute the cosIJK angle and store the distances
			const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);
			const double d[3]={dIJ,dIK,dJK};
			//set the basis
			const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
			BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
			//compute the IJ,IK,JK contribution to all angular basis functions
			basisAijk_.symm(cosIJK,d);
			for(int na=0; na<basisAijk_.size(); ++na){
				symm_[II][offsetA_+na]+=basisAijk_.symm()[na];
			}
		}
	}
	
	//execute newtwork
	nnh_[II].nn().fp(symm_[II]);
	//compute the energy
	const double energy=nnh_[II].nn().out()[0]+nnh_[II].type().energy().val();
	//compute the electronegativity
	//struc.chi(i)=nnh_[II].nn().out()[1]+nnh_[II].type().chi().val();
	
	return energy;
}

/**
* Compute the forces on the atoms for a given structure
* @param nnp - the neural network potential
* @param struc - the structure which we will compute
* @param vlist - neighbor list for each atom (includes periodic images)
* The forces are computed by looping over all nearest-neighbor pairs
* and all unique nearest-neighbor triples.  Thus, the neighbor list must be set 
* for the structure, and the type must correspond to the index of the atomic species
* in the NNP.  In addition, the index for each neighbor must be set to -1 if it is
* a periodic image or within [0,natoms] in which case the Newton's third law force
* pair must be added to the neighbor atom.
*/
double PotNNPEQ::compute(Structure& struc, const verlet::List& vlist){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"NNP::force(Structure&,const verlet::List&):\n";
	double energy=0;
	//compute all atoms
	for(int i=0; i<struc.nAtoms(); ++i){
		//get the index of species i
		const int II=struc.type(i);
		
		//compute the symmetry functions
		symm_[II].setZero();
		for(int j=0; j<vlist.size(i); ++j){
			//get the index of species j
			const int jj=vlist.neigh(i,j).index();
			const int JJ=index(struc.name(jj));
			//get the distance from J to I
			Eigen::Vector3d rIJ;
			struc.diff(struc.posn(i),struc.posn(jj),rIJ);
			rIJ.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dIJ=rIJ.norm();
			{
			//set the basis
			const int offsetR_=nnh_[II].offsetR(JJ);
			BasisR& basisRij_=nnh_[II].basisR(JJ);
			//compute the IJ contribution to all radial basis functions
			basisRij_.symm(dIJ);
			for(int nr=0; nr<basisRij_.size(); ++nr){
				symm_[II][offsetR_+nr]+=basisRij_.symm()[nr];
			}
			}
			//loop over all unique triplets
			for(int k=j+1; k<vlist.size(i); ++k){
				//find the index of the species of atom k
				const int kk=vlist.neigh(i,k).index();
				const int KK=index(struc.name(kk));
				//get the distance from K to I
				Eigen::Vector3d rIK;
				struc.diff(struc.posn(i),struc.posn(jj),rIK);
				rIK.noalias()-=struc.R()*vlist.neigh(i,j).cell();
				const double dIK=rIK.norm();
				//get the distance from J to K
				const double dJK=(rIK-rIJ).norm();
				//compute the cosIJK angle and store the distances
				const double cosIJK=rIJ.dot(rIK)/(dIJ*dIK);
				const double d[3]={dIJ,dIK,dJK};
				//set the basis
				const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
				BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
				//compute the IJ,IK,JK contribution to all angular basis functions
				basisAijk_.symm(cosIJK,d);
				for(int na=0; na<basisAijk_.size(); ++na){
					symm_[II][offsetA_+na]+=basisAijk_.symm()[na];
				}
			}
		}
		
		//execute the network
		nnh_[II].nn().fpbp(symm_[II]);
		//compute the energy
		energy+=nnh_[II].nn().out()[0]+nnh_[II].type().energy().val();
		//compute the electronegativity
		struc.chi(i)=nnh_[II].nn().out()[1]+nnh_[II].type().chi().val();
		//compute the network gradient
		nnh_[II].dOdZ().grad(nnh_[II].nn());
		const Eigen::VectorXd& dEdG=nnh_[II].dOdZ().dodi().row(0);
		
		//compute the forces
		for(int j=0; j<vlist.size(i); ++j){
			//get the indices of the jth neighbor
			const int jj=vlist.neigh(i,j).index();
			const int JJ=index(struc.name(jj));
			//get the distance from J to I
			Eigen::Vector3d rIJ;
			struc.diff(struc.posn(i),struc.posn(jj),rIJ);
			rIJ.noalias()-=struc.R()*vlist.neigh(i,j).cell();
			const double dIJ=rIJ.norm();
			if(dIJ<rc_){
				const double dIJi=1.0/dIJ;
				bool jmin=(vlist.neigh(i,j).cell().norm()>0)?false:true;
				//compute the IJ contribution to the radial force
				{
				const int offsetR_=nnh_[II].offsetR(JJ);
				const double amp=nnh_[II].basisR(JJ).force(dIJ,dEdG.data()+offsetR_)*dIJi;
				struc.force(i).noalias()+=amp*rIJ;
				if(jmin) struc.force(jj).noalias()-=amp*rIJ;
				}
				//loop over all unique triplets
				for(int k=j+1; k<vlist.size(i); ++k){
					//find the index of the species of atom k
					const int kk=vlist.neigh(i,k).index();
					const int KK=index(struc.name(kk));
					//get the distance from K to I
					Eigen::Vector3d rIK;
					struc.diff(struc.posn(i),struc.posn(jj),rIK);
					rIK.noalias()-=struc.R()*vlist.neigh(i,j).cell();
					const double dIK=rIK.norm();
					if(dIK<rc_){
						const double dIKi=1.0/dIK;
						bool kmin=(vlist.neigh(i,k).cell().norm()>0)?false:true;
						//get the distance from J to K
						const Eigen::Vector3d rJK=(rIK-rIJ);
						const double dJK=rJK.norm();
						const double dJKi=1.0/dJK;
						//set the basis
						const int offsetA_=nnh_[II].nInputR()+nnh_[II].offsetA(JJ,KK);
						BasisA& basisAijk_=nnh_[II].basisA(JJ,KK);
						//compute the IJ,IK,JK contribution to the angular force
						double phi=0; double eta[3]={0,0,0};
						const double cosIJK=rIJ.dot(rIK)*dIJi*dIKi;
						const double d[3]={dIJ,dIK,dJK};
						nnh_[II].basisA(JJ,KK).force(phi,eta,cosIJK,d,dEdG.data()+offsetA_);
						struc.force(i).noalias()+=(phi*(dIKi-cosIJK*dIJi)+eta[0])*rIJ*dIJi;
						struc.force(i).noalias()+=(phi*(dIJi-cosIJK*dIKi)+eta[1])*rIK*dIKi;
						if(jmin){
							struc.force(jj).noalias()-=(-phi*cosIJK*dIJi+eta[0])*rIJ*dIJi+phi*dIJi*rIK*dIKi;
							struc.force(jj).noalias()-=eta[2]*rJK*dJKi;
						}
						if(kmin){
							struc.force(kk).noalias()-=(-phi*cosIJK*dIKi+eta[1])*rIK*dIKi+phi*dIKi*rIJ*dIJi;
							struc.force(kk).noalias()+=eta[2]*rJK*dJKi;
						}
					}
				}
			}
		}
	}
	
	struc.pe()+=energy;
	return energy;
}

//==== static functions ====

//read/write basis

/**
* Read the basis for a given species from file.
* @param file - the name of the file from which the object will be read
* @param nnp - the neural network potential to be written
* @param atomName - the species for which we will read the basis
*/
void PotNNPEQ::read_basis(const char* file, const char* atomName){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::read(const char*,PotNNPEQ&,const char*):\n";
	FILE* reader=NULL;
	reader=fopen(file,"r");
	if(reader!=NULL){
		PotNNPEQ::read_basis(reader,atomName);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("PotNNPEQ::read(const char*,PotNNPEQ&): Could not open nnp file: \"")+std::string(file)+std::string("\""));
}

/**
* Read the basis for a given species from file.
* @param reader - the file pointer from which the object will be read
* @param nnp - the neural network potential to be written
* @param atomName - the species for which we will read the basis
*/
void PotNNPEQ::read_basis(FILE* reader, const char* atomName){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::read_basis(FILE*,PotNNPEQ&,const char*):\n";
	//==== local function variables ====
	Token token;
	char* input=new char[string::M];
	//==== get atom species ====
	const int atomIndex=index(atomName);
	//==== global cutoff ====
	token.read(fgets(input,string::M,reader),string::WS);
	const double rc=std::atof(token.next().c_str());
	if(rc!=rc_) throw std::invalid_argument("PotNNPEQ::read_basis(FILE*,PotNNPEQ&,const char*): invalid cutoff.");
	//==== number of species ====
	token.read(fgets(input,string::M,reader),string::WS);
	const int nspecies=std::atoi(token.next().c_str());
	if(nspecies!=ntypes_) throw std::invalid_argument("PotNNPEQ::read_basis(FILE*,PotNNPEQ&,const char*): invalid number of species.");
	//==== central species ====
	token.read(fgets(input,string::M,reader),string::WS);
	const int II=index(token.next().c_str());
	//==== check indices ====
	if(atomIndex!=II) throw std::invalid_argument("PotNNPEQ::read_basis(FILE*,PotNNPEQ&,const char*): invalid central species.\n");
	//==== basis - radial ====
	for(int j=0; j<ntypes_; ++j){
		//read species
		token.read(fgets(input,string::M,reader),string::WS);
		const int JJ=index(token.next().c_str());
		//read basis
		BasisR::read(reader,nnh_[II].basisR(JJ));
	}
	//==== basis - angular ====
	for(int j=0; j<ntypes_; ++j){
		for(int k=j; k<ntypes_; ++k){
			//read species
			token.read(fgets(input,string::M,reader),string::WS);
			const int JJ=index(token.next().c_str());
			const int KK=index(token.next().c_str());
			//read basis
			BasisA::read(reader,nnh_[II].basisA(JJ,KK));
		}
	}
	//==== initialize the inputs ====
	nnh_[II].init_input();
	//==== clear local variables ====
	delete[] input;
}

//read/write nnp

/**
* Write the neural network to file
* @param file - the name of the file to which the object will be written
* @param nnp - the neural network potential to be written
*/
void PotNNPEQ::write(const char* file){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::write(const char*,const PotNNPEQ&):\n";
	FILE* writer=NULL;
	writer=fopen(file,"w");
	if(writer!=NULL){
		PotNNPEQ::write(writer);
		fclose(writer);
		writer=NULL;
	} else throw std::runtime_error(std::string("PotNNPEQ::write(const char*,const PotNNPEQ&): Could not write to nnh file: \"")+std::string(file)+std::string("\""));
}

/**
* Read the neural network from file
* @param file - the name of the file fro
* @param nnp - stores the neural network potential to be read
*/
void PotNNPEQ::read(const char* file){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::read(const char*,PotNNPEQ&):\n";
	FILE* reader=NULL;
	reader=fopen(file,"r");
	if(reader!=NULL){
		PotNNPEQ::read(reader);
		fclose(reader);
		reader=NULL;
	} else throw std::runtime_error(std::string("PotNNPEQ::read(const char*,PotNNPEQ&): Could not open nnp file: \"")+std::string(file)+std::string("\""));
}

/**
* Write the neural network to file
* @param writer - the file pointer used to write the object to file
* @param nnp - the neural network potential to be written
*/
void PotNNPEQ::write(FILE* writer){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::write(FILE*,const PotNNPEQ&):\n";
	//==== header ====
	fprintf(writer,"ann\n");
	//==== species ====
	fprintf(writer, "nspecies %i\n",ntypes_);
	for(int n=0; n<ntypes_; ++n){
		const Type& type=nnh_[n].type();
		fprintf(writer,"name %s mass %f energy %f charge %f chi %f\n",
			type.name().c_str(),type.mass().val(),type.energy().val(),type.charge().val(),type.chi().val()
		);
	}
	//==== cutoff ====
	fprintf(writer,"rc %f\n",rc_);
	//==== basis ====
	for(int i=0; i<ntypes_; ++i){
		//write central species
		fprintf(writer,"basis %s\n",nnh_[i].type().name().c_str());
		//write basis - radial
		for(int j=0; j<ntypes_; ++j){
			//write species
			fprintf(writer,"basis_radial %s\n",nnh_[j].type().name().c_str());
			//write basis
			BasisR::write(writer,nnh_[i].basisR(j));
		}
		//write basis - angular
		for(int j=0; j<ntypes_; ++j){
			for(int k=j; k<ntypes_; ++k){
				//write species
				fprintf(writer,"basis_angular %s %s\n",nnh_[j].type().name().c_str(),nnh_[k].type().name().c_str());
				//write basis
				BasisA::write(writer,nnh_[i].basisA(j,k));
			}
		}
	}
	//==== neural network ====
	for(int n=0; n<ntypes_; ++n){
		//write central species
		fprintf(writer,"nn %s\n",nnh_[n].type().name().c_str());
		//write the network
		NN::ANN::write(writer,nnh_[n].nn());
	}
}

/**
* Read the neural network from file
* @param reader - the file pointer used to read the object from file
* @param nnp - stores the neural network potential to be read
*/
void PotNNPEQ::read(FILE* reader){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"PotNNPEQ::read(FILE*,PotNNPEQ&):\n";
	//==== local function variables ====
	Token token;
	char* input=new char[string::M];
	//==== header ====
	fgets(input,string::M,reader);
	//==== number of species ====
	token.read(fgets(input,string::M,reader),string::WS); token.next();
	const int ntypes=std::atoi(token.next().c_str());
	if(ntypes<=0) throw std::invalid_argument("PotNNPEQ::read(FILE*,PotNNPEQ&): invalid number of species.");
	//==== species ====
	std::vector<Type> species(ntypes);
	for(int n=0; n<ntypes; ++n){
		Type::read(fgets(input,string::M,reader),species[n]);
	}
	//==== resize ====
	resize(species.size());
	for(int i=0; i<ntypes_; ++i){
		nnh_[i].type()=species[i];
		map_.add(string::hash(species[i].name()),i);
	}
	//==== global cutoff ====
	token.read(fgets(input,string::M,reader),string::WS); token.next();
	const double rc=std::atof(token.next().c_str());
	if(rc!=rc_) throw std::invalid_argument("PotNNPEQ::coeff(Token&): invalid cutoff.");
	//==== basis ====
	for(int i=0; i<ntypes; ++i){
		//read central species
		token.read(fgets(input,string::M,reader),string::WS); token.next();
		const int II=index(token.next());
		//read basis - radial
		for(int j=0; j<ntypes; ++j){
			//read species
			token.read(fgets(input,string::M,reader),string::WS); token.next();
			const int JJ=index(token.next());
			//read basis
			BasisR::read(reader,nnh_[II].basisR(JJ));
		}
		//read basis - angular
		for(int j=0; j<ntypes; ++j){
			for(int k=j; k<ntypes; ++k){
				//read species
				token.read(fgets(input,string::M,reader),string::WS); token.next();
				const int JJ=index(token.next());
				const int KK=index(token.next());
				//read basis
				BasisA::read(reader,nnh_[II].basisA(JJ,KK));
			}
		}
	}
	//==== initialize inputs ====
	for(int i=0; i<ntypes; ++i){
		nnh_[i].init_input();
	}
	//==== initialize symm ====
	symm_.resize(ntypes);
	for(int i=0; i<ntypes; ++i){
		std::cout<<"ninput["<<i<<"] = "<<nnh_[i].nInput()<<"\n";
		symm_[i].resize(nnh_[i].nInput());
	}
	//==== neural network ====
	for(int n=0; n<ntypes; ++n){
		//read species
		token.read(fgets(input,string::M,reader),string::WS); token.next();
		const int II=index(token.next());
		//read network
		NN::ANN::read(reader,nnh_[II].nn());
		//resize gradient object
		nnh_[II].dOdZ().resize(nnh_[II].nn());
	}
	//==== clear local variables ====
	delete[] input;
}

}

//************************************************************
// serialization
//************************************************************

namespace serialize{
	
//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const ptnl::PotNNPEQ& obj){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"nbytes(const PotNNPEQ&):\n";
	int size=0;
	size+=nbytes(static_cast<const ptnl::Pot&>(obj));
	//species
	size+=nbytes(obj.ntypes());
	size+=nbytes(obj.map());
	for(int i=0; i<obj.ntypes(); ++i){
		size+=nbytes(obj.nnh(i));
	}
	//cutoff
	size+=nbytes(obj.rc());
	//return the size
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const ptnl::PotNNPEQ& obj, char* arr){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"pack(const PotNNPEQ&,char*):\n";
	int pos=0;
	pos+=pack(static_cast<const ptnl::Pot&>(obj),arr+pos);
	//species
	pos+=pack(obj.ntypes(),arr+pos);
	pos+=pack(obj.map(),arr+pos);
	for(int i=0; i<obj.ntypes(); ++i){
		pos+=pack(obj.nnh(i),arr+pos);
	}
	//cutoff
	pos+=pack(obj.rc(),arr+pos);
	//return bytes written
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(ptnl::PotNNPEQ& obj, const char* arr){
	if(POT_NNPEQ_PRINT_FUNC>0) std::cout<<"unpack(ptnl::PotNNPEQ&,const char*):\n";
	int pos=0;
	pos+=unpack(static_cast<ptnl::Pot&>(obj),arr+pos);
	//species
	int ntypes=0;
	Map<int,int> map;
	pos+=unpack(ntypes,arr+pos);
	pos+=unpack(map,arr+pos);
	std::vector<NNH> nnh(ntypes);
	std::vector<Type> species(ntypes);
	for(int i=0; i<ntypes; ++i){
		pos+=unpack(nnh[i],arr+pos);
		species[i]=nnh[i].type();
	}
	obj.resize(species.size());
	for(int i=0; i<obj.ntypes(); ++i){
		obj.nnh(i)=nnh[i];
	}
	obj.map()=map;
	//cutoff
	pos+=unpack(obj.rc(),arr+pos);
	//return bytes read
	return pos;
}

}
