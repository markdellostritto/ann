// c libaries
#include <cstdlib>
// c++ libraries
#include <iostream>
#include <chrono>
// local libaries
#include "nn_pot.hpp"
#include "vasp.hpp"

int main(int argc, char* argv[]){

	//======== cutoff ========
	bool test_cut=true;
	//======== symmetry functions ========
	bool test_phir_g2=true;
	bool test_phia_g4=true;
	//======== basis ========
	bool test_basisR_g2=true;
	bool test_basisA_g4=true;
	//======== unit test ========
	bool test_unit=true;
	//======== serialization ========
	bool test_symm_radial_serialize=true;
	bool test_symm_angular_serialize=true;
	bool test_basis_radial_serialize=true;
	bool test_basis_angular_serialize=true;
	//======== i/o ========
	bool test_io=true;
	//======== forces ========
	bool test_force_radial=false;
	bool test_force_angular=false;
	bool test_force=true;
	//======== symm time ========
	bool test_symm_time=true;
	bool test_force_time=true;
	
	if(test_cut){
	std::cout<<"************************************************************\n";
	std::cout<<"************************ TEST - CUT ************************\n";
	try{
		//local variables
		double rCut=5;
		double rMax=1.5*rCut;
		unsigned int N=500;
		std::vector<double> r(N);
		std::vector<double> cos(N);
		std::vector<double> tanh(N);
		std::vector<double> cosD(N);
		std::vector<double> tanhD(N);
		FILE* writer=NULL;
		
		std::cout<<"Paramters:\n";
		std::cout<<"\tN     = "<<N<<"\n";
		std::cout<<"\tR_CUT = "<<rCut<<"\n";
		std::cout<<"\tR_MAX = "<<rMax<<"\n";
		
		std::cout<<"Calculating cutoff functions...\n";
		for(unsigned int n=0; n<N; ++n){
			r[n]=((double)n)/N*rMax;
			cos[n]=CutoffF::cut_cos(r[n],rCut);
			cosD[n]=CutoffFD::cut_cos(r[n],rCut);
			tanh[n]=CutoffF::cut_tanh(r[n],rCut);
			tanhD[n]=CutoffFD::cut_tanh(r[n],rCut);
		}
		
		std::cout<<"Printing cutoff functions...\n";
		writer=fopen("nn_pot_test_cut.dat","w");
		if(writer!=NULL){
			fprintf(writer,"R COS COSD TANH TANHD\n");
			for(unsigned int n=0; n<N; ++n){
				fprintf(writer,"%f %.10f %.10f %.10f %.10f\n", r[n], cos[n], cosD[n], tanh[n], tanhD[n]);
			}
			fclose(writer);
			writer=NULL;
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - CUT:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************************ TEST - CUT ************************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_phir_g2){
	std::cout<<"************************************************************\n";
	std::cout<<"********************* TEST - PHIR - G2 *********************\n";
	try{
		//local variables
		double rCut=6;//cutoff
		double rMax=1.5*rCut;//max radius
		unsigned int N=500;//number of radial points
		unsigned int nEta=6,nRS=5;//number of parameters to test
		double etaMin=0.5,etaMax=2.5;
		double rsMin=0.0,rsMax=6.0;
		std::vector<double> r(N);//radial points
		std::vector<std::vector<double> > phir_g2_eta(nEta);//g2 functions - different eta
		std::vector<std::vector<double> > phir_g2_rs(nRS);//g2 functions - different rs
		std::vector<double> phir_g2(N);
		std::vector<double> phir_g2d(N);
		FILE* writer=NULL;
		
		//print the parameters
		std::cout<<"PARAMETERS:\n";
		std::cout<<"\tR_CUT = "<<rCut<<"\n";
		std::cout<<"\tR_MAX = "<<rMax<<"\n";
		std::cout<<"\tN_R = "<<N<<"\n";
		std::cout<<"\tETA = ("<<etaMin<<","<<etaMax<<")\n";
		std::cout<<"\tRS = ("<<rsMin<<","<<rsMax<<")\n";
		
		//resize the vectors storing the functions
		std::cout<<"Resizing vectors...\n";
		for(unsigned n=0; n<nEta; ++n) phir_g2_eta[n].resize(N);
		for(unsigned n=0; n<nRS; ++n) phir_g2_rs[n].resize(N);
		
		//set the radial points
		std::cout<<"Setting the radial points...\n";
		for(unsigned int n=0; n<N; ++n) r[n]=((double)n)/N*rMax;
		
		//calculate the radial functions - function of eta
		std::cout<<"Calculating the radial functions as a function of eta...\n";
		for(unsigned int i=0; i<nEta; ++i){
			PhiR_G2 gr=PhiR_G2(CutoffN::COS,rCut,etaMin+(etaMax-etaMin)*((double)i/(nEta-1)),0.0);
			for(unsigned int n=0; n<N; ++n) phir_g2_eta[i][n]=gr(r[n]);
		}
		
		//calculate the radial functions - function of eta
		std::cout<<"Calculating the radial functions as a function of rs...\n";
		for(unsigned int i=0; i<nRS; ++i){
			PhiR_G2 gr=PhiR_G2(CutoffN::COS,rCut,1.0,rsMin+(rsMax-rsMin)*((double)i/(nRS-1)));
			for(unsigned int n=0; n<N; ++n) phir_g2_rs[i][n]=gr(r[n]);
		}
		
		//calculate the radial functions - shape
		std::cout<<"Calculating the radial functions - shape...\n";
		PhiR_G2 gr=PhiR_G2(CutoffN::COS,rCut,1.0,0.0);
		for(unsigned int i=0; i<N; ++i){
			phir_g2[i]=gr.val(r[i]);
			phir_g2d[i]=gr.grad(r[i]);
		}
		
		//print the radial functions - function of eta
		std::cout<<"Printing the radial functions as a function of eta...\n";
		writer=fopen("nn_pot_test_gr_eta.dat","w");
		if(writer!=NULL){
			fprintf(writer,"R ");
			for(unsigned int i=0; i<nEta; ++i) fprintf(writer,"eta%f ",etaMin+(etaMax-etaMin)*((double)i/(nEta-1)));
			fprintf(writer,"\n");
			for(unsigned int n=0; n<N; ++n){
				fprintf(writer,"%f ",r[n]);
				for(unsigned int i=0; i<nEta; ++i){
					fprintf(writer,"%f ",phir_g2_eta[i][n]);
				}
				fprintf(writer,"\n");
			}
			fclose(writer);
			writer=NULL;
		}
		
		//print the radial functions - function of eta
		std::cout<<"Printing the radial functions as a function of rs...\n";
		writer=fopen("nn_pot_test_gr_rs.dat","w");
		if(writer!=NULL){
			fprintf(writer,"R ");
			for(unsigned int i=0; i<nRS; ++i) fprintf(writer,"rs%f ",rsMin+(rsMax-rsMin)*((double)i/(nRS-1)));
			fprintf(writer,"\n");
			for(unsigned int n=0; n<N; ++n){
				fprintf(writer,"%f ",r[n]);
				for(unsigned int i=0; i<nRS; ++i){
					fprintf(writer,"%f ",phir_g2_rs[i][n]);
				}
				fprintf(writer,"\n");
			}
			fclose(writer);
			writer=NULL;
		}
		
		//print the radial functions - shape
		std::cout<<"Printing the radial functions - shape...\n";
		writer=fopen("nn_pot_test_gr.dat","w");
		if(writer!=NULL){
			fprintf(writer,"R G2 G2D\n");
			for(unsigned int n=0; n<N; ++n){
				fprintf(writer,"%f %f %f\n",r[n],phir_g2[n],phir_g2d[n]);
			}
			fclose(writer);
			writer=NULL;
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - GR:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************* TEST - PHIR - G2 *********************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_phia_g4){
	std::cout<<"************************************************************\n";
	std::cout<<"********************* TEST - PHIA - G4 *********************\n";
	try{
		//local variables
		double rCut=5;//cutoff
		static const double pi=3.14159;
		double thetaMin=0.0,thetaMax=pi;
		unsigned int nTheta=100;
		double eta=1.0;
		double zetaMin=5,zetaMax=20;
		unsigned int nZeta=5;
		short lambda=1;
		double theta0Min=0.0,theta0Max=pi;
		unsigned int nTheta0=5;
		std::vector<double> theta(nTheta);
		std::vector<std::vector<double> > phia_g4_zeta(nZeta);//ga functions - different zeta
		FILE* writer=NULL;
		
		//print the parameters
		std::cout<<"PARAMETERS:\n";
		std::cout<<"\tR_CUT = "<<rCut<<"\n";
		std::cout<<"\tTHETA = ("<<thetaMin<<","<<thetaMax<<")\n";
		std::cout<<"\tZETA = ("<<zetaMin<<","<<zetaMax<<")\n";
		std::cout<<"\tTHETA0 = ("<<theta0Min<<","<<theta0Max<<")\n";
		
		//resize the vectors storing the functions
		std::cout<<"Resizing vectors...\n";
		for(unsigned n=0; n<nZeta; ++n) phia_g4_zeta[n].resize(nTheta);
		
		//set the theta points
		std::cout<<"Setting the theta points...\n";
		for(unsigned int n=0; n<nTheta; ++n) theta[n]=thetaMin+((double)n)/nTheta*(thetaMax-thetaMin);
		
		//calculate the angular functions - function of zeta
		std::cout<<"Calculating the angular functions as a function of zeta...\n";
		for(unsigned int i=0; i<nZeta; ++i){
			PhiA_G4 g4=PhiA_G4(CutoffN::COS,rCut,eta,zetaMin+(zetaMax-zetaMin)*((double)i/(nZeta-1)),0.0);
			for(unsigned int n=0; n<nTheta; ++n) phia_g4_zeta[i][n]=g4(theta[n],1.0,1.0,1.0);
		}
		
		//print the angular functions - function of zeta
		std::cout<<"Printing the angular functions as a function of zeta...\n";
		writer=fopen("nn_pot_test_phia_g4_zeta.dat","w");
		if(writer!=NULL){
			fprintf(writer,"THETA ");
			for(unsigned int i=0; i<nZeta; ++i) fprintf(writer,"ZETA%f ",zetaMin+(zetaMax-zetaMin)*((double)i/(nZeta-1)));
			fprintf(writer,"\n");
			for(unsigned int n=0; n<nTheta; ++n){
				fprintf(writer,"%f ",theta[n]);
				for(unsigned int i=0; i<nZeta; ++i){
					fprintf(writer,"%f ",phia_g4_zeta[i][n]);
				}
				fprintf(writer,"\n");
			}
			fclose(writer);
			writer=NULL;
		}
		
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - PHIA - G4:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************* TEST - PHIA - G4 *********************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_phia_g4){
	std::cout<<"************************************************************\n";
	std::cout<<"********************* TEST - PHIA - G4 *********************\n";
	try{
		//local variables
		double rCut=5;//cutoff
		static const double pi=3.14159;
		double thetaMin=0.0,thetaMax=pi;
		unsigned int nTheta=100;
		double eta=1.0;
		double zetaMin=5,zetaMax=20;
		unsigned int nZeta=5;
		short lambda=1;
		double theta0Min=0.0,theta0Max=pi;
		unsigned int nTheta0=5;
		std::vector<double> theta(nTheta);
		std::vector<std::vector<double> > phia_g4_zeta(nZeta);//ga functions - different zeta
		std::vector<std::vector<double> > phia_g4_theta(nTheta0);//ga functions - different theta0
		FILE* writer=NULL;
		
		//print the parameters
		std::cout<<"PARAMETERS:\n";
		std::cout<<"\tR_CUT = "<<rCut<<"\n";
		std::cout<<"\tTHETA = ("<<thetaMin<<","<<thetaMax<<")\n";
		std::cout<<"\tZETA = ("<<zetaMin<<","<<zetaMax<<")\n";
		std::cout<<"\tTHETA0 = ("<<theta0Min<<","<<theta0Max<<")\n";
		
		//resize the vectors storing the functions
		std::cout<<"Resizing vectors...\n";
		for(unsigned n=0; n<nZeta; ++n) phia_g4_zeta[n].resize(nTheta);
		for(unsigned n=0; n<nTheta0; ++n) phia_g4_theta[n].resize(nTheta);
		
		//set the theta points
		std::cout<<"Setting the theta points...\n";
		for(unsigned int n=0; n<nTheta; ++n) theta[n]=thetaMin+((double)n)/nTheta*(thetaMax-thetaMin);
		
		//calculate the angular functions - function of zeta
		std::cout<<"Calculating the angular functions as a function of zeta...\n";
		for(unsigned int i=0; i<nZeta; ++i){
			PhiA_G4 g4=PhiA_G4(CutoffN::COS,rCut,eta,zetaMin+(zetaMax-zetaMin)*((double)i/(nZeta-1)),0.0);
			for(unsigned int n=0; n<nTheta; ++n) phia_g4_zeta[i][n]=g4(theta[n],1.0,1.0,1.0);
		}
		
		//calculate the radial functions - function of eta
		std::cout<<"Calculating the angular functions as a function of theta0...\n";
		for(unsigned int i=0; i<nTheta0; ++i){
			PhiA_G4 g4=PhiA_G4(CutoffN::COS,rCut,eta,20.0,theta0Min+(theta0Max-theta0Min)*((double)i/(nTheta0-1)));
			for(unsigned int n=0; n<nTheta; ++n) phia_g4_theta[i][n]=g4(theta[n],1.0,1.0,1.0);
		}
		
		//print the angular functions - function of zeta
		std::cout<<"Printing the angular functions as a function of zeta...\n";
		writer=fopen("nn_pot_test_phia_g4p_zeta.dat","w");
		if(writer!=NULL){
			fprintf(writer,"THETA ");
			for(unsigned int i=0; i<nZeta; ++i) fprintf(writer,"ZETA%f ",zetaMin+(zetaMax-zetaMin)*((double)i/(nZeta-1)));
			fprintf(writer,"\n");
			for(unsigned int n=0; n<nTheta; ++n){
				fprintf(writer,"%f ",theta[n]);
				for(unsigned int i=0; i<nZeta; ++i){
					fprintf(writer,"%f ",phia_g4_zeta[i][n]);
				}
				fprintf(writer,"\n");
			}
			fclose(writer);
			writer=NULL;
		}
		
		//print the angular functions - function of theta0
		std::cout<<"Printing the angular functions as a function of theta0...\n";
		writer=fopen("nn_pot_test_phia_g4p_theta0.dat","w");
		if(writer!=NULL){
			fprintf(writer,"THETA0 ");
			for(unsigned int i=0; i<nTheta0; ++i) fprintf(writer,"rs%f ",theta0Min+(theta0Max-theta0Min)*((double)i/(nTheta0-1)));
			fprintf(writer,"\n");
			for(unsigned int n=0; n<nTheta; ++n){
				fprintf(writer,"%f ",theta[n]);
				for(unsigned int i=0; i<nTheta0; ++i){
					fprintf(writer,"%f ",phia_g4_theta[i][n]);
				}
				fprintf(writer,"\n");
			}
			fclose(writer);
			writer=NULL;
		}
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - PHIA - G4:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"********************* TEST - PHIA - G4 *********************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_basisR_g2){
	std::cout<<"************************************************************\n";
	std::cout<<"****************** TEST - BASIS - R - G2 ******************\n";
	try{
		//local variables
		BasisR basis;
		unsigned int nR=5;
		double rcut=5.0;
		double rmin=0.5;
		
		//initialize basis
		std::cout<<"Initializing basis...\n";
		basis.init_G2(nR,CutoffN::COS,rmin,rcut);
		
		//print basis
		std::cout<<"Printing basis...\n";
		std::cout<<basis<<"\n";
		
		//print the basis functions to file
		std::cout<<"Printing basis function data to file...\n";
		FILE* writer=std::fopen("nn_pot_test_basisr_g2.dat","w");
		if(writer!=NULL){
			unsigned int N=500;
			double rmax=rcut*1.2;
			fprintf(writer,"R ");
			for(unsigned int i=0; i<basis.nfR(); ++i) fprintf(writer,"f%i ",i);
			fprintf(writer,"\n");
			for(unsigned int n=0; n<N; ++n){
				double r=rmax*n/N;
				fprintf(writer, "%f ",r);
				for(unsigned int i=0; i<basis.nfR(); ++i){
					fprintf(writer,"%f ",basis.fR(i).val(r));
				}
				fprintf(writer,"\n");
			}
			fclose(writer);
			writer=NULL;
		}
		
		//print the basis to file
		std::cout<<"Printing basis to file...\n";
		writer=std::fopen("nn_pot_test_basisr_g2.txt","w");
		if(writer!=NULL){
			BasisR::write(writer,basis);
			fclose(writer);
			writer=NULL;
		}
		
		//load the basis from file
		std::cout<<"Reading basis from file...\n";
		writer=std::fopen("nn_pot_test_basisr_g2.txt","r");
		if(writer!=NULL){
			BasisR::read(writer,basis);
			fclose(writer);
			writer=NULL;
		}
		
		//print basis
		std::cout<<"Printing basis...\n";
		std::cout<<basis<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - BASIS - R - G2:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"****************** TEST - BASIS - R - G2 ******************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_basisA_g4){
	std::cout<<"************************************************************\n";
	std::cout<<"****************** TEST - BASIS - A - G4 ******************\n";
	try{
		//local variables
		BasisA basis;
		unsigned int nA=5;
		double rcut=5.0;
		double rmin=0.5;
		
		//initialize basis
		std::cout<<"Initializing basis...\n";
		basis.init_G4(nA,CutoffN::COS,rcut);
		
		//print basis
		std::cout<<"Printing basis...\n";
		std::cout<<basis<<"\n";
		
		FILE* writer=fopen("nn_pot_test_basisa_g4.dat","w");
		if(writer!=NULL){
			unsigned int N=500;
			double thetaMax=3.14159;
			fprintf(writer,"THETA ");
			for(unsigned int i=0; i<basis.nfA(); ++i) fprintf(writer,"f%i ",i);
			fprintf(writer,"\n");
			for(unsigned int n=0; n<N; ++n){
				double theta=thetaMax*n/N;
				fprintf(writer, "%f ",theta);
				for(unsigned int i=0; i<basis.nfA(); ++i){
					fprintf(writer,"%f ",basis.fA(i).val(std::cos(theta),1.0,1.0,1.0));
				}
				fprintf(writer,"\n");
			}
			fclose(writer);
			writer=NULL;
		}
		
		//print the basis to file
		std::cout<<"Printing basis to file...\n";
		writer=std::fopen("nn_pot_test_basisa_g4.txt","w");
		if(writer!=NULL){
			BasisA::write(writer,basis);
			fclose(writer);
			writer=NULL;
		}
		
		//load the basis from file
		std::cout<<"Reading basis from file...\n";
		writer=std::fopen("nn_pot_test_basisa_g4.txt","r");
		if(writer!=NULL){
			BasisA::read(writer,basis);
			fclose(writer);
			writer=NULL;
		}
		
		//print basis
		std::cout<<"Printing basis...\n";
		std::cout<<basis<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - BASIS - A - G4:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"****************** TEST - BASIS - A - G4 ******************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_unit){
	std::cout<<"************************************************************\n";
	std::cout<<"*********************** TEST - UNIT ***********************\n";
	try{
		//local variables
		NNPot nnpot;
		NNPot::Init nnpotInit;
		std::vector<unsigned int> nh(1,20);
		
		//set the nn pot parameters
		std::cout<<"Setting the nn potential parameters...\n";
		nnpotInit.nR=5;
		nnpotInit.nA=8;
		nnpotInit.rc=6.0;
		nnpotInit.lambda=0.0;
		nnpotInit.nh=nh;
		nnpotInit.tcut=CutoffN::COS;
		nnpotInit.tfType=NN::TransferN::TANH;
		nnpotInit.phiRN=PhiRN::G2;
		nnpotInit.phiAN=PhiAN::G4;
		
		//initialize the species
		std::cout<<"Initializing species...\n";
		std::vector<std::string> speciesNames=std::vector<std::string>(1,std::string("H"));
		nnpot.resize(speciesNames);
		
		//initialize the nn pot 
		std::cout<<"Initializing the potential...\n";
		nnpot.init(nnpotInit);
		
		//print the potential
		std::cout<<nnpot<<"\n";
		
		//copy the potential
		std::cout<<"Copying - constructor.\n";
		NNPot nnpotcopy(nnpot);
		std::cout<<nnpotcopy<<"\n";
		nnpotcopy.clear();
		
		//assign the potential
		std::cout<<"Copying - assignment.\n";
		nnpotcopy=nnpot;
		std::cout<<nnpotcopy<<"\n";
		nnpotcopy.clear();
		
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - UNIT:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"*********************** TEST - UNIT ***********************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_symm_radial_serialize){
	std::cout<<"************************************************************\n";
	std::cout<<"************* TEST - SYMM - RADIAL - SERIALIZE *************\n";
	try{
		PhiR_G1 phirG11(CutoffN::COS,6);
		PhiR_G1 phirG12(CutoffN::TANH,1);
		PhiR_G2 phirG21(CutoffN::COS,6,0.248928,1.5472894);
		PhiR_G2 phirG22(CutoffN::COS,6,0.0,0.0);
		char* arrG1=NULL;
		char* arrG2=NULL;
		unsigned int nBytes=0;
		
		//serialize g1
		std::cout<<"Serializing G1...\n";
		nBytes=serialize::nbytes(phirG11);
		arrG1=new char[nBytes];
		std::cout<<"Packing G1...\n";
		serialize::pack(phirG11,arrG1);
		std::cout<<"Unpacking G1...\n";
		serialize::unpack(phirG12,arrG1);
		std::cout<<"error = "<<(phirG11.rc-phirG12.rc)+(phirG11.tcut-phirG12.tcut)<<"\n";
		
		//serialize g1
		std::cout<<"Serializing G2...\n";
		nBytes=serialize::nbytes(phirG21);
		arrG2=new char[nBytes];
		std::cout<<"Packing G2...\n";
		serialize::pack(phirG21,arrG2);
		std::cout<<"Unpacking G2...\n";
		serialize::unpack(phirG22,arrG2);
		std::cout<<"error = "<<std::fabs(phirG21.rc-phirG22.rc)
			+std::fabs(phirG21.tcut-phirG22.tcut)
			+std::fabs(phirG21.eta-phirG22.eta)
			+std::fabs(phirG21.rs-phirG22.rs)
		<<"\n";
		
		delete[] arrG1;
		delete[] arrG2;
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SYMM - RADIAL - SERIALIZE:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************* TEST - SYMM - RADIAL - SERIALIZE *************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_symm_angular_serialize){
	std::cout<<"************************************************************\n";
	std::cout<<"************ TEST - SYMM - ANGULAR - SERIALIZE ************\n";
	try{
		PhiA_G3 phirG31(CutoffN::COS,6,0.27884,3.589201,1);
		PhiA_G3 phirG32(CutoffN::TANH,1,0,0,0);
		PhiA_G4 phirG41(CutoffN::COS,6,0.248928,1.5472894,1);
		PhiA_G4 phirG42(CutoffN::TANH,1,0,0,0);
		char* arrG3=NULL;
		char* arrG4=NULL;
		unsigned int nBytes=0;
		
		//serialize g1
		std::cout<<"Serializing G3...\n";
		nBytes=serialize::nbytes(phirG31);
		arrG3=new char[nBytes];
		std::cout<<"Packing G3...\n";
		serialize::pack(phirG31,arrG3);
		std::cout<<"Unpacking G3...\n";
		serialize::unpack(phirG32,arrG3);
		std::cout<<"error = "<<(phirG31.rc-phirG32.rc)
			+(phirG31.tcut-phirG32.tcut)
			+(phirG31.eta-phirG32.eta)
			+(phirG31.zeta-phirG32.zeta)
			+(phirG31.lambda-phirG32.lambda)
		<<"\n";
		
		//serialize g1
		std::cout<<"Serializing G4...\n";
		nBytes=serialize::nbytes(phirG41);
		arrG4=new char[nBytes];
		std::cout<<"Packing G4...\n";
		serialize::pack(phirG41,arrG4);
		std::cout<<"Unpacking G4...\n";
		serialize::unpack(phirG42,arrG4);
		std::cout<<"error = "<<std::fabs(phirG41.rc-phirG42.rc)
			+(phirG41.tcut-phirG42.tcut)
			+(phirG41.eta-phirG42.eta)
			+(phirG41.zeta-phirG42.zeta)
			+(phirG41.lambda-phirG42.lambda)
		<<"\n";
		
		delete[] arrG3;
		delete[] arrG4;
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SYMM - ANGULAR - SERIALIZE:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************ TEST - SYMM - ANGULAR - SERIALIZE ************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_basis_radial_serialize){
	std::cout<<"************************************************************\n";
	std::cout<<"************ TEST - BASIS - RADIAL - SERIALIZE ************\n";
	try{
		BasisR basisR1,basisR2;
		basisR1.init_G2(6,CutoffN::COS,0.5,6.0);
		basisR2.init_G2(1,CutoffN::COS,0.5,2.0);
		char* arr=NULL;
		unsigned int nBytes=0;
		std::cout<<"basis = "<<basisR1<<"\n";
		
		//serialize basisR
		std::cout<<"Serializing basisR...\n";
		nBytes=serialize::nbytes(basisR1);
		std::cout<<"nBytes = "<<nBytes<<"\n";
		arr=new char[nBytes];
		std::cout<<"Packing basisR...\n";
		serialize::pack(basisR1,arr);
		std::cout<<"Unpacking basisR...\n";
		serialize::unpack(basisR2,arr);
		
		std::cout<<"basis = "<<basisR2<<"\n";
		
		std::cout<<"Calculating error...\n";
		double error=std::fabs(basisR1.phiRN()-basisR2.phiRN());
		for(unsigned int i=0; i<basisR1.nfR(); ++i){
			const PhiR_G2& phirG21=static_cast<const PhiR_G2&>(basisR1.fR(i));
			const PhiR_G2& phirG22=static_cast<const PhiR_G2&>(basisR2.fR(i));
			error+=std::fabs(phirG21.rc-phirG22.rc);
			error+=std::fabs(phirG21.tcut-phirG22.tcut);
			error+=std::fabs(phirG21.eta-phirG22.eta);
			error+=std::fabs(phirG21.rs-phirG22.rs);
		}
		std::cout<<"error = "<<error<<"\n";
		
		delete[] arr;
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - BASIS - RADIAL - SERIALIZE:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************ TEST - BASIS - RADIAL - SERIALIZE ************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_basis_angular_serialize){
	std::cout<<"************************************************************\n";
	std::cout<<"************ TEST - BASIS - ANGULAR - SERIALIZE ************\n";
	try{
		BasisA basisA1,basisA2;
		basisA1.init_G4(6,CutoffN::COS,6.0);
		basisA2.init_G4(1,CutoffN::COS,2.0);
		char* arr=NULL;
		unsigned int nBytes=0;
		std::cout<<"basis = "<<basisA1<<"\n";
		
		//serialize basisA
		std::cout<<"Serializing basisA...\n";
		nBytes=serialize::nbytes(basisA1);
		std::cout<<"nBytes = "<<nBytes<<"\n";
		arr=new char[nBytes];
		std::cout<<"Packing basisA...\n";
		serialize::pack(basisA1,arr);
		std::cout<<"Unpacking basisA...\n";
		serialize::unpack(basisA2,arr);
		
		std::cout<<"basis = "<<basisA2<<"\n";
		
		std::cout<<"Calculating error...\n";
		double error=std::fabs(basisA1.phiAN()-basisA2.phiAN());
		for(unsigned int i=0; i<basisA1.nfA(); ++i){
			const PhiA_G4& phirG41=static_cast<const PhiA_G4&>(basisA1.fA(i));
			const PhiA_G4& phirG42=static_cast<const PhiA_G4&>(basisA2.fA(i));
			error+=std::fabs(phirG41.rc-phirG42.rc);
			error+=std::fabs(phirG41.tcut-phirG42.tcut);
			error+=std::fabs(phirG41.eta-phirG42.eta);
			error+=std::fabs(phirG41.zeta-phirG42.zeta);
			error+=std::fabs(phirG41.lambda-phirG42.lambda);
		}
		std::cout<<"error = "<<error<<"\n";
		
		delete[] arr;
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - BASIS - ANGULAR - SERIALIZE:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************ TEST - BASIS - ANGULAR - SERIALIZE ************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_io){
	std::cout<<"************************************************************\n";
	std::cout<<"************************ TEST - IO ************************\n";
	try{
		NNPot nnpot,nnpot_copy;
		NNPot::Init nnpotInit;
		nnpotInit.nR=6;
		nnpotInit.nA=6;
		nnpotInit.phiRN=PhiRN::G2;
		nnpotInit.phiAN=PhiAN::G4;
		nnpotInit.rm=0.5;
		nnpotInit.rc=6;
		nnpotInit.tcut=CutoffN::COS;
		nnpotInit.lambda=0.0;
		nnpotInit.tfType=NN::TransferN::TANH;
		nnpotInit.nh.resize(1,15);
		
		nnpot_copy=nnpot;
		
		//initialize potential
		std::cout<<"Initialize potential...\n";
		std::vector<std::string> species(1,std::string("Si"));
		nnpot.resize(species);
		nnpot.init(nnpotInit);
		nnpot.energyAtom(0)=-6.42789;
		nnpot_copy.resize(species);
		
		//print the potential
		std::cout<<"Writing potential...\n";
		nnpot.write();
		
		//read the potential
		std::cout<<"Reading potential...\n";
		nnpot_copy.read();
		
		//print the potential
		std::cout<<"Writing potential...\n";
		nnpot_copy.header()="ann_copy_";
		nnpot_copy.write();
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - IO:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************************ TEST - IO ************************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_force_radial){
	std::cout<<"************************************************************\n";
	std::cout<<"****************** TEST - FORCE - RADIAL ******************\n";
	try{
		// local variables
		Structure<AtomT> struc;
		NNPot nnpot;
		Eigen::Vector3d ftot=Eigen::Vector3d::Zero();
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.read();
		nnpot.initSymm(struc);
		
		//calculate force for each atom
		std::cout<<"Calculating radial force...\n";
		nnpot.forces_radial(struc);
		ftot.setZero();
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).force().transpose()<<"\n";
		}
		std::cout<<"rc = "<<nnpot.rc()<<"\n";
		std::cout<<"ftot = "<<ftot.transpose()<<"\n";
		
		nnpot.clear();
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si_n2.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.read();
		nnpot.initSymm(struc);
		
		//calculate force for each atom
		std::cout<<"Calculating radial force...\n";
		nnpot.forces_radial(struc);
		ftot.setZero();
		std::cout<<struc<<"\n";
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).posn().transpose()<<"\n";
		}
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).force().transpose()<<"\n";
		}
		std::cout<<"rc = "<<nnpot.rc()<<"\n";
		std::cout<<"ftot = "<<ftot.transpose()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - FORCE - RADIAL:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"****************** TEST - FORCE - RADIAL ******************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_force_angular){
	std::cout<<"************************************************************\n";
	std::cout<<"****************** TEST - FORCE - ANGULAR ******************\n";
	try{
		// local variables
		Structure<AtomT> struc;
		NNPot nnpot;
		Eigen::Vector3d ftot=Eigen::Vector3d::Zero();
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.read();
		nnpot.initSymm(struc);
		
		//calculate force for each atom
		std::cout<<"Calculating angular force...\n";
		nnpot.rc()=6;
		nnpot.forces_angular(struc);
		ftot.setZero();
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).force().transpose()<<"\n";
		}
		std::cout<<"ftot = "<<ftot.transpose()<<"\n";
		
		nnpot.clear();
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si_n2.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.read();
		nnpot.initSymm(struc);
		
		//calculate force for each atom
		std::cout<<"Calculating angular force...\n";
		nnpot.forces_angular(struc);
		ftot.setZero();
		std::cout<<struc<<"\n";
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).posn().transpose()<<"\n";
		}
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).force().transpose()<<"\n";
		}
		std::cout<<"ftot = "<<ftot.transpose()<<"\n";
		
		nnpot.clear();
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si_n3.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.read();
		nnpot.initSymm(struc);
		
		//calculate force for each atom
		std::cout<<"Calculating angular force...\n";
		nnpot.forces_angular(struc);
		ftot.setZero();
		std::cout<<struc<<"\n";
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).posn().transpose()<<"\n";
		}
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).force().transpose()<<"\n";
		}
		std::cout<<"ftot = "<<ftot.transpose()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - FORCE - ANGULAR:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"****************** TEST - FORCE - ANGULAR ******************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_force){
	std::cout<<"************************************************************\n";
	std::cout<<"*********************** TEST - FORCE ***********************\n";
	try{
		// local variables
		Structure<AtomT> struc;
		NNPot nnpot;
		Eigen::Vector3d ftot=Eigen::Vector3d::Zero();
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.read();
		nnpot.initSymm(struc);
		
		//calculate force for each atom
		std::cout<<"Calculating force...\n";
		nnpot.rc()=6;
		nnpot.forces(struc);
		ftot.setZero();
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).force().transpose()<<"\n";
		}
		std::cout<<"ftot = "<<ftot.transpose()<<"\n";
		
		nnpot.clear();
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si_n2.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.read();
		nnpot.initSymm(struc);
		
		//calculate force for each atom
		std::cout<<"Calculating force...\n";
		nnpot.forces(struc);
		ftot.setZero();
		std::cout<<struc<<"\n";
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).posn().transpose()<<"\n";
		}
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).force().transpose()<<"\n";
		}
		std::cout<<"ftot = "<<ftot.transpose()<<"\n";
		
		nnpot.clear();
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si_n3.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.read();
		nnpot.initSymm(struc);
		
		//calculate force for each atom
		std::cout<<"Calculating force...\n";
		nnpot.forces(struc);
		ftot.setZero();
		std::cout<<struc<<"\n";
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).posn().transpose()<<"\n";
		}
		for(unsigned int n=0; n<struc.nAtoms(); ++n){
			ftot.noalias()+=struc.atom(n).force();
			std::cout<<struc.atom(n).name()<<" "<<struc.atom(n).force().transpose()<<"\n";
		}
		std::cout<<"ftot = "<<ftot.transpose()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - FORCE:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"*********************** TEST - FORCE ***********************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_symm_time){
	std::cout<<"************************************************************\n";
	std::cout<<"*********************** TEST - SYMM - TIME ***********************\n";
	try{
		// local variables
		Structure<AtomT> struc;
		NNPot nnpot;
		Eigen::Vector3d ftot=Eigen::Vector3d::Zero();
		unsigned int N=10;
		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point stop;
		std::chrono::duration<double> time;
		
		NNPot::Init nnPotInit;
		nnPotInit.nR=6;
		nnPotInit.nA=6;
		nnPotInit.phiRN=PhiRN::G2;
		nnPotInit.phiAN=PhiAN::G4;
		nnPotInit.rm=0.5;
		nnPotInit.rc=6;
		nnPotInit.tcut=CutoffN::COS;
		nnPotInit.lambda=0.0;
		nnPotInit.tfType=NN::TransferN::TANH;
		nnPotInit.nh.resize(1,15);
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.init(nnPotInit);
		nnpot.initSymm(struc);
		
		//calculate symmetry functions for each atom
		std::cout<<"Calculating symmetry functions...\n";
		start=std::chrono::high_resolution_clock::now();
		for(unsigned int i=0; i<N; ++i){
			std::cout<<"iteration "<<i<<"\n";
			nnpot.inputs_symm(struc);
		}
		stop=std::chrono::high_resolution_clock::now();
		time=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		std::cout<<"N "<<N<<" time - execution = "<<time.count()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SYMM - TIME:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"*********************** TEST - SYMM - TIME ***********************\n";
	std::cout<<"************************************************************\n";
	}
	
	if(test_force_time){
	std::cout<<"************************************************************\n";
	std::cout<<"*********************** TEST - FORCE - TIME ***********************\n";
	try{
		// local variables
		Structure<AtomT> struc;
		NNPot nnpot;
		Eigen::Vector3d ftot=Eigen::Vector3d::Zero();
		unsigned int N=10;
		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point stop;
		std::chrono::duration<double> time;
		
		NNPot::Init nnPotInit;
		nnPotInit.nR=6;
		nnPotInit.nA=6;
		nnPotInit.phiRN=PhiRN::G2;
		nnPotInit.phiAN=PhiAN::G4;
		nnPotInit.rm=0.5;
		nnPotInit.rc=6;
		nnPotInit.tcut=CutoffN::COS;
		nnPotInit.lambda=0.0;
		nnPotInit.tfType=NN::TransferN::TANH;
		nnPotInit.nh.resize(1,15);
		
		//load poscar file
		std::cout<<"Loading POSCAR file...\n";
		VASP::POSCAR::load("Si.poscar",struc);
		std::cout<<struc<<"\n";
		
		//initialize the nn potential
		std::cout<<"Initializing the nn potential...\n";
		nnpot.resize(struc);
		nnpot.init(nnPotInit);
		nnpot.initSymm(struc);
		
		//calculate symmetry functions for each atom
		std::cout<<"Calculating symmetry functions...\n";
		start=std::chrono::high_resolution_clock::now();
		for(unsigned int i=0; i<N; ++i){
			std::cout<<"iteration "<<i<<"\n";
			nnpot.forces(struc);
		}
		stop=std::chrono::high_resolution_clock::now();
		time=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		std::cout<<"N "<<N<<" time - execution = "<<time.count()<<"\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - FORCE - TIME:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"*********************** TEST - FORCE - TIME ***********************\n";
	std::cout<<"************************************************************\n";
	}
}