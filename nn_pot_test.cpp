// c libaries
#include <cstdlib>
// c++ libraries
#include <iostream>
#include "nn_pot_serial.hpp"
#include "vasp.hpp"

int main(int argc, char* argv[]){

	bool test_cut=true;
	bool test_phir_g2=true;
	bool test_phia_g4=true;
	bool test_basisR_g2=true;
	bool test_basisA_g4=true;
	bool test_unit=true;
	bool test_symm=true;
	
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
		BasisR::init_G2(basis,nR,CutoffN::COS,rmin,rcut);
		
		//print basis
		std::cout<<"Printing basis...\n";
		std::cout<<basis<<"\n";
		
		//print the basis functions to file
		std::cout<<"Printing basis functions to file...\n";
		FILE* writer=std::fopen("nn_pot_test_basisr_g2.dat","w");
		if(writer!=NULL){
			unsigned int N=500;
			double rmax=rcut*1.2;
			fprintf(writer,"R ");
			for(unsigned int i=0; i<basis.fR.size(); ++i) fprintf(writer,"f%i ",i);
			fprintf(writer,"\n");
			for(unsigned int n=0; n<N; ++n){
				double r=rmax*n/N;
				fprintf(writer, "%f ",r);
				for(unsigned int i=0; i<basis.fR.size(); ++i){
					fprintf(writer,"%f ",(*basis.fR[i])(r));
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
		BasisA::init_G4(basis,nA,CutoffN::COS,rcut);
		
		//print basis
		std::cout<<"Printing basis...\n";
		std::cout<<basis<<"\n";
		
		FILE* writer=fopen("nn_pot_test_basisa_g4.dat","w");
		if(writer!=NULL){
			unsigned int N=500;
			double thetaMax=3.14159;
			fprintf(writer,"THETA ");
			for(unsigned int i=0; i<basis.fA.size(); ++i) fprintf(writer,"f%i ",i);
			fprintf(writer,"\n");
			for(unsigned int n=0; n<N; ++n){
				double theta=thetaMax*n/N;
				fprintf(writer, "%f ",theta);
				for(unsigned int i=0; i<basis.fA.size(); ++i){
					fprintf(writer,"%f ",(*basis.fA[i])(theta,1.0,1.0,1.0));
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
		std::vector<unsigned int> nh(1,20);
		
		//set the nn pot parameters
		std::cout<<"Setting the nn potential parameters...\n";
		nnpot.nR()=5;
		nnpot.nA()=8;
		nnpot.rc()=6.0;
		nnpot.lambda()=0.0;
		nnpot.nh()=nh;
		nnpot.tcut()=CutoffN::COS;
		nnpot.tfType()=NN::TransferN::TANH;
		nnpot.phiRN()=PhiRN::G2;
		nnpot.phiAN()=PhiAN::G4;
		std::vector<std::string> speciesNames=std::vector<std::string>(1,std::string("H"));
		nnpot.setSpecies(speciesNames);
		
		//initialize the nn pot 
		std::cout<<"Initializing the potential...\n";
		nnpot.init();
		
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
	
	if(test_symm){
	std::cout<<"************************************************************\n";
	std::cout<<"*********************** TEST - SYMM ***********************\n";
	try{
		//local variables
		Structure<AtomT> struc;
		NNPot nnpot;
		std::vector<unsigned int> nh(1,20);
		
		//load simulation
		VASP::XML::load("Si.xml",struc);
		std::cout<<struc<<"\n";
		
		//initialize the potential
		//set the nn pot parameters
		std::cout<<"Setting the nn potential parameters...\n";
		nnpot.nR()=6;
		nnpot.nA()=8;
		nnpot.rc()=6.0;
		nnpot.lambda()=0.0;
		nnpot.nh()=nh;
		nnpot.phiRN()=PhiRN::G2;
		nnpot.phiAN()=PhiAN::G4;
		nnpot.tcut()=CutoffN::COS;
		nnpot.tfType()=NN::TransferN::TANH;
		nnpot.initSpecies(struc);
		
		//initialize the nn pot 
		std::cout<<"Initializing the potential...\n";
		nnpot.init();
		
		//initialize the symmetry functions
		std::cout<<"Initializing the symmetry functions...\n";
		nnpot.initSymm(struc);
		
		//print the potential
		std::cout<<nnpot<<"\n";
		
		//calculate the inputs
		std::cout<<"Calculating the inputs...\n";
		nnpot.inputs_symm(struc);
		
		//printing the bases
		std::cout<<"Printing the bases...\n";
		std::cout<<"basisR = "<<nnpot.basisR()[0]<<"\n";
		std::cout<<"basisA = "<<nnpot.basisA()(0,0)<<"\n";
		
		//print the inputs
		std::cout<<"Printing the inputs...\n";
		for(unsigned int i=0; i<struc.nAtoms(); ++i){
			std::cout<<struc.atom(i).name()<<struc.atom(i).index()+1<<": "<<struc.atom(i).symm().transpose()<<"\n";
		}
		
		//execute the network
		std::cout<<"Executing the network...\n";
		std::cout<<"energy = "<<nnpot.energy(struc)<<"\n";
		
		//print the network
		std::cout<<"Printing the nn potential...\n";
		//NNPot::print_chk("nn_pot_test_out.dat",nnpot);
		std::cout<<"nn potential printed.\n";
		
		//load the network
		std::cout<<"Loading the nn potential...\n";
		//NNPot::load_chk("nn_pot_test_out.dat",nnpot);
		std::cout<<"nn potential loaded.\n";
		
		std::cout<<"nnpot = "<<nnpot<<"\n";
		
		//print the network again
		std::cout<<"Printing the nn potential again...\n";
		//NNPot::print_chk("nn_pot_test_new_out.dat",nnpot);
		std::cout<<"nn potential printed.\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - SYMM:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"*********************** TEST - SYMM ***********************\n";
	std::cout<<"************************************************************\n";
	}
	
}