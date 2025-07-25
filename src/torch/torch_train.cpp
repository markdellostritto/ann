// omp
#include "thread/openmp.hpp"
// c++
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <random>
// nnp
#include "nnp/type.hpp"
#include "nnp/nnp.hpp"
#include "ml/nn.hpp"
// torch
#include "torch/engine.hpp"
// chem
#include "chem/units.hpp"
#include "chem/ptable.hpp"
// struc
#include "struc/structure.hpp"
#include "analysis/group.hpp"
// format
#include "format/format.hpp"
#include "format/file_struc.hpp"
// str
#include "str/string.hpp"
#include "str/token.hpp"
#include "str/print.hpp"
// torch
#include "torch/pot_factory.hpp"
#include "torch/dump.hpp"
#include "torch/set_property_factory.hpp"
// util
#include "util/time.hpp"

int main(int argc, char* argv[]){
	//units
		units::System unitsys=units::System::UNKNOWN;
	//files
		std::string fparam;
		std::string fstruc;
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		Token token;
	//struc
		AtomType atomT;
		Structure struc;
		FILE_FORMAT::type format;
		Eigen::Vector3i nlat;
		bool super=false;
		int nstep=0;
	//train
		int nprint=0;
		int tau=0;
		double T=0;
		double dt=0;
		double beta=0;
		double gamma=0;
	//property
		std::vector<std::shared_ptr<property::Base> > sprop;
	//groups
		std::vector<Group> groups;
	//engine
		Engine engine;
		Dump dump;
	//rand
		std::srand(std::time(NULL));
		int seed=std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
	//time
		Clock clock;
	//nnp
		std::string annfile;
		NNP nnp;
		std::vector<NN::DODP> dodp;
	//misc
		bool error=false;
	
	try{
		//==== check the arguments ====
		if(argc!=2) throw std::invalid_argument("Torch::main(int,char**): Invalid number of arguments.");
		
		//==== omp ====
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("OMP",strbuf)<<"\n";
		#pragma omp parallel
		{if(omp_get_thread_num()==0) std::cout<<"num threads = "<<omp_get_num_threads()<<"\n";}
		std::cout<<print::buf(strbuf)<<"\n";
		
		//==== open the parameter file ==== 
		fparam=argv[1];
		FILE* reader=fopen(fparam.c_str(),"r");
		if(reader==NULL) throw std::runtime_error("Torch::main(int,char**): Could not open parameter file.");
		
		//==== read the parameter file ==== 
		std::cout<<"reading general parameters\n";
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);//trim comments
			Token token(input,string::WS); //split line into tokens
			if(token.end()) continue; //skip empty lines
			std::string tag=string::to_upper(token.next());
			if(tag=="UNITS"){
				unitsys=units::System::read(string::to_upper(token.next()).c_str());
				units::consts::init(unitsys);
			} else if(tag=="ATOM_TYPE"){
				atomT=AtomType::read(token);
				if(atomT.posn==false) throw std::invalid_argument("torch::main(int,char**): Atom type missing position.");
				if(atomT.mass==false) throw std::invalid_argument("torch::main(int,char**): Atom type missing mass.");
				if(atomT.index==false) throw std::invalid_argument("torch::main(int,char**): Atom type missing index.");
				if(atomT.type==false) throw std::invalid_argument("torch::main(int,char**): Atom type missing type.");
				if(atomT.symm==false) throw std::invalid_argument("torch::main(int,char**): Atom type missing symm.");
			} else if(tag=="FORMAT"){//simulation format
				format=FILE_FORMAT::read(string::to_upper(token.next()).c_str());
			} else if(tag=="FSTRUC"){
				fstruc=token.next();
				read_struc(fstruc.c_str(),format,atomT,struc);
			} else if(tag=="SUPER"){
				nlat[0]=std::atoi(token.next().c_str());
				nlat[1]=std::atoi(token.next().c_str());
				nlat[2]=std::atoi(token.next().c_str());
				super=true;
			} else if(tag=="GROUP"){
				Group group;
				Group::read(token,group,struc);
				groups.push_back(group);
			} else if(tag=="ENGINE"){
				engine.read(token);
			} else if(tag=="POT"){
				std::shared_ptr<ptnl::Pot> pot;
				ptnl::read(pot,token);
				engine.pots().push_back(pot);
			} else if(tag=="PROPERTY"){
				std::shared_ptr<property::Base> prop;
				property::read(prop,token);
				sprop.push_back(prop);
			} else if(tag=="DUMP"){
				dump.read(token);
			} else if(tag=="NSTEP"){
				nstep=std::atoi(token.next().c_str());
			} else if(tag=="NNP"){
				annfile=token.next();
				NNP::read(annfile.c_str(),nnp);
			} else if(tag=="TEMP"){
				T=std::atof(token.next().c_str());
			} else if(tag=="TAU"){
				tau=std::atoi(token.next().c_str());
			} else if(tag=="DT"){
				dt=std::atof(token.next().c_str());
			} else if(tag=="BETA"){
				beta=std::atof(token.next().c_str());
			} else if(tag=="GAMMA"){
				gamma=std::atof(token.next().c_str());
			} else if(tag=="NPRINT"){
				nprint=std::atoi(token.next().c_str());
			}
		}
		
		//==== print ====
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("PHYSICAL CONSTANTS",strbuf)<<"\n";
		std::cout<<"ke = "<<units::consts::ke()<<"\n";
		std::cout<<"kb = "<<units::consts::kb()<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TORCH",strbuf)<<"\n";
		std::cout<<"UNITS  = "<<unitsys<<"\n";
		std::cout<<"ATOMT  = "<<atomT<<"\n";
		std::cout<<"NSTEP  = "<<nstep<<"\n";
		std::cout<<"FSTRUC = "<<fstruc<<"\n";
		std::cout<<"FORMAT = "<<format<<"\n";
		std::cout<<"DUMP   = "<<dump<<"\n";
		if(super) std::cout<<"NLAT   = "<<nlat.transpose()<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("TRAIN",strbuf)<<"\n";
		std::cout<<"NPRINT = "<<nprint<<"\n";
		std::cout<<"TEMP   = "<<T<<"\n";
		std::cout<<"TAU    = "<<tau<<"\n";
		std::cout<<"DT     = "<<dt<<"\n";
		std::cout<<"BETA   = "<<beta<<"\n";
		std::cout<<"GAMMA  = "<<gamma<<"\n";
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("GROUPS",strbuf)<<"\n";
		for(int i=0; i<groups.size(); ++i){
			std::cout<<groups[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<print::title("PROPERTIES",strbuf)<<"\n";
		for(int i=0; i<sprop.size(); ++i){
			std::cout<<sprop[i]<<"\n";
		}
		std::cout<<print::buf(strbuf)<<"\n";
		std::cout<<nnp<<"\n";
		
		//==== check the parameters ===
		if(beta<=0) throw std::invalid_argument("torch::main(int,char**): Invalid beta (must be greater than zero)."); 
		if(gamma<=0) throw std::invalid_argument("torch::main(int,char**): Invalid gamma (must be greater than zero)."); 
		if(dt<=0) throw std::invalid_argument("torch::main(int,char**): Invalid timestep (must be greater than zero)."); 
		if(T<=0) throw std::invalid_argument("torch::main(int,char**): Invalid temperature (must be greater than zero)."); 
		if(tau<=0) throw std::invalid_argument("torch::main(int,char**): Invalid tau (must be greater than zero)."); 
		if(nprint<=0) throw std::invalid_argument("torch::main(int,char**): Invalid nprint option (must be greater than zero).");
		
		//==== read the structure ====
		std::cout<<"reading the structure\n";
		read_struc(fstruc.c_str(),format,atomT,struc);
		
		//==== set the indices ====
		std::cout<<"setting the indices\n";
		for(int i=0; i<struc.nAtoms(); ++i){
			struc.index(i)=i;
		}
		
		//==== set initial properties ====
		std::cout<<"setting initial properties\n";
		for(int i=0; i<sprop.size(); ++i){
			//set group
			for(int j=0; j<groups.size(); ++j){
				if(sprop[i]->group().label()==groups[j].label()){
					sprop[i]->group()=groups[j];
					break;
				}
			}
			//set property
			sprop[i]->set(struc);
		}
		
		//==== make supercell ====
		if(super){
			std::cout<<"making supercell\n";
			Structure struc_super;
			Structure::super(struc,struc_super,nlat);
			struc=struc_super;
		}
		
		//==== print structure ====
		std::cout<<struc<<"\n";
		
		//==== resize the engine ====
		std::cout<<"resizing the engine\n";
		int ntypes=-1;
		for(int i=0; i<struc.nAtoms(); ++i){
			if(struc.type(i)>ntypes) ntypes=struc.type(i);
		}
		ntypes++;
		std::cout<<"ntypes = "<<ntypes<<"\n";
		engine.resize(ntypes);
		
		//==== read the coefficients ====
		std::cout<<"reading coefficients\n";
		std::rewind(reader);
		while(fgets(input,string::M,reader)!=NULL){
			string::trim_right(input,string::COMMENT);//trim comments
			Token token(input,string::WS); //split line into tokens
			if(token.end()) continue; //skip empty lines
			std::string tag=string::to_upper(token.next());
			if(tag=="COEFF"){
				ptnl::coeff(engine.pots(),token);
			} 
		}
		
		//==== initialize the engine ====
		std::cout<<"initializing the engine\n";
		engine.init();
		std::cout<<engine<<"\n";
		
		//==== close parameter file ==== 
		std::fclose(reader);
		reader=NULL;
		
		//==== compute ==== 
		clock.begin();
		Structure strucCMD=struc;
		Structure strucNNP=struc;
		NNP::init(nnp,strucNNP);
		dodp.resize(nnp.ntypes());
		std::vector<Eigen::VectorXd> grad;
		grad.resize(nnp.ntypes());
		for(int i=0; i<nnp.ntypes(); ++i){
			dodp[i].resize(nnp.nnh(i).nn());
			grad[i]=Eigen::VectorXd::Zero(nnp.nnh(i).nn().size());
		}
		{
			std::cout<<"JOB - MD\n";
			FILE* writer=fopen("out.dump","w");
			if(writer==NULL) throw std::runtime_error("Could not open dump file.");
			printf("N T KE PE PEC PEN TE EXP\n");
			for(int t=0; t<nstep; ++t){
				struc.t()=t;
				if(t%engine.vlist().stride()==0) engine.vlist().build(struc);
				//verlet - first half-step
				for(int n=0; n<struc.nAtoms(); ++n){
					struc.vel(n).noalias()+=0.5*dt*struc.force(n)/struc.mass(n);
					struc.posn(n).noalias()+=struc.vel(n)*dt;
					Cell::returnToCell(struc.posn(n),struc.posn(n),struc.R(),struc.RInv());
				}
				//reset force/energy
				struc.pe()=0; 
				for(int n=0; n<struc.nAtoms(); ++n) struc.force(n).setZero();
				//compute - classical
				strucCMD=struc;
				const double peCMD=engine.compute(strucCMD);
				//compute - nnp
				strucNNP=struc;
				NNP::init(nnp,strucNNP);
				NNP::symm(nnp,strucNNP,engine.vlist());
				NNP::compute(nnp,strucNNP,engine.vlist());
				const double peNNP=strucNNP.pe();
				//combine
				const double deltaE=(peNNP-peCMD);
				const double expf=exp(-beta*beta*deltaE*deltaE);
				struc.pe()=(1.0-expf)*peCMD+expf*peNNP;
				for(int n=0; n<struc.nAtoms(); ++n){
					struc.force(n).noalias()=(1.0-expf)*strucCMD.force(n)+expf*strucNNP.force(n);
				}
				//second half-step
				for(int n=0; n<struc.nAtoms(); ++n){
					struc.vel(n).noalias()+=0.5*dt*struc.force(n)/struc.mass(n);
				}
				//update parameters
				for(int i=0; i<nnp.ntypes(); ++i){
					grad[i].setZero();
				}
				for(int n=0; n<struc.nAtoms(); ++n){
					const int type=nnp.index(strucNNP.name(n));
					nnp.nnh(type).nn().fpbp(strucNNP.symm(n));
					dodp[type].grad(nnp.nnh(type).nn());
					const double fac=deltaE/std::fabs(deltaE)*(1.0-expf)/strucNNP.nAtoms();
					int c=0;
					for(int l=0; l<nnp.nnh(type).nn().nlayer(); ++l){
						for(int i=0; i<nnp.nnh(type).nn().b(l).size(); ++i){
							grad[type][c++]+=fac*dodp[type].dodb()[0][l][i];
						}
					}
					for(int l=0; l<nnp.nnh(type).nn().nlayer(); ++l){
						for(int i=0; i<nnp.nnh(type).nn().w(l).size(); ++i){
							grad[type][c++]+=fac*dodp[type].dodw()[0][l](i);
						}
					}
				}
				for(int j=0; j<nnp.ntypes(); ++j){
					int c=0;
					for(int l=0; l<nnp.nnh(j).nn().nlayer(); ++l){
						for(int i=0; i<nnp.nnh(j).nn().b(l).size(); ++i){
							nnp.nnh(j).nn().b(l)(i)-=gamma*grad[j][c++];
						}
					}
					for(int l=0; l<nnp.nnh(j).nn().nlayer(); ++l){
						for(int i=0; i<nnp.nnh(j).nn().w(l).size(); ++i){
							nnp.nnh(j).nn().w(l)(i)-=gamma*grad[j][c++];
						}
					}
				}
				//write nnp
				if(t%nprint==0){
					const std::string file_ann=annfile+"."+std::to_string(t);
					NNP::write(file_ann.c_str(),nnp);
				}
				//increment
				++struc.t();
				//compute KE, T
				struc.ke()=0;
				for(int n=0; n<struc.nAtoms(); ++n){
					struc.ke()+=struc.mass(n)*struc.vel(n).squaredNorm();
				}
				struc.ke()*=0.5;
				struc.temp()=struc.ke()*(2.0/3.0)/(struc.nAtoms()*units::consts::kb());
				//alter velocities
				if(struc.t()%tau==0){
					const double fac=sqrt(T/(struc.temp()+1e-6));
					for(int i=0; i<struc.nAtoms(); ++i){
						struc.vel(i)*=fac;
					}
				}
				//compute KE, T
				struc.ke()=0;
				for(int n=0; n<struc.nAtoms(); ++n){
					struc.ke()+=struc.mass(n)*struc.vel(n).squaredNorm();
				}
				struc.ke()*=0.5;
				struc.temp()=struc.ke()*(2.0/3.0)/(struc.nAtoms()*units::consts::kb());
				//print
				if(t%dump.nprint()==0) printf("%i %4.5f %4.5f %4.5f %4.5f %4.5f %4.5f %4.5f\n",t,struc.temp(),struc.ke(),struc.pe(),peCMD,peNNP,struc.ke()+struc.pe(),expf);
				//write
				if(t%dump.nwrite()==0) Dump::write(struc,writer);
			}
			fclose(writer); writer=NULL;
		}
		clock.end();
		std::cout<<"time = "<<clock.duration()<<"\n";
		
	}catch(std::exception& e){
		std::cout<<"ERROR in Torch::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	//free memory
	std::cout<<"freeing memory\n";
	delete[] input;
	delete[] strbuf;
	
	if(error) return 1;
	else return 0;
}