// c libraries
#include <cstdio>
// c++ libraries
#include <iostream>
// ann - string
#include "string.hpp"
// ann - input
#include "input.hpp"
// ann - mpi - utility
#include "mpi_util.hpp"
#include "parallel.hpp"
// ann - compiler
#include "compiler.hpp"
// ann - print
#include "print.hpp"
// ann - eigen
#include "eigen.hpp"
// ann - neural network
#include "nn.hpp"
#include "nn_train.hpp"
#include "optimize.hpp"
// ann - time
#include "time.hpp"

//************************************************************
// MPI Communicators
//************************************************************

parallel::Comm WORLD;//all processors

int main(int argc, char* argv[]){
	//==== global variables ====
	//file i/o
		char* input=new char[string::M];
		char* strbuf=new char[print::len_buf];
		std::string file_param;
		std::vector<std::string> file_train;
		std::vector<std::string> file_val;
		std::vector<std::string> file_test;
		std::string file_out="nn_fit_1D.dat";
		FILE* reader=NULL;
		FILE* writer=NULL;
	//function data
		int dim_in=0,dim_out=0;
		int ntrain=0,nval=0,ntest=0;
		std::shared_ptr<std::vector<VecXd> > inTG(new std::vector<VecXd>()),outTG(new std::vector<VecXd>()); //data - train - global
		std::shared_ptr<std::vector<VecXd> > inVG(new std::vector<VecXd>()),outVG(new std::vector<VecXd>()); //data - val   - global
		std::shared_ptr<std::vector<VecXd> > inSG(new std::vector<VecXd>()),outSG(new std::vector<VecXd>()); //data - test  - global
		std::shared_ptr<std::vector<VecXd> > inTL(new std::vector<VecXd>()),outTL(new std::vector<VecXd>()); //data - train - local
		std::shared_ptr<std::vector<VecXd> > inVL(new std::vector<VecXd>()),outVL(new std::vector<VecXd>()); //data - val   - local
		std::shared_ptr<std::vector<VecXd> > inSL(new std::vector<VecXd>()),outSL(new std::vector<VecXd>()); //data - test  - local
	//batch
		int nbatch=0;
	//neural network
		int nIn=0,nOut=0;
		std::vector<int> nh;
		std::shared_ptr<NeuralNet::ANN> nn(new NeuralNet::ANN());
		NNOpt nnopt;
		NeuralNet::ANNInit annInit;
	//mpi
		MPI_Group group_world; //the group associated with the WORLD communicator
		parallel::Dist dist_batch; //data distribution - batch
		parallel::Dist dist_train; //data distribution - training
		parallel::Dist dist_val;   //data distribution - validation
		parallel::Dist dist_test;  //data distribution - testing
	//timing
		Clock clock_wall;
		
	try{
		
		//************************************************************************************
		// INITIALIZATION
		//************************************************************************************
		
		//======== initialize mpi ========
		MPI_Init(&argc,&argv);
		WORLD.label()=MPI_COMM_WORLD;
		MPI_Comm_size(WORLD.label(),&WORLD.size());
		MPI_Comm_rank(WORLD.label(),&WORLD.rank());
		
		//======== start wall clock ========
		if(WORLD.rank()==0) clock_wall.begin();
		
		//======== print title ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::title("NN_FIT",strbuf,' ')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
			std::cout<<print::buf(strbuf,'*')<<"\n";
		}
		
		//======== print compiler information ========
		if(WORLD.rank()==0){
			std::cout<<"date     = "<<compiler::date()<<"\n";
			std::cout<<"time     = "<<compiler::time()<<"\n";
			std::cout<<"compiler = "<<compiler::name()<<"\n";
			std::cout<<"version  = "<<compiler::version()<<"\n";
			std::cout<<"standard = "<<compiler::standard()<<"\n";
			std::cout<<"arch     = "<<compiler::arch()<<"\n";
			std::cout<<"instr    = "<<compiler::instr()<<"\n";
			std::cout<<"os       = "<<compiler::os()<<"\n";
			std::cout<<"omp      = "<<compiler::omp()<<"\n";
		}
		
		//======== print mathematical constants ========
		if(WORLD.rank()==0){
			std::cout<<print::buf(strbuf)<<"\n";
			std::cout<<print::title("MATH CONSTANTS",strbuf)<<"\n";
			std::printf("PI    = %.15f\n",math::constant::PI);
			std::printf("RadPI = %.15f\n",math::constant::RadPI);
			std::printf("Rad2  = %.15f\n",math::constant::Rad2);
			std::cout<<print::title("MATH CONSTANTS",strbuf)<<"\n";
			std::cout<<print::buf(strbuf)<<"\n";
		}
		
		//======== set mpi data ========
		{
			int* ranks=new int[WORLD.size()];
			MPI_Gather(&WORLD.rank(),1,MPI_INT,ranks,1,MPI_INT,0,WORLD.label());
			if(WORLD.rank()==0){
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<print::title("MPI",strbuf)<<"\n";
				std::cout<<"world - size = "<<WORLD.size()<<"\n"<<std::flush;
				std::cout<<print::title("MPI",strbuf)<<"\n";
				std::cout<<print::buf(strbuf)<<"\n";
				std::cout<<std::flush;
			}
			delete[] ranks;
		}
		
		//************************************************************************************
		// PARAMETER DETERMINATION
		//************************************************************************************
		
		//==== read parameters ====
		if(WORLD.rank()==0){
			
			//======== check the arguments ========
			if(argc!=2) throw std::invalid_argument("Invalid number of arguments.");
			
			//======== load the parameter file ========
			std::cout<<"reading parameter file\n";
			file_param=argv[1];
			
			//==== open parameter file ====
			reader=fopen(file_param.c_str(),"r");
			if(reader==NULL) throw std::runtime_error("Unable to open parameter file.");
			
			//==== read parameters ====
			std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL)
			{
				std::vector<std::string> strlist;
				string::trim_right(input,string::COMMENT);
				string::split(input,string::WS,strlist);
				if(strlist.size()==0) continue;
				string::to_upper(strlist.at(0));
				//optimization
				if(strlist.at(0)=="PRE_COND"){
					nnopt.preCond()=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="POST_COND"){
					nnopt.postCond()=string::boolean(strlist.at(1).c_str());
				} else if(strlist.at(0)=="NBATCH"){
					nbatch=std::atoi(strlist.at(1).c_str());
				} else if(strlist.at(0)=="READ_RESTART"){
					nnopt.restart()=true;
					nnopt.file_restart()=strlist.at(1);
				} else if(strlist.at(0)=="FILE_ERROR"){
					nnopt.file_error()=strlist.at(1);
				}
				//data
				if(strlist.at(0)=="DATA_TRAIN"){//data - training
					file_train.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DATA_VAL"){//data - validation
					file_val.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DATA_TEST"){//data - testing
					file_test.push_back(strlist.at(1));
				} else if(strlist.at(0)=="DIM_IN"){
					dim_in=std::atoi(strlist.at(1).c_str());
				} else if(strlist.at(0)=="DIM_OUT"){
					dim_out=std::atoi(strlist.at(1).c_str());
				}
				//neural network
				if(strlist.at(0)=="N_HIDDEN"){
					int nl=strlist.size()-1;
					if(nl<=0) throw std::invalid_argument("Invalid hidden layer configuration.");
					nh.resize(nl);
					for(int i=0; i<nl; ++i){
						nh.at(i)=std::atoi(strlist.at(i+1).c_str());
						if(nh.at(i)==0) throw std::invalid_argument("Invalid hidden layer configuration.");
					}
				} else if(strlist.at(0)=="SIGMA"){//initialization deviation
					annInit.sigma()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="INIT"){//initialization
					annInit.initType()=NeuralNet::InitN::read(string::to_upper(strlist.at(1)).c_str());
				} else if(strlist.at(0)=="SEED"){//initialization
					annInit.seed()=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="TRANSFER"){//transfer function
					nn->tfType()=NeuralNet::TransferN::read(string::to_upper(strlist.at(1)).c_str());
				}
			}
			
			//==== read optimization data ====
			Opt::read(nnopt.data(),reader);
			
			//==== read optimization object ====
			switch(nnopt.data().algo()){
				case Opt::ALGO::SGD:
					nnopt.model().reset(new Opt::SGD());
					read(static_cast<Opt::SGD&>(*nnopt.model()),reader);
				break;
				case Opt::ALGO::SDM:
					nnopt.model().reset(new Opt::SDM());
					read(static_cast<Opt::SDM&>(*nnopt.model()),reader);
				break;
				case Opt::ALGO::NAG:
					nnopt.model().reset(new Opt::NAG());
					read(static_cast<Opt::NAG&>(*nnopt.model()),reader);
				break;
				case Opt::ALGO::ADAGRAD:
					nnopt.model().reset(new Opt::ADAGRAD());
					read(static_cast<Opt::ADAGRAD&>(*nnopt.model()),reader);
				break;
				case Opt::ALGO::ADADELTA:
					nnopt.model().reset(new Opt::ADADELTA());
					read(static_cast<Opt::ADADELTA&>(*nnopt.model()),reader);
				break;
				case Opt::ALGO::RMSPROP:
					nnopt.model().reset(new Opt::RMSPROP());
					read(static_cast<Opt::RMSPROP&>(*nnopt.model()),reader);
				break;
				case Opt::ALGO::ADAM:
					nnopt.model().reset(new Opt::ADAM());
					read(static_cast<Opt::ADAM&>(*nnopt.model()),reader);
				break;
				case Opt::ALGO::NADAM:
					nnopt.model().reset(new Opt::NADAM());
					read(static_cast<Opt::NADAM&>(*nnopt.model()),reader);
				break;
				case Opt::ALGO::BFGS:
					nnopt.model().reset(new Opt::BFGS());
					read(static_cast<Opt::BFGS&>(*nnopt.model()),reader);
				break;
				case Opt::ALGO::RPROP:
					nnopt.model().reset(new Opt::RPROP());
					read(static_cast<Opt::RPROP&>(*nnopt.model()),reader);
				break;
			}
		
			//==== close parameter file ====
			fclose(reader);
			reader=NULL;
			
			//==== check the parameters ====
			if(dim_in<=0) throw std::invalid_argument("Invalid input dimension");
			if(dim_out<=0) throw std::invalid_argument("Invalid output dimension");
			
			//==== initialize neural network ====
			if(!nnopt.restart()){
				std::cout<<"initializing neural network\n";
				nn->resize(annInit,dim_in,nh,dim_out);
			} else {
				std::cout<<"reading restart file\n";
				//nnopt.read_restart(nnopt.file_restart.c_str());
			}
			
			//==== print data ====
			std::cout<<nnopt<<"\n";
			std::cout<<*nn<<"\n";
			std::cout<<nnopt.data()<<"\n";
			switch(nnopt.data().algo()){
				case Opt::ALGO::SGD: std::cout<<static_cast<const Opt::SGD&>(*nnopt.model())<<"\n"; break;
				case Opt::ALGO::SDM: std::cout<<static_cast<const Opt::SDM&>(*nnopt.model())<<"\n"; break;
				case Opt::ALGO::NAG: std::cout<<static_cast<const Opt::NAG&>(*nnopt.model())<<"\n"; break;
				case Opt::ALGO::ADAGRAD: std::cout<<static_cast<const Opt::ADAGRAD&>(*nnopt.model())<<"\n"; break;
				case Opt::ALGO::ADADELTA: std::cout<<static_cast<const Opt::ADADELTA&>(*nnopt.model())<<"\n"; break;
				case Opt::ALGO::RMSPROP: std::cout<<static_cast<const Opt::RMSPROP&>(*nnopt.model())<<"\n"; break;
				case Opt::ALGO::ADAM: std::cout<<static_cast<const Opt::ADAM&>(*nnopt.model())<<"\n"; break;
				case Opt::ALGO::NADAM: std::cout<<static_cast<const Opt::NADAM&>(*nnopt.model())<<"\n"; break;
				case Opt::ALGO::BFGS: std::cout<<static_cast<const Opt::BFGS&>(*nnopt.model())<<"\n"; break;
				case Opt::ALGO::RPROP: std::cout<<static_cast<const Opt::RPROP&>(*nnopt.model())<<"\n"; break;
			}
			
			//==== read training data ====
			std::cout<<"reading training data\n";
			for(int i=0; i<file_train.size(); ++i){
				std::cout<<"\t\""<<file_train[i]<<"\"\n";
				reader=fopen(file_train[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Unable to open file - data - training.");
				Eigen::VectorXd x(dim_in),y(dim_out);
				fgets(input,string::M,reader);//skip first line
				while(fgets(input,string::M,reader)){
					std::vector<std::string> strlist;
					string::split(input,string::WS,strlist);
					if(strlist.size()!=dim_in+dim_out) throw std::runtime_error("Invalid data - incorrect dimension.\n");
					for(int i=0; i<dim_in; ++i) x[i]=std::atof(strlist[i].c_str());
					for(int i=0; i<dim_out; ++i) y[i]=std::atof(strlist[i+dim_in].c_str());
					inTG->push_back(x);
					outTG->push_back(y);
				}
				fclose(reader);
				reader=NULL;
			}
		
			//==== read validation data ====
			std::cout<<"reading validation data\n";
			for(int i=0; i<file_val.size(); ++i){
				std::cout<<"\t\""<<file_val[i]<<"\"\n";
				reader=fopen(file_val[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Unable to open file - data - validation.");
				Eigen::VectorXd x(dim_in),y(dim_out);
				fgets(input,string::M,reader);//skip first line
				while(fgets(input,string::M,reader)){
					std::vector<std::string> strlist;
					string::split(input,string::WS,strlist);
					if(strlist.size()!=dim_in+dim_out) throw std::runtime_error("Invalid data - incorrect dimension.\n");
					for(int i=0; i<dim_in; ++i) x[i]=std::atof(strlist[i].c_str());
					for(int i=0; i<dim_out; ++i) y[i]=std::atof(strlist[i+dim_in].c_str());
					inVG->push_back(x);
					outVG->push_back(y);
				}
				fclose(reader);
				reader=NULL;
			}
		
			//==== read testing data ====
			std::cout<<"reading testing data\n";
			for(int i=0; i<file_test.size(); ++i){
				std::cout<<"\t\""<<file_test[i]<<"\"\n";
				reader=fopen(file_test[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Unable to open file - data - testing.");
				Eigen::VectorXd x(dim_in),y(dim_out);
				fgets(input,string::M,reader);//skip first line
				while(fgets(input,string::M,reader)){
					std::vector<std::string> strlist;
					string::split(input,string::WS,strlist);
					if(strlist.size()!=dim_in+dim_out) throw std::runtime_error("Invalid data - incorrect dimension.\n");
					for(int i=0; i<dim_in; ++i) x[i]=std::atof(strlist[i].c_str());
					for(int i=0; i<dim_out; ++i) y[i]=std::atof(strlist[i+dim_in].c_str());
					inSG->push_back(x);
					outSG->push_back(y);
				}
				fclose(reader);
				reader=NULL;
			}
			
			ntrain=inTG->size();
			nval=inVG->size();
			ntest=inSG->size();
		}
		
		//==== gen thread dist + offset ====
		MPI_Bcast(&nbatch,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&ntrain,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&nval  ,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&ntest ,1,MPI_INT,0,MPI_COMM_WORLD);
		dist_batch.init(WORLD.size(),WORLD.rank(),nbatch);
		dist_train.init(WORLD.size(),WORLD.rank(),ntrain);
		dist_val.init(WORLD.size(),WORLD.rank(),nval);
		dist_test.init(WORLD.size(),WORLD.rank(),ntest);
		//print
		if(WORLD.rank()==0){
			//thread dist
			int* thread_dist_batch=new int[WORLD.size()];
			int* thread_dist_train=new int[WORLD.size()];
			int* thread_dist_val  =new int[WORLD.size()];
			int* thread_dist_test =new int[WORLD.size()];
			MPI_Gather(&dist_batch.size(),1,MPI_INT,thread_dist_batch,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_train.size(),1,MPI_INT,thread_dist_train,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_val.size(),1,MPI_INT,thread_dist_val,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_test.size(),1,MPI_INT,thread_dist_test,1,MPI_INT,0,WORLD.label());
			//thread offset
			int* thread_offset_train=new int[WORLD.size()];
			int* thread_offset_val  =new int[WORLD.size()];
			int* thread_offset_test =new int[WORLD.size()];
			MPI_Gather(&dist_train.offset(),1,MPI_INT,thread_offset_train,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_val.offset(),1,MPI_INT,thread_offset_val,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_test.offset(),1,MPI_INT,thread_offset_test,1,MPI_INT,0,WORLD.label());
			//print
			std::cout<<"n-batch             = "<<nbatch<<"\n";
			std::cout<<"n-train             = "<<ntrain<<"\n";
			std::cout<<"n-val               = "<<nval<<"\n";
			std::cout<<"n-test              = "<<ntest<<"\n";
			std::cout<<"thread_dist_batch   = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_dist_batch[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_train   = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_dist_train[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_val     = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_dist_val[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_test    = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_dist_test[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_train = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_offset_train[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_val   = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_offset_val[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_test  = "; for(int i=0; i<WORLD.size(); ++i) std::cout<<thread_offset_test[i]<<" "; std::cout<<"\n";
			//free
			delete[] thread_dist_batch;
			delete[] thread_dist_train;
			delete[] thread_dist_val;
			delete[] thread_dist_test;
			delete[] thread_offset_train;
			delete[] thread_offset_val;
			delete[] thread_offset_test;
		}
		
		//==== b-cast data ====
		if(WORLD.rank()==0) std::cout<<"broadcasting data\n";
		mpi_util::bcast(WORLD.label(),nnopt);
		mpi_util::bcast(WORLD.label(),*nn);
		if(ntrain>0){
			int* thread_dist=new int[WORLD.size()];
			int* thread_offset=new int[WORLD.size()];
			MPI_Gather(&dist_train.size(),1,MPI_INT,thread_dist,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_train.offset(),1,MPI_INT,thread_offset,1,MPI_INT,0,WORLD.label());
			inTL->resize(thread_dist[WORLD.rank()]);
			outTL->resize(thread_dist[WORLD.rank()]);
			mpi_util::scatterv(*inTG,*inTL,thread_dist,thread_offset);
			mpi_util::scatterv(*outTG,*outTL,thread_dist,thread_offset);
			delete[] thread_dist;
			delete[] thread_offset;
		}
		if(nval>0){
			int* thread_dist=new int[WORLD.size()];
			int* thread_offset=new int[WORLD.size()];
			MPI_Gather(&dist_val.size(),1,MPI_INT,thread_dist,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_val.offset(),1,MPI_INT,thread_offset,1,MPI_INT,0,WORLD.label());
			inVL->resize(thread_dist[WORLD.rank()]);
			outVL->resize(thread_dist[WORLD.rank()]);
			mpi_util::scatterv(*inVG,*inVL,thread_dist,thread_offset);
			mpi_util::scatterv(*outVG,*outVL,thread_dist,thread_offset);
			delete[] thread_dist;
			delete[] thread_offset;
		}
		if(ntest>0){
			int* thread_dist=new int[WORLD.size()];
			int* thread_offset=new int[WORLD.size()];
			MPI_Gather(&dist_test.size(),1,MPI_INT,thread_dist,1,MPI_INT,0,WORLD.label());
			MPI_Gather(&dist_test.offset(),1,MPI_INT,thread_offset,1,MPI_INT,0,WORLD.label());
			inSL->resize(thread_dist[WORLD.rank()]);
			outSL->resize(thread_dist[WORLD.rank()]);
			mpi_util::scatterv(*inSG,*inSL,thread_dist,thread_offset);
			mpi_util::scatterv(*outSG,*outSL,thread_dist,thread_offset);
			delete[] thread_dist;
			delete[] thread_offset;
		}
		
		//==== set the data ====
		if(WORLD.rank()==0) std::cout<<"setting data\n";
		nnopt.inT()=inTL;
		nnopt.outT()=outTL;
		nnopt.inV()=inVL;
		nnopt.outV()=outVL;
		
		//==== execute optimization ====
		if(WORLD.rank()==0) std::cout<<"executing optimization\n";
		nnopt.train(nn,dist_batch.size());
		
		//==== write the data ====
		if(WORLD.rank()==0){
			std::cout<<"writing data\n";
			//test
			writer=fopen("nn_fit_train.dat","w");
			if(writer!=NULL){
				//print header
				fprintf(writer,"#");
				for(int i=0; i<dim_in; ++i) fprintf(writer,"X%i ",i);
				for(int i=0; i<dim_out; ++i) fprintf(writer,"Y%i ",i);
				fprintf(writer,"\n");
				for(int i=0; i<inTG->size(); ++i){
					const Eigen::VectorXd& in=(*inTG)[i];
					const Eigen::VectorXd& out=nn->execute((*inTG)[i]);
					for(int j=0; j<dim_in; ++j) fprintf(writer,"%f ",in[j]);
					for(int j=0; j<dim_out; ++j) fprintf(writer,"%f ",out[j]);
					fprintf(writer,"\n");
				}
				fclose(writer);
				writer=NULL;
			} else std::cout<<"could not open data file\n";
			//test
			writer=fopen("nn_fit_val.dat","w");
			if(writer!=NULL){
				//print header
				fprintf(writer,"#");
				for(int i=0; i<dim_in; ++i) fprintf(writer,"X%i ",i);
				for(int i=0; i<dim_out; ++i) fprintf(writer,"Y%i ",i);
				fprintf(writer,"\n");
				for(int i=0; i<inVG->size(); ++i){
					const Eigen::VectorXd& in=(*inVG)[i];
					const Eigen::VectorXd& out=nn->execute((*inVG)[i]);
					for(int j=0; j<dim_in; ++j) fprintf(writer,"%f ",in[j]);
					for(int j=0; j<dim_out; ++j) fprintf(writer,"%f ",out[j]);
					fprintf(writer,"\n");
				}
				fclose(writer);
				writer=NULL;
			} else std::cout<<"could not open data file\n";
			//test
			writer=fopen("nn_fit_test.dat","w");
			if(writer!=NULL){
				//print header
				fprintf(writer,"#");
				for(int i=0; i<dim_in; ++i) fprintf(writer,"X%i ",i);
				for(int i=0; i<dim_out; ++i) fprintf(writer,"Y%i ",i);
				fprintf(writer,"\n");
				for(int i=0; i<inSG->size(); ++i){
					const Eigen::VectorXd& in=(*inSG)[i];
					const Eigen::VectorXd& out=nn->execute((*inSG)[i]);
					for(int j=0; j<dim_in; ++j) fprintf(writer,"%f ",in[j]);
					for(int j=0; j<dim_out; ++j) fprintf(writer,"%f ",out[j]);
					fprintf(writer,"\n");
				}
				fclose(writer);
				writer=NULL;
			} else std::cout<<"could not open data file\n";
		}
		
		//==== stop wall clock ====
		if(WORLD.rank()==0) clock_wall.end();
		if(WORLD.rank()==0) std::cout<<"duration = "<<clock_wall.duration()<<"\n";
		
	} catch(std::exception& e){
		std::cout<<"ERROR in nn_fit::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	//==== free memory ====
	delete[] input;
	delete[] strbuf;
}