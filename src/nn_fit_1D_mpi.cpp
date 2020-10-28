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

int main(int argc, char* argv[]){
	//==== global variables ====
	//file i/o
		char* input=new char[string::M];
		std::string file_param;
		std::vector<std::string> file_train;
		std::vector<std::string> file_val;
		std::vector<std::string> file_test;
		std::string file_out="nn_fit_1D.dat";
		FILE* reader=NULL;
		FILE* writer=NULL;
	//function data
		unsigned int nTrain=0,nVal=0,nTest=0;
		std::shared_ptr<VecList> inTG(new VecList()),outTG(new VecList()); //data - train - global
		std::shared_ptr<VecList> inVG(new VecList()),outVG(new VecList()); //data - val   - global
		std::shared_ptr<VecList> inSG(new VecList()),outSG(new VecList()); //data - test  - global
		std::shared_ptr<VecList> inTL(new VecList()),outTL(new VecList()); //data - train - local
		std::shared_ptr<VecList> inVL(new VecList()),outVL(new VecList()); //data - val   - local
		std::shared_ptr<VecList> inSL(new VecList()),outSL(new VecList()); //data - test  - local
	//batch
		unsigned int nBatch=0;
		double pbatch=1;
	//thread dist
		int* thread_dist_batch=NULL;   //subset - number of structures  - global - batch
		int* thread_dist_train=NULL;   //dist - data - training
		int* thread_dist_val=NULL;     //dist - data - validation
		int* thread_dist_test=NULL;    //dist - data - testing
		int* thread_offset_train=NULL; //offset - data - validation
		int* thread_offset_val=NULL;   //offset - data - training
		int* thread_offset_test=NULL;  //offset - data - testing
	//neural network
		double vfrac=0.25;
		std::vector<unsigned int> nh;
		std::shared_ptr<NN::Network> nn(new NN::Network());
		NN::NNOpt nnopt;
	//mpi
		int rank=0;
		int nprocs=1;
		
	try{
		std::cout<<"nn_fit_1D\n";
		
		//======== initialize mpi ========
		if(rank==0) std::cout<<"initializing mpi\n";
		MPI_Init(&argc,&argv);
		
		MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
		MPI_Comm_rank(MPI_COMM_WORLD,&rank);
		
		//======== print compiler information ========
		if(rank==0){
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
		
		//==== read command line arguments ====
		if(rank==0){
			Input inputs(argc,argv);
			if(inputs.find("-in")) file_param=inputs.get("-in");
			if(inputs.find("-out")) file_out=inputs.get("-out");
		}
		
		//==== read parameters ====
		if(rank==0){
			
			//==== open parameter file ====
			reader=fopen(file_param.c_str(),"r");
			if(reader==NULL) throw std::runtime_error("Unable to open parameter file.");
			
			//==== read parameters ====
			std::cout<<"reading parameters\n";
			while(fgets(input,string::M,reader)!=NULL){
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
				} else if(strlist.at(0)=="PBATCH"){
					pbatch=std::atof(strlist.at(1).c_str());
				} else if(strlist.at(0)=="NBATCH"){
					nBatch=std::atoi(strlist.at(1).c_str());
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
				} 
				//neural network
				if(strlist.at(0)=="N_HIDDEN"){
					int nl=strlist.size()-1;
					if(nl<=0) throw std::invalid_argument("Invalid hidden layer configuration.");
					nh.resize(nl);
					for(unsigned int i=0; i<nl; ++i){
						nh.at(i)=std::atoi(strlist.at(i+1).c_str());
						if(nh.at(i)==0) throw std::invalid_argument("Invalid hidden layer configuration.");
					}
				} else if(strlist.at(0)=="TRANSFER"){//transfer function
					nn->tfType()=NN::TransferN::read(string::to_upper(strlist.at(1)).c_str());
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
				
			//==== initialize neural network ====
			if(!nnopt.restart()){
				std::cout<<"initializing neural network\n";
				nn->resize(1,nh,1);
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
			for(unsigned int i=0; i<file_train.size(); ++i){
				std::cout<<file_train[i]<<"\n";
				reader=fopen(file_train[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Unable to open file - data - training.");
				Eigen::VectorXd x(1),y(1);
				fgets(input,string::M,reader);//skip first line
				while(fgets(input,string::M,reader)){
					std::sscanf(input,"%lf %lf\n",&x[0],&y[0]);
					inTG->push_back(x);
					outTG->push_back(y);
				}
				fclose(reader);
				reader=NULL;
			}
		
			//==== read validation data ====
			std::cout<<"reading validation data\n";
			for(unsigned int i=0; i<file_val.size(); ++i){
				reader=fopen(file_val[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Unable to open file - data - validation.");
				Eigen::VectorXd x(1),y(1);
				fgets(input,string::M,reader);//skip first line
				while(fgets(input,string::M,reader)){
					std::sscanf(input,"%lf %lf\n",&x[0],&y[0]);
					inVG->push_back(x);
					outVG->push_back(y);
				}
				fclose(reader);
				reader=NULL;
			}
		
			//==== read testing data ====
			std::cout<<"reading testing data\n";
			for(unsigned int i=0; i<file_test.size(); ++i){
				reader=fopen(file_test[i].c_str(),"r");
				if(reader==NULL) throw std::runtime_error("Unable to open file - data - testing.");
				Eigen::VectorXd x(1),y(1);
				fgets(input,string::M,reader);//skip first line
				while(fgets(input,string::M,reader)){
					std::sscanf(input,"%lf %lf\n",&x[0],&y[0]);
					inSG->push_back(x);
					outSG->push_back(y);
				}
				fclose(reader);
				reader=NULL;
			}
		
			//==== set the batch size ====
			std::cout<<"setting batch size\n";
			if(nBatch==0) nBatch=pbatch*inTG->size();
			
			//==== check parameters ====
			std::cout<<"checking parameters\n";
			if(pbatch<=0 || pbatch>1) throw std::invalid_argument("Invalid batch size.");
			if(nBatch<=0 || nBatch>inTG->size()) throw std::invalid_argument("Invalid batch size.");
			
			nTrain=inTG->size();
			nVal=inVG->size();
			nTest=inSG->size();
		}
		
		//==== gen thread dist + offset ====
		MPI_Bcast(&nBatch,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&nTrain,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&nVal  ,1,MPI_INT,0,MPI_COMM_WORLD);
		MPI_Bcast(&nTest ,1,MPI_INT,0,MPI_COMM_WORLD);
		//thread dist
		thread_dist_batch=new int[nprocs];
		thread_dist_train=new int[nprocs];
		thread_dist_val  =new int[nprocs];
		thread_dist_test =new int[nprocs];
		parallel::gen_thread_dist(nprocs,nBatch,thread_dist_batch);
		parallel::gen_thread_dist(nprocs,nTrain,thread_dist_train);
		parallel::gen_thread_dist(nprocs,nVal  ,thread_dist_val);
		parallel::gen_thread_dist(nprocs,nTest ,thread_dist_test);
		//thread offset
		thread_offset_train=new int[nprocs];
		thread_offset_val  =new int[nprocs];
		thread_offset_test =new int[nprocs];
		parallel::gen_thread_offset(nprocs,nTrain,thread_offset_train);
		parallel::gen_thread_offset(nprocs,nVal  ,thread_offset_val);
		parallel::gen_thread_offset(nprocs,nTest ,thread_offset_test);
		//print
		if(rank==0){
			std::cout<<"n-batch             = "<<nBatch<<"\n";
			std::cout<<"n-train             = "<<nTrain<<"\n";
			std::cout<<"n-val               = "<<nVal<<"\n";
			std::cout<<"n-test              = "<<nTest<<"\n";
			std::cout<<"thread_dist_batch   = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_dist_batch[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_train   = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_dist_train[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_val     = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_dist_val[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_dist_test    = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_dist_test[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_train = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_offset_train[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_val   = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_offset_val[i]<<" "; std::cout<<"\n";
			std::cout<<"thread_offset_test  = "; for(int i=0; i<nprocs; ++i) std::cout<<thread_offset_test[i]<<" "; std::cout<<"\n";
		}
		
		//==== b-cast data ====
		if(rank==0) std::cout<<"broadcasting parameters\n";
		mpi_util::bcast(nnopt);
		mpi_util::bcast(*nn);
		if(nTrain>0){
			inTL->resize(thread_dist_train[rank]);
			outTL->resize(thread_dist_train[rank]);
			mpi_util::scatterv(*inTG,*inTL,thread_dist_train,thread_offset_train);
			mpi_util::scatterv(*outTG,*outTL,thread_dist_train,thread_offset_train);
		}
		if(nVal>0){
			inVL->resize(thread_dist_val[rank]);
			outVL->resize(thread_dist_val[rank]);
			mpi_util::scatterv(*inVG,*inVL,thread_dist_val,thread_offset_val);
			mpi_util::scatterv(*outVG,*outVL,thread_dist_val,thread_offset_val);
		}
		if(nTest>0){
			inSL->resize(thread_dist_test[rank]);
			outSL->resize(thread_dist_test[rank]);
			mpi_util::scatterv(*inSG,*inSL,thread_dist_test,thread_offset_test);
			mpi_util::scatterv(*outSG,*outSL,thread_dist_test,thread_offset_test);
		}
		
		//==== set the data ====
		if(rank==0) std::cout<<"setting data\n";
		nnopt.inT()=inTL;
		nnopt.outT()=outTL;
		nnopt.inV()=inVL;
		nnopt.outV()=outVL;
		
		//==== execute optimization ====
		if(rank==0) std::cout<<"executing optimization\n";
		nnopt.nbatch()=thread_dist_batch[rank];
		nnopt.train(nn);
		
		//==== write the data ====
		if(rank==0){
			std::cout<<"writing data\n";
			writer=fopen(file_out.c_str(),"w");
			if(writer!=NULL){
				fprintf(writer,"#X Y NN\n");
				for(unsigned int i=0; i<inSG->size(); ++i){
					fprintf(writer,"%f %f %f\n",(*inSG)[i][0],(*outSG)[i][0],nn->execute((*inSG)[i])[0]);
				}
				fclose(writer);
				writer=NULL;
			} else std::cout<<"could not open data file\n";
		}
		
		//======== finalize mpi ========
		if(rank==0) std::cout<<"finalizing mpi\n";
		std::cout<<std::flush;
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
	}catch(std::exception& e){
		std::cout<<"ERROR in nn_fit_1D::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
	}
	
	delete[] input;
}