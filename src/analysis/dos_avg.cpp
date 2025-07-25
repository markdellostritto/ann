// c libraries
#include <cstring>
// c++ libraries
#include <limits>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
// io
#include "str/parse.hpp"
#include "str/string.hpp"
#include "str/token.hpp"

int main(int argc, char* argv[]){
	//data
		int ndos=0;
		double dE=0.01;
		std::vector<double> efermi;
		std::vector<int> nval;
		std::vector<std::vector<std::vector<double> > > dos;
	//files
		std::vector<std::string> fnames;
		std::string fout;
		char* input=new char[string::M];
		Token token;
	//arguments
		std::vector<input::Arg> args;
	//misc
		bool error=false;
		
	try{
		//check the number of arguments
		if(argc<=1) throw std::invalid_argument("Invalid number of arguments.");
		
		//parse the arguments
		std::cout<<"parsing the arguments\n";
		input::parse(argc,argv,args);
		for(int i=0; i<args.size(); ++i){
			if(args[i].key()=="de") dE=std::atof(args[i].val(0).c_str());
			else if(args[i].key()=="out") fout=args[i].val(0);
			else if(args[i].key()=="dos") fnames=args[i].vals();
		}
		
		//read each dos
		std::cout<<"reading the dos\n";
		ndos=fnames.size();
		efermi.resize(ndos,0.0);
		nval.resize(ndos,0);
		dos.resize(ndos);
		for(int i=0; i<ndos; ++i){
			//open the file
			FILE* reader=fopen(fnames[i].c_str(),"r");
			if(reader==NULL) throw std::invalid_argument("Could not open dos file.");
			//read the header
			token.read(fgets(input,string::M,reader),string::WS);
			for(int j=0; j<9; ++j) token.next();
			efermi[i]=std::atof(token.next().c_str());
			char* pch=std::strchr(input,'(');
			while(pch!=NULL){nval[i]++; pch=std::strchr(pch+1,'(');}
			//read the data
			while(fgets(input,string::M,reader)!=NULL){
				token.read(input,string::WS);
				dos[i].push_back(std::vector<double>(4,0.0));
				for(int j=0; j<nval[i]; ++j){
					dos[i].back()[j]=std::atof(token.next().c_str());
				}
			}
			//close the file
			fclose(reader);
			reader=NULL;
		}
		
		//check the dos
		std::cout<<"checking the dos\n";
		for(int i=1; i<ndos; ++i){
			if(nval[i]!=nval[0]) throw std::invalid_argument("Incompatible number of data in each file.");
		}
		
		//set the fermi energy
		std::cout<<"setting the fermi energy\n";
		for(int i=0; i<ndos; ++i){
			for(int j=0; j<dos[i].size(); ++j){
				dos[i][j][0]-=efermi[i];
				//std::cout<<"dos["<<i<<"]["<<j<<"] = "; for(int k=0; k<dos[i][j].size(); ++k) std::cout<<dos[i][j][k]<<" "; std::cout<<"\n";
			}
		}
		//find the energy min/max
		std::cout<<"finding energy min/max\n";
		double emin=std::numeric_limits<double>::max();
		double emax=std::numeric_limits<double>::max()*-1;
		for(int i=0; i<ndos; ++i){
			for(int j=0; j<dos[i].size(); ++j){
				if(dos[i][j][0]<emin) emin=dos[i][j][0];
				if(dos[i][j][0]>emax) emax=dos[i][j][0];
			}
		}
		//avg the dos
		std::cout<<"averaging the dos\n";
		const int size=(emax-emin)/dE;
		std::vector<std::vector<double> > dosa(size,std::vector<double>(nval[0],0.0));
		for(int j=0; j<size; ++j){
			dosa[j][0]=emin+dE*j;
		}
		for(int i=0; i<ndos; ++i){
			for(int j=0; j<size; ++j){
				const double energy=emin+dE*j;
				if(dos[i].front()[0]<energy && energy<dos[i].back()[0]){
					int id=-1;
					for(int n=0; n<dos[i].size()-1; ++n){
						if(dos[i][n][0]<=energy && energy<=dos[i][n+1][0]){
							id=n; break;
						}
					}
					for(int k=1; k<nval[i]; ++k){
						dosa[j][k]+=(dos[i][id+1][k]-dos[i][id][k])/(dos[i][id+1][0]-dos[i][id][0])*(energy-dos[i][id][0])+dos[i][id][k];
					}
				}
			}
		}
		for(int j=0; j<size; ++j){
			for(int k=1; k<nval[0]; ++k){
				dosa[j][k]/=ndos;
			}
		}
		
		//write the avg dos
		std::cout<<"writing the dos\n";
		FILE* writer=fopen(fout.c_str(),"w");
		if(writer==NULL) throw std::invalid_argument("Unable to open output file."); 
		for(int j=0; j<size; ++j){
			for(int k=0; k<nval[0]; ++k){
				fprintf(writer,"%f ",dosa[j][k]);
			}
			fprintf(writer,"\n");
		}
		fclose(writer);
		writer=NULL;
	}catch(std::exception& e){
		std::cout<<"ERROR in dos_avg::main(int,char**):\n";
		std::cout<<e.what()<<"\n";
		error=true;
	}
	
	delete[] input;
	
	if(error) return 1;
	else return 0;
}