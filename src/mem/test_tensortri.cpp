#include <iostream>
#include <Eigen/Dense>
#include "mem/tensor.hpp"
#include "mem/tensortri.hpp"

int main(int argc, char* argv[]){
	
	{
		std::cout<<"============= TEST - RANK 1 =============\n";
		const int rank=1;
		const int dim=4;
		TensorTri<rank,int> tensortri(dim);
		std::cout<<"dim = "<<dim<<"\n";
		for(int i=0; i<dim; ++i){
			tensortri(i)=i;
		}
		for(int i=0; i<dim; ++i){
			std::cout<<"tensortri["<<i<<"] = "<<tensortri(i)<<"\n";
		}
	}
	
	{
		std::cout<<"============= TEST - RANK 2 =============\n";
		const int rank=2;
		const int dim=4;
		Tensor<rank,int> tensor(dim);
		TensorTri<rank,int> tensortri(dim);
		std::cout<<"dim = "<<dim<<"\n";
		std::cout<<"size = "<<tensortri.size()<<"\n";
		int c;
		std::cout<<"half-filled matrix\n";
		tensor*=0;
		tensortri*=0;
		c=0;
		for(int i=0; i<dim; ++i){
			for(int j=0; j<=i; ++j){
				tensor(i,j)=c;
				tensortri(i,j)=c;
				++c;
			}
		}
		std::cout<<"tensor    = \n";
		for(int i=0; i<dim; ++i){
			for(int j=0; j<dim; ++j){
				std::cout<<tensor(i,j)<<" ";
			}
			std::cout<<"\n";
		}
		std::cout<<"tensor ut = \n";
		for(int i=0; i<dim; ++i){
			for(int j=0; j<dim; ++j){
				std::cout<<tensortri(i,j)<<" ";
			}
			std::cout<<"\n";
		}
		std::cout<<"full-filled matrix\n";
		tensor*=0;
		tensortri*=0;
		c=0;
		for(int i=0; i<dim; ++i){
			for(int j=0; j<dim; ++j){
				tensor(i,j)=c;
				tensortri(i,j)=c;
				++c;
			}
		}
		std::cout<<"tensor    = \n";
		for(int i=0; i<dim; ++i){
			for(int j=0; j<dim; ++j){
				std::cout<<tensor(i,j)<<" ";
			}
			std::cout<<"\n";
		}
		std::cout<<"tensor ut = \n";
		for(int i=0; i<dim; ++i){
			for(int j=0; j<dim; ++j){
				std::cout<<tensortri(i,j)<<" ";
			}
			std::cout<<"\n";
		}
		std::cout<<"testing index\n";
		for(int i=0; i<dim; ++i){
			for(int j=0; j<=i; ++j){
				std::cout<<"loc["<<i<<","<<j<<"] = "<<tensortri.index(i,j)<<"\n";
			}
		}
	}
	
	{
		std::cout<<"============= TEST - RANK 3 =============\n";
		const int rank=3;
		const int dim=4;
		Tensor<rank,int> tensor(dim);
		TensorTri<rank,int> tensortri(dim);
		for(int i=0; i<dim; ++i){
			for(int j=0; j<=i; ++j){
				for(int k=0; k<=j; ++k){
					std::cout<<"loc["<<i<<","<<j<<","<<k<<"] = "<<tensortri.index(i,j,k)<<"\n";
				}
			}
		}
	}
	return 0;
}