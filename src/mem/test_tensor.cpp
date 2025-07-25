#include <iostream>
#include <Eigen/Dense>
#include "mem/tensor.hpp"

int main(int argc, char* argv[]){
	
	{
		const int rank=1;
		Eigen::VectorXi dim=Eigen::VectorXi::Constant(rank,4);
		Tensor<rank,int> tensor(dim);
		std::cout<<"dim = "<<dim<<"\n";
		for(int i=0; i<dim[0]; ++i){
			Eigen::VectorXi index=Eigen::VectorXi::Constant(rank,i);
			tensor(index)=i;
		}
		for(int i=0; i<dim[0]; ++i){
			Eigen::VectorXi index=Eigen::VectorXi::Constant(rank,i);
			std::cout<<"tensor["<<i<<"] = "<<tensor(index)<<"\n";
		}
	}
	
	{
		const int rank=2;
		const int size=4;
		Eigen::VectorXi dim=Eigen::VectorXi::Constant(rank,size);
		Tensor<rank,int> tensor(dim);
		std::cout<<"dim = "<<dim.transpose()<<"\n";
		int c=0;
		for(int i=0; i<size; ++i){
			for(int j=0; j<size; ++j){
				Eigen::VectorXi index=Eigen::VectorXi::Constant(rank,0); index<<i,j;
				tensor(index)=c++;
			}
		}
		for(int i=0; i<size; ++i){
			for(int j=0; j<size; ++j){
				std::cout<<tensor(i,j)<<" ";
			}
			std::cout<<"\n";
		}
		for(int i=0; i<size; ++i){
			for(int j=0; j<size; ++j){
				Eigen::VectorXi index=Eigen::VectorXi::Constant(rank,0); index<<i,j;
				std::cout<<tensor(index)<<" ";
			}
			std::cout<<"\n";
		}
	}
	
	return 0;
}