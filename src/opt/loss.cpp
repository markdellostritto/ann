// c
#include <cstring>
// c++
#include <iostream>
// opt
#include "opt/loss.hpp"

//***************************************************
// loss function
//***************************************************

namespace opt{

//==== type ====

std::ostream& operator<<(std::ostream& out, const Loss& loss){
	switch(loss){
		case Loss::MSE: out<<"MSE"; break;
		case Loss::MAE: out<<"MAE"; break;
		case Loss::HUBER: out<<"HUBER"; break;
		case Loss::ASINH: out<<"ASINH"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

const char* Loss::name(const Loss& loss){
	switch(loss){
		case Loss::MSE: return "MSE";
		case Loss::MAE: return "MAE";
		case Loss::HUBER: return "HUBER";
		case Loss::ASINH: return "ASINH";
		default: return "UNKNOWN";
	}
}

Loss Loss::read(const char* str){
	if(std::strcmp(str,"MSE")==0) return Loss::MSE;
	else if(std::strcmp(str,"MAE")==0) return Loss::MAE;
	else if(std::strcmp(str,"HUBER")==0) return Loss::HUBER;
	else if(std::strcmp(str,"ASINH")==0) return Loss::ASINH;
	else return Loss::UNKNOWN;
}

//==== error ====

double Loss::error(Loss loss, const Eigen::VectorXd& value, const Eigen::VectorXd& target){
	//compute the error
	double err=0;
	switch(loss){
		case Loss::MSE:{
			err=0.5*(value-target).squaredNorm();
		} break;
		case Loss::MAE:{
			err=(value-target).lpNorm<1>();
		} break;
		case Loss::HUBER:{
			err=std::sqrt(1.0+(value-target).squaredNorm())-1.0;
		} break;
		case Loss::ASINH:{
			const int size=value.size();
			for(int i=0; i<size; i++){
				const double arg=value[i]-target[i];
				const double sqrtf=sqrt(1.0+arg*arg);
				err+=1.0-sqrtf+arg*log(arg+sqrtf);
			}
		} break;
		default:{
			err=0;
		} break;
	}
	//return the error
	return err;
}

double Loss::error2(Loss loss, double delta, const Eigen::VectorXd& value, const Eigen::VectorXd& target){
	//compute the error
	double err=0;
	switch(loss){
		case Loss::MSE:{
			err=0.5*(value-target).squaredNorm();
		} break;
		case Loss::MAE:{
			err=(value-target).lpNorm<1>();
		} break;
		case Loss::HUBER:{
			err=delta*delta*(std::sqrt(1.0+((value-target)/delta).squaredNorm())-1.0);
		} break;
		case Loss::ASINH:{
			const int size=value.size();
			for(int i=0; i<size; i++){
				const double arg=(value[i]-target[i])/delta;
				const double sqrtf=sqrt(1.0+arg*arg);
				err+=delta*delta*(1.0-sqrtf+arg*log(arg+sqrtf));
			}
		} break;
		default:{
			err=0;
		} break;
	}
	//return the error
	return err;
}

double Loss::error(Loss loss, const Eigen::VectorXd& value, const Eigen::VectorXd& target, Eigen::VectorXd& grad){
	//compute the error
	double err=0;
	switch(loss){
		case Loss::MSE:{
			grad.noalias()=(value-target);
			err=0.5*grad.squaredNorm();
		} break;
		case Loss::MAE:{
			grad.noalias()=(value-target);
			err=grad.lpNorm<1>();
			for(int i=0; i<grad.size(); ++i) grad[i]/=std::fabs(grad[i]);
		} break;
		case Loss::HUBER:{
			grad.noalias()=(value-target);
			err=std::sqrt(1.0+grad.squaredNorm())-1.0;
			grad/=(err+1.0);
		} break;
		case Loss::ASINH:{
			const int size=value.size();
			for(int i=0; i<size; ++i){
				const double arg=value[i]-target[i];
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				err+=1.0-sqrtf+arg*logf;
				grad[i]=logf;
			}
		} break;
		default:{
			err=0; 
			grad.setZero();
		} break;
	}
	//return the error
	return err;
}

double Loss::error2(Loss loss, double delta, const Eigen::VectorXd& value, const Eigen::VectorXd& target, Eigen::VectorXd& grad){
	//compute the error
	double err=0;
	switch(loss){
		case Loss::MSE:{
			grad.noalias()=(value-target);
			err=0.5*grad.squaredNorm();
		} break;
		case Loss::MAE:{
			grad.noalias()=(value-target);
			err=grad.lpNorm<1>();
			for(int i=0; i<grad.size(); ++i) grad[i]/=std::fabs(grad[i]);
		} break;
		case Loss::HUBER:{
			grad.noalias()=(value-target);
			const double fsqrt=std::sqrt(1.0+(grad/delta).squaredNorm());
			err=delta*delta*(fsqrt-1.0);
			grad*=1.0/fsqrt;
		} break;
		case Loss::ASINH:{
			const int size=value.size();
			for(int i=0; i<size; ++i){
				const double arg=(value[i]-target[i])/delta;
				const double sqrtf=sqrt(1.0+arg*arg);
				const double logf=log(arg+sqrtf);
				err+=delta*delta*(1.0-sqrtf+arg*logf);
				grad[i]=delta*logf;
			}
		} break;
		default:{
			err=0; 
			grad.setZero();
		} break;
	}
	//return the error
	return err;
}

}
