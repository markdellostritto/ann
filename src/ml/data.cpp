#include "ml/data.hpp"

void MLData::clear(){
	size_=0;
	dInp_=0;
	dOut_=0;
	inp_.clear();
	out_.clear();
}

void MLData::resize(int dInp, int dOut){
	if(dInp<=0) throw std::invalid_argument("MLData::resize(int): invalid dimension - input.");
	if(dOut<=0) throw std::invalid_argument("MLData::resize(int): invalid dimension - output.");
	clear();
	dInp_=dInp;
	dOut_=dOut;
}

void MLData::resize(int size, int dInp, int dOut){
	if(size<=0) throw std::invalid_argument("MLData::resize(int): invalid size.");
	if(dInp<=0) throw std::invalid_argument("MLData::resize(int): invalid dimension - input.");
	if(dOut<=0) throw std::invalid_argument("MLData::resize(int): invalid dimension - output.");
	clear();
	size_=size;
	dInp_=dInp;
	dOut_=dOut;
	inp_.resize(size,Eigen::VectorXd::Zero(dInp_));
	out_.resize(size,Eigen::VectorXd::Zero(dOut_));
}

void MLData::push(const Eigen::VectorXd& inp, const Eigen::VectorXd& out){
	if(inp.size()!=dInp_) throw std::invalid_argument("MLData::resize(int): invalid dimension - input.");
	if(out.size()!=dOut_) throw std::invalid_argument("MLData::resize(int): invalid dimension - output.");
	inp_.push_back(inp);
	out_.push_back(out);
	++size_;
}