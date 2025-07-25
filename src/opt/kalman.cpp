#include "opt/kaman.hpp"

void resize(const NN::ANN& ann){
	nOut_=ann.nOut();
	nP_=ann.size();
	w_.resize(nP_);
	dy_=Eigen::VectorXd::Zero(nOut_);
	H_=Eigen::MatrixXd::Zero(nP_,nOut_);
	K_=Eigen::MatrixXd::Zero(nP_,nOut_);
	A_=Eigen::MatrixXd::Zero(nOut_,nOut_);
	AI_=Eigen::MatrixXd::Zero(nOut_,nOut_);
	P_=Eigen::MatrixXd::Zero(nP_,nP_);
	Q_=Eigen::MatrixXd::Identity(nP_,nP_);
}

void step(double g){
	AI_=Eigen::MatrixXd::Identity(nOut_,nOut_)*1.0/g;
	AI_.noalias()+=H_*P_.transpose()*H_;
	A_=AI_.inverse();
	K_.noalias()=P_*H_*A_;
	w_.noalias()+=K_*dy_;
	P_.noalias()+=K_*H_.transpose()*K_;
}