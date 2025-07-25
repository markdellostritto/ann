#pragma once
#ifndef KALMAN_HPP
#define KALMAN_HPP

class KalmanFilter{
private:
	int nOut_;//number of NN outputs
	int nP_;//number of NN parameters
	double q_;//noise parameter
	Eigen::VectorXd dy_;//output difference
	Eigen::VectorXd w_;//parameters
	Eigen::MatrixXd H_;
	Eigen::MatrixXd K_;//gain matrix
	Eigen::MatrixXd A_;
	Eigen::MatrixXd AI_;
	Eigen::MatrixXd P_;
	Eigen::MatrixXd Q_;//noise matrix
public:
	KalmanFilter(){}
	KalmanFilter(const NN::ANN& ann){resize(ann);}
	
	//==== access ====
	const int& nOut()const{return nOut_;}
	const int& nP()const{return nP_;}
	double& q(){return q_;}
	const double& q()const{return q_;}
	const Eigen::VectorXd& dy()const{return dy_;}
	const Eigen::VectorXd& w()const{return w_;}
	const Eigen::MatrixXd& H()const{return H_;}
	const Eigen::MatrixXd& A()const{return A_;}
	const Eigen::MatrixXd& P()const{return P_;}
	double& dy(int i){return dy_[i];}
	const double& dy(int i)const{return dy_[i];}
	double& H(int i, int j){return H_(i,j);}
	const double& H(int i, int j)const{return H_(i,j);}
	double& A(int i, int j){return A_(i,j);}
	const double& A(int i, int j)const{return A_(i,j);}
	double& P(int i, int j){return P_(i,j);}
	const double& P(int i, int j)const{return P_(i,j);}
	
	//==== member functions ====
	void resize(const NN::ANN& ann);
};

#endif