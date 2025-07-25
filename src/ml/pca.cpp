#include "ml/pca.hpp"

void PCA::resize(int nobs, int nvars){
	if(nobs<=0) throw std::invalid_argument("Invalid number of observations.");
	if(nvars<=0) throw std::invalid_argument("Invalid number of variables.");
	nobs_=nobs;
	nvars_=nvars;
	X_.resize(nobs_,nvars_);
	S_.resize(nobs_,nvars_);
	W_.resize(nobs_,nvars_);
	w_.resize(nvars_);
}

void PCA::compute(){
	for(int i=0; i<nvars_; ++i){
		const double mean=X_.col(i).mean();
		for(int j=0; j<nobs_; ++j){
			X_(j,i)-=mean;
		}
	}
	S_.noalias()=X_.transpose()*X_;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(S_);
	w_=solver.eigenvalues();
	W_=solver.eigenvectors();
}