// c++
#include <stdexcept>
// str
#include "str/string.hpp"
#include "str/print.hpp"
#include "str/token.hpp"
// math
#include "math/const.hpp"
#include "math/func.hpp"
// torch
#include "torch/phonon.hpp"

namespace phonon{
	
//******************************************************************
// KPath
//******************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const KPath& kpath){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("KPATH",str)<<"\n";
	for(int i=0; i<kpath.m_; ++i){
		out<<kpath.kpts_[i].transpose()<<" "<<kpath.npts_[i]<<"\n";
	}
	out<<print::title("KPATH",str)<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

//==== member functions ====

void KPath::resize(int m){
	if(m<=0) throw std::invalid_argument("Invalid number of k-points.");
	m_=m;
	npts_.resize(m_);
	kpts_.resize(m_);
}

void KPath::init(){
	N_=0;
	for(int i=0; i<m_-1; ++i){
		const Eigen::Vector3d diff=(kpts_[i+1]-kpts_[i])/npts_[i];
		for(int j=0; j<npts_[i]; ++j){
			const Eigen::Vector3d tmp=kpts_[i]+j*diff;
			kpath_.push_back(tmp);
		}
		N_+=npts_[i];
	}
	kpath_.push_back(kpts_.back());
}

//******************************************************************
// DynMat
//******************************************************************

//==== member functions ====

void DynMat::resize(const Structure& pcell, const Eigen::Vector3i& nlat){
	if(PHONON_PRINT_FUNC>0) std::cout<<"DynMat::resize(const Structure&,const Eigen::Vector3i&):\n";
	if(nlat[0]<=0 || nlat[1]<=0 || nlat[2]<=0) throw std::invalid_argument("Invalid lattice dimension.");
	//k-space lattice
	if(PHONON_PRINT_STATUS>0) std::cout<<"resizing k-space lattice\n";
	nlat_=nlat;
	np_=nlat_.prod();
	radnpi_=1.0/sqrt(1.0*np_);
	std::cout<<"nlat = "<<nlat_.transpose()<<"\n";
	std::cout<<"np_ = "<<np_<<"\n";
	
	//dynamical matrix
	if(PHONON_PRINT_STATUS>0) std::cout<<"resizing dynamical matrix\n";
	nModes_=pcell.nAtoms()*3;
	mat_.resize(3,nlat_,Eigen::MatrixXcd::Zero(nModes_,nModes_));
	evals_.resize(3,nlat_,Eigen::VectorXcd::Zero(nModes_));
	omega_.resize(3,nlat_,Eigen::VectorXd::Zero(nModes_));
	
	//vibrational modes
	if(PHONON_PRINT_STATUS>0) std::cout<<"setting supercell for phonon calculations\n";
	//resize the supercell
	pcell_=pcell;
	const int nAtomsT=pcell_.nAtoms()*np_;
	scell_.resize(nAtomsT,pcell_.atomType());
	//set the atomic properties
	int c=0;
	const AtomType& atomT=pcell_.atomType();
	map_.resize(pcell_.nAtoms(),Tensor<int>(3,nlat_));
	for(int i=0; i<nlat_[0]; ++i){
		for(int j=0; j<nlat_[1]; ++j){
			for(int k=0; k<nlat_[2]; ++k){
				Eigen::Vector3d R=i*pcell_.R().col(0)+j*pcell_.R().col(1)+k*pcell_.R().col(2);
				for(int n=0; n<pcell_.nAtoms(); ++n){
					//set map
					Eigen::Vector3i index; index<<i,j,k;
					map_[n](index)=c;
					//basic properties
					if(atomT.name)	scell_.name(c)=pcell_.name(n);
					if(atomT.an)	scell_.an(c)=pcell_.an(n);
					if(atomT.type)	scell_.type(c)=pcell_.type(n);
					if(atomT.index) scell_.index(c)=pcell_.index(n);
					//serial properties
					if(atomT.mass)	scell_.mass(c)=pcell_.mass(n);
					//vector properties
					if(atomT.posn)	scell_.posn(c)=pcell_.posn(n)+R;
					if(atomT.force) scell_.force(c)=pcell_.force(n);
					//increment
					c++;
				}
			}
		}
	}
	Eigen::MatrixXd Rnew=pcell.R();
	Rnew.col(0)*=nlat_[0];
	Rnew.col(1)*=nlat_[1];
	Rnew.col(2)*=nlat_[2];
	static_cast<Cell&>(scell_).init(Rnew);
	
	if(PHONON_PRINT_DATA>1){
		std::cout<<"map = \n";
		for(int n=0; n<map_.size(); ++n){
			for(int i=0; i<nlat_[0]; ++i){
				for(int j=0; j<nlat_[1]; ++j){
					for(int k=0; k<nlat_[2]; ++k){
						Eigen::VectorXi index(3); index<<i,j,k;
						std::cout<<i<<" "<<j<<" "<<k<<" "<<n<<" "<<map_[n](index)<<"\n";
					}
				}
			}
		}
	}
}

void DynMat::compute(int nsample, double dr, Engine& engine){
	if(PHONON_PRINT_FUNC>0) std::cout<<"DynMat::compute(int,double,Engine&):\n";
	//==== local variables ====
	Tensor<Eigen::VectorXcd> uk;//fourier transform of deviations (N x (nprim*3))
	Tensor<Eigen::VectorXcd> Fk;//fourier transform of forces (N x (nprim*3))
	Tensor<Eigen::MatrixXcd> ukuk;//fourier transform of product of deviations (N x (nprim*3 x nprim*3))
	Tensor<Eigen::MatrixXcd> Fkuk;//fourier transform of product of force and deviation (N x (nprim*3 x nprim*3))
	fftw_plan fftpx,fftpy,fftpz;//fftw plans
	fftw_complex *inx,*iny,*inz;//fftw input arrays
	fftw_complex *outx,*outy,*outz;//fftw output arrays
	
	//==== resize ====
	if(PHONON_PRINT_STATUS>0) std::cout<<"resizing utility vectors\n";
	uk.resize(3,nlat_,Eigen::VectorXcd::Zero(nModes_));
	Fk.resize(3,nlat_,Eigen::VectorXcd::Zero(nModes_));
	ukuk.resize(3,nlat_,Eigen::MatrixXcd::Zero(nModes_,nModes_));
	Fkuk.resize(3,nlat_,Eigen::MatrixXcd::Zero(nModes_,nModes_));
	
	//==== initialize fft ====
	if(PHONON_PRINT_STATUS>0) std::cout<<"initializing fft\n";
	inx=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*np_);
	iny=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*np_);
	inz=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*np_);
	outx=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*np_);
	outy=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*np_);
	outz=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*np_);
	fftpx=fftw_plan_dft_3d(nlat_[0],nlat_[1],nlat_[2],inx,outx,FFTW_BACKWARD,FFTW_ESTIMATE);
	fftpy=fftw_plan_dft_3d(nlat_[0],nlat_[1],nlat_[2],iny,outy,FFTW_BACKWARD,FFTW_ESTIMATE);
	fftpz=fftw_plan_dft_3d(nlat_[0],nlat_[1],nlat_[2],inz,outz,FFTW_BACKWARD,FFTW_ESTIMATE);
	
	//==== compute ukuk and FkFk ====
	if(PHONON_PRINT_STATUS>0) std::cout<<"computing ukuk and FkFk\n";
	Eigen::Vector3d rtmp;
	Structure struc;
	std::cout<<"scell = "<<scell_<<"\n";
	std::cout<<"dr = "<<dr<<"\n";
	scell_.energy()=0;
	for(int n=0; n<scell_.nAtoms(); ++n) scell_.force(n).setZero();
	for(int t=0; t<nsample; ++t){
		//randomize the structural positions
		std::cout<<"randomizing positions\n";
		struc=scell_;
		for(int n=0; n<struc.nAtoms(); ++n){
			struc.posn(n).noalias()+=Eigen::Vector3d::Random()*dr/math::constant::Rad3;
		}
		//compute forces
		std::cout<<"computing forces\n";
		engine.nlist().build(struc);
		engine.compute(struc);
		//loop over all atoms in the primitive cell
		std::cout<<"looping over atoms in primitive cell\n";
		for(int n=0; n<pcell_.nAtoms(); ++n){
			//== fourier transform of deviations ==
			//record the deviation from the average in the fftw input
			for(int i=0; i<np_; ++i){
				struc.diff(struc.posn(map_[n][i]),scell_.posn(map_[n][i]),rtmp);
				inx[i][0]=rtmp[0]; inx[i][1]=0.0;
				iny[i][0]=rtmp[1]; iny[i][1]=0.0;
				inz[i][0]=rtmp[2]; inz[i][1]=0.0;
			}
			//perform the fourier transform
			fftw_execute(fftpx);
			fftw_execute(fftpy);
			fftw_execute(fftpz);
			//record the fourier transforms
			for(int i=0; i<np_; ++i){
				uk[i][n*3+0]=std::complex<double>(outx[i][0]*radnpi_,outx[i][1]*radnpi_);
				uk[i][n*3+1]=std::complex<double>(outy[i][0]*radnpi_,outy[i][1]*radnpi_);
				uk[i][n*3+2]=std::complex<double>(outz[i][0]*radnpi_,outz[i][1]*radnpi_);
			}
			//== fourier transform of forces ==
			//record the force
			for(int i=0; i<np_; ++i){
				const Eigen::Vector3d& f=struc.force(map_[n][i]);
				inx[i][0]=f[0]; inx[i][1]=0.0;
				iny[i][0]=f[1]; iny[i][1]=0.0;
				inz[i][0]=f[2]; inz[i][1]=0.0;
			}
			//perform the fourier transform
			fftw_execute(fftpx);
			fftw_execute(fftpy);
			fftw_execute(fftpz);
			//record the fourier transforms
			for(int i=0; i<np_; ++i){
				Fk[i][n*3+0]=std::complex<double>(outx[i][0]*radnpi_,outx[i][1]*radnpi_);
				Fk[i][n*3+1]=std::complex<double>(outy[i][0]*radnpi_,outy[i][1]*radnpi_);
				Fk[i][n*3+2]=std::complex<double>(outz[i][0]*radnpi_,outz[i][1]*radnpi_);
			}
		}
		//== record products ==
		for(int i=0; i<np_; ++i){
			for(int n=0; n<nModes_; ++n){
				for(int m=0; m<nModes_; ++m){
					ukuk[i](n,m)+=std::conj(uk[i][n])*uk[i][m];
					Fkuk[i](n,m)+=std::conj(Fk[i][n])*uk[i][m];
				}
			}
		}
	}
	
	//==== free memory ====
	if(PHONON_PRINT_STATUS>0) std::cout<<"freeing fftw memory\n";
	fftw_free(inx);
	fftw_free(iny);
	fftw_free(inz);
	fftw_free(outx);
	fftw_free(outy);
	fftw_free(outz);
	fftw_destroy_plan(fftpx);
	fftw_destroy_plan(fftpy);
	fftw_destroy_plan(fftpz);
	
	//==== normalize ====
	if(PHONON_PRINT_STATUS>0) std::cout<<"normalizing\n";
	const double norm=1.0/(np_*1.0);
	for(int i=0; i<np_; ++i){
		//ukuk[i]*=norm;
		//Fkuk[i]*=norm;
	}
	
	//==== compute the dynamical matrix ====
	if(PHONON_PRINT_STATUS>0) std::cout<<"computing the dynamical matrix\n";
	mvec_.resize(nModes_);
	for(int i=0; i<nModes_; i+=3){
		mvec_[i+0]=pcell_.mass(i/3);
		mvec_[i+1]=pcell_.mass(i/3);
		mvec_[i+2]=pcell_.mass(i/3);
	}
	for(int i=0; i<np_; ++i){
		mat_[i]=Fkuk[i]*ukuk[i].inverse();
		for(int n=0; n<nModes_; ++n){
			for(int m=0; m<nModes_; ++m){
				mat_[i](n,m)*=1.0/std::sqrt(mvec_[n]*mvec_[m]);
			}
		}
	}
	mvec_.resize(0);
	
	//==== compute the eigenvalues ====
	if(PHONON_PRINT_STATUS>0) std::cout<<"computing the eigenvalues\n";
	double error_im=0;
	//compute eigenvalues for all k points
	wmin_=0; wmax_=0;
	for(int i=0; i<np_; ++i){
		Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(mat_[i]);
		evals_[i]=solver.eigenvalues();
		//std::cout<<"evals_["<<i<<"] = "<<evals_[i].transpose()<<"\n";
		for(int j=0; j<nModes_; ++j) omega_[i][j]=std::sqrt(std::fabs(evals_[i][j].real()));
		for(int j=0; j<nModes_; ++j) error_im+=std::fabs(evals_[i][j].imag()/evals_[i][j].real());
		if(omega_[i].minCoeff()<wmin_) wmin_=omega_[i].minCoeff();
		if(omega_[i].maxCoeff()>wmax_) wmax_=omega_[i].maxCoeff();
		//std::cout<<"omega["<<i<<"] = "<<omega_[i].transpose()<<"\n";
	}
	error_im/=(np_*nModes_);
	if(PHONON_PRINT_DATA>0) std::cout<<"w-lim = "<<wmin_<<" "<<wmax_<<"\n";
	if(PHONON_PRINT_DATA>0) std::cout<<"error_im = "<<error_im<<"\n";
	//enforce asr
	double error_asr=0;
	std::vector<Eigen::Matrix3cd> asr(pcell_.nAtoms(),Eigen::Matrix3cd::Zero());
	for(int nn=0; nn<nasr_; ++nn){
		//reset the asr vector
		for(int n=0; n<pcell_.nAtoms(); ++n) asr[n]=Eigen::Matrix3cd::Zero();
		//compute the asr vector
		for(int i=0; i<3; ++i){
			for(int j=0; j<3; ++j){
				for(int n=0; n<pcell_.nAtoms(); ++n){
					for(int m=0; m<pcell_.nAtoms(); ++m){
						asr[n](i,j)+=mat_[0](n*3+i,m*3+j);
					}
				}
			}
		}
		//subtract the asr vector from the matrix
		for(int i=0; i<3; ++i){
			for(int j=0; j<3; ++j){
				for(int n=0; n<pcell_.nAtoms(); ++n){
					for(int m=0; m<pcell_.nAtoms(); ++m){
						std::complex<double> denom=std::complex<double>(pcell_.nAtoms()*1.0,0.0);
						mat_[0](n*3+i,m*3+j)-=asr[n](i,j)/denom;
					}
				}
			}
		}
		Eigen::MatrixXcd dynmat0=Eigen::MatrixXcd::Zero(nModes_,nModes_);
		for(int i=0; i<nModes_; ++i){
			for(int j=0; j<nModes_; ++j){
				dynmat0(i,j)=0.5*(mat_[0](i,j)+mat_[0](j,i));
			}
		}
		mat_[0]=dynmat0;
		//reset the asr vector
		for(int n=0; n<pcell_.nAtoms(); ++n) asr[n]=Eigen::Matrix3cd::Zero();
		//compute the asr error
		error_asr=0;
		for(int i=0; i<3; ++i){
			for(int j=0; j<3; ++j){
				for(int n=0; n<pcell_.nAtoms(); ++n){
					for(int m=0; m<pcell_.nAtoms(); ++m){
						asr[n](i,j)+=mat_[0](n*3+i,m*3+j);
					}
				}
			}
		}
		for(int n=0; n<pcell_.nAtoms(); ++n){
			error_asr+=asr[n].norm();
		}
		error_asr/=pcell_.nAtoms();
		Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(mat_[0]);
		evals_[0]=solver.eigenvalues();
		for(int j=0; j<omega_[0].size(); ++j) omega_[0][j]=std::sqrt(std::fabs(evals_[0][j].real()))*-1.0*math::func::sgn(evals_[0][j].real());
	}
	std::cout<<"error_asr = "<<error_asr<<"\n";
	if(PHONON_PRINT_DATA>0) std::cout<<"w-lim = "<<wmin_<<" "<<wmax_<<"\n";
	
}

//******************************************************************
// DOS
//******************************************************************

void DOS::compute(const DynMat& dynmat){
	if(PHONON_PRINT_FUNC>0) std::cout<<"DOS::compute(const DynMat&):\n";
	//==== compute the phonon density of states ====
	if(wlmax_<0) wlmax_=dynmat.wmax();
	if(wlmin_<0) wlmin_=dynmat.wmin();
	if(sigma_<0) sigma_=0.01;//default value
	if(dw_<0) dw_=(wlmax_-wlmin_)/1000.0;
	const int ndos=static_cast<int>((wlmax_-wlmin_)/dw_)+1;
	std::cout<<"wlim  = "<<wlmin_<<" "<<wlmax_<<"\n";
	std::cout<<"dw    = "<<dw_<<"\n";
	std::cout<<"sigma = "<<sigma_<<"\n";
	std::cout<<"ndos  = "<<ndos<<"\n";
	dos_.resize(ndos,Eigen::Vector2d::Zero());
	for(int i=0; i<ndos; ++i){
		const double norm=1.0/(sigma_*sqrt(2.0*math::constant::PI));
		const double w=wlmin_+i*dw_;
		dos_[i][0]=w;
		for(int n=0; n<dynmat.np(); ++n){
			for(int m=0; m<dynmat.nModes(); ++m){
				dos_[i][1]+=norm*std::exp(-(w-dynmat.omega()[n][m])*(w-dynmat.omega()[n][m])/(2.0*sigma_*sigma_));
			}
		}
	}
	for(int i=0; i<ndos; ++i){
		dos_[i][1]/=(dynmat.np()*dynmat.nModes());
	}
}

}