#include "math.hpp"

LinSolver::type LinSolver::load(const char* str){
	if(std::strcmp(str,"LDLT")==0) return LinSolver::LDLT;
	else if(std::strcmp(str,"LLT")==0) return LinSolver::LLT;
	else if(std::strcmp(str,"PARTIALPIVLU")==0) return LinSolver::PartialPivLU;
	else if(std::strcmp(str,"FULLPIVLU")==0) return LinSolver::FullPivLU;
	else if(std::strcmp(str,"HQR")==0) return LinSolver::HQR;
	else if(std::strcmp(str,"COLPIVHQR")==0) return LinSolver::ColPivHQR;
	else return UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, const LinSolver::type& t){
	if(t==LinSolver::LDLT) out<<"LDLT";
	else if(t==LinSolver::LLT) out<<"LLT";
	else if(t==LinSolver::PartialPivLU) out<<"PartialPivLU";
	else if(t==LinSolver::FullPivLU) out<<"FullPivLU";
	else if(t==LinSolver::HQR) out<<"HQR";
	else if(t==LinSolver::ColPivHQR) out<<"ColPivHQR";
	else out<<"UNKNOWN";
	return out;
}

namespace function{
	
	double poly(double x, const std::vector<double>& a){
		int j=a.size()-1;
		double result=a[j];
		while(j>0) result=x*result+a[--j];
		return result;
	}
	
	double poly(double x, const double* a, unsigned int s){
		double result=a[--s];
		while(s>0) result=x*result+a[--s];
		return result;
	}
	
	double StepCos::operator()(double x)const{
		if(x<p_[0]) return 0;
		else if(x>p_[1]) return 1;
		else return 0.5*(1.0-std::cos(num_const::PI*(x-p_[0])/(p_[1]-p_[0])));
	}
	
}

namespace special{
	
	//**************************************************************
	//Kummer's (confluent hypergeometric) function 
	//**************************************************************
	
	double M(double a, double b, double z, double prec){
		if(DEBUG_MATH>0) std::cout<<"M(unsigned int,unsigned int,double):\n";
		unsigned int nMax=1e8;
		double fac=1,result=1;
		for(unsigned int n=1; n<=nMax; ++n){
			fac*=a*z/(n*b);
			result+=fac;
			++a; ++b;
			if(std::fabs(fac/result)*100<prec) break;
		}
		return result;
	}
	
	//**************************************************************
	//Legendre Polynomials
	//**************************************************************
	
	std::vector<double>& legendre(unsigned int n, std::vector<double>& c){
		if(n==0) c.resize(n+1,1.0);
		else if(n==1){c.resize(n+1,0.0); c[1]=1.0;}
		else {
			c.resize(n+1,0.0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1.0; ct2[1]=1.0;
			for(unsigned int m=2; m<=n; ++m){
				for(int l=m; l>0; --l) c[l]=(2.0*m-1.0)/m*ct2[l-1];
				c[0]=0.0;
				for(int l=m; l>=0; --l) c[l]-=(m-1.0)/m*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
	
	//**************************************************************
	//Chebyshev Polynomials
	//**************************************************************
	
	double chebyshev1r(unsigned int n, double x){
		if(n==0) return 1;
		else if(n==1) return x;
		else return 2*x*chebyshev1r(n-1,x)-chebyshev1r(n-2,x);
	}
	
	double chebyshev1l(unsigned int n, double x){
		if(n==0) return 1;
		double rnm1=x,rnm2=1,rn=x;
		for(unsigned int i=2; i<=n; ++i){
			rn=2*x*rnm1-rnm2;
			rnm2=rnm1; rnm1=rn;
		}
		return rn;
	}
	
	std::vector<double>& chebyshev1l(unsigned int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=x;
			for(unsigned int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	double chebyshev2r(unsigned int n, double x){
		if(n==0) return 1;
		else if(n==1) return 2*x;
		else return 2*x*chebyshev2r(n-1,x)-chebyshev2r(n-2,x);
	}
	
	double chebyshev2l(unsigned int n, double x){
		if(n==0) return 1;
		double rnm1=2*x,rnm2=1,rn=2*x;
		for(unsigned int i=2; i<=n; ++i){
			rn=2*x*rnm1-rnm2;
			rnm2=rnm1; rnm1=rn;
		}
		return rn;
	}
	
	std::vector<double>& chebyshev2l(unsigned int n, double x, std::vector<double>& r){
		if(r.size()!=n+1) throw std::invalid_argument("Invalid vector size.");
		r[0]=1;
		if(n>=1){
			r[1]=2*x;
			for(unsigned int i=2; i<=n; ++i){
				r[i]=2*x*r[i-1]-r[i-2];
			}
		}
		return r;
	}
	
	std::vector<double>& chebyshev1_root(unsigned int n, std::vector<double>& r){
		r.resize(n);
		for(unsigned int i=0; i<n; i++) r[i]=std::cos((2.0*i+1.0)/(2.0*n)*num_const::PI);
		return r;
	}
	
	std::vector<double>& chebyshev2_root(unsigned int n, std::vector<double>& r){
		r.resize(n);
		for(unsigned int i=0; i<n; i++) r[i]=std::cos((i+1.0)/(n+1.0)*num_const::PI);
		return r;
	}
	
	//**************************************************************
	//Jacobi Polynomials
	//**************************************************************
	
	double jacobi(unsigned int n, double a, double b, double x){
		if(n==0) return 1;
		else if(n==1) return 0.5*(2*(a+1)+(a+b+2)*(x-1));
		else return 
			(2*n+a+b-1)*((2*n+a+b)*(2*n+a+b-2)*x+a*a-b*b)/(2*n*(n+a+b)*(2*n+a+b-2))*jacobi(n-1,a,b,x)
			-(n+a-1)*(n+b-1)*(2*n+a+b)/(n*(n+a+b)*(2*n+a+b-2))*jacobi(n-2,a,b,x);
	}
	
	std::vector<double>& jacobi(unsigned int n, double a, double b, std::vector<double>& c){
		if(n==0) c.resize(1,1);
		else if(n==1){
			c.resize(2,0);
			c[0]=0.5*(a-b);
			c[1]=0.5*(a+b+2);
		} else {
			c.resize(n+1,0.0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1.0;
			ct2[0]=0.5*(a-b);
			ct2[1]=0.5*(a+b+2);
			for(unsigned int m=2; m<=n; ++m){
				c[0]=0.0;
				for(int l=m; l>0; --l) c[l]=(2*m+a+b-1)*(2*m+a+b)/(2*m*(m+a+b))*ct2[l-1];
				for(int l=m; l>=0; --l) c[l]+=(2*m+a+b-1)*(a*a-b*b)/(2*m*(m+a+b)*(2*m+a+b-2))*ct2[l];
				for(int l=m; l>=0; --l) c[l]-=(m+a-1)*(m+b-1)*(2*m+a+b)/(m*(m+a+b)*(2*m+a+b-2))*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
	
	//**************************************************************
	//Laguerre Polynomials
	//**************************************************************
	
	std::vector<double>& laguerre(unsigned int n, std::vector<double>& c){
		if(n==0) c.resize(1,1);
		else if(n==1){
			c.resize(2,1);
			c[1]=-1;
		} else {
			c.resize(n+1,0);
			std::vector<double> ct1(n+1,0.0),ct2(n+1,0.0);
			ct1[0]=1;
			ct2[0]=1; ct2[1]=-1;
			for(unsigned int m=2; m<=n; ++m){
				c[0]=0;
				for(int l=m; l>0; --l) c[l]=-1.0/m*ct2[l-1];
				for(int l=m; l>=0; --l) c[l]+=(2.0*m-1.0)/m*ct2[l];
				for(int l=m; l>=0; --l) c[l]-=(m-1.0)/m*ct1[l];
				ct1=ct2; ct2=c;
			}
		}
		return c;
	}
}

namespace roots{
	
	double Root::prec=1E-4;
	unsigned int Root::nMax=50;
	unsigned int Root::N=0;
	
	double Root::bisect(const func& f, double xMin, double xMax){
		if(DEBUG_MATH>0) std::cout<<"bisect(const func&,double,double):\n";
		if(xMax<=xMin) throw std::invalid_argument("Found max abscissa lt/e min abscissa.");
		double yMin=f(xMin),yMax=f(xMax);
		double xMid,yMid;
		if(yMin*yMax>=0) throw std::invalid_argument("Equal sign at edges, cannot find root.");
		for(int i=0; i<nMax; i++){
			yMid=f(xMid=0.5*(xMax+xMin));
			if(yMid*yMin>=0) xMin=xMid;//same on left, move right
			else xMax=xMid;//same on right, move left
			if(yMid==0 || (xMax-xMin)<prec) {N=i; return yMid;}
		}
		throw std::runtime_error("Root Search Failed: Max Iteration Limit Reached.");
	}
	
	double Root::ridder(const func& f, double xMin, double xMax){
		if(DEBUG_MATH>0) std::cout<<"ridder(const func&,double,double):\n";
		if(xMax<=xMin) throw std::invalid_argument("Found max abscissa lt/e min abscissa.");
		double yMin=f(xMin),yMax=f(xMax);
		double xMid,yMid;
		if(yMin*yMax>=0) throw std::invalid_argument("Equal sign at edges, cannot find root.");
		for(int i=0; i<nMax; i++){
			yMid=f(xMid=0.5*(xMin+xMax));
			double s=std::sqrt(yMid*yMid-yMin*yMax);
			if(s==0) throw std::runtime_error("Zero Denominator.");
			double xNew=xMid+(xMid-xMin)*misc::sign(yMin-yMax)*yMid/s;
			double yNew=f(xNew);
			if(yNew==0) {N=i; return xNew;}
			if(yMid*yNew<0){
				xMin=xMid; yMin=yMid;
				xMax=xNew; yMax=yNew;
			} else if(yMin*yNew<0){
				xMax=xNew; yMax=yNew;
			} else if(yMax*yNew<0){
				xMin=xNew; yMin=yNew;
			} else throw std::runtime_error("Badly conditioned function.");
			if(std::fabs(xMax-xMin)<=prec) {N=i; return xNew;}
		}
		throw std::runtime_error("Root Search Failed: Max Iteration Limit Reached.");
	}
	
	double Root::NR(const func& f, const func& df, double x1, double x2){
		if(DEBUG_MATH>0) std::cout<<"NR(const func&,const func&,double,double):\n";
		double yMin=f(x1),yMax=f(x2);
		double xMin,xMax;
		double xMid,yMid;
		if(yMin*yMax>=0) throw std::invalid_argument("Equal sign at edges, cannot find root.");
		if(yMin<0){xMin=x1; xMax=x2;}
		else{xMin=x2; xMax=x1;}
		double root=0.5*(xMax+xMin);
		double dxOld=std::fabs(xMax-xMin);
		double dx=dxOld;
		double F=f(root);
		double dF=df(root);
		for(int i=0; i<nMax; i++){
			//if the root is out of range, or if the change is too small, bisect
			if(((root-xMax)*dF-F)*((root-xMin)*dF-F)>0 || std::fabs(2*F)>std::fabs(dxOld*dF)){
				//bisection
				dxOld=dx;
				dx=0.5*(xMax-xMin);
				root=xMin+dx;
			} else {
				//Newton-Raphson
				dxOld=dx;
				dx=F/dF;
				double temp=root;
				root-=dx;
			}
			if(std::fabs(dx)<prec) {N=i; return root;}
			F=f(root);
			dF=df(root);
			if(F<0) xMin=root;
			else xMax=root;
		}
		throw std::runtime_error("Root Search Failed: Max Iteration Limit Reached.");
	}
	
	std::vector<double>& RootPoly::eigen(const std::vector<double>& a, std::vector<double>& r){
		if(DEBUG_MATH>0) std::cout<<"RootPoly::eigen(const std::vector<double>,std::vector<double>):\n";
		if(a.size()==0) throw std::invalid_argument("Polynomial has no coefficients.");
		else if(a.size()==1) throw std::invalid_argument("Constant function has no roots.");
		else if(a.size()==2) r.resize(1,-a[0]/a[1]);
		else{
			unsigned int s=a.size()-1;
			r.resize(s);
			Eigen::MatrixXd mat=Eigen::MatrixXd::Zero(s,s);
			for(int i=0; i<s; i++){
				mat(0,i)=-a[a.size()-1-(i+1)]/a[a.size()-1];
			}
			for(int i=0; i<s-1; i++){
				mat(i+1,i)=1;
			}
			Eigen::EigenSolver<Eigen::MatrixXd> solver(mat);
			if(solver.info()!=Eigen::Success) throw std::runtime_error("Badly conditioned polynomial: no solution.");
			for(int i=0; i<s; i++){
				if(std::fabs(solver.eigenvalues()[i].imag())>num_const::ZERO) throw std::runtime_error("Badly conditioned polynomial: imaginary roots.");
				else r[i]=solver.eigenvalues()[i].real();
			}
		}
		return r;
	}
	
}

namespace integration{
	
	//***************************************************************
	//Quadrature class
	//***************************************************************
	
	double Quadrature::error(boost::function<double(double)>& f, double a, double b, unsigned int n){
		return -(b-a)*(f(b)-f(a))/(12*std::log2(n)*std::log2(n));
	}
	
	double Quadrature::init(boost::function<double(double)>& f, double& a, double& b, double& s){
		n_=1;
		return s=0.5*(b-a)*(f(b)+f(a));
	}
	
	double Quadrature::next(boost::function<double(double)>& f, double& a, double& b, double& s){
		//add 2^n points
		double dx=(b-a)/n_;
		double x=a+0.5*dx;
		double sum=0;
		for(int i=0; i<n_; i++){
			sum+=f(x);
			x+=dx;
		}
		s=0.5*(s+(b-a)*sum/n_);
		n_*=2;
		return s;
	}
	
	//***************************************************************
	//Trapezoid class
	//***************************************************************
	
	double Trapezoid::integrate(boost::function<double(double)>& f, double a, double b){
		double sOld,s;
		init(f,a,b,s);
		for(int n=0; n<nMin_; n++){
			next(f,a,b,s);
		}
		sOld=s;
		for(int n=nMin_; n<nMax_; n++){
			next(f,a,b,s);
			if(std::fabs(sOld-s)<prec_*std::fabs(sOld)) return s;
			else sOld=s;
		}
		throw std::runtime_error("Inaccurate Quadrature: Achieved Iteration Limit.");
	}
	
	//***************************************************************
	//Simpson class
	//***************************************************************
	
	double Simpson::integrate(boost::function<double(double)>& f, double a, double b){
		double sOld=0,sOldT=0,sT=0,s;
		init(f,a,b,sT);
		s=(4.0*sT-sOldT)/3.0;
		sOld=s; sOldT=sT;
		for(int n=0; n<nMin_; n++){
			next(f,a,b,sT);
			s=(4.0*sT-sOldT)/3.0;
			sOld=s; sOldT=sT;
		}
		for(int n=nMin_; n<nMax_; n++){
			next(f,a,b,sT);
			s=(4.0*sT-sOldT)/3.0;
			if(std::fabs(sOld-s)<prec_*std::fabs(sOld)) return s;
			sOld=s; sOldT=sT;
		}
		throw std::runtime_error("Inaccurate Quadrature: Achieved Iteration Limit.");
	}
	
	//***************************************************************
	//Romberg class
	//***************************************************************
	
	double Romberg::integrate(boost::function<double(double)>& f, double a, double b){
		if(DEBUG_MATH>0) std::cout<<"Romberg::integrate(boost::function<double(double)>&,double,double):\n";
		unsigned int order=4;//number of points used in extrapolation (order of polynomial)
		//s = results, h = step sizes (extrapolating to zero)
		std::vector<double> sV(nMax_),hV(nMax_+1);
		double s,ss,ssOld=0;
		hV[0]=1.0;
		init(f,a,b,s); sV[0]=s;
		hV[1]=0.25*hV[0];
		for(int i=2; i<order+1; i++){
			next(f,a,b,s); sV[i-1]=s;
			hV[i]=0.25*hV[i-1];
			if(DEBUG_MATH>0) std::cout<<"int-est = "<<s<<"\n";
		}
		for(int i=order+1; i<nMax_; i++){
			next(f,a,b,s); sV[i-1]=s;
			ss=Interp::interpPoly(0,i-(order+1),order,hV,sV);
			if(std::fabs(ssOld-ss)<prec_*std::fabs(ssOld)) return ss;
			ssOld=ss;
			if(DEBUG_MATH>0) std::cout<<"int-est = "<<ss<<"\n";
			hV[i]=0.25*hV[i-1];
		}
		throw std::runtime_error("Inaccurate Quadrature: Achieved Iteration Limit.");
	}
	
	//***************************************************************
	//Gauss-Legendre Quadrature class
	//***************************************************************
	
	//constructors/destructors
	
	QuadGaussLegendre::QuadGaussLegendre():prec_(1E-5),order_(7){
		//find the coefficients
		special::legendre(order_,coeffs_);
		//find the zeros
		roots::RootPoly::eigen(coeffs_,x_);
		//find the weights
		QuadGaussLegendre::weights(order_,x_,w_);
	};
	
	QuadGaussLegendre::QuadGaussLegendre(unsigned int order):prec_(1E-5),order_(order){
		//find the coefficients
		special::legendre(order_,coeffs_);
		//find the zeros
		roots::RootPoly::eigen(coeffs_,x_);
		//find the weights
		QuadGaussLegendre::weights(order_,x_,w_);
	};
	
	QuadGaussLegendre::QuadGaussLegendre(double prec, unsigned int order):prec_(prec),order_(order){
		//find the coefficients
		special::legendre(order_,coeffs_);
		//find the zeros
		roots::RootPoly::eigen(coeffs_,x_);
		//find the weights
		QuadGaussLegendre::weights(order_,x_,w_);
	};
		
	//member functions
	
	void QuadGaussLegendre::weights(unsigned int n, std::vector<double>& x, std::vector<double>& w){
		//generate the coeffs of the Legendre polynomial of order n
		std::vector<double> coeffs;
		special::legendre(n,coeffs);
		//generate the zeros
		roots::RootPoly::eigen(coeffs,x);
		//generate the values of the weigths
		w.resize(x.size());
		for(int i=0; i<w.size(); i++){
			w[i]=2*(1.0-x[i]*x[i])/((n+1)*(n+1)*boost::math::legendre_p(n+1,x[i])*boost::math::legendre_p(n+1,x[i]));
		}
	}
	
	double QuadGaussLegendre::integrate(boost::function<double(double)>& f, double a, double b){
		double diff=0.5*(b-a);
		double sum=0.5*(b+a);
		double integral=0;
		for(int i=0; i<x_.size(); i++){
			integral+=w_[i]*f(diff*x_[i]+sum);
		}
		integral*=diff;
		return integral;
	}
	
	//***************************************************************
	//Gauss-Laguerre Quadrature class
	//***************************************************************
	
	//constructors/destructors
	
	QuadGaussLaguerre::QuadGaussLaguerre():prec_(1E-5),order_(7){
		//find the coefficients
		special::laguerre(order_,coeffs_);
		//find the zeros
		roots::RootPoly::eigen(coeffs_,x_);
		//find the weights
		QuadGaussLaguerre::weights(order_,x_,w_);
	};
	
	QuadGaussLaguerre::QuadGaussLaguerre(unsigned int order):prec_(1E-5),order_(order){
		//find the coefficients
		special::laguerre(order_,coeffs_);
		//find the zeros
		roots::RootPoly::eigen(coeffs_,x_);
		//find the weights
		QuadGaussLaguerre::weights(order_,x_,w_);
	};
	
	QuadGaussLaguerre::QuadGaussLaguerre(double prec, unsigned int order):prec_(prec),order_(order){
		//find the coefficients
		special::laguerre(order_,coeffs_);
		//find the zeros
		roots::RootPoly::eigen(coeffs_,x_);
		//find the weights
		QuadGaussLaguerre::weights(order_,x_,w_);
	};
		
	//member functions
	
	void QuadGaussLaguerre::weights(unsigned int n, std::vector<double>& x, std::vector<double>& w){
		//generate the coeffs of the Legendre polynomial of order n
		std::vector<double> coeffs;
		special::laguerre(n,coeffs);
		//generate the zeros
		roots::RootPoly::eigen(coeffs,x);
		//generate the values of the weigths
		w.resize(x.size());
		for(unsigned int i=0; i<w.size(); ++i){
			w[i]=x[i]/((n+1)*(n+1)*boost::math::laguerre(n+1,x[i])*boost::math::laguerre(n+1,x[i]));
		}
	}
	
	double QuadGaussLaguerre::integrate(boost::function<double(double)>& f){
		double integral=0;
		for(unsigned int i=0; i<x_.size(); ++i) integral+=w_[i]*f(x_[i]);
		return integral;
	}
	
	//***************************************************************
	//Gauss-Jacobi Quadrature class
	//***************************************************************
	
	//constructors/destructors
	
	QuadGaussJacobi::QuadGaussJacobi():prec_(1E-5),order_(7),a_(1),b_(0){
		//find the coefficients
		special::jacobi(order_,a_,b_,coeffs_);
		//find the zeros
		roots::RootPoly::eigen(coeffs_,x_);
		//find the weights
		QuadGaussJacobi::weights(order_,a_,b_,x_,w_);
	};
	
	QuadGaussJacobi::QuadGaussJacobi(unsigned int order, double a, double b):prec_(1E-5),order_(order),a_(a),b_(b){
		//find the coefficients
		std::cout<<"a_ = "<<a_<<", b_ = "<<b_<<"\n";
		special::jacobi(order_,a_,b_,coeffs_);
		//find the zeros
		roots::RootPoly::eigen(coeffs_,x_);
		//find the weights
		QuadGaussJacobi::weights(order_,a_,b_,x_,w_);
	};
	
	QuadGaussJacobi::QuadGaussJacobi(double prec, unsigned int order, double a, double b):prec_(prec),order_(order),a_(a),b_(b){
		//find the coefficients
		special::jacobi(order_,a_,b_,coeffs_);
		//find the zeros
		roots::RootPoly::eigen(coeffs_,x_);
		//find the weights
		QuadGaussJacobi::weights(order_,a_,b_,x_,w_);
	};
		
	//member functions
	
	void QuadGaussJacobi::weights(unsigned int n, double a, double b, std::vector<double>& x, std::vector<double>& w){
		//generate the coeffs of the Legendre polynomial of order n
		std::vector<double> coeffs;
		special::jacobi(n,a,b,coeffs);
		//generate the zeros
		roots::RootPoly::eigen(coeffs,x);
		//generate the values of the weigths
		w.resize(x.size());
		for(unsigned int i=0; i<w.size(); ++i){
			w[i]=-1.0*(2.0*n+a+b+2.0)/(n+a+b+1.0)
				*boost::math::tgamma_ratio(n+a+1.0,n+a+b+2.0)
				*boost::math::tgamma_ratio(n+b+1.0,n+2.0)
				*std::pow(2.0,a+b+1)/(special::jacobi(n-1,a+1,b+1,x[i])*special::jacobi(n+1,a,b,x[i]));
		}
	}
	
	double QuadGaussJacobi::integrate(boost::function<double(double)>& f, double a, double b){
		double diff=0.5*(b-a);
		double sum=0.5*(b+a);
		double integral=0;
		for(unsigned int i=0; i<x_.size(); ++i){
			integral+=w_[i]*f(diff*x_[i]+sum);
		}
		integral*=diff;
		return integral;
	}
	
}

namespace geom{

	Eigen::Matrix3d& cosineM(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Matrix3d& mat, bool norm){
		mat.noalias()=v2*v1.transpose();
		if(norm) mat/=(v1.norm()*v2.norm());
		return mat;
	}
	
	Eigen::Matrix3d& RX(double theta, Eigen::Matrix3d& mat){
		mat.setZero();
		mat(0,0)=1; mat(1,0)=0; mat(2,0)=0;
		mat(0,1)=0; mat(1,1)=std::cos(theta); mat(2,1)=std::sin(theta);
		mat(0,2)=0; mat(1,2)=-std::sin(theta); mat(2,2)=std::cos(theta);
		return mat;
	}
	
	Eigen::Matrix3d& RY(double theta, Eigen::Matrix3d& mat){
		mat.setZero();
		mat(0,0)=std::cos(theta); mat(1,0)=0; mat(2,0)=-std::sin(theta);
		mat(0,1)=0; mat(1,1)=1; mat(2,1)=0;
		mat(0,2)=std::sin(theta); mat(1,2)=0; mat(2,2)=std::cos(theta);
		return mat;
	}
	
	Eigen::Matrix3d& RZ(double theta, Eigen::Matrix3d& mat){
		mat(0,0)=std::cos(theta); mat(1,0)=std::sin(theta); mat(2,0)=0;
		mat(0,1)=-std::sin(theta); mat(1,1)=std::cos(theta); mat(2,1)=0;
		mat(0,2)=0; mat(1,2)=0; mat(2,2)=1;
	}
}