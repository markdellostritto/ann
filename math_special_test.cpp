#include <iostream>
#include <cstdlib>
#include <chrono>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include "math_special.hpp"
#include "math_function.hpp"

int main(int argc, char* argv[]){
	
	bool test_sigmoid=true;
	bool test_legendre=false;
	bool test_chebyshev=false;
	bool test_laguerre=false;
	bool test_jacobi=false;
	bool test_kummer=false;
	
	if(test_sigmoid){
	std::cout<<"*********************************************************\n";
	std::cout<<"****** TEST -  SGIMOID ******\n";
	try{
		unsigned int N=10000;
		std::vector<double> x(N);
		std::vector<double> y(N);
		double xMin=-10,xMax=10;
		
		for(unsigned int i=0; i<N; ++i){
			double xx=(xMax-xMin)*i/N+xMin;
			double yy=special::sigmoid(xx);
			x[i]=xx; y[i]=yy;
		}
		
		FILE* writer=fopen("sigmoid_test.dat","w");
		if(writer!=NULL){
			fprintf(writer,"X Y\n");
			for(unsigned int i=0; i<N; ++i) fprintf(writer,"%f %f\n",x[i],y[i]);
		}
	}catch(std::exception& e){
		std::cout<<"Error in TEST - SIGMOID :\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"****** TEST -  SGIMOID ******\n";
	std::cout<<"*********************************************************\n";	
	}
	
	if(test_legendre){
	std::cout<<"*********************************************************\n";
	std::cout<<"****** TEST - POLYNOMIAL - LEGENDRE ******\n";
	try{
		std::cout<<"Test of Legendre Polynomial coefficient generator...\n";
		unsigned int nMax=20;
		for(int i=0; i<=nMax; i++){
			std::vector<double> a;
			special::legendre(i,a);
			std::cout<<"\t";
			for(unsigned int j=0; j<a.size(); ++j) std::cout<<a[j]<<" ";
			std::cout<<"\n";
		}
		std::cout<<"Test of Legendre Polynomial coefficients: evaluation of polynomial:\n";
		double error=0;
		unsigned int nRand=10000;
		std::srand(std::time(NULL));
		for(int n=0; n<=nMax; n++){
			std::vector<double> a;
			special::legendre(n,a);
			for(int i=0; i<nRand; i++){
				double x=-1+((double)std::rand())/RAND_MAX*2.0;
				error+=std::fabs((function::poly(x,a)-boost::math::legendre_p(n,x))/boost::math::legendre_p(n,x))*100;
			}
			error/=nRand;
			std::cout<<"P("<<n<<") % error = "<<error<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"Error in TEST - POLYNOMIAL - LEGENDRE:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"****** TEST - POLYNOMIAL - LEGENDRE ******\n";
	std::cout<<"*********************************************************\n";
	}
	
	if(test_laguerre){
	std::cout<<"*********************************************************\n";
	std::cout<<"****** TEST - POLYNOMIAL - LAGUERRE ******\n";
	try{
		std::cout<<"Test of Laguerre Polynomial coefficient generator...\n";
		unsigned int nMax=20;
		for(int i=0; i<=nMax; i++){
			std::vector<double> a;
			special::laguerre(i,a);
			std::cout<<"\t";
			for(int j=0; j<a.size(); j++){
				std::cout<<a[j]<<" ";
			}
			std::cout<<"\n";
		}
		std::cout<<"Test of Laguerre Polynomial coefficients: evaluation of polynomial:\n";
		double error=0;
		unsigned int nRand=10000;
		std::srand(std::time(NULL));
		for(int n=0; n<=nMax; n++){
			std::vector<double> a;
			special::laguerre(n,a);
			for(int i=0; i<nRand; i++){
				double x=-1+((double)std::rand())/RAND_MAX*2.0;
				error+=std::fabs((function::poly(x,a)-boost::math::laguerre(n,x))/boost::math::laguerre(n,x))*100;
			}
			error/=nRand;
			std::cout<<"P("<<n<<") % error = "<<error<<"\n";
		}
	}catch(std::exception& e){
		std::cout<<"Error in TEST - POLYNOMIAL - LAGUERRE:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"****** TEST - POLYNOMIAL - LAGUERRE ******\n";
	std::cout<<"*********************************************************\n";
	}
	
	if(test_chebyshev){
	std::cout<<"*********************************************************\n";
	std::cout<<"****** TEST - POLYNOMIAL - CHEBYSHEV ******\n";
	try{
		std::cout<<"Test of Chebyshev Polynomials...\n";
		FILE* writer=fopen("chebyshev.dat","w");
		if(writer!=NULL){
			fprintf(writer, "x o0 o1 o2 o3 o4 o5\n");
			unsigned int nPoints=501;
			double a=-1,b=1;
			double diff=(b-a)/nPoints;
			for(int i=0; i<=nPoints; i++){
				double x=a+diff*i;
				fprintf(writer,"%f %f %f %f %f %f %f\n", x,
					special::chebyshev1r(0,x),
					special::chebyshev1r(1,x),
					special::chebyshev1r(2,x),
					special::chebyshev1r(3,x),
					special::chebyshev1r(4,x),
					special::chebyshev1r(5,x)
				);
			}
			fclose(writer);
		}
	}catch(std::exception& e){
		std::cout<<"Error in TEST - POLYNOMIAL - CHEBYSHEV:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"****** TEST - POLYNOMIAL - CHEBYSHEV ******\n";
	std::cout<<"*********************************************************\n";
	}
	
	if(test_jacobi){
	std::cout<<"*********************************************************\n";
	std::cout<<"****** TEST - POLYNOMIAL - JACOBI ******\n";
	try{
		unsigned int N=100;
		unsigned int nMax=5;
		double diff=0;
		std::cout<<"Test for (a,b)=(0,0)\n";
		for(unsigned int n=0; n<nMax; ++n){
			diff=0;
			for(unsigned int i=0; i<N; ++i){
				double x=-1+((double)std::rand())/RAND_MAX*2.0;
				double exact=boost::math::legendre_p(n,x);
				double approx=special::jacobi(n,0,0,x);
				diff+=std::fabs((exact-approx)/exact)*100;
			}
			std::cout<<"\terror("<<n<<") = "<<diff/N<<"\n";
		}
		std::cout<<"Test for (a,b)=(-1/2,-1/2)\n";
		for(unsigned int n=0; n<nMax; ++n){
			diff=0;
			for(unsigned int i=0; i<N; ++i){
				double x=-1+((double)std::rand())/RAND_MAX*2.0;
				double exact=special::chebyshev1r(n,x);
				double approx=special::jacobi(n,-0.5,-0.5,x)/special::jacobi(n,-0.5,-0.5,1.0);
				diff+=std::fabs((exact-approx)/exact)*100;
			}
			std::cout<<"\terror("<<n<<") = "<<diff/N<<"\n";
		}
		std::cout<<"Test for n=1\n";
		diff=0;
		for(unsigned int ia=0; ia<N; ++ia){
			for(unsigned int ib=0; ib<N; ++ib){
				double x=-1+((double)std::rand())/RAND_MAX*2.0;
				double a=((double)std::rand())/RAND_MAX*5.0;
				double b=((double)std::rand())/RAND_MAX*5.0;
				double exact=0.5*(2*(a+1)+(a+b+2)*(x-1));
				double approx=special::jacobi(1,a,b,x);
				diff+=std::fabs((exact-approx)/exact)*100;
			}
		}
		std::cout<<"\terror("<<1<<") = "<<diff/(N*N)<<"\n";
		std::cout<<"Test for n=2\n";
		diff=0;
		for(unsigned int ia=0; ia<N; ++ia){
			for(unsigned int ib=0; ib<N; ++ib){
				double x=-1+((double)std::rand())/RAND_MAX*2.0;
				double a=((double)std::rand())/RAND_MAX*5.0;
				double b=((double)std::rand())/RAND_MAX*5.0;
				double exact=1.0/8.0*(4*(a+1)*(a+2)+4*(a+b+3)*(a+2)*(x-1)+(a+b+3)*(a+b+4)*(x-1)*(x-1));
				double approx=special::jacobi(2,a,b,x);
				diff+=std::fabs((exact-approx)/exact)*100;
			}
		}
		std::cout<<"\terror("<<1<<") = "<<diff/(N*N)<<"\n";
		std::cout<<"Test Polynomial Coefficients\n";
		{
			unsigned int n=2;
			double a=-2;
			double b=-3;
			std::vector<double> c;
			special::jacobi(0,a,b,c);
			std::cout<<"\tc(0) = "; for(unsigned int i=0; i<1; ++i) std::cout<<c[i]<<" "; std::cout<<"\n";
			special::jacobi(1,a,b,c);
			std::cout<<"\tc(1) = "; for(unsigned int i=0; i<2; ++i) std::cout<<c[i]<<" "; std::cout<<"\n";
			special::jacobi(2,a,b,c);
			std::cout<<"\tc(2) = "; for(unsigned int i=0; i<3; ++i) std::cout<<c[i]<<" "; std::cout<<"\n";
			special::jacobi(3,a,b,c);
			std::cout<<"\tc(3) = "; for(unsigned int i=0; i<4; ++i) std::cout<<c[i]<<" "; std::cout<<"\n";
		}
		std::cout<<"Test Polynomial Coefficients\n";
		diff=0;
		for(unsigned int i=0; i<N; ++i){
			unsigned int n=std::rand()%5;
			double x=-1+((double)std::rand())/RAND_MAX*2.0;
			double a=((double)std::rand())/RAND_MAX*5.0;
			double b=((double)std::rand())/RAND_MAX*5.0;
			double exact=special::jacobi(n,a,b,x);
			std::vector<double> c; special::jacobi(n,a,b,c);
			double approx=function::poly(x,c);
			diff+=std::fabs((exact-approx)/exact)*100;
		}
		std::cout<<"\tdiff = "<<diff/N<<"\n";
	}catch(std::exception& e){
		std::cout<<"Error in TEST - POLYNOMIAL - JACOBI:\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"****** TEST - POLYNOMIAL - JACOBI ******\n";
	std::cout<<"*********************************************************\n";
	}
	
	if(test_kummer){
	std::cout<<"*********************************************************\n";
	std::cout<<"************ TEST - KUMMER HYPERGEOMETRIC ************\n";
	try{
		//local function variables
		unsigned int N=10000;
		double a,b,z,diff;
		double aMin=0,aMax=5;
		double bMin=0,bMax=5;
		double zMin=-5,zMax=5;
		std::srand(std::time(NULL));
		std::cout<<"Special values...\n";
		//a=0
		diff=0;
		for(unsigned int i=0; i<N; ++i){
			a=0;
			b=bMin+((double)std::rand())/RAND_MAX*(bMax-bMin);
			z=zMin+((double)std::rand())/RAND_MAX*(zMax-zMin);
			//std::cout<<"M(a,b,z) = M("<<a<<","<<b<<","<<z<<")\n";
			double result=special::M(a,b,z);
			double exact=1.0;
			diff+=std::fabs((result-exact)/exact*100);
			//std::cout<<"M = ("<<std::setprecision(6)<<exact<<","<<result<<","<<std::fabs((result-exact)/exact*100)<<")\n";
		}
		std::cout<<"M(0,b,z): diff = "<<std::setprecision(6)<<diff/N<<"\n";
		//a=b
		diff=0;
		for(unsigned int i=0; i<N; ++i){
			a=aMin+((double)std::rand())/RAND_MAX*(aMax-aMin);
			b=a;
			z=zMin+((double)std::rand())/RAND_MAX*(zMax-zMin);
			//std::cout<<"M(a,b,z) = M("<<a<<","<<b<<","<<z<<")\n";
			double result=special::M(a,b,z);
			double exact=std::exp(z);
			diff+=std::fabs((result-exact)/exact*100);
			//std::cout<<"M = ("<<std::setprecision(6)<<exact<<","<<result<<","<<std::fabs((result-exact)/exact*100)<<")\n";
		}
		std::cout<<"M(a,a,z): diff = "<<std::setprecision(6)<<diff/N<<"\n";
		//a=1/2,b=1/2,-z^2
		diff=0;
		for(unsigned int i=0; i<N; ++i){
			a=0.5;
			b=1.5;
			z=zMin+((double)std::rand())/RAND_MAX*(zMax-zMin);
			//std::cout<<"M(a,b,z) = M("<<a<<","<<b<<","<<z<<")\n";
			double result=special::M(a,b,-z*z);
			double exact=std::sqrt(num_const::PI)/(2*z)*boost::math::erf(z);
			diff+=std::fabs((result-exact)/exact*100);
			//std::cout<<"M = ("<<std::setprecision(6)<<exact<<","<<result<<","<<std::fabs((result-exact)/exact*100)<<")\n";
		}
		std::cout<<"M(1/2,3/2,-z^2): diff = "<<std::setprecision(6)<<diff/N<<"\n";
		//b=2a
		diff=0;
		for(unsigned int i=0; i<N; ++i){
			a=aMin+((double)std::rand())/RAND_MAX*(aMax-aMin);
			b=2*a;
			z=((double)std::rand())/RAND_MAX*(zMax);
			//std::cout<<"M(a,b,z) = M("<<a<<","<<b<<","<<z<<")\n";
			double result=special::M(a,b,z);
			double exact=std::exp(0.5*z)*std::pow(0.25*z,0.5-a)*boost::math::tgamma(a+0.5)*boost::math::cyl_bessel_i(a-0.5,0.5*z);
			diff+=std::fabs((result-exact)/exact*100);
			//std::cout<<"M = ("<<std::setprecision(6)<<exact<<","<<result<<","<<std::fabs((result-exact)/exact*100)<<")\n";
		}
		std::cout<<"M(a,2a,z): diff = "<<std::setprecision(6)<<diff/N<<"\n";
		//1,s+1
		diff=0;
		for(unsigned int i=0; i<N; ++i){
			double s=bMin+((double)std::rand())/RAND_MAX*(bMax-bMin);
			a=1;
			b=1+s;
			z=((double)std::rand())/RAND_MAX*(zMax);
			//std::cout<<"M(a,b,z) = M("<<a<<","<<b<<","<<z<<")\n";
			double result=special::M(a,b,z);
			double exact=s*std::exp(z)/std::pow(z,s)*boost::math::gamma_p(s,z);
			diff+=std::fabs((result-exact)/exact*100);
			//std::cout<<"M = ("<<std::setprecision(6)<<exact<<","<<result<<","<<std::fabs((result-exact)/exact*100)<<")\n";
		}
		std::cout<<"M(1,s+1,z): diff = "<<std::setprecision(6)<<diff/N<<"\n";
		//timing
		std::chrono::high_resolution_clock::time_point start=std::chrono::high_resolution_clock::now();
		for(unsigned int i=0; i<N; ++i){
			a=0;
			b=bMin+((double)std::rand())/RAND_MAX*(bMax-bMin);
			z=zMin+((double)std::rand())/RAND_MAX*(zMax-zMin);
			//std::cout<<"M(a,b,z) = M("<<a<<","<<b<<","<<z<<")\n";
			double result=special::M(a,b,z);
		}
		std::chrono::high_resolution_clock::time_point stop=std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time=std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
		std::cout<<"M("<<N<<") completed in "<<time.count()<<" seconds.\n";
		std::cout<<"M average = "<<time.count()/N<<" seconds.\n";
	}catch(std::exception& e){
		std::cout<<"ERROR in TEST - KUMMER HYPERGEOMETRIC\n";
		std::cout<<e.what()<<"\n";
	}
	std::cout<<"************ TEST - KUMMER HYPERGEOMETRIC ************\n";
	std::cout<<"*********************************************************\n";
	}
}