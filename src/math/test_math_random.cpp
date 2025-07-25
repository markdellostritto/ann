// c++ libraries
#include <iostream>
#include <chrono>
// math
#include <math/random.hpp>
#include <math/hist.hpp>
#include <math/const.hpp>
//str
#include "str/print.hpp"

using math::constant::PI;

void test_dist_uniform(){
	//generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	//distribution
	const double a=-1.0,b=1.0;
	rng::dist::Uniform dist(a,b);
	//compute
	const int nsample=10000000;
	const int nbins=100;
	const double xmin=-2.0;
	const double xmax=2.0;
	const double xdelta=(xmax-xmin)/nbins;
	Histogram hist(xmin,xmax,nbins);
	for(int i=0; i<nsample; ++i){
		hist.push(dist.rand(gen));
	}
	//integral
	double intg=0;
	for(int i=0; i<nbins; ++i){
		intg+=xdelta*hist.ordinate(i)*nbins;
	}
	//pdf
	double errpdf=0;
	for(int i=0; i<nbins; ++i){
		const double x=hist.abscissa(i);
		const double pdfe=dist.pdf(x);
		const double pdfa=hist.hist(i)/(hist.c()-hist.m());
		errpdf+=fabs(pdfe-pdfa);
	}
	errpdf/=nbins;
	//write
	FILE* writer=fopen("test_math_random_uniform.dat","w");
	fprintf(writer,"#x pdf hist\n");
	for(int i=0; i<nbins; ++i){
		fprintf(writer,"%f %f\n",hist.abscissa(i),hist.ordinate(i)*nbins);
	}
	fclose(writer);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - MATH - RANDOM - UNIFORM",str)<<"\n";
	std::cout<<"a      = "<<a<<"\n";
	std::cout<<"b      = "<<b<<"\n";
	std::cout<<"xlim   = "<<xmin<<":"<<xmax<<":"<<xdelta<<"\n";
	std::cout<<"error - intg = "<<std::fabs(1.0-intg)<<"\n";
	std::cout<<"error - pdf  = "<<errpdf<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_dist_laplace(){
	//generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	//distribution
	const double lambda=(double)gen()/gen.max()*(5.0-1.0)+1.0;
	rng::dist::Laplace dist(lambda);
	//compute
	const int nsample=10000000;
	const int nbins=100;
	const double xmin=-5.0*lambda;
	const double xmax=5.0*lambda;
	const double xdelta=(xmax-xmin)/nbins;
	Histogram hist(xmin,xmax,nbins);
	for(int i=0; i<nsample; ++i){
		hist.push(dist.rand(gen));
	}
	//integral
	double intg=0;
	for(int i=0; i<nbins; ++i){
		intg+=hist.hist(i)/(hist.c()-hist.m());
	}
	//pdf
	double errpdf=0;
	for(int i=0; i<nbins; ++i){
		const double x=hist.abscissa(i);
		const double pdfe=dist.pdf(x);
		const double pdfa=hist.hist(i)/(hist.c()-hist.m());
		errpdf+=fabs(pdfe-pdfa);
	}
	errpdf/=nbins;
	//write
	FILE* writer=fopen("test_math_random_laplace.dat","w");
	fprintf(writer,"#x pdf hist\n");
	for(int i=0; i<nbins; ++i){
		const double x=hist.abscissa(i);
		fprintf(writer,"%f %f %f\n",x,dist.pdf(x),hist.hist(i)/(hist.c()-hist.m())/xdelta);
	}
	fclose(writer);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - MATH - RANDOM - LAPLACE",str)<<"\n";
	std::cout<<"lambda = "<<lambda<<"\n";
	std::cout<<"xlim   = "<<xmin<<":"<<xmax<<":"<<xdelta<<"\n";
	std::cout<<"error - intg = "<<std::fabs(1.0-intg)<<"\n";
	std::cout<<"error - pdf  = "<<errpdf<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_dist_normal(){
	//generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	//distribution
	const double mu=0.0;
	//const double sigma=(double)gen()/gen.max()*(5.0-1.0)+1.0;
	const double sigma=1.0;
	rng::dist::Normal dist(mu,sigma);
	//compute
	const int nsample=10000000;
	const int nbins=100;
	const double xmin=-4.0*sigma;
	const double xmax=4.0*sigma;
	const double xdelta=(xmax-xmin)/nbins;
	Histogram hist(xmin,xmax,nbins);
	for(int i=0; i<nsample; ++i){
		hist.push(dist.rand(gen));
	}
	//integral
	double intg=0;
	for(int i=0; i<nbins; ++i){
		intg+=hist.hist(i)/(hist.c()-hist.m());
	}
	//pdf
	double errpdf=0;
	for(int i=0; i<nbins; ++i){
		const double x=hist.abscissa(i);
		const double pdfe=dist.pdf(x);
		const double pdfa=hist.hist(i)/(hist.c()-hist.m());
		errpdf+=fabs(pdfe-pdfa);
	}
	errpdf/=nbins;
	//write
	FILE* writer=fopen("test_math_random_normal.dat","w");
	fprintf(writer,"#x p p\n");
	for(int i=0; i<nbins; ++i){
		const double x=hist.abscissa(i);
		fprintf(writer,"%f %f %f\n",x,dist.pdf(x),hist.hist(i)/(hist.c()-hist.m())/xdelta);
	}
	fclose(writer);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - MATH - RANDOM - NORMAL",str)<<"\n";
	std::cout<<"nsample = "<<nsample<<"\n";
	std::cout<<"mu      = "<<mu<<"\n";
	std::cout<<"sigma   = "<<sigma<<"\n";
	std::cout<<"xlim    = "<<xmin<<":"<<xmax<<":"<<xdelta<<"\n";
	std::cout<<"error - intg = "<<std::fabs(1.0-intg)<<"\n";
	std::cout<<"error - pdf  = "<<errpdf<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_dist_normal2(){
	//generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	//distribution
	const double mu=0.0;
	const double sigma=(double)gen()/gen.max()*(5.0-1.0)+1.0;
	rng::dist::Normal dist1(mu,sigma);
	rng::dist::Normal dist2(mu,2.0*sigma);
	//compute
	const int nsample=10000000;
	const int nbins=100;
	const double xmin=-2.0*4.0*sigma;
	const double xmax=2.0*4.0*sigma;
	const double xdelta=(xmax-xmin)/nbins;
	Histogram hist1(xmin,xmax,nbins);
	Histogram hist2(xmin,xmax,nbins);
	for(int i=0; i<nsample; ++i){
		hist1.push(dist1.rand(gen));
		hist2.push(dist1.rand(gen)*2.0);
	}
	//integral
	double intg=0;
	for(int i=0; i<nbins; ++i){
		intg+=hist1.hist(i)/(hist1.c()-hist1.m());
	}
	//pdf
	double errpdf=0;
	for(int i=0; i<nbins; ++i){
		const double x=hist1.abscissa(i);
		const double pdfe=dist1.pdf(x);
		const double pdfa=hist1.hist(i)/(hist1.c()-hist1.m());
		errpdf+=fabs(pdfe-pdfa);
	}
	errpdf/=nbins;
	//write
	FILE* writer=fopen("test_math_random_normal2.dat","w");
	fprintf(writer,"#x pdf1 pdf2 hist1 hist2\n");
	for(int i=0; i<nbins; ++i){
		const double x=hist1.abscissa(i);
		fprintf(writer,"%f %f %f %f %f\n",x,dist1.pdf(x),dist2.pdf(x),hist1.hist(i)/(hist1.c()-hist1.m())/xdelta,hist2.hist(i)/(hist2.c()-hist2.m())/xdelta);
	}
	fclose(writer);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - MATH - RANDOM - NORMAL2",str)<<"\n";
	std::cout<<"nsample = "<<nsample<<"\n";
	std::cout<<"mu      = "<<mu<<"\n";
	std::cout<<"sigma   = "<<sigma<<"\n";
	std::cout<<"xlim    = "<<xmin<<":"<<xmax<<":"<<xdelta<<"\n";
	std::cout<<"error - intg = "<<std::fabs(1.0-intg)<<"\n";
	std::cout<<"error - pdf  = "<<errpdf<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_dist_logistic(){
	//generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	//distribution
	const double mu=0.0;
	//const double sigma=(double)gen()/gen.max()*(5.0-1.0)+1.0;
	const double sigma=sqrt(3.0/(PI*PI));
	rng::dist::Logistic dist(mu,sigma);
	//compute
	const int nsample=10000000;
	const int nbins=100;
	const double xmin=-7.0*sigma;
	const double xmax=7.0*sigma;
	const double xdelta=(xmax-xmin)/nbins;
	Histogram hist(xmin,xmax,nbins);
	for(int i=0; i<nsample; ++i){
		hist.push(dist.rand(gen));
	}
	//integral
	double intg=0;
	for(int i=0; i<nbins; ++i){
		intg+=hist.hist(i)/(hist.c()-hist.m());
	}
	//write
	FILE* writer=fopen("test_math_random_logistic.dat","w");
	fprintf(writer,"#x pdf hist\n");
	for(int i=0; i<nbins; ++i){
		const double x=hist.abscissa(i);
		fprintf(writer,"%f %f %f\n",x,dist.pdf(x),hist.hist(i)/(hist.c()-hist.m())/xdelta);
	}
	fclose(writer);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - MATH - RANDOM - LOGISTIC",str)<<"\n";
	std::cout<<"mu    = "<<mu<<"\n";
	std::cout<<"sigma = "<<sigma<<"\n";
	std::cout<<"xlim  = "<<xmin<<":"<<xmax<<":"<<xdelta<<"\n";
	std::cout<<"error - intg = "<<std::fabs(1.0-intg)<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_dist_sech(){
	//generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	//distribution
	const double mu=0.0;
	//const double sigma=(double)gen()/gen.max()*(5.0-1.0)+1.0;
	const double sigma=1.0;
	rng::dist::Sech dist(mu,sigma);
	//compute
	const int nsample=10000000;
	const int nbins=100;
	const double xmin=-7.0*sigma;
	const double xmax=7.0*sigma;
	const double xdelta=(xmax-xmin)/nbins;
	Histogram hist(xmin,xmax,nbins);
	for(int i=0; i<nsample; ++i){
		hist.push(dist.rand(gen));
	}
	//integral
	double intg=0;
	for(int i=0; i<nbins; ++i){
		intg+=hist.hist(i)/(hist.c()-hist.m());
	}
	//write
	FILE* writer=fopen("test_math_random_sech.dat","w");
	fprintf(writer,"#x pdf hist\n");
	for(int i=0; i<nbins; ++i){
		const double x=hist.abscissa(i);
		fprintf(writer,"%f %f %f\n",x,dist.pdf(x),hist.hist(i)/(hist.c()-hist.m())/xdelta);
	}
	fclose(writer);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - MATH - RANDOM - SECH",str)<<"\n";
	std::cout<<"mu    = "<<mu<<"\n";
	std::cout<<"sigma = "<<sigma<<"\n";
	std::cout<<"xlim  = "<<xmin<<":"<<xmax<<":"<<xdelta<<"\n";
	std::cout<<"error - intg = "<<std::fabs(1.0-intg)<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

void test_dist_cosine(){
	//generator
	unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);
	//distribution
	const double mu=0.0;
	//const double sigma=(double)gen()/gen.max()*(5.0-1.0)+1.0;
	const double sigma=2.0;
	rng::dist::Cosine dist(mu,sigma);
	//compute
	const int nsample=10000000;
	const int nbins=100;
	const double xmin=-2.0*sigma;
	const double xmax=2.0*sigma;
	const double xdelta=(xmax-xmin)/nbins;
	Histogram hist(xmin,xmax,nbins);
	for(int i=0; i<nsample; ++i){
		hist.push(dist.rand(gen));
	}
	//integral
	double intg=0;
	for(int i=0; i<nbins; ++i){
		intg+=hist.hist(i)/(hist.c()-hist.m());
	}
	//pdf
	double errpdf=0;
	for(int i=0; i<nbins; ++i){
		const double x=hist.abscissa(i);
		const double pdfe=dist.pdf(x);
		const double pdfa=hist.hist(i)/(hist.c()-hist.m());
		errpdf+=fabs(pdfe-pdfa);
	}
	errpdf/=nbins;
	//write
	FILE* writer=fopen("test_math_random_cosine.dat","w");
	fprintf(writer,"#x pdf hist\n");
	for(int i=0; i<nbins; ++i){
		const double x=hist.abscissa(i);
		fprintf(writer,"%f %f %f\n",x,dist.pdf(x),hist.hist(i)/(hist.c()-hist.m())/xdelta);
	}
	fclose(writer);
	//print
	char* str=new char[print::len_buf];
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	std::cout<<print::title("TEST - MATH - RANDOM - COSINE",str)<<"\n";
	std::cout<<"mu    = "<<mu<<"\n";
	std::cout<<"sigma = "<<sigma<<"\n";
	std::cout<<"xlim  = "<<xmin<<":"<<xmax<<":"<<xdelta<<"\n";
	std::cout<<"error - intg = "<<std::fabs(1.0-intg)<<"\n";
	std::cout<<"error - pdf  = "<<errpdf<<"\n";
	std::cout<<print::buf(str,print::char_buf)<<"\n";
	delete[] str;
}

int main(int argc, char*[]){
	
	test_dist_uniform();
	test_dist_laplace();
	test_dist_normal();
	test_dist_normal2();
	test_dist_logistic();
	test_dist_sech();
	test_dist_cosine();
	
	return 0;

}