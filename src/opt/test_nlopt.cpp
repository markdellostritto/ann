//c
#include <cstdlib>
#include <ctime>
//c++
#include <iostream>
//eigen
#include <Eigen/Dense>
//nlopt
#include <nlopt.hpp>

//**********************************************************************
// Rosenberg function
//**********************************************************************

struct Rosen{
	double a,b;
	Rosen():a(1.0),b(100.0){};
	Rosen(double aa, double bb):a(aa),b(bb){};
	double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& g){
		g[0]=-2.0*(a-x[0])-4.0*b*x[0]*(x[1]-x[0]*x[0]);
		g[1]=2.0*b*(x[1]-x[0]*x[0]);
		return (a-x[0])*(a-x[0])+b*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
	};
};

int count=0;
int step=10;
Rosen rosen;

double func_obj(const std::vector<double>& x, std::vector<double>& g, void* f_data){
    Eigen::VectorXd xv=Eigen::VectorXd::Zero(x.size());
    Eigen::VectorXd gv=Eigen::VectorXd::Zero(x.size());
    for(int i=0; i<x.size(); ++i) xv[i]=x[i];
    double val=rosen(xv,gv);
    if(count%step==0) std::cout<<"count "<<count<<" val "<<val<<"\n";
    ++count;
    return val;
}

int main(int argc, char* arvg[]){

    //optimization constants
    int miter=1000;
    double tol=1e-12;
    const int dim=2;
    std::cout<<"**** optimization constants ****\n";
    std::cout<<"miter = "<<miter<<"\n";
    std::cout<<"tol = "<<tol<<"\n";
    std::cout<<"dim = "<<dim<<"\n";

    // set bounds and parameters
    std::cout<<"**** setting bounds and parameters ****\n";
    std::vector<double> x(dim),lb(dim),ub(dim);
    std::srand(std::time(NULL));
    lb[0]=-10.0; lb[1]=-10.0;
    ub[0]=10.0; ub[1]=10.0;
    x[0]=std::rand()*1.0/RAND_MAX*(ub[0]-lb[0])+lb[0];
    x[1]=std::rand()*1.0/RAND_MAX*(ub[1]-lb[1])+lb[1];

    // making optimizer
    std::cout<<"**** making optimizer ****\n";
    //nlopt::opt opt(nlopt::LN_SBPLX,dim); 
    nlopt::opt opt(nlopt::LN_BOBYQA,dim); 
    opt.set_min_objective(func_obj,NULL);
    opt.set_ftol_abs(tol);
    opt.set_maxeval(miter);
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    double minf;

    // optimize
    std::cout<<"**** optimize ****\n";
    nlopt::result result=opt.optimize(x,minf);
    if(result>=0) std::cout<<"optimization successful\n";
    else std::cout<<"optimization unsuccessful\n";
    const double val=opt.last_optimum_value();
    std::cout<<"value = "<<val<<"\n";
    std::cout<<"x = "; for(int i=0; i<dim; ++i) std::cout<<x[i]<<" "; std::cout<<"\n";

    return 0;
}