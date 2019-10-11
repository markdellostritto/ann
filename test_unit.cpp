#include "test_unit.hpp"

//**********************************************
// math_special
//**********************************************

void test_math_special_cos(){
	//local variables
	const unsigned int N=10000;
	const double xmin=0;
	const double xmax=num_const::PI;
	std::vector<double> x(N,0);
	std::vector<double> cos_exact(N,0);
	std::vector<double> cos_approx(N,0);
	double time_exact,time_approx;
	clock_t start,stop;
	
	//generate abscissae
	for(int i=N-1; i>=0; --i) x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
	//generate exact ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) cos_exact[i]=std::cos(x[i]);
	stop=std::clock();
	time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
	//generate approximate ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) cos_approx[i]=special::cos(x[i]);
	stop=std::clock();
	time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
	//compute error
	double err_avg=0,err_max=0;
	for(int i=N-1; i>=0; --i){
		double err=std::fabs(cos_approx[i]-cos_exact[i]);
		err_avg+=err;
		err_max=(err>err_max)?err:err_max;
	}
	err_avg/=N;
	
	time_exact*=1e9/N;
	time_approx*=1e9/N;
	
	//print results
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - MATH - SPECIAL - COS\n";
	std::cout<<"N             = "<<N<<"\n";
	std::cout<<"interval      = "<<xmin<<" "<<xmax<<"\n";
	std::cout<<"error - avg   = "<<err_avg<<"\n";
	std::cout<<"error - max   = "<<err_max<<"\n";
	std::cout<<"time - exact  = "<<time_exact<<"\n";
	std::cout<<"time - approx = "<<time_approx<<"\n";
	std::cout<<"====================================================\n";
}

void test_math_special_sin(){
	//local variables
	const unsigned int N=10000;
	const double xmin=0;
	const double xmax=num_const::PI;
	std::vector<double> x(N,0);
	std::vector<double> sin_exact(N,0);
	std::vector<double> sin_approx(N,0);
	double time_exact,time_approx;
	clock_t start,stop;
	
	//generate abscissae
	for(int i=N-1; i>=0; --i) x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
	//generate exact ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) sin_exact[i]=std::sin(x[i]);
	stop=std::clock();
	time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
	//generate approximate ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) sin_approx[i]=special::sin(x[i]);
	stop=std::clock();
	time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
	//compute error
	double err_avg=0,err_max=0;
	for(int i=N-1; i>=0; --i){
		double err=std::fabs(sin_approx[i]-sin_exact[i]);
		err_avg+=err;
		err_max=(err>err_max)?err:err_max;
	}
	err_avg/=N;
	
	time_exact*=1e9/N;
	time_approx*=1e9/N;
	
	//print results
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - MATH - SPECIAL - SIN\n";
	std::cout<<"N             = "<<N<<"\n";
	std::cout<<"interval      = "<<xmin<<" "<<xmax<<"\n";
	std::cout<<"error - avg   = "<<err_avg<<"\n";
	std::cout<<"error - max   = "<<err_max<<"\n";
	std::cout<<"time - exact  = "<<time_exact<<"\n";
	std::cout<<"time - approx = "<<time_approx<<"\n";
	std::cout<<"====================================================\n";
}

void test_math_special_logp1(){
	//local variables
	const unsigned int N=10000;
	const double xmin=-0.5;
	const double xmax=1.0;
	std::vector<double> x(N,0);
	std::vector<double> exact(N,0);
	std::vector<double> approx(N,0);
	double time_exact,time_approx;
	clock_t start,stop;
	
	//generate abscissae
	for(int i=N-1; i>=0; --i) x[i]=(xmax-xmin)*i/(1.0*N)+xmin;
	//generate exact ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) exact[i]=std::log(x[i]+1.0);
	stop=std::clock();
	time_exact=((double)(stop-start))/CLOCKS_PER_SEC;
	//generate approximate ordinates
	start=std::clock();
	for(int i=N-1; i>=0; --i) approx[i]=special::logp1(x[i]);
	stop=std::clock();
	time_approx=((double)(stop-start))/CLOCKS_PER_SEC;
	//compute error
	double err_avg=0,err_max=0;
	for(int i=N-1; i>=0; --i){
		double err=std::fabs(approx[i]-exact[i]);
		err_avg+=err;
		err_max=(err>err_max)?err:err_max;
	}
	err_avg/=N;
	
	time_exact*=1e9/N;
	time_approx*=1e9/N;
	
	//print results
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - MATH - SPECIAL - LOGP1\n";
	std::cout<<"N             = "<<N<<"\n";
	std::cout<<"interval      = "<<xmin<<" "<<xmax<<"\n";
	std::cout<<"error - avg   = "<<err_avg<<"\n";
	std::cout<<"error - max   = "<<err_max<<"\n";
	std::cout<<"time - exact  = "<<time_exact<<"\n";
	std::cout<<"time - approx = "<<time_approx<<"\n";
	std::cout<<"====================================================\n";
}

//**********************************************
// cutoff
//**********************************************

void test_cutoff_cos(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double integral=0.5*(CutoffF::cut_cos(0.0,rc)+CutoffF::cut_cos(rc,rc));
	for(int i=N-2; i>=1; --i){
		integral+=CutoffF::cut_cos(i/(N-1.0)*rc,rc);
	}
	integral*=dr;
	double errg=0;
	for(int i=N-2; i>=1; --i){
		const double g=0.5*(CutoffF::cut_cos((i+1.0)/(N-1.0)*rc,rc)-CutoffF::cut_cos((i-1.0)/(N-1.0)*rc,rc))/dr;
		errg+=std::fabs(g-CutoffFD::cut_cos(i/(N-1.0)*rc,rc));
	}
	errg/=(N-1.0);
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - CUTOFF - COS\n";
	std::cout<<"err - integral = "<<std::fabs(0.5*rc-integral)<<"\n";
	std::cout<<"err - gradient = "<<errg<<"\n";
	std::cout<<"====================================================\n";
}

//**********************************************
// symm
//**********************************************

void test_symm_t1(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double errg=0,r;
	PhiR_T1 t1(1.4268,0.56278905);
	for(int i=N-2; i>=1; --i){
		r=(i+1.0)/(N-1.0)*rc;
		const double f2=t1.val(r,CutoffF::cut_cos(r,rc));
		r=(i-1.0)/(N-1.0)*rc;
		const double f1=t1.val(r,CutoffF::cut_cos(r,rc));
		const double g=0.5*(f2-f1)/dr;
		r=i/(N-1.0)*rc;
		const double cut=CutoffF::cut_cos(r,rc);
		const double gcut=CutoffFD::cut_cos(r,rc);
		errg+=std::fabs(g-t1.grad(r,cut,gcut));
	}
	errg/=(N-1.0);
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - SYMM - T1\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"====================================================\n";
}

void test_symm_g1(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double errg=0,r;
	PhiR_G1 g1;
	for(int i=N-2; i>=1; --i){
		r=(i+1.0)/(N-1.0)*rc;
		const double f2=g1.val(r,CutoffF::cut_cos(r,rc));
		r=(i-1.0)/(N-1.0)*rc;
		const double f1=g1.val(r,CutoffF::cut_cos(r,rc));
		const double g=0.5*(f2-f1)/dr;
		r=i/(N-1.0)*rc;
		const double cut=CutoffF::cut_cos(r,rc);
		const double gcut=CutoffFD::cut_cos(r,rc);
		errg+=std::fabs(g-g1.grad(r,cut,gcut));
	}
	errg/=(N-1.0);
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - SYMM - G1\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"====================================================\n";
}

void test_symm_g2(){
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double errg=0,r;
	PhiR_G2 g2(1.4268,0.56278905);
	for(int i=N-2; i>=1; --i){
		r=(i+1.0)/(N-1.0)*rc;
		const double f2=g2.val(r,CutoffF::cut_cos(r,rc));
		r=(i-1.0)/(N-1.0)*rc;
		const double f1=g2.val(r,CutoffF::cut_cos(r,rc));
		const double g=0.5*(f2-f1)/dr;
		r=i/(N-1.0)*rc;
		const double cut=CutoffF::cut_cos(r,rc);
		const double gcut=CutoffFD::cut_cos(r,rc);
		errg+=std::fabs(g-g2.grad(r,cut,gcut));
	}
	errg/=(N-1.0);
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - SYMM - G2\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad = "<<errg<<"\n";
	std::cout<<"====================================================\n";
}

void test_symm_g3(){
	//local variables
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double errgd[3]={0,0,0};
	double errga=0;
	double r[3],c[3];
	PhiA_G3 g3(1.4268,2.5,1);
	const double cos=1.0/std::sqrt(2.0);
	//grad - dist - 0
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=CutoffF::cut_cos(r[0],rc); c[1]=CutoffF::cut_cos(r[1],rc); c[2]=CutoffF::cut_cos(r[2],rc);
		//first point
		r[0]=(i+1.0)/(N-1.0)*rc; c[0]=CutoffF::cut_cos(r[0],rc);
		const double f2=g3.val(cos,r,c);
		//second point
		r[0]=(i-1.0)/(N-1.0)*rc; c[0]=CutoffF::cut_cos(r[0],rc);
		const double f1=g3.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[0]=i/(N-1.0)*rc; c[0]=CutoffF::cut_cos(r[0],rc);
		const double gcut=CutoffFD::cut_cos(r[0],rc);
		//error
		errgd[0]+=std::fabs(g-g3.grad_dist_0(r,c,gcut)*g3.angle(cos));
	}
	errgd[0]/=(N-1.0);
	//grad - dist - 1
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=CutoffF::cut_cos(r[0],rc); c[1]=CutoffF::cut_cos(r[1],rc); c[2]=CutoffF::cut_cos(r[2],rc);
		//first point
		r[1]=(i+1.0)/(N-1.0)*rc; c[1]=CutoffF::cut_cos(r[1],rc);
		const double f2=g3.val(cos,r,c);
		//second point
		r[1]=(i-1.0)/(N-1.0)*rc; c[1]=CutoffF::cut_cos(r[1],rc);
		const double f1=g3.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[1]=i/(N-1.0)*rc; c[1]=CutoffF::cut_cos(r[1],rc);
		const double gcut=CutoffFD::cut_cos(r[1],rc);
		//error
		errgd[1]+=std::fabs(g-g3.grad_dist_1(r,c,gcut)*g3.angle(cos));
	}
	errgd[1]/=(N-1.0);
	//grad - dist - 2
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=CutoffF::cut_cos(r[0],rc); c[1]=CutoffF::cut_cos(r[1],rc); c[2]=CutoffF::cut_cos(r[2],rc);
		//first point
		r[2]=(i+1.0)/(N-1.0)*rc; c[2]=CutoffF::cut_cos(r[2],rc);
		const double f2=g3.val(cos,r,c);
		//second point
		r[2]=(i-1.0)/(N-1.0)*rc; c[2]=CutoffF::cut_cos(r[2],rc);
		const double f1=g3.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[2]=i/(N-1.0)*rc; c[2]=CutoffF::cut_cos(r[2],rc);
		const double gcut=CutoffFD::cut_cos(r[2],rc);
		//error
		errgd[2]+=std::fabs(g-g3.grad_dist_2(r,c,gcut)*g3.angle(cos));
	}
	errgd[2]/=(N-1.0);
	//grad - angle
	for(int i=N-2; i>=1; --i){
		double cosv;
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=CutoffF::cut_cos(r[0],rc); c[1]=CutoffF::cut_cos(r[1],rc); c[2]=CutoffF::cut_cos(r[2],rc);
		//first point
		cosv=(i+1.0)/(N-1.0)*num_const::PI;
		const double f2=g3.val(cosv,r,c);
		//second point
		cosv=(i-1.0)/(N-1.0)*num_const::PI;
		const double f1=g3.val(cosv,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		cosv=i/(N-1.0)*num_const::PI;
		const double gcut=CutoffFD::cut_cos(r[2],rc);
		//error
		errga+=std::fabs(g-g3.grad_angle(cosv)*g3.dist(r,c));
	}
	errga/=(N-1.0);
	//print
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - SYMM - G3\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - grad - dist[0] = "<<errgd[0]<<"\n";
	std::cout<<"err - grad - dist[1] = "<<errgd[1]<<"\n";
	std::cout<<"err - grad - dist[2] = "<<errgd[2]<<"\n";
	std::cout<<"err - grad - angle   = "<<errga<<"\n";
	std::cout<<"====================================================\n";
}

void test_symm_g4(){
	//local variables
	const double rc=6.0;
	const int N=100;
	const double dr=rc/(N-1.0);
	double errgd[3]={0,0,0};
	double errga=0;
	double r[3],c[3];
	PhiA_G4 g4(1.4268,2.5,1);
	const double cos=1.0/std::sqrt(2.0);
	//grad - dist - 0
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=CutoffF::cut_cos(r[0],rc); c[1]=CutoffF::cut_cos(r[1],rc); c[2]=CutoffF::cut_cos(r[2],rc);
		//first point
		r[0]=(i+1.0)/(N-1.0)*rc; c[0]=CutoffF::cut_cos(r[0],rc);
		const double f2=g4.val(cos,r,c);
		//second point
		r[0]=(i-1.0)/(N-1.0)*rc; c[0]=CutoffF::cut_cos(r[0],rc);
		const double f1=g4.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[0]=i/(N-1.0)*rc; c[0]=CutoffF::cut_cos(r[0],rc);
		const double gcut=CutoffFD::cut_cos(r[0],rc);
		//error
		errgd[0]+=std::fabs(g-g4.grad_dist_0(r,c,gcut)*g4.angle(cos));
	}
	errgd[0]/=(N-1.0);
	//grad - dist - 1
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=CutoffF::cut_cos(r[0],rc); c[1]=CutoffF::cut_cos(r[1],rc); c[2]=CutoffF::cut_cos(r[2],rc);
		//first point
		r[1]=(i+1.0)/(N-1.0)*rc; c[1]=CutoffF::cut_cos(r[1],rc);
		const double f2=g4.val(cos,r,c);
		//second point
		r[1]=(i-1.0)/(N-1.0)*rc; c[1]=CutoffF::cut_cos(r[1],rc);
		const double f1=g4.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[1]=i/(N-1.0)*rc; c[1]=CutoffF::cut_cos(r[1],rc);
		const double gcut=CutoffFD::cut_cos(r[1],rc);
		//error
		errgd[1]+=std::fabs(g-g4.grad_dist_1(r,c,gcut)*g4.angle(cos));
	}
	errgd[1]/=(N-1.0);
	//grad - dist - 2
	for(int i=N-2; i>=1; --i){
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=CutoffF::cut_cos(r[0],rc); c[1]=CutoffF::cut_cos(r[1],rc); c[2]=CutoffF::cut_cos(r[2],rc);
		//first point
		r[2]=(i+1.0)/(N-1.0)*rc; c[2]=CutoffF::cut_cos(r[2],rc);
		const double f2=g4.val(cos,r,c);
		//second point
		r[2]=(i-1.0)/(N-1.0)*rc; c[2]=CutoffF::cut_cos(r[2],rc);
		const double f1=g4.val(cos,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		r[2]=i/(N-1.0)*rc; c[2]=CutoffF::cut_cos(r[2],rc);
		const double gcut=CutoffFD::cut_cos(r[2],rc);
		errgd[2]+=std::fabs(g-g4.grad_dist_2(r,c,gcut)*g4.angle(cos));
	}
	errgd[2]/=(N-1.0);
	//grad - angle
	for(int i=N-2; i>=1; --i){
		double cosv;
		//assign arrays
		r[0]=1.472896; r[1]=1.5728; r[2]=1.9587892;
		c[0]=CutoffF::cut_cos(r[0],rc); c[1]=CutoffF::cut_cos(r[1],rc); c[2]=CutoffF::cut_cos(r[2],rc);
		//first point
		cosv=(i+1.0)/(N-1.0)*num_const::PI;
		const double f2=g4.val(cosv,r,c);
		//second point
		cosv=(i-1.0)/(N-1.0)*num_const::PI;
		const double f1=g4.val(cosv,r,c);
		//gradient - approx
		const double g=0.5*(f2-f1)/dr;
		//gradient - exact
		cosv=i/(N-1.0)*num_const::PI;
		errga+=std::fabs(g-g4.grad_angle(cosv)*g4.dist(r,c));
	}
	errga/=(N-1.0);
	//print
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - SYMM - G4\n";
	std::cout<<"N  = "<<N<<"\n";
	std::cout<<"rc = "<<rc<<"\n";
	std::cout<<"err - gradient - dist[0] = "<<errgd[0]<<"\n";
	std::cout<<"err - gradient - dist[1] = "<<errgd[1]<<"\n";
	std::cout<<"err - gradient - dist[2] = "<<errgd[2]<<"\n";
	std::cout<<"err - gradient - angle   = "<<errga<<"\n";
	std::cout<<"====================================================\n";
}

//**********************************************
// eigen
//**********************************************

void test_unit_eigen_vec3d(){
	unsigned int memsize=0;
	char* memarr=NULL;
	Eigen::Vector3d vec1=Eigen::Vector3d::Random();
	Eigen::Vector3d vec2=Eigen::Vector3d::Zero();
	memsize=serialize::nbytes(vec1);
	memarr=new char[memsize];
	serialize::pack(vec1,memarr);
	serialize::unpack(vec2,memarr);
	delete memarr;
	double err=(vec1-vec2).norm();
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - EIGEN - VECTOR3D\n";
	std::cout<<"err - serialization = "<<err<<"\n";
	std::cout<<"====================================================\n";
}

void test_unit_eigen_vecxd(){
	unsigned int memsize=0;
	char* memarr=NULL;
	Eigen::VectorXd vec1=Eigen::VectorXd::Random(7);
	Eigen::VectorXd vec2;
	memsize=serialize::nbytes(vec1);
	memarr=new char[memsize];
	serialize::pack(vec1,memarr);
	serialize::unpack(vec2,memarr);
	delete memarr;
	double err=(vec1-vec2).norm();
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - EIGEN - VECTORXD\n";
	std::cout<<"err - serialization = "<<err<<"\n";
	std::cout<<"====================================================\n";
}

void test_unit_eigen_mat3d(){
	unsigned int memsize=0;
	char* memarr=NULL;
	Eigen::Matrix3d vec1=Eigen::Matrix3d::Random();
	Eigen::Matrix3d vec2=Eigen::Matrix3d::Zero();
	memsize=serialize::nbytes(vec1);
	memarr=new char[memsize];
	serialize::pack(vec1,memarr);
	serialize::unpack(vec2,memarr);
	delete memarr;
	double err=(vec1-vec2).norm();
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - EIGEN - MATRIX3D\n";
	std::cout<<"err - serialization = "<<err<<"\n";
	std::cout<<"====================================================\n";
}

void test_unit_eigen_matxd(){
	unsigned int memsize=0;
	char* memarr=NULL;
	Eigen::MatrixXd vec1=Eigen::MatrixXd::Random(7,7);
	Eigen::MatrixXd vec2;
	memsize=serialize::nbytes(vec1);
	memarr=new char[memsize];
	serialize::pack(vec1,memarr);
	serialize::unpack(vec2,memarr);
	delete memarr;
	double err=(vec1-vec2).norm();
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - EIGEN - MATRIXXD\n";
	std::cout<<"err - serialization = "<<err<<"\n";
	std::cout<<"====================================================\n";
}

//**********************************************
// optimize
//**********************************************

void test_unit_opt_sgd(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::SGD;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::SGD sgd(data.dim());
	sgd.decay()=1;
	sgd.gamma()=1e-3;
	//optimize
	for(unsigned int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		sgd.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		if(data.optVal()==Opt::VAL::FTOL_REL && data.dv()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::XTOL_REL && data.dp()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::FTOL_ABS && data.val()<data.tol()) break;
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - SGD\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"====================================================\n";
}

void test_unit_opt_sdm(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::SDM;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::SDM sdm(data.dim());
	sdm.decay()=1;
	sdm.gamma()=1e-3;
	sdm.eta()=0.9;
	//optimize
	for(unsigned int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		sdm.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		if(data.optVal()==Opt::VAL::FTOL_REL && data.dv()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::XTOL_REL && data.dp()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::FTOL_ABS && data.val()<data.tol()) break;
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - SDM\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"====================================================\n";
}

void test_unit_opt_nag(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::NAG;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::NAG nag(data.dim());
	nag.decay()=1;
	nag.gamma()=1e-3;
	nag.eta()=0.9;
	//optimize
	for(unsigned int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		nag.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		if(data.optVal()==Opt::VAL::FTOL_REL && data.dv()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::XTOL_REL && data.dp()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::FTOL_ABS && data.val()<data.tol()) break;
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - NAG\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"====================================================\n";
}

void test_unit_opt_adam(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::ADAM;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::ADAM adam(data.dim());
	adam.decay()=1;
	adam.gamma()=1e-3;
	//optimize
	for(unsigned int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		adam.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		if(data.optVal()==Opt::VAL::FTOL_REL && data.dv()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::XTOL_REL && data.dp()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::FTOL_ABS && data.val()<data.tol()) break;
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - ADAM\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"====================================================\n";
}

void test_unit_opt_nadam(){
	//rosenberg function
	Rosen rosen;
	//init data
	Opt::Data data(2);
	data.p()[0]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.p()[1]=2.0*(1.0*std::rand()/RAND_MAX-0.5);
	data.pOld()=data.p();
	data.max()=100000;
	data.nPrint()=100;
	data.algo()=Opt::ALGO::NADAM;
	data.optVal()=Opt::VAL::FTOL_ABS;
	data.tol()=1e-8;
	//init optimizer
	Opt::NADAM nadam(data.dim());
	nadam.decay()=1;
	nadam.gamma()=1e-3;
	//optimize
	for(unsigned int i=0; i<data.max(); ++i){
		//compute the value and gradient
		data.val()=rosen(data.p(),data.g());
		//compute the new position
		nadam.step(data);
		//calculate the difference
		data.dv()=std::fabs(data.val()-data.valOld());
		data.dp()=(data.p()-data.pOld()).norm();
		//check the break condition
		if(data.optVal()==Opt::VAL::FTOL_REL && data.dv()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::XTOL_REL && data.dp()<data.tol()) break;
		else if(data.optVal()==Opt::VAL::FTOL_ABS && data.val()<data.tol()) break;
		//print the status
		data.pOld().noalias()=data.p();//set "old" p value
		data.gOld().noalias()=data.g();//set "old" g value
		data.valOld()=data.val();//set "old" value
		//update step
		++data.step();
	}
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - OPTIMIZE - NADAM\n";
	std::cout<<"n_steps = "<<data.step()<<"\n";
	std::cout<<"val     = "<<data.val()<<"\n";
	std::cout<<"x       = "<<data.p().transpose()<<"\n";
	std::cout<<"====================================================\n";
}

//**********************************************
// ewald
//**********************************************

void test_unit_ewald_madelung(){
	//generate NaCl crystal
	units::System::type unitsys=units::System::AU;
	units::consts::init(unitsys);
	double a0=5.6199998856;
	if(unitsys==units::System::AU) a0*=units::BOHRpANG;
	AtomType atomT; 
	atomT.name=true; atomT.an=true; atomT.index=true; atomT.type=true;
	atomT.charge=true; atomT.posn=true;
	Structure strucg;
	const unsigned int nspecies=2;
	std::vector<unsigned int> natoms(nspecies,4);
	std::vector<std::string> names(nspecies);
	names[0]="Na"; names[1]="Cl";
	strucg.resize(natoms,names,atomT);
	Eigen::Matrix3d lv=Eigen::Matrix3d::Identity()*a0;
	strucg.init(lv);
	strucg.posn(0)<<0.000000000,0.000000000,0.000000000;
	strucg.posn(1)<<0.000000000,0.500000000,0.500000000;
	strucg.posn(2)<<0.500000000,0.000000000,0.500000000;
	strucg.posn(3)<<0.500000000,0.500000000,0.000000000;
	strucg.posn(4)<<0.500000000,0.500000000,0.500000000;
	strucg.posn(5)<<0.500000000,0.000000000,0.000000000;
	strucg.posn(6)<<0.000000000,0.500000000,0.000000000;
	strucg.posn(7)<<0.000000000,0.000000000,0.500000000;
	for(unsigned int i=0; i<8; ++i) strucg.posn(i)*=a0;
	for(unsigned int i=0; i<4; ++i) strucg.charge(i)=1;
	for(unsigned int i=4; i<8; ++i) strucg.charge(i)=-1;
	for(unsigned int i=0; i<4; ++i) strucg.name(i)="Na";
	for(unsigned int i=4; i<8; ++i) strucg.name(i)="Cl";
	//ewald
	Ewald3D::Coulomb ewald;
	static const double mc=1.74756;
	//modify the positions by the nn distance
	double rmin=100;
	for(unsigned int i=0; i<strucg.nAtoms(); ++i){
		for(unsigned int j=i+1; j<strucg.nAtoms(); ++j){
			double r=Cell::dist(strucg.posn(i),strucg.posn(j),strucg.R(),strucg.RInv());
			if(r<rmin) rmin=r;
		}
	}
	//initialize the ewald object
	ewald.init(strucg,1e-6);
	//compute the total energy
	const double mce=-2*ewald.energy(strucg)/strucg.nAtoms()*rmin/units::consts::ke();
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - EWALD - MADELUNG\n";
	std::cout<<"madelung constant = "<<mc<<"\n";
	std::cout<<"madelung (ewald)  = "<<mce<<"\n";
	std::cout<<"error (percent)   = "<<std::fabs((mce-mc)/mc*100.0)<<"\n";
	std::cout<<"====================================================\n";
}

//**********************************************
// string
//**********************************************

void test_unit_string_hash(){
	const char* str1="lnt$*(nGnlj#o*t^nuoVho$&lks";
	const char* str2="lnt$*(nGnlj#o*t^nuoVho$&lks";
	const char* str3="lnt$*(nGnlj#o*t^nCoVho$&lks";
	const unsigned int hash1=string::hash(str1);
	const unsigned int hash2=string::hash(str2);
	const unsigned int hash3=string::hash(str3);
	std::cout<<"====================================================\n";
	std::cout<<"TEST - UNIT - STRING - HASH\n";
	std::cout<<"err -     equal = "<<std::fabs(1.0*hash1-1.0*hash2)<<"\n";
	std::cout<<"err - non-equal = "<<1.0/std::fabs(1.0*hash1-1.0*hash3)<<"\n";
	std::cout<<"====================================================\n";
}

//**********************************************
// nn
//**********************************************

void test_unit_nn_out(){
	//local function variables
	NN::Network nn;
	//resize the nn
	nn.tfType()=NN::TransferN::TANH;
	std::vector<unsigned int> nh(2);
	nh[0]=7; nh[1]=5;
	nn.resize(2,nh,3);
	//init rand
	std::srand(std::time(NULL));
	const unsigned int N=100;
	double errv=0;
	for(unsigned int n=0; n<N; ++n){
		//initialize the input nodes
		for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=std::rand()/RAND_MAX-0.5;
		//execute the network
		nn.execute();
		//compute error
		Eigen::VectorXd vec1,vec2,vec3;
		Eigen::VectorXd grad1,grad2,grad3;
		vec1=nn.edge(0)*nn.input()+nn.bias(0); grad1=vec1;
		nn.tfdv(0)(vec1,grad1);
		vec2=nn.edge(1)*vec1+nn.bias(1); grad2=vec2;
		nn.tfdv(1)(vec2,grad2);
		vec3=nn.edge(2)*vec2+nn.bias(2); grad3=vec3;
		nn.tfdv(2)(vec3,grad3);
		errv+=(vec3-nn.output()).norm();
	}
	errv/=N;
	std::cout<<"====================================================\n";
	std::cout<<"TEST - NN - OUT\n";
	std::cout<<"transfer = "<<nn.tfType()<<"\n";
	std::cout<<"config   = "<<nn.nInput()<<" "; for(unsigned int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOutput()<<"\n";
	std::cout<<"err      = "<<errv<<"\n";
	std::cout<<"====================================================\n";
}

void test_unit_nn_grad(){
	//local function variables
	const unsigned int N=100;
	NN::Network nn;
	//resize the nn
	Eigen::MatrixXd dOutExact,dOutApprox;
	//nn.tfType()=NN::TransferN::LINEAR;
	nn.tfType()=NN::TransferN::TANH;
	std::vector<unsigned int> nh(2);
	nh[0]=7; nh[1]=5;
	nn.resize(4,nh,3);
	//initialize the input nodes
	for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=std::rand()/RAND_MAX-0.5;
	Eigen::VectorXd input=nn.input();
	//execute the network, compute analytic gradient
	nn.execute();
	nn.grad_out();
	dOutApprox=nn.dOut(0);
	//compute brute force gradient
	double err=0;
	dOutExact=Eigen::MatrixXd::Zero(nn.nOutput(),nn.nInput());
	for(unsigned int m=0; m<N; ++m){
		Eigen::VectorXd delta=Eigen::VectorXd::Random(nn.nInput())/100.0;
		for(unsigned int n=0; n<nn.nInput(); ++n){
			//point 1
			for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=input[i];
			nn.input(n)+=delta[n];
			nn.execute();
			Eigen::VectorXd outNew1=nn.output();
			//point 2
			for(unsigned int i=0; i<nn.nInput(); ++i) nn.input(i)=input[i];
			nn.input(n)-=delta[n];
			nn.execute();
			Eigen::VectorXd outNew2=nn.output();
			//gradient
			for(unsigned int i=0; i<nn.output().size(); ++i){
				dOutExact(i,n)=0.5*(outNew1[i]-outNew2[i])/delta[n];
			}
		}
		err+=(dOutExact-dOutApprox).norm();
	}
	err/=N;
	std::cout<<"====================================================\n";
	std::cout<<"TEST - NN - GRAD\n";
	std::cout<<"transfer = "<<nn.tfType()<<"\n";
	std::cout<<"config   = "<<nn.nInput()<<" "; for(unsigned int i=0; i<nh.size(); ++i) std::cout<<nh[i]<<" "; std::cout<<nn.nOutput()<<"\n";
	std::cout<<"err      = "<<err<<"\n";
	std::cout<<"====================================================\n";
}

//**********************************************
// main
//**********************************************

int main(int argc, char* argv[]){
	
	std::srand(std::time(NULL));
	
	std::cout<<"*********************************************************\n";
	std::cout<<"********************* MATH - CONST *********************\n";
	std::cout<<"ZERO  = "<<num_const::ZERO<<"\n";
	std::cout<<"PI    = "<<num_const::PI<<"\n";
	std::cout<<"RadPI = "<<num_const::RadPI<<"\n";
	std::cout<<"Rad2  = "<<num_const::Rad2<<"\n";
	std::cout<<"Rad3  = "<<num_const::Rad3<<"\n";
	std::cout<<"E     = "<<num_const::E<<"\n";
	std::cout<<"PHI   = "<<num_const::PHI<<"\n";
	std::cout<<"LOG2  = "<<num_const::LOG2<<"\n";
	std::cout<<"********************* MATH - CONST *********************\n";
	std::cout<<"*********************************************************\n";
	
	std::cout<<"*********************************************************\n";
	std::cout<<"******************** MATH - SPECIAL ********************\n";
	test_math_special_cos();
	test_math_special_sin();
	test_math_special_logp1();
	std::cout<<"******************** MATH - SPECIAL ********************\n";
	std::cout<<"*********************************************************\n";
	
	std::cout<<"*********************************************************\n";
	std::cout<<"************************ CUTOFF ************************\n";
	test_cutoff_cos();
	std::cout<<"************************ CUTOFF ************************\n";
	std::cout<<"*********************************************************\n";
	
	std::cout<<"*********************************************************\n";
	std::cout<<"************************* SYMM *************************\n";
	test_symm_t1();
	test_symm_g1();
	test_symm_g2();
	test_symm_g3();
	test_symm_g4();
	std::cout<<"************************* SYMM *************************\n";
	std::cout<<"*********************************************************\n";
	
	std::cout<<"*********************************************************\n";
	std::cout<<"************************* EIGEN *************************\n";
	test_unit_eigen_vec3d();
	test_unit_eigen_vecxd();
	test_unit_eigen_mat3d();
	test_unit_eigen_matxd();
	std::cout<<"************************* EIGEN *************************\n";
	std::cout<<"*********************************************************\n";
	
	std::cout<<"*********************************************************\n";
	std::cout<<"*********************** OPTIMIZE ***********************\n";
	std::cout<<"ROSENBERG\n";
	std::cout<<"OPT - V = 0.0\n";
	std::cout<<"OPT - X = (1,1)\n";
	//test_unit_opt_sgd();
	test_unit_opt_sdm();
	test_unit_opt_nag();
	test_unit_opt_adam();
	test_unit_opt_nadam();
	std::cout<<"*********************** OPTIMIZE ***********************\n";
	std::cout<<"*********************************************************\n";
	
	std::cout<<"*********************************************************\n";
	std::cout<<"************************* EWALD *************************\n";
	test_unit_ewald_madelung();
	std::cout<<"************************* EWALD *************************\n";
	std::cout<<"*********************************************************\n";
	
	std::cout<<"*********************************************************\n";
	std::cout<<"************************ STRING ************************\n";
	test_unit_string_hash();
	std::cout<<"************************ STRING ************************\n";
	std::cout<<"*********************************************************\n";
	
	std::cout<<"*********************************************************\n";
	std::cout<<"************************** NN **************************\n";
	test_unit_nn_out();
	test_unit_nn_grad();
	std::cout<<"************************** NN **************************\n";
	std::cout<<"*********************************************************\n";
	
}