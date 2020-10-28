
static const char char_buf='-';

struct LJ{
private:
	double eps_,sigma_;
public:
	//==== constructors/destructors ====
	LJ(){}
	LJ(double eps, double sigma):eps_(eps),sigma_(sigma){}
	~LJ(){}
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& sigma(){return sigma_;}
	const double& sigma()const{return sigma_;}
	
	//==== operators ====
	double operator()(double r){
		const double x=sigma_/r;
		const double x6=x*x*x*x*x*x;
		return 4.0*eps_*(x6*x6-x6);
	}
};

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
	
//**********************************************
// math_special
//**********************************************

void test_math_special_cos();
void test_math_special_sin();
void test_math_special_logp1();

//**********************************************
// accumulator - 1D
//**********************************************

void test_acc_1D();
void test_acc_2D();

//**********************************************
// cutoff
//**********************************************

void test_cutoff_cos();

//**********************************************
// symm
//**********************************************

void test_symm_g2();
void test_symm_t1();
void test_symm_g3();

//**********************************************
// eigen
//**********************************************

void test_unit_eigen_vec3d();
void test_unit_eigen_vecxd();
void test_unit_eigen_mat3d();
void test_unit_eigen_matxd();

//**********************************************
// optimize
//**********************************************

void test_unit_opt_sgd();
void test_unit_opt_sdm();
void test_unit_opt_nag();
void test_unit_opt_adam();
void test_unit_opt_nadam();

//**********************************************
// ewald
//**********************************************

void test_unit_ewald_madelung();
void test_unit_ewald_potential();

//**********************************************
// string
//**********************************************

void test_unit_string_hash();

//**********************************************
// random
//**********************************************

void test_random_time();
void test_random_dist();

//**********************************************
// nn
//**********************************************

void test_unit_nn();
void test_unit_nn_tfunc();
void test_unit_nn_out();
void test_unit_nn_grad();
void test_unit_nn_time();

//**********************************************
// nn pot
//**********************************************

void test_unit_nnh();
void test_unit_nnp();

//**********************************************
// structure
//**********************************************

void test_unit_struc();
void test_cell_list_square();