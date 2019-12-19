
static const char char_buf='-';

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
// nn
//**********************************************

void test_unit_nn_out();
void test_unit_nn_grad();