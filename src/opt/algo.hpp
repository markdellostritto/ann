#pragma once
#ifndef ALGO_HPP
#define ALGO_HPP

// c++
#include <iosfwd>
#include <memory>
// eigen
#include <Eigen/Dense>
// memory
#include "mem/serialize.hpp"
// opt
#include "opt/objective.hpp"
// str
#include "str/string.hpp"
#include "str/token.hpp"

#ifndef OPT_ALGO_PRINT_FUNC
#define OPT_ALGO_PRINT_FUNC 0
#endif

namespace opt{
namespace algo{

//***************************************************
// name
//***************************************************

class Name{
public:
	enum Type{
		UNKNOWN,
		SGD,
		SDM,
		NAG,
		ADAGRAD,
		ADADELTA,
		RMSPROP,
		ADAM,
		ADAMW,
		ADAB,
		YOGI,
		NOGI,
		NADAM,
		AMSGRAD,
		BFGS,
		RPROP,
		CG
	};
	//constructor
	Name():t_(Type::UNKNOWN){}
	Name(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Name read(const char* str);
	static const char* name(const Name& name);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Name& name);

class Base{
protected:
	int dim_;
	Name name_;
public:
	//==== constructors/destructors ====
	Base():name_(Name::UNKNOWN),dim_(0){}
	Base(Name name):name_(name),dim_(0){}
	virtual ~Base(){}
	
	//==== access ====
	const int& dim()const{return dim_;}
	const Name& name()const{return name_;}
	
	//==== member functions ====
	virtual void read(Token& token){};
	virtual void resize(int dim);
	virtual void step(Objective& obj)=0;
};

//***************************************************
// SGD
//***************************************************

class SGD: public Base{
private:
public:
	//==== constructors/destructors ====
	SGD():Base(Name::SGD){}
	~SGD(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const SGD& sgd);
	
	//==== member functions ====
	void read(Token& token){};
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// SDM
//***************************************************

class SDM: public Base{
private:
	double eta_;
	Eigen::VectorXd dx_;
public:
	//==== constructors/destructors ====
	SDM():Base(Name::SDM),eta_(0.9){}
	~SDM(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const SDM& sdm);
	
	//==== access ====
	double& eta(){return eta_;}
	const double& eta()const{return eta_;}
	Eigen::VectorXd& dx(){return dx_;}
	const Eigen::VectorXd& dx()const{return dx_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// NAG
//***************************************************

class NAG: public Base{
private:
	double eta_;
	Eigen::VectorXd dx_;
public:
	//==== constructors/destructors ====
	NAG():Base(Name::NAG),eta_(0.9){}
	~NAG(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NAG& nag);
	
	//==== access ====
	double& eta(){return eta_;}
	const double& eta()const{return eta_;}
	Eigen::VectorXd& dx(){return dx_;}
	const Eigen::VectorXd& dx()const{return dx_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// ADAGRAD
//***************************************************

class ADAGRAD: public Base{
private:
	double eps_;
	Eigen::VectorXd mgrad2_;
public:
	//==== constructors/destructors ====
	ADAGRAD():Base(Name::ADAGRAD),eps_(1.0e-16){}
	~ADAGRAD(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ADAGRAD& adagrad);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// ADADELTA
//***************************************************

class ADADELTA: public Base{
private:
	double eta_;
	double eps_;
	Eigen::VectorXd mgrad2_;
	Eigen::VectorXd mdx2_;
	Eigen::VectorXd dx_;
public:
	//==== constructors/destructors ====
	ADADELTA():Base(Name::ADADELTA),eta_(0.9),eps_(1.0e-16){}
	~ADADELTA(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ADADELTA& adadelta);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& eta(){return eta_;}
	const double& eta()const{return eta_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	Eigen::VectorXd& mdx2(){return mdx2_;}
	Eigen::VectorXd& dx(){return dx_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	const Eigen::VectorXd& mdx2()const{return mdx2_;}
	const Eigen::VectorXd& dx()const{return dx_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// RMSPROP
//***************************************************

class RMSPROP: public Base{
private:
	double eps_;
	Eigen::VectorXd mgrad2_;
public:
	//==== constructors/destructors ====
	RMSPROP():Base(Name::RMSPROP),eps_(1.0e-16){}
	~RMSPROP(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const RMSPROP& rmsprop);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// ADAM
//***************************************************

class ADAM: public Base{
private:
	static const double beta1_,beta2_;
	double beta1i_,beta2i_;
	double eps_;
	Eigen::VectorXd mgrad_;
	Eigen::VectorXd mgrad2_;
public:
	//==== constructors/destructors ====
	ADAM():Base(Name::ADAM),eps_(1.0e-16){}
	~ADAM(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ADAM& adam);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	Eigen::VectorXd& mgrad(){return mgrad_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad()const{return mgrad_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// ADAMW
//***************************************************

class ADAMW: public Base{
private:
	static const double beta1_,beta2_;
	double beta1i_,beta2i_;
	double eps_,w_;
	Eigen::VectorXd mgrad_;
	Eigen::VectorXd mgrad2_;
public:
	//==== constructors/destructors ====
	ADAMW():Base(Name::ADAMW),eps_(1.0e-16),w_(1.0e-3){}
	~ADAMW(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ADAMW& adamw);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	double& w(){return w_;}
	const double& w()const{return w_;}
	Eigen::VectorXd& mgrad(){return mgrad_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad()const{return mgrad_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// ADAB
//***************************************************

class ADAB: public Base{
private:
	static const double beta1_,beta2_;
	double beta1i_,beta2i_;
	double eps_;
	Eigen::VectorXd mgrad_;
	Eigen::VectorXd mgrad2_;
public:
	//==== constructors/destructors ====
	ADAB():Base(Name::ADAB),eps_(1.0e-16){}
	~ADAB(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const ADAB& adab);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	Eigen::VectorXd& mgrad(){return mgrad_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad()const{return mgrad_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// YOGI
//***************************************************

class YOGI: public Base{
private:
	static const double beta1_,beta2_;
	double beta1i_,beta2i_;
	double eps_;
	Eigen::VectorXd mgrad_;
	Eigen::VectorXd mgrad2_;
public:
	//==== constructors/destructors ====
	YOGI():Base(Name::YOGI),eps_(1.0e-16){}
	~YOGI(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const YOGI& yogi);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	Eigen::VectorXd& mgrad(){return mgrad_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad()const{return mgrad_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// NOGI
//***************************************************

class NOGI: public Base{
private:
	static const double beta1_,beta2_;
	double beta1i_,beta2i_;
	double eps_;
	Eigen::VectorXd mgrad_;
	Eigen::VectorXd mgrad2_;
public:
	//==== constructors/destructors ====
	NOGI():Base(Name::NOGI),eps_(1.0e-16){}
	~NOGI(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NOGI& yoni);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	Eigen::VectorXd& mgrad(){return mgrad_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad()const{return mgrad_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// NADAM
//***************************************************

class NADAM: public Base{
private:
	static const double beta1_,beta2_;
	double beta1i_,beta2i_;
	double eps_,eps2_;
	Eigen::VectorXd mgrad_;
	Eigen::VectorXd mgrad2_;
public:
	//==== constructors/destructors ====
	NADAM():Base(Name::NADAM),eps_(1.0e-16){}
	~NADAM(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const NADAM& nadam);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	Eigen::VectorXd& mgrad(){return mgrad_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	const Eigen::VectorXd& mgrad()const{return mgrad_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// AMSGRAD
//***************************************************

class AMSGRAD: public Base{
private:
	static const double beta1_,beta2_;
	double beta1i_,beta2i_;
	double eps_;
	Eigen::VectorXd mgrad_;
	Eigen::VectorXd mgrad2_;
	Eigen::VectorXd mgrad2m_;
public:
	//==== constructors/destructors ====
	AMSGRAD():Base(Name::AMSGRAD),eps_(1.0e-16){}
	~AMSGRAD(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const AMSGRAD& amsgrad);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	Eigen::VectorXd& mgrad(){return mgrad_;}
	Eigen::VectorXd& mgrad2(){return mgrad2_;}
	Eigen::VectorXd& mgrad2m(){return mgrad2m_;}
	const Eigen::VectorXd& mgrad()const{return mgrad_;}
	const Eigen::VectorXd& mgrad2()const{return mgrad2_;}
	const Eigen::VectorXd& mgrad2m()const{return mgrad2m_;}
	
	//==== member functions ====
	void read(Token& token);
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// BFGS
//***************************************************

class BFGS: public Base{
private:
	Eigen::MatrixXd B_,BOld_;
	Eigen::VectorXd s_,y_;
public:
	//==== constructors/destructors ====
	BFGS():Base(Name::BFGS){}
	~BFGS(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const BFGS& bfgs);
	
	//==== access ====
	Eigen::MatrixXd& B(){return B_;}
	Eigen::MatrixXd& BOld(){return BOld_;}
	Eigen::VectorXd& s(){return s_;}
	Eigen::VectorXd& y(){return y_;}
	const Eigen::MatrixXd& B()const{return B_;}
	const Eigen::MatrixXd& BOld()const{return BOld_;}
	const Eigen::VectorXd& s()const{return s_;}
	const Eigen::VectorXd& y()const{return y_;}
	
	//==== member functions ====
	void read(Token& token){};
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// RPROP
//***************************************************

class RPROP: public Base{
private:
	static const double etaP;
	static const double etaM;
	static const double deltaMax;
	static const double deltaMin;
	Eigen::VectorXd delta_;
	Eigen::VectorXd dx_;
public:
	//==== constructors/destructors ====
	RPROP():Base(Name::RPROP){}
	~RPROP(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const RPROP& rprop);
	
	//==== access ====
	Eigen::VectorXd& dx(){return dx_;}
	Eigen::VectorXd& delta(){return delta_;}
	const Eigen::VectorXd& dx()const{return dx_;}
	const Eigen::VectorXd& delta()const{return delta_;}
	
	//==== member functions ====
	void read(Token& token){};
	void resize(int dim);
	void step(Objective& obj);
};

//***************************************************
// CG
//***************************************************

class CG: public Base{
private:
	double eps_;
	Eigen::VectorXd cgd_;//cg direction
public:
	//==== constructors/destructors ====
	CG():Base(Name::CG){}
	~CG(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const RPROP& rprop);
	
	//==== access ====
	double& eps(){return eps_;}
	const double& eps()const{return eps_;}
	Eigen::VectorXd& cgd(){return cgd_;}
	const Eigen::VectorXd& cgd()const{return cgd_;}
	
	//==== member functions ====
	void read(Token& token){};
	void resize(int dim);
	void step(Objective& obj);
};


//***************************************************
// Factory
//***************************************************

std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Base>& obj);
std::shared_ptr<Base>& make(std::shared_ptr<Base>& obj, Name name);
std::shared_ptr<Base>& read(std::shared_ptr<Base>& obj, Token& token);

}
}

namespace serialize{

	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const std::shared_ptr<opt::algo::Base>& obj);
	template <> int nbytes(const opt::algo::Base& obj);
	template <> int nbytes(const opt::algo::SGD& obj);
	template <> int nbytes(const opt::algo::SDM& obj);
	template <> int nbytes(const opt::algo::NAG& obj);
	template <> int nbytes(const opt::algo::ADAGRAD& obj);
	template <> int nbytes(const opt::algo::ADADELTA& obj);
	template <> int nbytes(const opt::algo::RMSPROP& obj);
	template <> int nbytes(const opt::algo::ADAM& obj);
	template <> int nbytes(const opt::algo::ADAMW& obj);
	template <> int nbytes(const opt::algo::ADAB& obj);
	template <> int nbytes(const opt::algo::YOGI& obj);
	template <> int nbytes(const opt::algo::NOGI& obj);
	template <> int nbytes(const opt::algo::NADAM& obj);
	template <> int nbytes(const opt::algo::AMSGRAD& obj);
	template <> int nbytes(const opt::algo::BFGS& obj);
	template <> int nbytes(const opt::algo::RPROP& obj);
	template <> int nbytes(const opt::algo::CG& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const std::shared_ptr<opt::algo::Base>& obj, char* arr);
	template <> int pack(const opt::algo::Base& obj, char* arr);
	template <> int pack(const opt::algo::SGD& obj, char* arr);
	template <> int pack(const opt::algo::SDM& obj, char* arr);
	template <> int pack(const opt::algo::NAG& obj, char* arr);
	template <> int pack(const opt::algo::ADAGRAD& obj, char* arr);
	template <> int pack(const opt::algo::ADADELTA& obj, char* arr);
	template <> int pack(const opt::algo::RMSPROP& obj, char* arr);
	template <> int pack(const opt::algo::ADAM& obj, char* arr);
	template <> int pack(const opt::algo::ADAMW& obj, char* arr);
	template <> int pack(const opt::algo::ADAB& obj, char* arr);
	template <> int pack(const opt::algo::YOGI& obj, char* arr);
	template <> int pack(const opt::algo::NOGI& obj, char* arr);
	template <> int pack(const opt::algo::NADAM& obj, char* arr);
	template <> int pack(const opt::algo::AMSGRAD& obj, char* arr);
	template <> int pack(const opt::algo::BFGS& obj, char* arr);
	template <> int pack(const opt::algo::RPROP& obj, char* arr);
	template <> int pack(const opt::algo::CG& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(std::shared_ptr<opt::algo::Base>& obj, const char* arr);
	template <> int unpack(opt::algo::Base& obj, const char* arr);
	template <> int unpack(opt::algo::SGD& obj, const char* arr);
	template <> int unpack(opt::algo::SDM& obj, const char* arr);
	template <> int unpack(opt::algo::NAG& obj, const char* arr);
	template <> int unpack(opt::algo::ADAGRAD& obj, const char* arr);
	template <> int unpack(opt::algo::ADADELTA& obj, const char* arr);
	template <> int unpack(opt::algo::RMSPROP& obj, const char* arr);
	template <> int unpack(opt::algo::ADAM& obj, const char* arr);
	template <> int unpack(opt::algo::ADAMW& obj, const char* arr);
	template <> int unpack(opt::algo::ADAB& obj, const char* arr);
	template <> int unpack(opt::algo::YOGI& obj, const char* arr);
	template <> int unpack(opt::algo::NOGI& obj, const char* arr);
	template <> int unpack(opt::algo::NADAM& obj, const char* arr);
	template <> int unpack(opt::algo::AMSGRAD& obj, const char* arr);
	template <> int unpack(opt::algo::BFGS& obj, const char* arr);
	template <> int unpack(opt::algo::RPROP& obj, const char* arr);
	template <> int unpack(opt::algo::CG& obj, const char* arr);
	
}

#endif