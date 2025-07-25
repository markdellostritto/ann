#pragma once
#ifndef LOSS_HPP
#define LOSS_HPP

// c++
#include <iosfwd>
// eigen
#include <Eigen/Dense>

//***************************************************
// loss function
//***************************************************

namespace opt{

struct Loss{
public:
	enum Type{
		UNKNOWN,
		MSE,
		MAE,
		HUBER,
		ASINH
	};
	//constructor
	Loss():t_(Type::UNKNOWN){}
	Loss(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Loss read(const char* str);
	static const char* name(const Loss& loss);
	//error
	static double error(Loss loss, const Eigen::VectorXd& value, const Eigen::VectorXd& target);
	static double error2(Loss loss, double delta, const Eigen::VectorXd& value, const Eigen::VectorXd& target);
	static double error(Loss loss, const Eigen::VectorXd& value, const Eigen::VectorXd& target, Eigen::VectorXd& grad);
	static double error2(Loss loss, double delta, const Eigen::VectorXd& value, const Eigen::VectorXd& target, Eigen::VectorXd& grad);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Loss& loss);

}

#endif 