// c libraries
#include <cstdlib>
//c++
#include <iostream>

//***********************************************************************
// OPERATION
//***********************************************************************

class Operation{
public:
	//enum
	enum Type{
		UNKNOWN,
		SORT,
		ROTATE,
		TRANSLATE,
		DILATE,
		REPLICATE,
		BOX
	};
	//constructor
	Operation():t_(Type::UNKNOWN){}
	Operation(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
	//member functions
	static Operation read(const char* str);
	static const char* name(const Operation& op);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Operation& op);

//***********************************************************************
// SORT
//***********************************************************************

class Sort{
public:
	enum Type{
		UNKNOWN,
		MOLECULE,
		TYPE
	};
	//constructor
	Sort():t_(Type::UNKNOWN){}
	Sort(Type t):t_(t){}
	//operators
	operator Type()const{return t_;}
//member functions
	static Sort read(const char* str);
	static const char* name(const Sort& sort);
private:
	Type t_;
	//prevent automatic conversion for other built-in types
	//template<typename T> operator T() const;
};
std::ostream& operator<<(std::ostream& out, const Sort& sys);

//***********************************************************************
// Modify
//***********************************************************************

class Modify{
private:
public:
	virtual Structure& mod(Structure& struc)const=0;
};

//***********************************************************************
// Translation
//***********************************************************************

class Translate: public Modify{
private:
	Eigen::Vector3d dr_;
public:
	Eigen::Vector3d& dr(){return dr_;}
	const Eigen::Vector3d dr()const{return dr_;}
	Structure& mod(Structure& struc)const;
};

//***********************************************************************
// Rotation
//***********************************************************************

struct Rotate: public Modify{
private:
	double angle_;
	Eigen::Vector3d origin_;
	Eigen::Vector3d axis_;
public:
	double& angle(){return angle_;}
	const double& angle()const{return angle_;}
	Eigen::Vector3d& origin(){return origin_;}
	const Eigen::Vector3d& origin()const{return origin_;}
	Eigen::Vector3d& axis(){return axis_;}
	const Eigen::Vector3d& axis()const{return axis_;}
	Structure& mod(Structure& struc)const;
};

//***********************************************************************
// Dilation
//***********************************************************************

struct Dilate: public Modify{
private:
	double s_;
public:
	double& s(){return s_;}
	const double& s()const{return s_;}
	Structure& mod(Structure& struc)const;
};

//***********************************************************************
// Replication
//***********************************************************************

struct Replicate: public Modify{
private:
	Eigen::Vector3i s_;
public:
	Eigen::Vector3i& s(){return s_;}
	const Eigen::Vector3i& s()const{return s_;}
	Structure& mod(Structure& struc)const;
};

//***********************************************************************
// Box
//***********************************************************************

struct Box: public Modify{
private:
	Eigen::Vector3d c1_;//bottom corner
	Eigen::Vector3d c2_;//top corner
public:
	Eigen::Vector3d& c1(){return c1_;}
	const Eigen::Vector3d& c1()const{return c1_;}
	Eigen::Vector3d& c2(){return c2_;}
	const Eigen::Vector3d& c2()const{return c2_;}
	Structure& mod(Structure& struc)const;
};