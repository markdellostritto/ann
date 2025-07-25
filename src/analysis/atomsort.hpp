#ifndef ATOM_SORT_HPP
#define ATOM_SORT_HPP

class Sort{
public:
	enum Type{
		MOLECULE,
		UNKNOWN
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
	
Structure& sort_mol(Structure& struc);

#endif