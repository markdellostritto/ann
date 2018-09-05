#ifndef MATH_CMP_HPP
#define MATH_CMP_HPP

namespace cmp{
	
	template <class T> T max(T x1, T x2){return (x1>x2)?x1:x2;}
	template <class T> T min(T x1, T x2){return (x1>x2)?x2:x1;}
	template <class T> unsigned int delta(const T& x1, const T& x2){return (x1==x2);}
	
};

#endif