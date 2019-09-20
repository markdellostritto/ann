#pragma once
#ifndef MATH_CMP_HPP
#define MATH_CMP_HPP

namespace cmp{
	
	template <class T> inline T max(T x1, T x2)noexcept{return (x1>x2)?x1:x2;}
	template <class T> inline T min(T x1, T x2)noexcept{return (x1>x2)?x2:x1;}
	template <class T> inline unsigned int delta(const T& x1, const T& x2)noexcept{return (x1==x2);}
	
};

#endif