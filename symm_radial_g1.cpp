#include "symm_radial_g1.hpp"

//Behler G1
double PhiR_G1::operator()(double r)const noexcept{
	return CutoffF::funcs[tcut](r,rc);
}
double PhiR_G1::val(double r)const noexcept{
	return CutoffF::funcs[tcut](r,rc);
}
double PhiR_G1::amp(double r)const noexcept{
	return 1;
}
double PhiR_G1::cut(double r)const noexcept{
	return CutoffF::funcs[tcut](r,rc);
}
double PhiR_G1::grad(double r)const noexcept{
	return CutoffFD::funcs[tcut](r,rc);
}
std::ostream& operator<<(std::ostream& out, const PhiR_G1& f){
	return out<<static_cast<const PhiR&>(f)<<" G1";
}
