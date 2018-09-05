#include "symm_radial_g2.hpp"

//Behler G2
double PhiR_G2::operator()(double r)const noexcept{
	return std::exp(-eta*(r-rs)*(r-rs))*CutoffF::funcs[tcut](r,rc);
}
double PhiR_G2::val(double r)const noexcept{
	return std::exp(-eta*(r-rs)*(r-rs))*CutoffF::funcs[tcut](r,rc);
}
double PhiR_G2::amp(double r)const noexcept{
	return std::exp(-eta*(r-rs)*(r-rs));
}
double PhiR_G2::cut(double r)const noexcept{
	return CutoffF::funcs[tcut](r,rc);
}
double PhiR_G2::grad(double r)const noexcept{
	return std::exp(-eta*(r-rs)*(r-rs))*(-2.0*eta*(r-rs)*CutoffF::funcs[tcut](r,rc)+CutoffFD::funcs[tcut](r,rc));
}
std::ostream& operator<<(std::ostream& out, const PhiR_G2& f){
	return out<<static_cast<const PhiR&>(f)<<" G2 "<<f.eta<<" "<<f.rs<<" ";
}
