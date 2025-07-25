#pragma once
#ifndef COMPUTE_STATE_KE_HPP
#define COMPUTE_STATE_KE_CPP

namespace compute{

namespace state{

class KE: public Base{
private:
public:
	//==== constructors/destructors ====
	KE():Base(Name::KE){}
	~KE(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const KE& ke);
	
	//==== member functions ====
	void read(Token& token){}
	double compute(const Struc& struc);
};

}

}

#endif