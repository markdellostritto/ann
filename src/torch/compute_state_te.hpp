#pragma once
#ifndef COMPUTE_STATE_TE_HPP
#define COMPUTE_STATE_TE_CPP

namespace compute{

namespace state{

class TE: public Base{
private:
public:
	//==== constructors/destructors ====
	TE():Base(Name::TE){}
	~TE(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const TE& te);
	
	//==== member functions ====
	void read(Token& token){}
	double compute(const Struc& struc);
};

}

}

#endif