#pragma once
#ifndef COMPUTE_STATE_PE_HPP
#define COMPUTE_STATE_PE_CPP

namespace compute{

namespace state{

class PE: public Base{
private:
public:
	//==== constructors/destructors ====
	PE():Base(Name::PE){}
	~PE(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const PE& pe);
	
	//==== member functions ====
	void read(Token& token){}
	double compute(const Struc& struc);
};

}

}

#endif